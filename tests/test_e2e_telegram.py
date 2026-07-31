"""
End-to-end tests through Telegram handler mocks.

Exercises the real handler → FSM → CRUD stack on an in-memory SQLite DB.
"""
from __future__ import annotations

import os
import sys
import unittest
from datetime import timedelta
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import select

from bot.handlers import doctor as doctor_handler
from bot.handlers import patient as patient_handler
from database import crud
from database.models import Appointment, Doctor, Patient, Session as DoctorSession, Slot
from fsm.patient_fsm import State
from nlp.gemini_client import gemini
from tests.helpers import make_test_engine, make_test_session, run_async, seed_doctor, seed_patient, seed_slot, use_test_db
from tests.telegram_mocks import last_reply_text, make_context, make_start_update, make_text_update
from utils.datetime_utils import utcnow


PATIENT_ID = 910_001
DOCTOR_TG_ID = 920_001


class TestPatientBookingE2E(unittest.TestCase):
    """Full patient journey: /start → data collection → confirm → DB appointment."""

    def setUp(self):
        self.engine = make_test_engine()
        self.db = make_test_session(self.engine)
        self.doctor = seed_doctor(
            self.db,
            specialty="general_practice",
            clinic_code="CLINIC-E2E-GP",
            clinic_name="عيادة E2E",
        )
        self.slot_when = (utcnow() + timedelta(days=2)).replace(hour=10, minute=0, second=0, microsecond=0)
        self.slot = seed_slot(self.db, self.doctor, when=self.slot_when, priority_class="P3")
        self.db.commit()
        patient_handler._sessions.clear()

    def tearDown(self):
        self.db.close()

    def _send(self, text: str):
        update, message = make_text_update(PATIENT_ID, text)
        with use_test_db(self.db):
            run_async(patient_handler.handle_text(update, make_context()))
        return message

    @patch.object(gemini, "_available", False)
    @patch("bot.handlers.patient.TTS_ENABLED", False)
    def test_full_booking_flow_creates_confirmed_appointment(self):
        update, _ = make_start_update(PATIENT_ID)
        with use_test_db(self.db):
            run_async(patient_handler.handle_start(update, make_context()))

        self._send("أحمد محمود")  # GREETING → COLLECT_NAME
        self._send("أحمد محمود")  # save name → COLLECT_COMPLAINT
        self._send("عندي زكام بسيط من يومين")
        self._send("🟢 روتيني / عادي")
        confirm_msg = self._send("بكرا")
        self.assertIn("وجدت موعد", last_reply_text(confirm_msg))

        fsm = patient_handler._sessions[PATIENT_ID]
        self.assertEqual(fsm.state, State.CONFIRM)

        final_msg = self._send("✅ تأكيد الحجز")
        self.assertIn("تم تأكيد حجزك", last_reply_text(final_msg))
        self.assertEqual(fsm.state, State.FINALIZED)

        appt = self.db.scalar(
            select(Appointment)
            .join(Patient, Appointment.patient_id == Patient.patient_id)
            .where(Patient.telegram_id == PATIENT_ID)
            .order_by(Appointment.created_at.desc())
        )
        self.assertIsNotNone(appt)
        self.assertEqual(appt.status, "confirmed")
        self.assertEqual(appt.slot_id, self.slot.slot_id)

        booked_slot = self.db.get(Slot, self.slot.slot_id)
        self.assertEqual(booked_slot.status, "booked")

        profile = crud.get_profile(self.db, PATIENT_ID)
        self.assertEqual(profile.get("name"), "أحمد محمود")


class TestDoctorSessionE2E(unittest.TestCase):
    """Doctor journey: session note → review → confirm → linked clinical session."""

    def setUp(self):
        self.engine = make_test_engine()
        self.db = make_test_session(self.engine)
        self.doctor = seed_doctor(
            self.db,
            specialty="general_practice",
            clinic_code="CLINIC-E2E-DOC",
            telegram_id=DOCTOR_TG_ID,
            name="د. E2E",
        )
        self.patient = seed_patient(self.db, telegram_id=880_001, name="ليلى أبو علي")
        self.db.commit()
        doctor_handler._sessions.clear()

    def tearDown(self):
        self.db.close()

    def _send(self, text: str):
        update, message = make_text_update(DOCTOR_TG_ID, text)
        with use_test_db(self.db):
            run_async(doctor_handler.handle_doctor_text(update, make_context()))
        return message

    @patch("bot.handlers.patient.TTS_ENABLED", False)
    def test_doctor_session_note_saved_and_linked_to_patient(self):
        note = (
            "المريض اسمه ليلى أبو علي، شاكي من سعال، "
            "التشخيص التهاب bronchi، ايبوبروفين 400 مرتين، متابعة بعد اسبوع"
        )
        self._send("/session")
        review_msg = self._send(note)
        self.assertIn("ملخص الجلسة", last_reply_text(review_msg))

        fsm = doctor_handler._sessions[DOCTOR_TG_ID]
        from fsm.doctor_fsm import DoctorState

        self.assertEqual(fsm.state, DoctorState.REVIEW)

        save_msg = self._send("تأكيد")
        self.assertIn("تم حفظ الجلسة", last_reply_text(save_msg))
        self.assertEqual(fsm.state, DoctorState.SAVED)

        session = self.db.scalar(
            select(DoctorSession)
            .where(DoctorSession.doctor_id == self.doctor.doctor_id)
            .order_by(DoctorSession.session_id.desc())
        )
        self.assertIsNotNone(session)
        self.assertEqual(session.patient_id, self.patient.patient_id)
        self.assertEqual(session.patient_name, "ليلى أبو علي")
        self.assertIsNotNone(session.chief_complaint)


class TestRouterE2E(unittest.TestCase):
    """Router sends doctors vs patients to the correct handler."""

    def setUp(self):
        self.engine = make_test_engine()
        self.db = make_test_session(self.engine)
        seed_doctor(self.db, clinic_code="CLINIC-RT-DOC", telegram_id=DOCTOR_TG_ID)
        self.db.commit()
        patient_handler._sessions.clear()
        doctor_handler._sessions.clear()

    def tearDown(self):
        self.db.close()

    @patch.object(gemini, "_available", False)
    @patch("bot.handlers.patient.TTS_ENABLED", False)
    def test_route_start_patient_gets_booking_greeting(self):
        from bot.router import route_start

        update, message = make_start_update(PATIENT_ID)
        with use_test_db(self.db):
            run_async(route_start(update, make_context()))
        self.assertIn("مساعد الحجز", last_reply_text(message))

    @patch("bot.handlers.patient.TTS_ENABLED", False)
    def test_route_start_doctor_gets_doctor_greeting(self):
        from bot.router import route_start

        update, message = make_start_update(DOCTOR_TG_ID)
        with use_test_db(self.db):
            run_async(route_start(update, make_context()))
        self.assertIn("دكتور", last_reply_text(message))


if __name__ == "__main__":
    unittest.main(verbosity=2)
