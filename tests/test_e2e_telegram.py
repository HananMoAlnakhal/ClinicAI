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
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import select

from bot.handlers import doctor as doctor_handler
from bot.handlers import patient as patient_handler
from database import crud
from database.models import Appointment, Patient, Session as DoctorSession, Slot
from fsm.patient_fsm import PatientFSM, State
from nlp.gemini_client import gemini
from tests.helpers import make_test_engine, make_test_session, run_async, seed_doctor, seed_patient, seed_slot, use_test_db, use_test_db
from tests.telegram_mocks import last_reply_text, make_context, make_start_update, make_text_update
from utils.datetime_utils import utcnow


PATIENT_ID = 910_001
DOCTOR_TG_ID = 920_001


def _clear_patient_session(db, user_id: int = PATIENT_ID) -> None:
    crud.delete_fsm_session(db, user_id, role="patient")


class TestPatientBookingE2E(unittest.TestCase):
    """Full patient journey: /start → data collection → confirm → DB appointment."""

    def setUp(self):
        self.engine = make_test_engine()
        self.db = make_test_session(self.engine)
        _clear_patient_session(self.db)
        self.doctor = seed_doctor(
            self.db,
            specialty="general_practice",
            clinic_code="CLINIC-E2E-GP",
            clinic_name="عيادة E2E",
        )
        self.slot_when = (utcnow() + timedelta(days=2)).replace(hour=10, minute=0, second=0, microsecond=0)
        self.slot = seed_slot(self.db, self.doctor, when=self.slot_when, priority_class="P3")
        self.db.commit()

    def tearDown(self):
        self.db.close()

    def _send(self, text: str):
        update, message = make_text_update(PATIENT_ID, text)
        with use_test_db(self.db):
            run_async(patient_handler.handle_text(update, make_context()))
        return message

    @patch("scheduler.classifier.classify_specialty", side_effect=lambda *a, **k: {
        "specialty": "general_practice",
        "specialty_ar": "الطب العام",
        "method": "rule",
        "confidence": 0.95,
    })
    @patch.object(gemini, "_available", False)
    @patch("bot.handlers.patient.TTS_ENABLED", False)
    def test_full_booking_flow_creates_confirmed_appointment(self, _mock_classify):
        update, _ = make_start_update(PATIENT_ID)
        with use_test_db(self.db):
            run_async(patient_handler.handle_start(update, make_context()))

        self._send("أحمد محمود")
        self._send("أحمد محمود")  # save name → COLLECT_COMPLAINT
        self._send("صداع خفيف من يومين")
        self._send("🟢 روتيني / عادي")
        confirm_msg = self._send("بكرا")
        self.assertIn("وجدت موعد", last_reply_text(confirm_msg))

        with use_test_db(self.db):
            fsm = patient_handler._get_fsm(PATIENT_ID)
        self.assertEqual(fsm.state, State.CONFIRM)

        final_msg = self._send("✅ تأكيد الحجز")
        self.assertIn("تم تأكيد حجزك", last_reply_text(final_msg))
        with use_test_db(self.db):
            fsm = patient_handler._get_fsm(PATIENT_ID)
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

    @patch.object(gemini, "_available", False)
    @patch("bot.handlers.patient.TTS_ENABLED", False)
    def test_confirm_cancel_does_not_cancel_db_appointment(self):
        """❌ إلغاء in CONFIRM must not call cancel_latest_patient_appointment."""
        patient = seed_patient(self.db, telegram_id=PATIENT_ID, name="ليلى")
        existing_slot = seed_slot(
            self.db,
            self.doctor,
            when=(utcnow() + timedelta(days=5)).replace(hour=12, minute=0, second=0, microsecond=0),
            priority_class="P3",
            status="booked",
        )
        self.db.add(
            Appointment(
                appt_id="appt-existing-e2e",
                patient_id=patient.patient_id,
                slot_id=existing_slot.slot_id,
                appt_datetime=existing_slot.slot_datetime,
                specialty="general_practice",
                status="confirmed",
            )
        )
        self.db.commit()
        _clear_patient_session(self.db)

        fsm = PatientFSM(user_id=PATIENT_ID)
        fsm.state = State.CONFIRM
        fsm.data = {
            "name": "ليلى",
            "complaint": {"raw": "صداع"},
            "urgency_score": 0.3,
            "time_pref": {"date": str(self.slot_when.date()), "phrase": "بكرا"},
            "specialty_hint": "general_practice",
            "specialty_ar": "الطب العام",
            "priority_class": "P3",
        }
        fsm.slot = {
            "slot_id": self.slot.slot_id,
            "slot_datetime": self.slot_when,
            "doctor_name": "د. اختبار",
            "clinic_name": "عيادة E2E",
        }
        from scheduler.priority import PriorityResult

        fsm.priority = PriorityResult(score=0.3, priority_class="P3", label_ar="روتيني", label_color="green", breakdown={})
        with use_test_db(self.db):
            crud.upsert_fsm_session(
                self.db,
                PATIENT_ID,
                role="patient",
                state=fsm.state.name,
                data_json=fsm.data,
                slot_json={
                    **fsm.slot,
                    "slot_datetime": fsm.slot["slot_datetime"].isoformat(),
                },
                priority_json={"priority_class": "P3", "score": 0.3, "label_ar": "روتيني", "breakdown": {}},
            )

        with patch.object(crud, "cancel_latest_patient_appointment") as mock_cancel:
            msg = self._send("❌ إلغاء")
            mock_cancel.assert_not_called()

        self.assertIn("تم الإلغاء", last_reply_text(msg))
        with use_test_db(self.db):
            reloaded = patient_handler._get_fsm(PATIENT_ID)
        self.assertEqual(reloaded.state, State.CANCELLED)

        appt = self.db.scalar(
            select(Appointment).where(Appointment.appt_id == "appt-existing-e2e")
        )
        self.assertEqual(appt.status, "confirmed")

    @patch.object(gemini, "_available", False)
    @patch("bot.handlers.patient.TTS_ENABLED", False)
    def test_session_persists_across_handler_reload(self):
        update, _ = make_start_update(PATIENT_ID)
        with use_test_db(self.db):
            run_async(patient_handler.handle_start(update, make_context()))
        self._send("سارة محمود")
        self._send("سارة محمود")
        with use_test_db(self.db):
            row = crud.get_fsm_session(self.db, PATIENT_ID, role="patient")
        self.assertIsNotNone(row)
        self.assertEqual(row.state, State.COLLECT_COMPLAINT.name)

        with use_test_db(self.db):
            reloaded = patient_handler._get_fsm(PATIENT_ID)
        self.assertEqual(reloaded.state, State.COLLECT_COMPLAINT)
        self.assertEqual(reloaded.data.get("name"), "سارة محمود")


class TestDoctorSessionE2E(unittest.TestCase):
    """Doctor journey: session note → review → confirm → linked clinical session."""

    def setUp(self):
        self.engine = make_test_engine()
        self.db = make_test_session(self.engine)
        crud.delete_fsm_session(self.db, DOCTOR_TG_ID, role="doctor")
        self.doctor = seed_doctor(
            self.db,
            specialty="general_practice",
            clinic_code="CLINIC-E2E-DOC",
            telegram_id=DOCTOR_TG_ID,
            name="د. E2E",
        )
        self.patient = seed_patient(self.db, telegram_id=880_001, name="ليلى أبو علي")
        self.db.commit()

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

        with use_test_db(self.db):
            fsm = doctor_handler._get_fsm(self.doctor)
        from fsm.doctor_fsm import DoctorState

        self.assertEqual(fsm.state, DoctorState.REVIEW)

        save_msg = self._send("تأكيد")
        self.assertIn("تم حفظ الجلسة", last_reply_text(save_msg))
        with use_test_db(self.db):
            row = crud.get_fsm_session(self.db, DOCTOR_TG_ID, role="doctor")
        self.assertEqual(row.state, DoctorState.SAVED.name)

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
        _clear_patient_session(self.db)
        crud.delete_fsm_session(self.db, DOCTOR_TG_ID, role="doctor")
        seed_doctor(self.db, clinic_code="CLINIC-RT-DOC", telegram_id=DOCTOR_TG_ID)
        self.db.commit()

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
