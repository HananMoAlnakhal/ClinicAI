"""Unit tests for database/crud.py — patients, slots, booking, sessions."""
import os
import sys
import unittest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database import crud
from database.models import Appointment, PatientProfile, Slot
from tests.helpers import make_test_session, seed_doctor, seed_patient, seed_slot
from utils.datetime_utils import utcnow


def _booking_data(**overrides):
    base = {
        "name": "أحمد محمود",
        "complaint": {"raw": "صداع", "category": "general", "urgency_score": 0.3},
        "urgency_score": 0.4,
        "time_pref": {"date": None, "phrase": "أي وقت"},
        "specialty_hint": "general_practice",
        "specialty_ar": "الطب العام",
        "priority_class": "P3",
        "priority_score": 0.35,
    }
    base.update(overrides)
    return base


class TestPatientCrud(unittest.TestCase):
    def setUp(self):
        self.db = make_test_session()

    def tearDown(self):
        self.db.close()

    def test_get_or_create_patient_creates_then_updates_name(self):
        p1 = crud.get_or_create_patient(self.db, telegram_id=1001, name="أحمد")
        self.db.commit()
        p2 = crud.get_or_create_patient(self.db, telegram_id=1001, name="أحمد محمود")
        self.db.commit()

        self.assertEqual(p1.patient_id, p2.patient_id)
        self.assertEqual(p2.name, "أحمد محمود")

    def test_search_patient_by_partial_name(self):
        crud.get_or_create_patient(self.db, telegram_id=1002, name="سارة خالد")
        self.db.commit()
        results = crud.search_patient(self.db, "سارة")
        self.assertEqual(len(results), 1)
        self.assertIn("سارة", results[0].name)


class TestSlotFinding(unittest.TestCase):
    def setUp(self):
        self.db = make_test_session()
        self.doctor = seed_doctor(self.db, specialty="general_practice", clinic_code="CLINIC-GP-CRUD")
        self.future = utcnow() + timedelta(days=2)
        self.slot = seed_slot(self.db, self.doctor, when=self.future.replace(hour=9, minute=0), priority_class="P3")

    def tearDown(self):
        self.db.close()

    def test_find_next_available_slot_returns_future_slot(self):
        found = crud.find_next_available_slot(
            self.db,
            specialty="general_practice",
            priority_class="P3",
            telegram_id=9999,
        )
        self.assertIsNotNone(found)
        self.assertEqual(found.slot_id, self.slot.slot_id)

    def test_find_slot_skips_booked_slots(self):
        self.slot.status = "booked"
        self.db.commit()
        found = crud.find_next_available_slot(
            self.db,
            specialty="general_practice",
            priority_class="P3",
        )
        self.assertIsNone(found)


class TestBookingFlow(unittest.TestCase):
    def setUp(self):
        self.db = make_test_session()
        self.doctor = seed_doctor(self.db, specialty="general_practice", clinic_code="CLINIC-GP-BOOK")
        self.when = (utcnow() + timedelta(days=3)).replace(hour=11, minute=0, second=0, microsecond=0)
        self.slot = seed_slot(self.db, self.doctor, when=self.when, priority_class="P3")

    def tearDown(self):
        self.db.close()

    def test_create_patient_file_and_book_success(self):
        result = crud.create_patient_file_and_book(
            self.db,
            telegram_id=2001,
            data=_booking_data(),
            slot_id=self.slot.slot_id,
        )
        self.assertIsNotNone(result["patient"])
        self.assertIsNotNone(result["appointment"])
        self.assertFalse(result["slot_conflict"])
        self.assertIsNone(result["booking_conflict"])

        appt = result["appointment"]
        self.assertEqual(appt.status, "confirmed")
        self.assertEqual(appt.slot_id, self.slot.slot_id)

        refreshed_slot = self.db.get(Slot, self.slot.slot_id)
        self.assertEqual(refreshed_slot.status, "booked")

        from sqlalchemy import select

        profile = self.db.scalar(select(PatientProfile).where(PatientProfile.telegram_id == 2001))
        self.assertEqual(profile.data.get("name"), "أحمد محمود")

    def test_booking_detects_slot_conflict(self):
        self.slot.status = "booked"
        self.db.commit()

        result = crud.create_patient_file_and_book(
            self.db,
            telegram_id=2002,
            data=_booking_data(),
            slot_id=self.slot.slot_id,
        )
        self.assertTrue(result["slot_conflict"])
        self.assertIsNone(result["appointment"])

    def test_waitlist_creates_appointment_without_slot(self):
        result = crud.create_waitlist_appointment(
            self.db,
            telegram_id=2003,
            data=_booking_data(priority_class="P2"),
        )
        self.assertIsNotNone(result["appointment"])
        self.assertEqual(result["appointment"].status, "waitlisted")
        self.assertIsNone(result["appointment"].slot_id)


class TestBookingConflicts(unittest.TestCase):
    def setUp(self):
        self.db = make_test_session()
        self.doctor = seed_doctor(self.db, specialty="cardiology", clinic_code="CLINIC-CARD-BOOK")
        self.patient = seed_patient(self.db, telegram_id=3001)
        self.when = (utcnow() + timedelta(days=4)).replace(hour=10, minute=0, second=0, microsecond=0)
        self.slot1 = seed_slot(self.db, self.doctor, when=self.when)
        self.slot2 = seed_slot(
            self.db,
            self.doctor,
            when=self.when + timedelta(minutes=15),
        )

    def tearDown(self):
        self.db.close()

    def test_time_overlap_conflict_blocks_second_booking(self):
        first = crud.reserve_slot_and_create_appointment(
            self.db,
            _booking_data(specialty_hint="cardiology"),
            self.slot1.slot_id,
            self.patient,
        )
        self.db.commit()

        conflict = crud.find_patient_booking_conflict(
            self.db,
            self.patient.patient_id,
            self.slot2.slot_datetime,
            "cardiology",
        )
        self.assertIsNotNone(conflict)
        self.assertEqual(conflict["type"], "time_overlap")
        self.assertIsNotNone(first["appointment"])


class TestDoctorSessions(unittest.TestCase):
    def setUp(self):
        self.db = make_test_session()
        self.doctor = seed_doctor(self.db, clinic_code="CLINIC-DOC-SESS")
        self.patient = seed_patient(self.db, telegram_id=4001, name="ليلى أبو علي")

    def tearDown(self):
        self.db.close()

    def test_create_session_links_patient_by_name(self):
        session = crud.create_session(
            self.db,
            {
                "patient_name": "ليلى أبو علي",
                "chief_complaint": "سعال",
                "diagnosis": "التهاب bronchi",
                "medications": [{"name": "Paracetamol", "dose": "500mg"}],
                "followup_days": 7,
                "raw_transcription": "ملاحظة صوتية",
            },
            doctor_id=self.doctor.doctor_id,
        )
        self.assertEqual(session.patient_id, self.patient.patient_id)
        self.assertEqual(session.patient_name, "ليلى أبو علي")

    def test_update_appointment_status_cancels_and_releases_slot(self):
        when = (utcnow() + timedelta(days=5)).replace(hour=14, minute=0, second=0, microsecond=0)
        slot = seed_slot(self.db, self.doctor, when=when)
        booking = crud.reserve_slot_and_create_appointment(
            self.db,
            _booking_data(),
            slot.slot_id,
            self.patient,
        )
        self.db.commit()
        appt_id = booking["appointment"].appt_id

        updated = crud.update_appointment_status(self.db, appt_id, "cancelled")
        self.assertEqual(updated.status, "cancelled")

        released = self.db.get(Slot, slot.slot_id)
        self.assertEqual(released.status, "available")


class TestProfilesAndMessages(unittest.TestCase):
    def setUp(self):
        self.db = make_test_session()

    def tearDown(self):
        self.db.close()

    def test_upsert_profile_merges_data(self):
        crud.upsert_profile(self.db, telegram_id=5001, data={"name": "خالد", "last_complaint": "صداع"})
        crud.upsert_profile(self.db, telegram_id=5001, data={"last_specialty": "neurology"})
        self.db.commit()

        profile = crud.get_profile(self.db, 5001)
        self.assertEqual(profile["name"], "خالد")
        self.assertEqual(profile["last_specialty"], "neurology")

    def test_log_message_creates_conversation(self):
        log = crud.log_message(self.db, 5002, "inbound", "text", "مرحبا")
        self.assertIsNotNone(log.log_id)
        conversations = crud.get_conversations(self.db, limit=10)
        self.assertTrue(any(c.telegram_id == 5002 for c in conversations))


class TestSchedulingPolicy(unittest.TestCase):
    """Verify slot priority rules and same-day specialty guard."""

    def setUp(self):
        self.db = make_test_session()
        self.doctor = seed_doctor(self.db, specialty="general_practice", clinic_code="CLINIC-POLICY")
        self.when = (utcnow() + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
        self.p1_slot = seed_slot(self.db, self.doctor, when=self.when, priority_class="P1")
        self.p3_slot = seed_slot(
            self.db,
            self.doctor,
            when=self.when.replace(hour=10),
            priority_class="P3",
        )
        self.db.commit()

    def tearDown(self):
        self.db.close()

    def test_p3_cannot_take_p1_reserved_slot(self):
        found = crud.find_next_available_slot(
            self.db,
            specialty="general_practice",
            priority_class="P3",
        )
        self.assertIsNotNone(found)
        self.assertEqual(found.slot_id, self.p3_slot.slot_id)
        self.assertNotEqual(found.slot_id, self.p1_slot.slot_id)

    def test_p1_can_take_p1_reserved_slot(self):
        found = crud.find_next_available_slot(
            self.db,
            specialty="general_practice",
            priority_class="P1",
        )
        self.assertEqual(found.slot_id, self.p1_slot.slot_id)

    def test_same_specialty_same_day_conflict(self):
        patient = seed_patient(self.db, telegram_id=6001)
        first = crud.reserve_slot_and_create_appointment(
            self.db,
            _booking_data(specialty_hint="general_practice"),
            self.p3_slot.slot_id,
            patient,
        )
        self.db.commit()

        other_slot = seed_slot(
            self.db,
            self.doctor,
            when=self.when.replace(hour=15),
            priority_class="P3",
        )
        conflict = crud.find_patient_booking_conflict(
            self.db,
            patient.patient_id,
            other_slot.slot_datetime,
            "general_practice",
        )
        self.assertIsNotNone(conflict)
        self.assertEqual(conflict["type"], "same_specialty_same_day")
        self.assertIsNotNone(first["appointment"])


class TestNewSpecialtyDoctors(unittest.TestCase):
    """Keyboard specialties must have dedicated clinic doctors and bookable slots."""

    def setUp(self):
        self.db = make_test_session()

    def tearDown(self):
        self.db.close()

    def test_gastroenterology_chronic_and_elderly_have_available_slots(self):
        from datetime import timedelta

        specialties = ("gastroenterology", "chronic_diseases", "elderly")
        codes = ("CLINIC-GI-TEST", "CLINIC-CHR-TEST", "CLINIC-ELD-TEST")
        for specialty, code in zip(specialties, codes):
            doctor = seed_doctor(self.db, specialty=specialty, clinic_code=code)
            when = (utcnow() + timedelta(days=7)).replace(hour=10, minute=0, second=0, microsecond=0)
            seed_slot(self.db, doctor, when=when, priority_class="P3")
        self.db.commit()

        for specialty in specialties:
            with self.subTest(specialty=specialty):
                slot = crud.find_next_available_slot(self.db, specialty=specialty, priority_class="P3")
                self.assertIsNotNone(slot)
                self.assertEqual(slot.doctor.specialty, specialty)


if __name__ == "__main__":
    unittest.main(verbosity=2)
