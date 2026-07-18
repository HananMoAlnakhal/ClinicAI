"""Unit tests for database/models.py — schema, defaults, and relationships."""
import os
import sys
import unittest
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import inspect

from database.models import (
    Appointment,
    Conversation,
    Doctor,
    MessageLog,
    Patient,
    PatientProfile,
    Session as DoctorSession,
    Slot,
)
from tests.helpers import make_test_session, seed_doctor, seed_patient, seed_slot


class TestModelTables(unittest.TestCase):
    def setUp(self):
        self.db = make_test_session()

    def tearDown(self):
        self.db.close()

    def test_all_expected_tables_exist(self):
        names = set(inspect(self.db.bind).get_table_names())
        expected = {
            "patients",
            "doctors",
            "appointments",
            "sessions",
            "slots",
            "conversations",
            "message_logs",
            "patient_profiles",
        }
        self.assertTrue(expected.issubset(names))

    def test_patient_defaults_and_unique_telegram(self):
        p1 = Patient(telegram_id=111, name="أحمد")
        p2 = Patient(telegram_id=222, name="سارة")
        self.db.add_all([p1, p2])
        self.db.commit()

        self.assertIsNotNone(p1.patient_id)
        self.assertIsNotNone(p1.created_at)
        self.assertEqual(p1.name, "أحمد")

    def test_doctor_requires_clinic_identity(self):
        doctor = seed_doctor(self.db, specialty="cardiology", clinic_code="CLINIC-CARD-TEST")
        self.assertTrue(doctor.is_active)
        self.assertEqual(doctor.specialty, "cardiology")
        self.assertIsNone(doctor.telegram_id)

    def test_slot_belongs_to_doctor(self):
        doctor = seed_doctor(self.db)
        slot = seed_slot(self.db, doctor, status="available")
        self.assertEqual(slot.doctor_id, doctor.doctor_id)
        self.assertEqual(slot.status, "available")
        self.assertEqual(slot.doctor.specialty, doctor.specialty)

    def test_appointment_links_patient_and_slot(self):
        patient = seed_patient(self.db, telegram_id=333)
        doctor = seed_doctor(self.db, specialty="neurology", clinic_code="CLINIC-NEURO-T")
        slot = seed_slot(self.db, doctor)
        appt = Appointment(
            appt_id="appt_test_001",
            patient_id=patient.patient_id,
            slot_id=slot.slot_id,
            appt_datetime=slot.slot_datetime,
            specialty=doctor.specialty,
            status="confirmed",
        )
        self.db.add(appt)
        self.db.commit()

        self.assertEqual(appt.patient.patient_id, patient.patient_id)
        self.assertEqual(appt.slot.slot_id, slot.slot_id)

    def test_doctor_session_optional_patient_and_appointment(self):
        doctor = seed_doctor(self.db, clinic_code="CLINIC-GP-T2")
        session = DoctorSession(
            doctor_id=doctor.doctor_id,
            patient_name="مريض بدون ملف",
            chief_complaint="صداع",
            diagnosis="صداع توتري",
        )
        self.db.add(session)
        self.db.commit()

        self.assertIsNone(session.patient_id)
        self.assertIsNone(session.appointment_id)
        self.assertEqual(session.doctor.doctor_id, doctor.doctor_id)

    def test_patient_profile_json_payload(self):
        patient = seed_patient(self.db, telegram_id=444)
        profile = PatientProfile(
            patient_id=patient.patient_id,
            telegram_id=patient.telegram_id,
            data={"name": patient.name, "last_complaint": "ألم ظهر"},
        )
        self.db.add(profile)
        self.db.commit()

        loaded = self.db.get(PatientProfile, profile.profile_id)
        self.assertEqual(loaded.data["last_complaint"], "ألم ظهر")
        self.assertEqual(loaded.patient.patient_id, patient.patient_id)

    def test_conversation_and_message_log(self):
        conversation = Conversation(telegram_id=555, role="patient")
        self.db.add(conversation)
        self.db.flush()

        log = MessageLog(
            conversation_id=conversation.conversation_id,
            telegram_id=555,
            direction="inbound",
            message_type="text",
            content="مرحبا",
        )
        self.db.add(log)
        self.db.commit()

        self.assertEqual(len(conversation.messages), 1)
        self.assertEqual(conversation.messages[0].content, "مرحبا")


if __name__ == "__main__":
    unittest.main(verbosity=2)
