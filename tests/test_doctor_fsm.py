"""Unit tests for fsm/doctor_fsm.py — doctor session note FSM."""
import os
import sys
import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fsm.doctor_fsm import DoctorFSM, DoctorState, EDITABLE_FIELDS
from tests.helpers import run_async


SAMPLE_SESSION = {
    "patient_name": "أحمد خالد",
    "chief_complaint": "ألم ركبة",
    "symptom_duration": "3 أيام",
    "diagnosis": "التهاب مفصل",
    "medications": [{"name": "Ibuprofen", "dose": "400mg", "frequency": "مرتين يومياً", "duration": "5 أيام"}],
    "investigations": [{"name_ar": "تحليل دم", "name_en": "CBC"}],
    "followup_days": 14,
}


class TestDoctorFSMFlow(unittest.TestCase):
    def setUp(self):
        self.fsm = DoctorFSM(doctor_id=1, telegram_id=70001)

    def test_idle_moves_to_listening_with_instructions(self):
        reply = run_async(self.fsm.handle("أي نص"))
        self.assertEqual(self.fsm.state, DoctorState.LISTENING)
        self.assertIn("ملاحظات الجلسة", reply)

    def test_listening_extracts_and_shows_review(self):
        self.fsm.state = DoctorState.LISTENING

        with patch("nlp.doctor_extractor.extract_session_fields", return_value=SAMPLE_SESSION):
            reply = run_async(self.fsm.handle("المريض اسمه أحمد خالد، شاكي من ألم ركبة"))

        self.assertEqual(self.fsm.state, DoctorState.REVIEW)
        self.assertIn("أحمد خالد", reply)
        self.assertIn("Ibuprofen", reply)
        self.assertIn("تأكيد", reply)

    def test_review_confirm_saves_session(self):
        self.fsm.state = DoctorState.REVIEW
        self.fsm.session = dict(SAMPLE_SESSION)
        self.fsm.session["raw_transcription"] = "note"

        saved = MagicMock()
        saved.patient_id = 10
        saved.appointment_id = "appt_1"

        @contextmanager
        def fake_get_db():
            yield MagicMock()

        with patch("database.db.get_db", fake_get_db):
            with patch("database.crud.create_session", return_value=saved) as mock_create:
                reply = run_async(self.fsm.handle("تأكيد"))

        mock_create.assert_called_once()
        self.assertEqual(self.fsm.state, DoctorState.SAVED)
        self.assertEqual(self.fsm.session, {})
        self.assertIn("تم حفظ الجلسة", reply)
        self.assertIn("ربطها بملف المريض", reply)

    def test_review_edit_field_updates_summary(self):
        self.fsm.state = DoctorState.REVIEW
        self.fsm.session = dict(SAMPLE_SESSION)

        reply = run_async(self.fsm.handle("الشكوى: وجع ظهر"))

        self.assertEqual(self.fsm.state, DoctorState.REVIEW)
        self.assertEqual(self.fsm.session["chief_complaint"], "وجع ظهر")
        self.assertIn("تم تحديث", reply)
        self.assertIn("وجع ظهر", reply)

    def test_review_unknown_input_shows_save_hint(self):
        self.fsm.state = DoctorState.REVIEW
        self.fsm.session = dict(SAMPLE_SESSION)

        reply = run_async(self.fsm.handle("عدّل شي"))

        self.assertIn("تأكيد", reply)
        self.assertIn("الحقل", reply)

    def test_editing_unknown_field_shows_help(self):
        self.fsm.state = DoctorState.REVIEW
        self.fsm.session = dict(SAMPLE_SESSION)

        reply = run_async(self.fsm.handle("عدّل شي"))

        self.assertIn("تأكيد", reply)

    def test_saved_state_prompts_new_session(self):
        self.fsm.state = DoctorState.SAVED
        reply = run_async(self.fsm.handle("مرحبا"))
        self.assertIn("/session", reply)


class TestDoctorFSMConstants(unittest.TestCase):
    def test_editable_fields_cover_summary_labels(self):
        labels = set(EDITABLE_FIELDS.keys())
        self.assertIn("اسم المريض", labels)
        self.assertIn("الشكوى", labels)
        self.assertIn("المتابعة", labels)

    def test_medication_edit_updates_medications_list(self):
        fsm = DoctorFSM(doctor_id=1, telegram_id=70002)
        fsm.state = DoctorState.REVIEW
        fsm.session = dict(SAMPLE_SESSION)

        run_async(fsm.handle("الدواء: Paracetamol, Ibuprofen"))

        self.assertEqual(
            fsm.session["medications"],
            [{"name": "Paracetamol"}, {"name": "Ibuprofen"}],
        )

    def test_followup_edit_parses_integer(self):
        fsm = DoctorFSM(doctor_id=1, telegram_id=70003)
        fsm.state = DoctorState.REVIEW
        fsm.session = dict(SAMPLE_SESSION)

        run_async(fsm.handle("المتابعة: 21"))

        self.assertEqual(fsm.session["followup_days"], 21)

    def test_session_command_resets_to_listening(self):
        fsm = DoctorFSM(doctor_id=1, telegram_id=70004)
        fsm.state = DoctorState.SAVED
        fsm.session = {"patient_name": "old"}

        reply = run_async(fsm.handle("/session"))

        self.assertEqual(fsm.state, DoctorState.LISTENING)
        self.assertEqual(fsm.session, {})
        self.assertIn("ملاحظات الجلسة", reply)


if __name__ == "__main__":
    unittest.main(verbosity=2)
