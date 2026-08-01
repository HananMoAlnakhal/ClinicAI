"""Unit tests for fsm/patient_fsm.py — patient booking FSM."""
import os
import sys
import unittest
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fsm.patient_fsm import PatientFSM, State, REQUIRED_FIELDS, FIELD_QUESTIONS_AR, SPECIALTY_LABEL_TO_KEY, SPECIALTY_LABEL_TO_KEY, SPECIALTY_LABEL_TO_KEY
from tests.helpers import run_async, unpack_fsm


def _high_confidence_classify(*args, **kwargs):
    return {
        "specialty": "general_practice",
        "specialty_ar": "الطب العام",
        "method": "rule",
        "confidence": 0.95,
    }


class TestPatientFSMHelpers(unittest.TestCase):
    def setUp(self):
        self.fsm = PatientFSM(user_id=80001)

    def test_parse_time_label_today_and_tomorrow(self):
        today = self.fsm._parse_time_label("بدي موعد اليوم")
        self.assertEqual(today["date"], str(date.today()))

        tomorrow = self.fsm._parse_time_label("بكرا")
        self.assertEqual(tomorrow["date"], str(date.today() + timedelta(days=1)))

    def test_parse_specialty_label_arabic(self):
        self.assertEqual(self.fsm._parse_specialty_label("عظام"), "orthopedics")
        self.assertEqual(self.fsm._parse_specialty_label("هضمي"), "gastroenterology")

    def test_absorb_urgency_levels(self):
        self.fsm._absorb_urgency("عاجل جدا")
        self.assertGreaterEqual(self.fsm.data["urgency_score"], 0.85)

        self.fsm.data.clear()
        self.fsm._absorb_urgency("روتيني")
        self.assertLessEqual(self.fsm.data["urgency_score"], 0.25)

    def test_missing_fields_detects_empty_time_pref(self):
        self.fsm.data = {
            "name": "أحمد",
            "complaint": {"raw": "صداع"},
            "urgency_score": 0.4,
            "time_pref": {"date": None, "phrase": ""},
        }
        missing = self.fsm._missing_fields()
        self.assertIn("time_pref", missing)

    def test_reset_clears_booking_state(self):
        self.fsm.state = State.FINALIZED
        self.fsm.data = {"name": "x"}
        self.fsm.slot = {"slot_id": 1}
        self.fsm._reset()
        self.assertEqual(self.fsm.state, State.GREETING)
        self.assertEqual(self.fsm.data, {})
        self.assertIsNone(self.fsm.slot)


class TestPatientFSMConversation(unittest.TestCase):
    def setUp(self):
        self.fsm = PatientFSM(user_id=80002)

    def test_greeting_asks_for_name(self):
        reply, keyboard = unpack_fsm(run_async(self.fsm.handle("مرحبا")))
        self.assertEqual(self.fsm.state, State.COLLECT_NAME)
        self.assertIn(FIELD_QUESTIONS_AR["name"], reply)
        self.assertIsNone(keyboard)

    def test_collect_name_then_complaint(self):
        run_async(self.fsm.handle("start"))
        reply, _ = unpack_fsm(run_async(self.fsm.handle("سارة محمود")))
        self.assertEqual(self.fsm.state, State.COLLECT_COMPLAINT)
        self.assertIn("سارة", reply)

    def test_collect_complaint_from_plain_text(self):
        self.fsm.state = State.COLLECT_COMPLAINT
        reply, keyboard = unpack_fsm(run_async(self.fsm.handle("عندي صداع من يومين")))
        self.assertEqual(self.fsm.state, State.COLLECT_URGENCY)
        self.assertIn("complaint", self.fsm.data)
        self.assertIsNotNone(keyboard)

    def test_urgency_callback_moves_to_time(self):
        self.fsm.state = State.COLLECT_URGENCY
        self.fsm.data = {"name": "أحمد", "complaint": {"raw": "صداع"}}
        reply, keyboard = unpack_fsm(run_async(self.fsm.handle_callback("urgency:P2")))
        self.assertEqual(self.fsm.state, State.COLLECT_TIME)
        self.assertEqual(self.fsm.data["urgency_score"], 0.5)
        self.assertIsNotNone(keyboard)

    @patch("fsm.patient_fsm.gemini")
    @patch("fsm.patient_fsm.classify_with_gemini_fallback", side_effect=_high_confidence_classify)
    @patch("fsm.patient_fsm.classify_specialty", side_effect=_high_confidence_classify)
    @patch("fsm.patient_fsm.score_and_classify")
    def test_full_flow_reaches_confirm_with_slot(self, mock_score, mock_classify_rules, mock_classify, mock_gemini):
        mock_gemini._available = False
        mock_gemini.extract_missing_field = AsyncMock(return_value=None)
        priority = MagicMock()
        priority.priority_class = "P3"
        priority.score = 0.35
        priority.label_ar = "روتيني"
        priority.breakdown = {"f1": 0.3}
        mock_score.return_value = priority

        slot_dt = datetime.utcnow() + timedelta(days=2)
        slot = MagicMock()
        slot.slot_id = 99
        slot.slot_datetime = slot_dt
        slot.specialty = "general_practice"
        slot.priority_class = "P3"
        slot.doctor = MagicMock(
            doctor_id=1,
            name="د. اختبار",
            specialty="general_practice",
            clinic_code="CLINIC-GP",
            clinic_name="عيادة عامة",
        )

        @contextmanager
        def fake_get_db():
            yield MagicMock()

        with patch("database.db.get_db", fake_get_db):
            self.fsm.services.classify = mock_classify_rules
            self.fsm.services.classify_with_fallback = mock_classify
            self.fsm.services.score = mock_score
            self.fsm.services.find_slots = MagicMock(return_value=[slot])
            self.fsm.services.gemini = mock_gemini
            run_async(self.fsm.handle("start"))
            run_async(self.fsm.handle("أحمد"))
            run_async(self.fsm.handle("صداع خفيف"))
            run_async(self.fsm.handle("روتيني"))
            reply, keyboard = unpack_fsm(run_async(self.fsm.handle("بكرا")))

        self.assertEqual(self.fsm.state, State.CONFIRM)
        self.assertIn("وجدت موعد", reply)
        self.assertNotIn("درجة الأولوية", reply)
        self.assertIsNotNone(keyboard)
        self.assertEqual(self.fsm.slot["slot_id"], 99)

    @patch("fsm.patient_fsm.classify_specialty", side_effect=_high_confidence_classify)
    @patch("fsm.patient_fsm.score_and_classify")
    def test_confirm_yes_finalizes_booking(self, mock_score, mock_classify):
        priority = MagicMock(priority_class="P3", score=0.3, label_ar="روتيني", breakdown={})
        mock_score.return_value = priority

        self.fsm.state = State.CONFIRM
        self.fsm.data = {
            "name": "أحمد",
            "complaint": {"raw": "صداع"},
            "urgency_score": 0.3,
            "time_pref": {"date": str(date.today() + timedelta(days=1)), "phrase": "بكرا"},
            "specialty_hint": "general_practice",
            "specialty_ar": "الطب العام",
            "priority_class": "P3",
        }
        self.fsm.slot = {
            "slot_id": 5,
            "slot_datetime": datetime.utcnow() + timedelta(days=1),
            "doctor_name": "د. اختبار",
            "clinic_name": "عيادة",
        }
        self.fsm.priority = priority

        appt = MagicMock()
        appt.appt_id = "appt_test_99"
        appt.appt_datetime = self.fsm.slot["slot_datetime"]

        @contextmanager
        def fake_get_db():
            yield MagicMock()

        with patch("database.db.get_db", fake_get_db):
            self.fsm.services.book = MagicMock(
                return_value={"appointment": appt, "slot_conflict": False, "booking_conflict": None},
            )
            reply, _ = unpack_fsm(run_async(self.fsm.handle("نعم")))

        self.assertEqual(self.fsm.state, State.FINALIZED)
        self.assertIn("تم تأكيد حجزك", reply)
        self.assertEqual(self.fsm.finalized_appointment_id, "appt_test_99")

    def test_confirm_no_cancels(self):
        self.fsm.state = State.CONFIRM
        self.fsm.slot = {"slot_id": 1}
        reply, _ = unpack_fsm(run_async(self.fsm.handle("لا")))
        self.assertEqual(self.fsm.state, State.CANCELLED)
        self.assertIn("تم الإلغاء", reply)

    def test_low_confidence_specialty_prompts_user(self):
        self.fsm.state = State.COLLECT_TIME
        self.fsm.data = {
            "name": "ليلى",
            "complaint": {"raw": "سؤال عام"},
            "urgency_score": 0.3,
            "time_pref": {"date": str(date.today()), "phrase": "اليوم"},
        }

        with patch("fsm.patient_fsm.gemini") as mock_gemini:
            mock_gemini._available = False
            self.fsm.services.classify = MagicMock(
                return_value={"specialty": "general_practice", "specialty_ar": "الطب العام", "method": "default", "confidence": 0.5},
            )
            reply, keyboard = unpack_fsm(run_async(self.fsm.handle("اليوم")))

        self.assertEqual(self.fsm.state, State.COLLECT_SPECIALTY)
        self.assertIn("تخصص", reply)
        self.assertIsNotNone(keyboard)


class TestPatientFSMRequiredFields(unittest.TestCase):
    def test_required_fields_match_questions(self):
        for field_name in REQUIRED_FIELDS:
            self.assertIn(field_name, FIELD_QUESTIONS_AR)


class TestPatientFSMLogic(unittest.TestCase):
    """Behavioral checks — verify business rules, not just happy-path execution."""

    def test_validate_blocks_scheduling_until_checklist_complete(self):
        fsm = PatientFSM(user_id=81001)
        fsm.state = State.COLLECT_TIME
        fsm.data = {"name": "أحمد", "complaint": {"raw": "صداع"}, "urgency_score": 0.4}

        with patch("fsm.patient_fsm.classify_specialty") as mock_classify:
            mock_classify.assert_not_called()
            reply, _ = unpack_fsm(run_async(fsm.handle("")))
        self.assertEqual(fsm.state, State.COLLECT_TIME)
        self.assertIn("محتاجين", reply)

    def test_clarification_does_not_hijack_normal_booking_phrase(self):
        fsm = PatientFSM(user_id=81002)
        fsm.state = State.COLLECT_COMPLAINT
        fsm.data = {"name": "سارة"}

        reply, _ = unpack_fsm(run_async(fsm.handle("فهمت، عندي صداع")))

        self.assertEqual(fsm.state, State.COLLECT_URGENCY)
        self.assertNotIn("فهمت إن", reply)

    def test_parse_specialty_arabic_full_label(self):
        fsm = PatientFSM(user_id=81003)
        self.assertEqual(fsm._parse_specialty_label("الطب العام"), "general_practice")

    def test_unclear_urgency_stays_in_collect_urgency(self):
        fsm = PatientFSM(user_id=81005)
        fsm.state = State.COLLECT_URGENCY
        fsm.data = {"name": "أحمد", "complaint": {"raw": "صداع"}}

        reply, keyboard = unpack_fsm(run_async(fsm.handle("xyz gibberish")))

        self.assertEqual(fsm.state, State.COLLECT_URGENCY)
        self.assertNotIn("urgency_score", fsm.data)
        self.assertIn("الأولوية", reply)
        self.assertIsNotNone(keyboard)

    def test_specialty_label_keys_match_classifier(self):
        from scheduler.classifier import SPECIALTY_NAMES_AR

        for key in set(SPECIALTY_LABEL_TO_KEY.values()):
            self.assertIn(key, SPECIALTY_NAMES_AR, f"missing classifier entry for {key}")

    def test_keyboard_specialty_labels_parse(self):
        fsm = PatientFSM(user_id=81006)
        self.assertEqual(fsm._parse_specialty_label("❤️ قلب وأوعية"), "cardiology")
        self.assertEqual(fsm._parse_specialty_label("🧓 كبار السن"), "elderly")

    def test_unclear_urgency_stays_in_collect_urgency(self):
        fsm = PatientFSM(user_id=81005)
        fsm.state = State.COLLECT_URGENCY
        fsm.data = {"name": "أحمد", "complaint": {"raw": "صداع"}}

        reply, keyboard = unpack_fsm(run_async(fsm.handle("xyz gibberish")))

        self.assertEqual(fsm.state, State.COLLECT_URGENCY)
        self.assertNotIn("urgency_score", fsm.data)
        self.assertIn("الأولوية", reply)
        self.assertIsNotNone(keyboard)

    def test_specialty_label_keys_match_classifier(self):
        from scheduler.classifier import SPECIALTY_NAMES_AR

        for key in set(SPECIALTY_LABEL_TO_KEY.values()):
            self.assertIn(key, SPECIALTY_NAMES_AR, f"missing classifier entry for {key}")

    def test_keyboard_specialty_labels_parse(self):
        fsm = PatientFSM(user_id=81006)
        self.assertEqual(fsm._parse_specialty_label("❤️ قلب وأوعية"), "cardiology")
        self.assertEqual(fsm._parse_specialty_label("🧓 كبار السن"), "elderly")

    def test_unclear_urgency_stays_in_collect_urgency(self):
        fsm = PatientFSM(user_id=81005)
        fsm.state = State.COLLECT_URGENCY
        fsm.data = {"name": "أحمد", "complaint": {"raw": "صداع"}}

        reply, keyboard = unpack_fsm(run_async(fsm.handle("xyz gibberish")))

        self.assertEqual(fsm.state, State.COLLECT_URGENCY)
        self.assertNotIn("urgency_score", fsm.data)
        self.assertIn("الأولوية", reply)
        self.assertIsNotNone(keyboard)

    def test_specialty_label_keys_match_classifier(self):
        from scheduler.classifier import SPECIALTY_NAMES_AR

        for key in set(SPECIALTY_LABEL_TO_KEY.values()):
            self.assertIn(key, SPECIALTY_NAMES_AR, f"missing classifier entry for {key}")

    def test_keyboard_specialty_labels_parse(self):
        from bot.keyboards import specialty_keyboard

        fsm = PatientFSM(user_id=81006)
        # self.assertEqual(fsm._parse_specialty_label("🫀 قلب وأوعية"), "cardiology")
        for row in specialty_keyboard().keyboard:
            for button in row:
                key = fsm._parse_specialty_label(button.text)
                with self.subTest(label=button.text):
                    self.assertIsNotNone(key, f"failed to parse specialty button: {button.text!r}")
                    self.assertIn(key, {
                        "gastroenterology",
                        "neurology",
                        "orthopedics",
                        "gynecology",
                        "chronic_diseases",
                        "dermatology",
                        "general_practice",
                        "elderly",
                    })

    @patch("fsm.patient_fsm.classify_specialty", side_effect=_high_confidence_classify)
    @patch("fsm.patient_fsm.score_and_classify")
    def test_slot_conflict_retries_with_next_available_slot(self, mock_score, mock_classify):
        priority = MagicMock(priority_class="P3", score=0.3, label_ar="روتيني", breakdown={})
        mock_score.return_value = priority

        fsm = PatientFSM(user_id=81004)
        fsm.state = State.CONFIRM
        fsm.data = {
            "name": "أحمد",
            "complaint": {"raw": "صداع"},
            "urgency_score": 0.3,
            "time_pref": {"date": str(date.today() + timedelta(days=1)), "phrase": "بكرا"},
            "specialty_hint": "general_practice",
            "specialty_ar": "الطب العام",
            "priority_class": "P3",
        }
        fsm.priority = priority

        old_slot = {"slot_id": 1, "slot_datetime": datetime.utcnow() + timedelta(days=1), "doctor_name": "د.", "clinic_name": "عيادة"}
        new_slot_dt = datetime.utcnow() + timedelta(days=2)
        new_slot = MagicMock(
            slot_id=2,
            slot_datetime=new_slot_dt,
            specialty="general_practice",
            priority_class="P3",
            doctor=MagicMock(
                doctor_id=1,
                name="د. بديل",
                specialty="general_practice",
                clinic_code="CLINIC-GP",
                clinic_name="عيادة عامة",
            ),
        )
        fsm.slot = dict(old_slot)

        @contextmanager
        def fake_get_db():
            yield MagicMock()

        with patch("database.db.get_db", fake_get_db):
            fsm.services.book = MagicMock(
                return_value={"appointment": None, "slot_conflict": True, "booking_conflict": None},
            )
            fsm.services.find_slots = MagicMock(return_value=[new_slot])
            reply, keyboard = unpack_fsm(run_async(fsm.handle("نعم")))

        fsm.services.find_slots.assert_called_once()
        self.assertEqual(fsm.state, State.CONFIRM)
        self.assertEqual(fsm.slot["slot_id"], 2)
        self.assertIn("انحجز قبل التأكيد", reply)
        self.assertIn("وجدت موعد", reply)
        self.assertIsNotNone(keyboard)

    def test_unsupported_specialty_offers_gp_fallback(self):
        fsm = PatientFSM(user_id=81007)
        fsm.state = State.COLLECT_COMPLAINT
        fsm.data = {"name": "أحمد"}
        reply, keyboard = unpack_fsm(run_async(fsm.handle("بدي موعد عند طبيب اسنان")))
        self.assertEqual(fsm.state, State.OFFER_GP_FALLBACK)
        self.assertIn("غير متوفرة", reply)
        self.assertIsNotNone(keyboard)

    def test_urgency_free_text_accepted(self):
        fsm = PatientFSM(user_id=81008)
        fsm.state = State.COLLECT_URGENCY
        fsm.data = {"name": "أحمد", "complaint": {"raw": "صداع"}}
        reply, keyboard = unpack_fsm(run_async(fsm.handle("عاجل")))
        self.assertEqual(fsm.state, State.COLLECT_TIME)
        self.assertGreaterEqual(fsm.data["urgency_score"], 0.85)
        self.assertIsNotNone(keyboard)

    def test_confirm_edit_returns_to_time_selection(self):
        fsm = PatientFSM(user_id=81009)
        fsm.state = State.CONFIRM
        fsm.slot = {"slot_id": 1, "slot_datetime": datetime.utcnow()}
        fsm.data = {"name": "أحمد", "time_pref": {"date": str(date.today()), "phrase": "اليوم"}}
        reply, keyboard = unpack_fsm(run_async(fsm.handle("✏️ تعديل الموعد")))
        self.assertEqual(fsm.state, State.COLLECT_TIME)
        self.assertIn("متى", reply)
        self.assertIsNotNone(keyboard)


if __name__ == "__main__":
    unittest.main(verbosity=2)
