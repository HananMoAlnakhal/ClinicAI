"""Tests for scheduler.plan_appointment orchestrator."""
from __future__ import annotations

import os
import sys
import unittest
from datetime import timedelta
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scheduler.scheduler import plan_appointment
from tests.helpers import make_test_session, run_async, seed_doctor, seed_slot, use_test_db
from utils.datetime_utils import utcnow


def _high_confidence_classify(*args, **kwargs):
    return {
        "specialty": "general_practice",
        "specialty_ar": "الطب العام",
        "method": "rule",
        "confidence": 0.95,
    }


class TestPlanAppointment(unittest.TestCase):
    def setUp(self):
        from tests.helpers import make_test_engine

        self.engine = make_test_engine()
        self.db = make_test_session(self.engine)
        self.doctor = seed_doctor(self.db, specialty="general_practice", clinic_code="PLAN-GP")
        self.when = (utcnow() + timedelta(days=2)).replace(hour=9, minute=0, second=0, microsecond=0)
        self.slot = seed_slot(self.db, self.doctor, when=self.when, priority_class="P3")
        self.db.commit()

    def tearDown(self):
        self.db.close()

    @patch("scheduler.scheduler.classify_with_gemini_fallback", side_effect=_high_confidence_classify)
    def test_plan_appointment_returns_best_slot(self, _mock_classify):
        data = {
            "complaint": {"raw": "صداع خفيف"},
            "urgency_score": 0.25,
            "time_pref": {"date": str(self.when.date()), "phrase": "بكرا"},
            "telegram_id": 902_001,
        }

        with use_test_db(self.db):
            decision = run_async(plan_appointment(data, self.db, gemini_client=None))

        self.assertFalse(decision.waitlisted)
        self.assertIsNotNone(decision.slot)
        self.assertEqual(decision.slot.slot_id, self.slot.slot_id)
        self.assertEqual(decision.priority_class, "P3")

    @patch("scheduler.scheduler.classify_with_gemini_fallback", side_effect=_high_confidence_classify)
    def test_plan_appointment_waitlists_when_no_slots(self, _mock_classify):
        self.slot.status = "booked"
        self.db.add(self.slot)
        self.db.commit()

        data = {
            "complaint": {"raw": "صداع"},
            "urgency_score": 0.25,
            "time_pref": {"date": None, "phrase": "أي وقت"},
            "telegram_id": 902_002,
        }

        with use_test_db(self.db):
            decision = run_async(plan_appointment(data, self.db))

        self.assertTrue(decision.waitlisted)
        self.assertIsNone(decision.slot)
        self.assertIsNotNone(decision.waitlist)
        self.assertGreaterEqual(decision.waitlist.position, 1)


class TestPlanAppointmentContract(unittest.TestCase):
    """FSM slot lookup and plan_appointment should agree on best slot."""

    def setUp(self):
        from tests.helpers import make_test_engine

        self.engine = make_test_engine()
        self.db = make_test_session(self.engine)
        self.doctor = seed_doctor(self.db, specialty="general_practice", clinic_code="PLAN-CTR")
        self.when = (utcnow() + timedelta(days=3)).replace(hour=14, minute=0, second=0, microsecond=0)
        self.slot = seed_slot(self.db, self.doctor, when=self.when, priority_class="P3")
        self.db.commit()

    def tearDown(self):
        self.db.close()

    @patch("scheduler.scheduler.classify_with_gemini_fallback", side_effect=_high_confidence_classify)
    def test_crud_and_plan_agree_on_best_slot(self, _mock_classify):
        from database import crud

        data = {
            "complaint": {"raw": "متابعة دورية"},
            "urgency_score": 0.2,
            "time_pref": {"date": str(self.when.date()), "phrase": "الأسبوع الجاي"},
            "priority_class": "P3",
        }

        with use_test_db(self.db):
            crud_slots = crud.find_available_slots(
                self.db,
                specialty="general_practice",
                priority_class="P3",
                preferred_date=str(self.when.date()),
                limit=1,
            )
            decision = run_async(plan_appointment(data, self.db))

        self.assertTrue(crud_slots)
        self.assertIsNotNone(decision.slot)
        self.assertEqual(crud_slots[0].slot_id, decision.slot.slot_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
