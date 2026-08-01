"""Tests for scheduler/slot_policy.py unified slot selection."""
from __future__ import annotations

import os
import sys
import unittest
from datetime import timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scheduler.slot_policy import (
    SlotView,
    filter_by_block_rules,
    filter_by_wave_rules,
    rank_slots,
    select_slots,
)
from tests.helpers import make_test_session, seed_doctor, seed_patient, seed_slot, use_test_db
from utils.datetime_utils import utcnow


class TestSlotPolicy(unittest.TestCase):
    def setUp(self):
        from tests.helpers import make_test_engine

        self.engine = make_test_engine()
        self.db = make_test_session(self.engine)
        self.doctor = seed_doctor(self.db, specialty="general_practice", clinic_code="POL-GP")
        self.patient = seed_patient(self.db, telegram_id=901_001, name="مريض سياسة")
        self.now = utcnow()

    def tearDown(self):
        self.db.close()

    def _view(self, slot) -> SlotView:
        return SlotView.from_orm(slot)

    def test_p3_cannot_take_p1_only_slot(self):
        p1_slot = self._view(
            seed_slot(
                self.db,
                self.doctor,
                when=self.now + timedelta(days=1),
                priority_class="P1",
            )
        )
        p3_open = self._view(
            seed_slot(
                self.db,
                self.doctor,
                when=self.now + timedelta(days=1, hours=1),
                priority_class="P3",
            )
        )
        self.db.commit()

        result = filter_by_block_rules([p1_slot, p3_open], "P3")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].priority_class, "P3")

    def test_p1_wave_blocks_slot_beyond_two_days(self):
        near = self._view(seed_slot(self.db, self.doctor, when=self.now + timedelta(days=1), priority_class="P3"))
        far = self._view(seed_slot(self.db, self.doctor, when=self.now + timedelta(days=5), priority_class="P3"))
        self.db.commit()

        result = filter_by_wave_rules([near, far], "P1", now=self.now)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].slot_id, near.slot_id)

    def test_p3_ranking_prefers_less_loaded_day(self):
        from database.models import Appointment

        busy_day = (self.now + timedelta(days=3)).replace(hour=10, minute=0, second=0, microsecond=0)
        quiet_day = (self.now + timedelta(days=4)).replace(hour=10, minute=0, second=0, microsecond=0)
        busy_slot = seed_slot(self.db, self.doctor, when=busy_day, priority_class="P3")
        quiet_slot = seed_slot(self.db, self.doctor, when=quiet_day, priority_class="P3")
        for i in range(3):
            self.db.add(
                Appointment(
                    appt_id=f"load-busy-{i}",
                    patient_id=self.patient.patient_id,
                    slot_id=busy_slot.slot_id,
                    appt_datetime=busy_day.replace(hour=10 + i),
                    specialty="general_practice",
                    status="confirmed",
                )
            )
        self.db.commit()

        with use_test_db(self.db):
            slots = select_slots(
                self.db,
                specialty="general_practice",
                priority_class="P3",
                limit=2,
            )
        self.assertGreaterEqual(len(slots), 2)
        self.assertEqual(slots[0].slot_id, quiet_slot.slot_id)

    def test_select_slots_skips_patient_conflict(self):
        when = (self.now + timedelta(days=2)).replace(hour=11, minute=0, second=0, microsecond=0)
        conflict_slot = seed_slot(self.db, self.doctor, when=when, priority_class="P3")
        alt_slot = seed_slot(
            self.db,
            self.doctor,
            when=when + timedelta(days=1),
            priority_class="P3",
        )
        from database.models import Appointment

        self.db.add(
            Appointment(
                appt_id="appt-conflict-1",
                patient_id=self.patient.patient_id,
                slot_id=conflict_slot.slot_id,
                appt_datetime=when,
                specialty="general_practice",
                status="confirmed",
            )
        )
        self.db.commit()

        with use_test_db(self.db):
            slots = select_slots(
                self.db,
                specialty="general_practice",
                priority_class="P3",
                patient_id=self.patient.patient_id,
                limit=3,
            )
        ids = {s.slot_id for s in slots}
        self.assertNotIn(conflict_slot.slot_id, ids)
        self.assertIn(alt_slot.slot_id, ids)

    def test_gp_fallback_when_specialty_empty(self):
        ortho = seed_doctor(self.db, specialty="orthopedics", clinic_code="POL-ORTHO")
        gp_slot = seed_slot(
            self.db,
            self.doctor,
            when=self.now + timedelta(days=2),
            priority_class="P3",
        )
        self.db.commit()

        with use_test_db(self.db):
            slots = select_slots(
                self.db,
                specialty="orthopedics",
                priority_class="P3",
                limit=1,
            )
        self.assertEqual(len(slots), 1)
        self.assertEqual(slots[0].slot_id, gp_slot.slot_id)
        self.assertEqual(slots[0].doctor.specialty, "general_practice")


if __name__ == "__main__":
    unittest.main(verbosity=2)
