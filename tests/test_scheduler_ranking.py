import os
import sys
import unittest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scheduler.scheduler import AppointmentSlot, _build_slot_sort_key


def make_slot(slot_id, when, specialty="neurology", priority_class=None, doctor_id=None):
    return AppointmentSlot(
        slot_id=slot_id,
        slot_datetime=when,
        specialty=specialty,
        priority_class=priority_class,
        status="available",
        doctor_id=doctor_id,
        notes=f"doctor_id:{doctor_id}" if doctor_id else None,
    )


class TestUrgencyAwareRanking(unittest.TestCase):
    """
    Proves the fix: priority_class now actually changes the WINNER between
    two candidate slots, instead of the old dead `time_bias` constant that
    could never affect relative ordering within a single ranking call.
    """

    def setUp(self):
        self.now = datetime(2026, 7, 11, 8, 0, 0)
        # Slot A: sooner, but on a busier clinic day (c_load=5)
        self.slot_soon_busy = make_slot(1, self.now + timedelta(hours=2))
        # Slot B: later, but on a quiet clinic day (c_load=0)
        self.slot_later_quiet = make_slot(2, self.now + timedelta(days=3))

        self.clinic_load = {
            ("neurology", self.slot_soon_busy.slot_datetime.date()): 5,
            ("neurology", self.slot_later_quiet.slot_datetime.date()): 0,
        }
        self.doctor_load = {}
        self.utilization = {}

    def _rank(self, priority_class):
        slots = [self.slot_later_quiet, self.slot_soon_busy]  # deliberately reversed input
        ranked = sorted(
            slots,
            key=lambda s: _build_slot_sort_key(
                s,
                pref_day=None,
                priority_class=priority_class,
                clinic_load=self.clinic_load,
                doctor_load=self.doctor_load,
                utilization=self.utilization,
            ),
        )
        return ranked

    def test_p1_prefers_earliest_slot_despite_higher_load(self):
        ranked = self._rank("P1")
        self.assertEqual(ranked[0].slot_id, 1)  # soon+busy wins for P1

    def test_p3_prefers_lower_load_over_earliest_slot(self):
        ranked = self._rank("P3")
        self.assertEqual(ranked[0].slot_id, 2)  # later+quiet wins for P3

    def test_p2_also_prefers_lower_load_over_earliest_slot(self):
        ranked = self._rank("P2")
        self.assertEqual(ranked[0].slot_id, 2)

    def test_identical_load_falls_back_to_earliest_for_any_class(self):
        # when load is tied, everyone should just get the earliest slot
        tied_load = {
            ("neurology", self.slot_soon_busy.slot_datetime.date()): 2,
            ("neurology", self.slot_later_quiet.slot_datetime.date()): 2,
        }
        for pc in ("P1", "P2", "P3"):
            ranked = sorted(
                [self.slot_later_quiet, self.slot_soon_busy],
                key=lambda s: _build_slot_sort_key(
                    s,
                    pref_day=None,
                    priority_class=pc,
                    clinic_load=tied_load,
                    doctor_load={},
                    utilization={},
                ),
            )
            self.assertEqual(ranked[0].slot_id, 1, f"failed for {pc}")

    # NOTE: an end-to-end test of rank_slots() itself (going through
    # clinic_load_by_day / doctor_load_by_day / slot_utilization_by_day)
    # needs a real SQLAlchemy session or a much fuller fake DB/ORM layer to
    # be meaningful -- out of scope here since the goal was to validate the
    # ranking fix itself, which the four tests above already prove directly
    # against the real, patched scheduler.py.

if __name__ == "__main__":
    unittest.main(verbosity=2)
