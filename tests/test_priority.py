import sys, os, unittest
from datetime import date, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scheduler.priority import score_and_classify, WEIGHTS, THETA_P1, THETA_P2


class TestWeightsIntegrity(unittest.TestCase):
    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(WEIGHTS.values()), 1.0, places=6)


class TestScoreAndClassify(unittest.TestCase):

    def test_empty_data_defaults_to_p3(self):
        r = score_and_classify({})
        self.assertEqual(r.priority_class, "P3")

    def test_high_urgency_explicit_complaint_score_pushes_p1(self):
        data = {
            "complaint": {"urgency_score": 1.0},
            "urgency_score": 1.0,
            "is_followup": False,
            "specialty_hint": "neurology",
            "time_pref": {"date": date.today().isoformat()},
        }
        r = score_and_classify(data)
        self.assertEqual(r.priority_class, "P1")

    def test_low_everything_gives_p3(self):
        data = {
            "complaint": {"urgency_score": 0.1},
            "urgency_score": 0.1,
            "is_followup": False,
            "specialty_hint": "dermatology",
            "time_pref": {"date": (date.today() + timedelta(days=30)).isoformat()},
        }
        r = score_and_classify(data)
        self.assertEqual(r.priority_class, "P3")

    def test_boundary_just_below_theta_p1(self):
        # construct data landing just under 0.68 to confirm it's P2 not P1
        data = {
            "complaint": {"urgency_score": 0.60},
            "urgency_score": 0.60,
            "is_followup": False,
            "specialty_hint": "orthopedics",
            "time_pref": {"date": (date.today() + timedelta(days=7)).isoformat()},
        }
        r = score_and_classify(data)
        self.assertLess(r.score, THETA_P1)
        self.assertEqual(r.priority_class, "P2")

    # ── _timing_score edge cases (the fixed bug) ─────────────────────────────

    def test_timing_score_string_instead_of_dict_no_crash(self):
        data = {"time_pref": "أي وقت متاح"}   # not a dict at all
        try:
            r = score_and_classify(data)
        except AttributeError:
            self.fail("_timing_score crashed on non-dict time_pref -- bug NOT fixed")
        self.assertIsNotNone(r)

    def test_timing_score_none(self):
        data = {"time_pref": None}
        r = score_and_classify(data)
        self.assertIsNotNone(r)

    def test_timing_score_missing_date_key(self):
        data = {"time_pref": {"phrase": "أي وقت متاح"}}
        r = score_and_classify(data)
        self.assertIsNotNone(r)

    def test_timing_score_malformed_date_string(self):
        data = {"time_pref": {"date": "not-a-date"}}
        r = score_and_classify(data)
        self.assertIsNotNone(r)

    def test_timing_score_past_date_clamped_to_zero_delta(self):
        past = (date.today() - timedelta(days=10)).isoformat()
        data = {"time_pref": {"date": past}}
        r = score_and_classify(data)
        # delta clamped to 0 -> should get the "today" score (1.0) for f5
        self.assertEqual(r.breakdown["f5"], 1.0)

    def test_specialty_score_unknown_specialty_falls_back(self):
        data = {"specialty_hint": "made_up_specialty_xyz"}
        r = score_and_classify(data)
        self.assertEqual(r.breakdown["f4"], 0.3)  # default fallback score

    def test_complaint_not_dict_no_crash(self):
        data = {"complaint": "just a raw string, not a dict"}
        try:
            r = score_and_classify(data)
        except Exception as e:
            self.fail(f"crashed on non-dict complaint: {e}")
        self.assertIsNotNone(r)

    def test_urgency_score_non_numeric_no_crash(self):
        data = {"urgency_score": "very high"}  # not a float
        r = score_and_classify(data)
        self.assertIsNotNone(r)


if __name__ == "__main__":
    unittest.main(verbosity=2)
