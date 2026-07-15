import sys, os, unittest
from unittest.mock import AsyncMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scheduler.classifier import (
    classify_specialty,
    classify_with_gemini_fallback,
    SPECIALTY_KEYS,
)


SPECIALTY_RULE_CASES = [
    ("عندي صداع شديد جداً مع تنميل بإيدي", "neurology"),
    ("صار عندي كسر بذراعي بعد ما وقعت", "orthopedics"),
    ("أنا بالشهر السابع من الحمل وصار عندي نزيف رحمي", "gynecology"),
    ("طلع عندي طفح جلدي على إيدي", "dermatology"),
    ("عندي مغص وإسهال شديد من الصبح", "gastroenterology"),
    ("السكر عندي طلع فوق 400 اليوم", "chronic_diseases"),
    ("جدي عمره 78 سنة وبده فحص دوري", "elderly"),
    ("عندي زكام بسيط من أيام", "general_practice"),
]


class TestRuleBasedClassifier(unittest.TestCase):

    def test_all_eight_specialties(self):
        for text, expected in SPECIALTY_RULE_CASES:
            with self.subTest(specialty=expected):
                r = classify_specialty(text)
                self.assertEqual(r["specialty"], expected)
                self.assertEqual(r["method"], "rule")
                self.assertIn(r["specialty"], SPECIALTY_KEYS)

    def test_no_match_falls_to_default(self):
        r = classify_specialty("بدي أسأل سؤال عادي مش طبي")
        self.assertEqual(r["specialty"], "general_practice")
        self.assertEqual(r["method"], "default")
        self.assertEqual(r["confidence"], 0.5)

    def test_empty_and_none_text_do_not_crash(self):
        for text in ("", None):
            with self.subTest(text=repr(text)):
                r = classify_specialty(text)
                self.assertEqual(r["method"], "default")

    def test_orthopedics_wataar_word_boundary(self):
        r = classify_specialty("عندي شد بمنطقة وتر الركبة")
        self.assertEqual(r["specialty"], "orthopedics")


class TestGeminiFallback(unittest.IsolatedAsyncioTestCase):

    _VAGUE = "بدي أسأل سؤال عادي مش طبي"

    async def test_gemini_used_only_when_rules_default(self):
        mock_client = AsyncMock()
        mock_client.ask.return_value = "chronic_diseases"
        r = await classify_with_gemini_fallback(self._VAGUE, mock_client)
        self.assertEqual(r["method"], "gemini")
        self.assertEqual(r["specialty"], "chronic_diseases")
        mock_client.ask.assert_awaited_once()

    async def test_gemini_not_called_when_rule_already_matched(self):
        mock_client = AsyncMock()
        r = await classify_with_gemini_fallback("صار عندي كسر بذراعي", mock_client)
        self.assertEqual(r["method"], "rule")
        mock_client.ask.assert_not_called()

    async def test_gemini_returns_unknown_key_falls_back_to_default(self):
        mock_client = AsyncMock()
        mock_client.ask.return_value = "cardiology"
        r = await classify_with_gemini_fallback(self._VAGUE, mock_client)
        self.assertEqual(r["method"], "default")
        self.assertEqual(r["specialty"], "general_practice")

    async def test_gemini_parses_trailing_punctuation(self):
        mock_client = AsyncMock()
        mock_client.ask.return_value = "chronic_diseases."
        r = await classify_with_gemini_fallback(self._VAGUE, mock_client)
        self.assertEqual(r["method"], "gemini")
        self.assertEqual(r["specialty"], "chronic_diseases")

    async def test_gemini_parses_prefixed_arabic_text(self):
        mock_client = AsyncMock()
        mock_client.ask.return_value = "الجواب: chronic_diseases"
        r = await classify_with_gemini_fallback(self._VAGUE, mock_client)
        self.assertEqual(r["method"], "gemini")
        self.assertEqual(r["specialty"], "chronic_diseases")

    async def test_gemini_empty_response_handled_gracefully(self):
        mock_client = AsyncMock()
        mock_client.ask.return_value = ""
        r = await classify_with_gemini_fallback(self._VAGUE, mock_client)
        self.assertEqual(r["method"], "default")

    async def test_gemini_raises_exception_handled_gracefully(self):
        mock_client = AsyncMock()
        mock_client.ask.side_effect = TimeoutError("network down")
        r = await classify_with_gemini_fallback(self._VAGUE, mock_client)
        self.assertEqual(r["method"], "default")


if __name__ == "__main__":
    unittest.main(verbosity=2)
