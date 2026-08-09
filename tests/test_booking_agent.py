"""Unit tests for nlp/booking_agent.py."""
from nlp.booking_agent import (
    apply_extracted_to_data,
    fallback_reply,
    merge_rule_extracted,
    missing_required_fields,
    run_booking_turn,
)
from tests.helpers import run_async


def test_merge_rule_extracted_name():
    out = merge_rule_extracted("انا اسمي أحمد", {})
    assert out.get("name")
    assert "حمد" in out["name"]


def test_apply_extracted_complaint_and_urgency():
    data: dict = {}
    apply_extracted_to_data(
        data,
        {"complaint": "صداع", "urgency": "روتيني", "time_pref": "بكرا"},
        score_from_label=lambda _: 0.2,
    )
    assert data["complaint"]["raw"] == "صداع"
    assert data["urgency_score"] == 0.2
    assert data["time_pref"]["phrase"] == "بكرا"


def test_missing_required_fields():
    data = {"name": "سارة", "complaint": {"raw": "كحة"}}
    missing = missing_required_fields(data, ["name", "complaint", "urgency_score", "time_pref"])
    assert "urgency_score" in missing
    assert "time_pref" in missing


def test_fallback_reply_asks_for_name():
    turn = fallback_reply({}, ["name", "complaint", "urgency_score", "time_pref"], "مرحبا")
    assert "اسم" in turn.reply
    assert turn.intent == "continue"


def test_run_booking_turn_offline():
    turn = run_async(run_booking_turn("أحمد", "CHATTING", {}, []))
    assert turn.reply
    assert turn.intent == "continue"
