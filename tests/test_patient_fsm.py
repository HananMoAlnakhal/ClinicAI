"""Unit tests for fsm/patient_fsm.py — patient booking FSM."""
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.keyboards import specialty_keyboard
from fsm.patient_fsm import FIELD_QUESTIONS_AR, REQUIRED_FIELDS, SPECIALTY_LABEL_TO_KEY, PatientFSM, State
from scheduler.classifier import SPECIALTY_NAMES_AR
from tests.helpers import run_async, unpack_fsm


def _high_confidence_classify(*args, **kwargs):
    return {
        "specialty": "general_practice",
        "specialty_ar": "الطب العام",
        "method": "rule",
        "confidence": 0.95,
    }


@pytest.fixture
def patient_fsm():
    return PatientFSM(user_id=80002)


def test_parse_time_label_today_and_tomorrow():
    fsm = PatientFSM(user_id=80001)
    today = fsm._parse_time_label("بدي موعد اليوم")
    assert today["date"] == str(date.today())
    tomorrow = fsm._parse_time_label("بكرا")
    assert tomorrow["date"] == str(date.today() + timedelta(days=1))


def test_parse_specialty_label_arabic():
    fsm = PatientFSM(user_id=80001)
    assert fsm._parse_specialty_label("عظام") == "orthopedics"
    assert fsm._parse_specialty_label("هضمي") == "gastroenterology"


def test_absorb_urgency_levels():
    fsm = PatientFSM(user_id=80001)
    fsm._absorb_urgency("عاجل جدا")
    assert fsm.data["urgency_score"] >= 0.85

    fsm.data.clear()
    fsm._absorb_urgency("روتيني")
    assert fsm.data["urgency_score"] <= 0.25


def test_missing_fields_detects_empty_time_pref():
    fsm = PatientFSM(user_id=80001)
    fsm.data = {
        "name": "أحمد",
        "complaint": {"raw": "صداع"},
        "urgency_score": 0.4,
        "time_pref": {"date": None, "phrase": ""},
    }
    assert "time_pref" in fsm._missing_fields()


def test_reset_clears_booking_state():
    fsm = PatientFSM(user_id=80001)
    fsm.state = State.FINALIZED
    fsm.data = {"name": "x"}
    fsm.slot = {"slot_id": 1}
    fsm._reset()
    assert fsm.state == State.GREETING
    assert fsm.data == {}
    assert fsm.slot is None


def test_greeting_asks_for_name(patient_fsm):
    reply, keyboard = unpack_fsm(run_async(patient_fsm.handle("مرحبا")))
    assert patient_fsm.state == State.COLLECT_NAME
    assert FIELD_QUESTIONS_AR["name"] in reply
    assert keyboard is None


def test_collect_name_then_complaint(patient_fsm):
    run_async(patient_fsm.handle("start"))
    reply, _ = unpack_fsm(run_async(patient_fsm.handle("سارة محمود")))
    assert patient_fsm.state == State.COLLECT_COMPLAINT
    assert "سارة" in reply


def test_collect_complaint_from_plain_text(patient_fsm):
    patient_fsm.state = State.COLLECT_COMPLAINT
    reply, keyboard = unpack_fsm(run_async(patient_fsm.handle("عندي صداع من يومين")))
    assert patient_fsm.state == State.COLLECT_URGENCY
    assert "complaint" in patient_fsm.data
    assert keyboard is not None


def test_urgency_callback_moves_to_time(patient_fsm):
    patient_fsm.state = State.COLLECT_URGENCY
    patient_fsm.data = {"name": "أحمد", "complaint": {"raw": "صداع"}}
    reply, keyboard = unpack_fsm(run_async(patient_fsm.handle_callback("urgency:P2")))
    assert patient_fsm.state == State.COLLECT_TIME
    assert patient_fsm.data["urgency_score"] == 0.5
    assert keyboard is not None


@patch("fsm.patient_fsm.gemini")
@patch("fsm.patient_fsm.classify_with_gemini_fallback", side_effect=_high_confidence_classify)
@patch("fsm.patient_fsm.classify_specialty", side_effect=_high_confidence_classify)
@patch("fsm.patient_fsm.score_and_classify")
def test_full_flow_reaches_confirm_with_slot(mock_score, _mock_classify_rules, mock_classify, mock_gemini, patient_fsm):
    mock_gemini._available = False
    mock_gemini.extract_missing_field = AsyncMock(return_value=None)
    priority = MagicMock(
        priority_class="P3",
        score=0.35,
        label_ar="روتيني",
        breakdown={"f1": 0.3},
    )
    mock_score.return_value = priority

    slot_dt = datetime.utcnow() + timedelta(days=2)
    slot = MagicMock(
        slot_id=99,
        slot_datetime=slot_dt,
        specialty="general_practice",
        priority_class="P3",
        doctor=MagicMock(
            doctor_id=1,
            name="د. اختبار",
            specialty="general_practice",
            clinic_code="CLINIC-GP",
            clinic_name="عيادة عامة",
        ),
    )

    @contextmanager
    def fake_get_db():
        yield MagicMock()

    with patch("database.db.get_db", fake_get_db):
        patient_fsm.services.classify = _high_confidence_classify
        patient_fsm.services.classify_with_fallback = mock_classify
        patient_fsm.services.score = mock_score
        patient_fsm.services.find_slots = MagicMock(return_value=[slot])
        patient_fsm.services.gemini = mock_gemini
        run_async(patient_fsm.handle("start"))
        run_async(patient_fsm.handle("أحمد"))
        run_async(patient_fsm.handle("صداع خفيف"))
        run_async(patient_fsm.handle("روتيني"))
        reply, keyboard = unpack_fsm(run_async(patient_fsm.handle("بكرا")))

    assert patient_fsm.state == State.CONFIRM
    assert "وجدت موعد" in reply
    assert "درجة الأولوية" not in reply
    assert keyboard is not None
    assert patient_fsm.slot["slot_id"] == 99


@patch("fsm.patient_fsm.classify_specialty", side_effect=_high_confidence_classify)
@patch("fsm.patient_fsm.score_and_classify")
def test_confirm_yes_finalizes_booking(mock_score, _mock_classify, patient_fsm):
    priority = MagicMock(priority_class="P3", score=0.3, label_ar="روتيني", breakdown={})
    mock_score.return_value = priority

    patient_fsm.state = State.CONFIRM
    patient_fsm.data = {
        "name": "أحمد",
        "complaint": {"raw": "صداع"},
        "urgency_score": 0.3,
        "time_pref": {"date": str(date.today() + timedelta(days=1)), "phrase": "بكرا"},
        "specialty_hint": "general_practice",
        "specialty_ar": "الطب العام",
        "priority_class": "P3",
    }
    patient_fsm.slot = {
        "slot_id": 5,
        "slot_datetime": datetime.utcnow() + timedelta(days=1),
        "doctor_name": "د. اختبار",
        "clinic_name": "عيادة",
    }
    patient_fsm.priority = priority

    appt = MagicMock()
    appt.appt_id = "appt_test_99"
    appt.appt_datetime = patient_fsm.slot["slot_datetime"]

    @contextmanager
    def fake_get_db():
        yield MagicMock()

    with patch("database.db.get_db", fake_get_db):
        patient_fsm.services.book = MagicMock(
            return_value={"appointment": appt, "slot_conflict": False, "booking_conflict": None},
        )
        reply, _ = unpack_fsm(run_async(patient_fsm.handle("نعم")))

    assert patient_fsm.state == State.FINALIZED
    assert "تم تأكيد حجزك" in reply
    assert patient_fsm.finalized_appointment_id == "appt_test_99"


def test_confirm_no_cancels(patient_fsm):
    patient_fsm.state = State.CONFIRM
    patient_fsm.slot = {"slot_id": 1}
    reply, _ = unpack_fsm(run_async(patient_fsm.handle("لا")))
    assert patient_fsm.state == State.CANCELLED
    assert "تم الإلغاء" in reply


def test_low_confidence_specialty_prompts_user(patient_fsm):
    patient_fsm.state = State.COLLECT_TIME
    patient_fsm.data = {
        "name": "ليلى",
        "complaint": {"raw": "سؤال عام"},
        "urgency_score": 0.3,
        "time_pref": {"date": str(date.today()), "phrase": "اليوم"},
    }

    with patch("fsm.patient_fsm.gemini") as mock_gemini:
        mock_gemini._available = False
        patient_fsm.services.classify = MagicMock(
            return_value={
                "specialty": "general_practice",
                "specialty_ar": "الطب العام",
                "method": "default",
                "confidence": 0.5,
            },
        )
        reply, keyboard = unpack_fsm(run_async(patient_fsm.handle("اليوم")))

    assert patient_fsm.state == State.COLLECT_SPECIALTY
    assert "تخصص" in reply
    assert keyboard is not None


def test_required_fields_match_questions():
    for field_name in REQUIRED_FIELDS:
        assert field_name in FIELD_QUESTIONS_AR


def test_validate_blocks_scheduling_until_checklist_complete():
    fsm = PatientFSM(user_id=81001)
    fsm.state = State.COLLECT_TIME
    fsm.data = {"name": "أحمد", "complaint": {"raw": "صداع"}, "urgency_score": 0.4}

    with patch("fsm.patient_fsm.classify_specialty") as mock_classify:
        mock_classify.assert_not_called()
        reply, _ = unpack_fsm(run_async(fsm.handle("")))
    assert fsm.state == State.COLLECT_TIME
    assert "محتاجين" in reply


def test_clarification_does_not_hijack_normal_booking_phrase():
    fsm = PatientFSM(user_id=81002)
    fsm.state = State.COLLECT_COMPLAINT
    fsm.data = {"name": "سارة"}

    reply, _ = unpack_fsm(run_async(fsm.handle("فهمت، عندي صداع")))

    assert fsm.state == State.COLLECT_URGENCY
    assert "فهمت إن" not in reply


def test_parse_specialty_arabic_full_label():
    fsm = PatientFSM(user_id=81003)
    assert fsm._parse_specialty_label("الطب العام") == "general_practice"


def test_unclear_urgency_stays_in_collect_urgency():
    fsm = PatientFSM(user_id=81005)
    fsm.state = State.COLLECT_URGENCY
    fsm.data = {"name": "أحمد", "complaint": {"raw": "صداع"}}

    reply, keyboard = unpack_fsm(run_async(fsm.handle("xyz gibberish")))

    assert fsm.state == State.COLLECT_URGENCY
    assert "urgency_score" not in fsm.data
    assert "الأولوية" in reply
    assert keyboard is not None


def test_specialty_label_keys_match_classifier():
    for key in set(SPECIALTY_LABEL_TO_KEY.values()):
        assert key in SPECIALTY_NAMES_AR, f"missing classifier entry for {key}"


def test_keyboard_specialty_labels_parse():
    fsm = PatientFSM(user_id=81006)
    allowed = {
        "gastroenterology",
        "neurology",
        "orthopedics",
        "gynecology",
        "chronic_diseases",
        "dermatology",
        "general_practice",
        "elderly",
    }
    for row in specialty_keyboard().keyboard:
        for button in row:
            key = fsm._parse_specialty_label(button.text)
            assert key is not None, f"failed to parse specialty button: {button.text!r}"
            assert key in allowed


@patch("fsm.patient_fsm.classify_specialty", side_effect=_high_confidence_classify)
@patch("fsm.patient_fsm.score_and_classify")
def test_slot_conflict_retries_with_next_available_slot(mock_score, _mock_classify):
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
    fsm.slot = {
        "slot_id": 1,
        "slot_datetime": datetime.utcnow() + timedelta(days=1),
        "doctor_name": "د.",
        "clinic_name": "عيادة",
    }

    new_slot = MagicMock(
        slot_id=2,
        slot_datetime=datetime.utcnow() + timedelta(days=2),
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
    assert fsm.state == State.CONFIRM
    assert fsm.slot["slot_id"] == 2
    assert "انحجز قبل التأكيد" in reply
    assert "وجدت موعد" in reply
    assert keyboard is not None


def test_unsupported_specialty_offers_gp_fallback():
    fsm = PatientFSM(user_id=81007)
    fsm.state = State.COLLECT_COMPLAINT
    fsm.data = {"name": "أحمد"}
    reply, keyboard = unpack_fsm(run_async(fsm.handle("بدي موعد عند طبيب اسنان")))
    assert fsm.state == State.OFFER_GP_FALLBACK
    assert "غير متوفرة" in reply
    assert keyboard is not None


def test_urgency_free_text_accepted():
    fsm = PatientFSM(user_id=81008)
    fsm.state = State.COLLECT_URGENCY
    fsm.data = {"name": "أحمد", "complaint": {"raw": "صداع"}}
    reply, keyboard = unpack_fsm(run_async(fsm.handle("عاجل")))
    assert fsm.state == State.COLLECT_TIME
    assert fsm.data["urgency_score"] >= 0.85
    assert keyboard is not None


def test_confirm_edit_returns_to_time_selection():
    fsm = PatientFSM(user_id=81009)
    fsm.state = State.CONFIRM
    fsm.slot = {"slot_id": 1, "slot_datetime": datetime.utcnow()}
    fsm.data = {"name": "أحمد", "time_pref": {"date": str(date.today()), "phrase": "اليوم"}}
    reply, keyboard = unpack_fsm(run_async(fsm.handle("✏️ تعديل الموعد")))
    assert fsm.state == State.COLLECT_TIME
    assert "متى" in reply
    assert keyboard is not None
