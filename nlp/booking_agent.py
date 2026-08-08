"""
nlp/booking_agent.py — LLM-first patient booking conversation turn.

Each turn returns a natural Arabic reply plus structured field extraction.
Falls back to rule-based extractor when the LLM is unavailable.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from nlp.extractor import extract_patient_fields
from nlp.gemini_client import OFF_TOPIC_REPLY, GeminiClient, gemini
from nlp.normalizer import normalize

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 12

VALID_INTENTS = frozenset({
    "continue",
    "confirm",
    "decline",
    "cancel",
    "accept_gp",
    "reject_gp",
    "inquiry",
    "contact",
    "new_booking",
    "next_slot",
    "slot_list",
    "edit_time",
    "off_topic",
})

FIELD_QUESTIONS_AR = {
    "name": "ما اسمك الكريم؟ 😊",
    "complaint": "سلامتك! شو الأعراض أو سبب الزيارة؟ 🩺",
    "urgency_score": "كيف شايف حالة المريض؟ (عاجل / متوسط / روتيني)",
    "time_pref": "متى يناسبك الموعد؟ (اليوم، بكرا، الأسبوع الجاي...)",
}


@dataclass
class BookingTurnResult:
    reply: str
    intent: str = "continue"
    extracted: dict[str, Any] = field(default_factory=dict)
    off_topic: bool = False


def trim_chat_history(history: list[dict]) -> list[dict]:
    if len(history) <= MAX_HISTORY_TURNS * 2:
        return history
    return history[-(MAX_HISTORY_TURNS * 2) :]


def append_history(history: list[dict], role: str, content: str) -> list[dict]:
    text = (content or "").strip()
    if not text:
        return history
    out = list(history) + [{"role": role, "content": text}]
    return trim_chat_history(out)


def _parse_json_response(raw: str) -> dict:
    clean = (raw or "").strip()
    if clean.startswith("```json"):
        clean = clean[7:]
    if clean.startswith("```"):
        clean = clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]
    return json.loads(clean.strip())


def _urgency_label_to_score(label: str | None) -> float | None:
    if not label:
        return None
    norm = normalize(label).lower()
    if norm in ("urgent", "عاجل", "طارئ", "فوري"):
        return 0.9
    if norm in ("medium", "متوسط", "عادي"):
        return 0.5
    if norm in ("routine", "روتيني"):
        return 0.2
    if any(w in norm for w in ("عاجل", "طارئ", "فوري", "خطير")):
        return 0.9
    if any(w in norm for w in ("متوسط", "خلال اسبوع", "عادي")):
        return 0.5
    if any(w in norm for w in ("روتيني", "مش عاجل", "اي وقت")):
        return 0.2
    return None


def _parse_time_phrase(text: str) -> dict | None:
    if not text:
        return None
    from datetime import date, timedelta

    t = normalize(text).lower()
    today = date.today()
    if "بعد بكرا" in t or "بعد غد" in t:
        return {"date": str(today + timedelta(days=2)), "phrase": "بعد بكرا"}
    if "اليوم" in t or "هلق" in t or "الان" in t:
        return {"date": str(today), "phrase": "اليوم"}
    if "بكرا" in t or "غدا" in t:
        return {"date": str(today + timedelta(days=1)), "phrase": "بكرا"}
    if "اسبوع" in t or "أسبوع" in t:
        return {"date": str(today + timedelta(days=7)), "phrase": "الأسبوع الجاي"}
    if "اي وقت" in t or "أي وقت" in t or "لا يهم" in t:
        return {"date": None, "phrase": "أي وقت متاح"}
    return {"date": None, "phrase": text.strip()}


def _validate_name(name: str | None) -> str | None:
    if not name:
        return None
    raw = name.strip()
    if not raw or raw.startswith("/"):
        return None
    if any(ch.isdigit() for ch in raw):
        return None
    words = raw.split()
    if not (1 <= len(words) <= 4) or len(raw) > 40:
        return None
    norm = normalize(raw)
    if any(w in norm for w in ("موعد", "حجز", "عاجل", "روتيني", "اليوم", "بكرا")):
        return None
    return raw


def _message_has_urgency_signal(text: str) -> bool:
    norm = normalize(text or "").lower()
    tokens = (
        "عاجل", "طارئ", "فوري", "خطير", "روتيني", "متوسط", "عادي",
        "p1", "p2", "p3", "🔴", "🟡", "🟢",
    )
    return any(t in norm for t in tokens)


def merge_rule_extracted(user_message: str, collected: dict) -> dict[str, Any]:
    """Rule-based extraction to merge with or replace LLM output."""
    fields = extract_patient_fields(user_message)
    out: dict[str, Any] = {}

    if not collected.get("name") and fields.get("name"):
        name = _validate_name(fields["name"])
        if name:
            out["name"] = name

    if not collected.get("complaint") and fields.get("complaint"):
        out["complaint"] = fields["complaint"].get("raw") or fields["complaint"]

    if collected.get("urgency_score") is None and fields.get("urgency_score") is not None:
        if _message_has_urgency_signal(user_message):
            out["urgency_score"] = fields["urgency_score"]

    tp = fields.get("time_pref") or {}
    if not collected.get("time_pref") and (tp.get("date") or tp.get("phrase")):
        out["time_pref"] = tp.get("phrase") or tp.get("date")

    return out


def apply_extracted_to_data(
    data: dict,
    extracted: dict[str, Any],
    *,
    score_from_label,
) -> None:
    """Merge agent/rule extracted fields into FSM data dict."""
    name = extracted.get("name")
    if name and not data.get("name"):
        valid = _validate_name(str(name))
        if valid:
            data["name"] = valid

    complaint = extracted.get("complaint")
    if complaint and not data.get("complaint"):
        raw = str(complaint).strip()
        if len(raw) >= 2:
            data["complaint"] = {
                "raw": raw,
                "category": "general",
                "urgency_score": 0.3,
                "specialty": "general_practice",
            }

    urgency = extracted.get("urgency") or extracted.get("urgency_score")
    if data.get("urgency_score") is None and urgency is not None:
        if isinstance(urgency, (int, float)):
            data["urgency_score"] = float(urgency)
        else:
            score = _urgency_label_to_score(str(urgency))
            if score is None and score_from_label:
                score = score_from_label(str(urgency))
            if score is not None:
                data["urgency_score"] = score

    time_pref = extracted.get("time_pref")
    if not data.get("time_pref") and time_pref:
        if isinstance(time_pref, dict):
            data["time_pref"] = time_pref
        else:
            mapped = _parse_time_phrase(str(time_pref))
            if mapped:
                data["time_pref"] = mapped

    if data.get("time_pref") and isinstance(data["time_pref"], str):
        mapped = _parse_time_phrase(data["time_pref"])
        if mapped:
            data["time_pref"] = mapped


def missing_required_fields(data: dict, required: list[str]) -> list[str]:
    missing = []
    for field_name in required:
        val = data.get(field_name)
        if val is None:
            missing.append(field_name)
        elif field_name == "time_pref" and isinstance(val, dict) and not (val.get("date") or val.get("phrase")):
            missing.append(field_name)
        elif field_name == "complaint" and not val:
            missing.append(field_name)
    return missing


def fallback_reply(
    collected: dict,
    required: list[str],
    user_message: str,
    *,
    rule_hint: str | None = None,
    phase: str = "CHATTING",
) -> BookingTurnResult:
    """Template reply when LLM is unavailable."""
    merged = merge_rule_extracted(user_message, collected)
    missing = missing_required_fields(collected, required)
    rule_intent = detect_rule_intent(user_message, rule_hint=rule_hint)

    if phase == "CONFIRM":
        return BookingTurnResult(
            reply="لسا معك بالحجز 🙂 بدك تأكيد الموعد، تشوف وقت ثاني، أو نلغي؟",
            intent=rule_intent or "continue",
            extracted=merged,
        )

    if phase in ("TERMINAL", "GP_FALLBACK"):
        if rule_intent in VALID_INTENTS:
            return BookingTurnResult(reply="", intent=rule_intent, extracted=merged)
        return BookingTurnResult(reply="", intent="continue", extracted=merged)

    if GeminiClient.looks_off_topic(user_message):
        first = missing[0] if missing else "name"
        return BookingTurnResult(
            reply=f"{OFF_TOPIC_REPLY}\n\n{FIELD_QUESTIONS_AR.get(first, FIELD_QUESTIONS_AR['name'])}",
            intent="off_topic",
            extracted=merged,
            off_topic=True,
        )

    if rule_intent in VALID_INTENTS:
        if rule_intent == "inquiry":
            return BookingTurnResult(
                reply="",
                intent="inquiry",
                extracted=merged,
            )
        if rule_intent == "contact":
            return BookingTurnResult(
                reply="📞 تواصلك وصل. يمكنك كتابة رسالتك هنا، وسيتم حفظها في سجل المحادثات للعيادة.",
                intent="contact",
                extracted=merged,
            )
        if rule_intent == "new_booking":
            return BookingTurnResult(
                reply="📅 تمام، خلينا نبدأ حجز جديد. " + FIELD_QUESTIONS_AR["name"],
                intent="new_booking",
                extracted=merged,
            )
        if rule_intent == "cancel":
            return BookingTurnResult(reply="", intent="cancel", extracted=merged)

    if not missing and merged:
        return BookingTurnResult(
            reply="تمام، خلينا نكمل الحجز. 👍",
            intent="continue",
            extracted=merged,
        )

    first = missing[0] if missing else "name"
    return BookingTurnResult(
        reply=FIELD_QUESTIONS_AR.get(first, FIELD_QUESTIONS_AR["name"]),
        intent=rule_intent or "continue",
        extracted=merged,
    )


def detect_rule_intent(user_message: str, *, rule_hint: str | None = None) -> str | None:
    """Rule-based intent hints merged with LLM output (operations stay rule-driven)."""
    if rule_hint:
        return rule_hint
    norm = normalize(user_message or "").lower()
    if not norm:
        return None
    cancel_tokens = ("الغاء", "إلغاء", "الغي", "إلغي", "كنسل", "cancel")
    if any(t in norm for t in cancel_tokens):
        return "cancel"
    if any(t in norm for t in ("حجز موعد", "موعد جديد", "ابدأ", "من جديد", "restart", "book")):
        return "new_booking"
    if "استعلام" in norm or "موعدي" in norm:
        return "inquiry"
    if "مواعيد" in norm and any(w in norm for w in ("موجود", "مسجل", "محجوز", "ضايل", "متبق", "باقي")):
        return "inquiry"
    if "موعد" in norm and any(w in norm for w in ("مسجل", "محجوز", "حجزي", "اخر", "آخر", "عندي", "وين")):
        return "inquiry"
    if "تواصل" in norm or "اتصل" in norm:
        return "contact"
    return None


async def run_booking_turn(
    user_message: str,
    phase: str,
    collected: dict,
    chat_history: list[dict],
    slot_context: dict | None = None,
    operation_context: dict | None = None,
    *,
    required_fields: list[str] | None = None,
    score_from_label=None,
) -> BookingTurnResult:
    """
    Run one LLM booking conversation turn.
    Returns reply + structured extraction for the orchestrator.
    """
    required = required_fields or ["name", "complaint", "urgency_score", "time_pref"]
    text = (user_message or "").strip()
    op_ctx = dict(operation_context or {})
    rule_intent = detect_rule_intent(text, rule_hint=op_ctx.get("rule_hint"))
    if rule_intent and "rule_hint" not in op_ctx:
        op_ctx["rule_hint"] = rule_intent

    if not gemini.is_ready:
        return fallback_reply(
            collected, required, text, rule_hint=op_ctx.get("rule_hint"), phase=phase
        )

    raw_json = await gemini.booking_turn(
        user_message=text,
        phase=phase,
        collected=collected,
        chat_history=chat_history,
        slot_context=slot_context,
        operation_context=op_ctx,
    )
    if not raw_json:
        return fallback_reply(
            collected, required, text, rule_hint=op_ctx.get("rule_hint"), phase=phase
        )

    try:
        parsed = _parse_json_response(raw_json)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("booking_turn JSON parse failed: %s (raw=%s)", exc, raw_json[:200])
        return fallback_reply(
            collected, required, text, rule_hint=op_ctx.get("rule_hint"), phase=phase
        )

    reply = (parsed.get("reply") or "").strip()
    intent = (parsed.get("intent") or "continue").strip().lower()
    if intent not in VALID_INTENTS:
        intent = "continue"
    if rule_intent in VALID_INTENTS and intent == "continue":
        intent = rule_intent

    off_topic = bool(parsed.get("off_topic")) or intent == "off_topic"
    extracted_raw = parsed.get("extracted") or {}
    if not isinstance(extracted_raw, dict):
        extracted_raw = {}

    extracted: dict[str, Any] = {}
    if not off_topic:
        if extracted_raw.get("name"):
            extracted["name"] = extracted_raw["name"]
        if extracted_raw.get("complaint"):
            extracted["complaint"] = extracted_raw["complaint"]
        if extracted_raw.get("urgency"):
            extracted["urgency"] = extracted_raw["urgency"]
        if extracted_raw.get("time_pref"):
            extracted["time_pref"] = extracted_raw["time_pref"]
        rule_merge = merge_rule_extracted(text, collected)
        for k, v in rule_merge.items():
            extracted.setdefault(k, v)

    if off_topic and not reply:
        first = missing_required_fields(collected, required)
        q = FIELD_QUESTIONS_AR.get(first[0] if first else "name", FIELD_QUESTIONS_AR["name"])
        reply = f"{OFF_TOPIC_REPLY}\n\n{q}"

    if not reply:
        fb = fallback_reply(
            collected, required, text, rule_hint=op_ctx.get("rule_hint"), phase=phase
        )
        reply = fb.reply
        if intent == "continue" and fb.intent != "continue":
            intent = fb.intent
        for k, v in fb.extracted.items():
            extracted.setdefault(k, v)

    return BookingTurnResult(
        reply=reply,
        intent=intent,
        extracted=extracted,
        off_topic=off_topic,
    )
