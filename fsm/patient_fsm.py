"""
fsm/patient_fsm.py
Patient appointment booking FSM:
  collect data → validate checklist → classify clinic → score priority
  → create/update patient file → check DB slots → confirm → reserve slot/book appointment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from nlp.extractor import extract_patient_fields
from nlp.normalizer import normalize
from scheduler.priority import score_and_classify
from scheduler.classifier import classify_specialty, SPECIALTY_NAMES_AR
from nlp.gemini_client import gemini
from bot.keyboards import urgency_keyboard, time_pref_keyboard, confirm_keyboard, specialty_keyboard


class State(Enum):
    GREETING = auto()
    COLLECT_NAME = auto()
    COLLECT_COMPLAINT = auto()
    COLLECT_URGENCY = auto()
    COLLECT_TIME = auto()
    VALIDATE = auto()          # requirements-loop checkpoint
    COLLECT_SPECIALTY = auto() # used when classifier confidence is low
    CLASSIFY = auto()          # specialty + priority
    FIND_SLOT = auto()
    CONFIRM = auto()
    FINALIZED = auto()
    WAITLISTED = auto()
    CANCELLED = auto()


# Required fields — the checklist before touching scheduling.
REQUIRED_FIELDS: list[str] = ["name", "complaint", "urgency_score", "time_pref"]

FIELD_QUESTIONS_AR: dict[str, str] = {
    "name": "ما اسمك الكريم؟",
    "complaint": "شو الشكوى أو سبب الزيارة؟",
    "urgency_score": "هل الموضوع عاجل، متوسط، أو روتيني؟",
    "time_pref": "متى تحب الموعد؟ (اليوم، بكرا، الأسبوع الجاي...)",
}

CONFIRM_WORDS = {"نعم", "ايوه", "آيوه", "تمام", "ماشي", "اوك", "ok", "yes", "يلا", "احجز", "تاكيد", "تأكيد", "تاكيد الحجز", "تأكيد الحجز"}
CANCEL_WORDS = {"لا", "الغي", "إلغي", "الغاء", "إلغاء", "بدي الغي", "مش حابب", "no"}

SPECIALTY_LABEL_TO_KEY = {
    "قلب": "cardiology",
    "اعصاب": "neurology",
    "أعصاب": "neurology",
    "عظام": "orthopedics",
    "مفاصل": "orthopedics",
    "نساء": "gynecology",
    "توليد": "gynecology",
    "اطفال": "pediatrics",
    "أطفال": "pediatrics",
    "اسنان": "dentistry",
    "أسنان": "dentistry",
    "عيون": "ophthalmology",
    "جلدية": "dermatology",
    "جلديه": "dermatology",
    "طب عام": "general_practice",
    "عام": "general_practice",
}


@dataclass
class PatientFSM:
    user_id: int
    state: State = State.GREETING
    data: dict = field(default_factory=dict)
    slot: Optional[dict] = None              # Plain dict, not detached ORM object
    priority: Optional[object] = None        # PriorityResult
    finalized_appointment_id: Optional[str] = None

    # ── Public entry ──────────────────────────────────────────────────────────

    async def handle(self, text: str) -> tuple[str, object | None]:
        """Process one message, advance state, return Arabic reply plus keyboard."""
        text = text or ""
        norm = normalize(text)

        if self.state in (State.FINALIZED, State.CANCELLED, State.WAITLISTED):
            if self._is_new_booking_request(text):
                self._reset()
            else:
                return self._reply("تم إنهاء الطلب السابق. إذا بدك حجز جديد اكتب: حجز موعد جديد 📅", None)

        await self._absorb(text)  # always try to extract from every message

        if self._is_clarification_request(text):
            return await self._reply_with_context(text)

        if self.state == State.GREETING:
            self.state = State.COLLECT_NAME
            return self._reply(
                "أهلاً وسهلاً 👋 أنا المساعد الذكي للحجز في العيادة.\n" + FIELD_QUESTIONS_AR["name"],
                None,
            )

        if self.state == State.COLLECT_NAME:
            if self.data.get("name"):
                self.state = State.COLLECT_COMPLAINT
                return self._reply(f"أهلاً {self.data['name']}! 😊\n" + FIELD_QUESTIONS_AR["complaint"], None)

            if text.strip() and not text.strip().startswith("/"):
                self.data["name"] = text.strip()
                self.state = State.COLLECT_COMPLAINT
                return self._reply(f"أهلاً {self.data['name']}! 😊\n" + FIELD_QUESTIONS_AR["complaint"], None)

            return self._reply(FIELD_QUESTIONS_AR["name"], None)

        if self.state == State.COLLECT_COMPLAINT:
            if self.data.get("complaint"):
                self.state = State.COLLECT_URGENCY
                return self._reply(FIELD_QUESTIONS_AR["urgency_score"], urgency_keyboard())

            complaint = await self._extract_complaint_from_text(text)
            if complaint:
                self.data["complaint"] = complaint
                self.state = State.COLLECT_URGENCY
                return self._reply(FIELD_QUESTIONS_AR["urgency_score"], urgency_keyboard())

            if text.strip():
                self.data["complaint"] = {
                    "raw": text.strip(),
                    "category": "general",
                    "urgency_score": 0.3,
                    "specialty": "general_practice",
                }
                self.state = State.COLLECT_URGENCY
                return self._reply(FIELD_QUESTIONS_AR["urgency_score"], urgency_keyboard())

            return self._reply("ممكن تخبرني أكثر عن سبب زيارتك؟", None)

        if self.state == State.COLLECT_URGENCY:
            self._absorb_urgency(norm)
            self.state = State.COLLECT_TIME
            return self._reply(FIELD_QUESTIONS_AR["time_pref"], time_pref_keyboard())

        if self.state == State.COLLECT_TIME:
            mapped = self._parse_time_label(text)
            if mapped:
                self.data["time_pref"] = mapped
            return await self._run_validate()

        if self.state == State.COLLECT_SPECIALTY:
            specialty_key = self._parse_specialty_label(text)
            if not specialty_key:
                return self._reply("اختاري/اختر التخصص الأقرب من الأزرار حتى أحجز الموعد في العيادة المناسبة.", specialty_keyboard())
            self.data["specialty_hint"] = specialty_key
            self.data["specialty_ar"] = SPECIALTY_NAMES_AR.get(specialty_key, specialty_key)
            self.data["specialty_confirmed_by_patient"] = True
            return await self._score_and_find_slot()

        if self.state == State.CONFIRM:
            return await self._handle_confirm(norm)

        return self._reply("عفواً، ما فهمت. ممكن تعيد؟", None)

    async def handle_callback(self, data: str) -> tuple[str, object | None]:
        # Kept for compatibility if inline keyboards are added later.
        if data.startswith("urgency:"):
            level = data.split(":", 1)[1]
            self._set_urgency_from_label(level)
            self.state = State.COLLECT_TIME
            return self._reply(FIELD_QUESTIONS_AR["time_pref"], time_pref_keyboard())

        if data.startswith("time:"):
            self.data["time_pref"] = self._map_time_selection(data.split(":", 1)[1])
            return await self._run_validate()

        if data.startswith("spec:"):
            specialty_key = data.split(":", 1)[1]
            if specialty_key in SPECIALTY_NAMES_AR:
                self.data["specialty_hint"] = specialty_key
                self.data["specialty_ar"] = SPECIALTY_NAMES_AR[specialty_key]
                self.data["specialty_confirmed_by_patient"] = True
                return await self._score_and_find_slot()

        if data.startswith("confirm:"):
            return await self._handle_confirm("نعم" if data.endswith("yes") else "لا")

        return self._reply("عفواً، لم أفهم اختيارك. حاول مرة أخرى.", None)

    def _reply(self, text: str, keyboard: object | None = None) -> tuple[str, object | None]:
        return text, keyboard

    # ── Extraction / validation ───────────────────────────────────────────────

    async def _absorb(self, text: str):
        """Merge newly extracted fields and optionally enrich with AI."""
        extracted = extract_patient_fields(text)
        for k, v in extracted.items():
            if v is not None and not self.data.get(k):
                self.data[k] = v

        if text.strip() and len(text.strip()) > 20 and any(self.data.get(f) is None for f in REQUIRED_FIELDS):
            await self._try_ai_extraction(text)

    async def _extract_complaint_from_text(self, original_text: str) -> dict | None:
        complaint = extract_patient_fields(original_text).get("complaint")
        if complaint:
            return complaint

        if gemini._available:
            complaint_text = await gemini.extract_missing_field(original_text, "complaint")
            if complaint_text:
                return {
                    "raw": complaint_text.strip(),
                    "category": "general",
                    "urgency_score": 0.3,
                    "specialty": "general_practice",
                }
        return None

    def _missing_fields(self) -> list[str]:
        missing = []
        for field_name in REQUIRED_FIELDS:
            val = self.data.get(field_name)
            if val is None:
                missing.append(field_name)
            elif field_name == "time_pref" and isinstance(val, dict) and not (val.get("date") or val.get("phrase")):
                missing.append(field_name)
            elif field_name == "complaint" and not val:
                missing.append(field_name)
        return missing

    async def _run_validate(self) -> tuple[str, object | None]:
        """Checklist loop: never schedule until all required data is available."""
        self.state = State.VALIDATE
        missing = self._missing_fields()
        if missing:
            first_missing = missing[0]
            self.state = {
                "name": State.COLLECT_NAME,
                "complaint": State.COLLECT_COMPLAINT,
                "urgency_score": State.COLLECT_URGENCY,
                "time_pref": State.COLLECT_TIME,
            }[first_missing]
            keyboard = urgency_keyboard() if first_missing == "urgency_score" else time_pref_keyboard() if first_missing == "time_pref" else None
            return (f"بعدنا محتاجين معلومة واحدة 📋\n{FIELD_QUESTIONS_AR[first_missing]}", keyboard)

        return await self._classify_and_schedule()

    # ── Classify + priority + schedule ────────────────────────────────────────

    async def _classify_and_schedule(self) -> tuple[str, object | None]:
        self.state = State.CLASSIFY

        norm_complaint = normalize(self.data.get("complaint", {}).get("raw", ""))
        spec_result = classify_specialty(norm_complaint)
        self.data["specialty_hint"] = spec_result["specialty"]
        self.data["specialty_ar"] = spec_result["specialty_ar"]
        self.data["specialty_method"] = spec_result.get("method")
        self.data["specialty_confidence"] = spec_result.get("confidence")

        # If the classifier only used the default fallback, ask the patient to choose.
        if spec_result.get("method") == "default" or float(spec_result.get("confidence", 0)) < 0.75:
            self.state = State.COLLECT_SPECIALTY
            return self._reply(
                "حتى أحجزك في العيادة المناسبة، اختاري/اختر أقرب تخصص للشكوى:",
                specialty_keyboard(),
            )

        return await self._score_and_find_slot()

    async def _score_and_find_slot(self) -> tuple[str, object | None]:
        self.priority = score_and_classify(self.data)
        self.data["priority_class"] = self.priority.priority_class
        self.data["priority_score"] = self.priority.score
        self.data["priority_breakdown"] = self.priority.breakdown

        self.state = State.FIND_SLOT
        return await self._find_slot()

    async def _find_slot(self) -> tuple[str, object | None]:
        from database.db import get_db
        from database import crud

        with get_db() as db:
            slot = crud.find_next_available_slot(
                db,
                specialty=self.data.get("specialty_hint", "general_practice"),
                priority_class=self.priority.priority_class,
                preferred_date=self.data.get("time_pref", {}).get("date"),
                telegram_id=self.user_id,
            )
            if slot:
                self.slot = {
                    "slot_id": slot.slot_id,
                    "slot_datetime": slot.slot_datetime,
                    "specialty": slot.doctor.specialty if slot.doctor else slot.specialty,
                    "priority_class": slot.priority_class,
                    "doctor_id": slot.doctor.doctor_id if slot.doctor else None,
                    "doctor_name": slot.doctor.name if slot.doctor else None,
                    "clinic_code": slot.doctor.clinic_code if slot.doctor else None,
                    "clinic_name": slot.doctor.clinic_name if slot.doctor else None,
                }

        if not self.slot:
            await self._save_waitlist()
            self.state = State.WAITLISTED
            return (
                "عفواً، ما في مواعيد متاحة حالياً في هذا الاختصاص. 😔\n"
                "تم حفظ ملفك وإضافتك لقائمة الانتظار، وسنتواصل معك بأقرب وقت.",
                None,
            )

        self.state = State.CONFIRM
        dt = self.slot["slot_datetime"].strftime("%A، %d/%m/%Y — %H:%M")
        return (
            f"وجدت موعد مناسب! 📅\n\n"
            f"📆 {dt}\n"
            f"🏥 التخصص: {self.data.get('specialty_ar', '')}\n"
            f"👨‍⚕️ الطبيب: {self.slot.get('doctor_name') or '—'}\n"
            f"🏢 العيادة: {self.slot.get('clinic_name') or '—'} ({self.slot.get('clinic_code') or '—'})\n"
            f"{self.priority.label_ar} — درجة الأولوية: {self.priority.score:.2f}\n\n"
            f"تأكد الحجز؟",
            confirm_keyboard(),
        )

    async def _handle_confirm(self, norm: str) -> tuple[str, object | None]:
        if any(w in norm for w in CONFIRM_WORDS):
            result = await self._finalize()
            if result.get("slot_conflict"):
                self.slot = None
                self.state = State.FIND_SLOT
                return (
                    "للأسف الموعد انحجز قبل التأكيد بثواني. رح أبحث لك عن أقرب موعد بديل الآن.",
                    None,
                )

            if result.get("booking_conflict"):
                conflict = result["booking_conflict"]
                existing = conflict.get("appointment")
                if isinstance(existing, dict):
                    existing_dt = existing.get("appt_datetime")
                    when = existing_dt.strftime("%A، %d/%m/%Y — %H:%M") if existing_dt else "موعد سابق"
                    specialty = existing.get("specialty_ar") or existing.get("specialty") or "نفس التخصص"
                else:
                    when = existing.appt_datetime.strftime("%A، %d/%m/%Y — %H:%M") if existing and existing.appt_datetime else "موعد سابق"
                    specialty = (existing.specialty_ar or existing.specialty or "نفس التخصص") if existing else "نفس التخصص"
                self.state = State.FINALIZED
                if conflict.get("type") == "time_overlap":
                    return (
                        "ما بقدر أثبت هذا الموعد لأن عندك موعد آخر بنفس الوقت أو وقت متداخل. ⏰\n"
                        f"موعدك الحالي: {when} — {specialty}.\n"
                        "ممكن تحجز موعدًا بتخصص مختلف في نفس اليوم بشرط يكون بوقت آخر غير متداخل.",
                        None,
                    )
                return (
                    "عندك موعد فعال مسبقًا لنفس التخصص في نفس اليوم، لذلك ما حجزت موعدًا ثانيًا. ✅\n"
                    f"موعدك الحالي: {when} — {specialty}.\n"
                    "لو بدك تشوف تخصصًا مختلفًا، ممكن تحجز موعدًا آخر بوقت غير متداخل.",
                    None,
                )

            if not result.get("appointment"):
                self.state = State.WAITLISTED
                return (
                    "تم حفظ ملفك، لكن لم أستطع تثبيت الموعد حالياً. أضفتك لقائمة الانتظار وسنتواصل معك. 🌿",
                    None,
                )

            appt = result["appointment"]
            self.finalized_appointment_id = appt.appt_id
            self.state = State.FINALIZED
            return (
                "✅ تم تأكيد حجزك وحفظ ملفك في النظام!\n"
                f"رقم الحجز: {appt.appt_id}\n"
                f"📆 {appt.appt_datetime.strftime('%A، %d/%m/%Y — %H:%M')}\n"
                f"👨‍⚕️ الطبيب: {self.slot.get('doctor_name') or '—'}\n"
                f"🏢 العيادة: {self.slot.get('clinic_name') or '—'}\n"
                "سيظهر الموعد تلقائياً في لوحة التحكم كموعد محجوز. نتمنى لك الشفاء 🌿",
                None,
            )

        if any(w in norm for w in CANCEL_WORDS):
            self.state = State.CANCELLED
            return ("تم الإلغاء. إذا احتجت أي شيء، أنا هون. 👋", None)

        return ("اكتب نعم لتأكيد الحجز، أو لا للإلغاء.", confirm_keyboard())

    async def _finalize(self) -> dict:
        from database.db import get_db
        from database import crud

        with get_db() as db:
            return crud.create_patient_file_and_book(
                db=db,
                telegram_id=self.user_id,
                data=self.data,
                slot_id=self.slot["slot_id"] if self.slot else None,
            )

    async def _save_waitlist(self) -> None:
        from database.db import get_db
        from database import crud

        with get_db() as db:
            crud.create_waitlist_appointment(db, self.user_id, self.data)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _absorb_urgency(self, norm: str):
        if any(w in norm for w in ["عاجل", "طارئ", "فوري", "خطير"]):
            self.data["urgency_score"] = max(float(self.data.get("urgency_score", 0)), 0.85)
        elif any(w in norm for w in ["روتيني", "مش عاجل", "اي وقت", "أي وقت"]):
            self.data["urgency_score"] = min(float(self.data.get("urgency_score", 0.5)), 0.25)
        else:
            self.data.setdefault("urgency_score", 0.4)

    def _score_from_label(self, label: str | None) -> float | None:
        if not label:
            return None
        label = label.lower()
        if any(w in label for w in ["عاجل", "طارئ", "فوري", "خطر"]):
            return 0.9
        if any(w in label for w in ["متوسط", "خلال أسبوع", "خلال اسبوع", "عادي"]):
            return 0.5
        if any(w in label for w in ["روتيني", "مش عاجل", "أي وقت", "اي وقت"]):
            return 0.2
        return None

    def _set_urgency_from_label(self, label: str):
        if label == "P1":
            score = 0.9
        elif label == "P2":
            score = 0.5
        elif label == "P3":
            score = 0.2
        else:
            score = self._score_from_label(label)
        if score is not None:
            self.data["urgency_score"] = score

    def _map_time_selection(self, selection: str) -> dict:
        from datetime import date, timedelta

        today = date.today()
        if selection == "today":
            return {"date": str(today), "phrase": "اليوم"}
        if selection == "tomorrow":
            return {"date": str(today + timedelta(days=1)), "phrase": "بكرا"}
        if selection == "day_after":
            return {"date": str(today + timedelta(days=2)), "phrase": "بعد بكرا"}
        if selection == "next_week":
            return {"date": str(today + timedelta(days=7)), "phrase": "الأسبوع الجاي"}
        return {"date": None, "phrase": "أي وقت متاح"}

    def _parse_time_label(self, text: str) -> dict | None:
        if not text:
            return None
        t = text.lower()
        from datetime import date, timedelta

        today = date.today()
        if "بعد بكرا" in t or "بعد غد" in t:
            return {"date": str(today + timedelta(days=2)), "phrase": "بعد بكرا"}
        if "اليوم" in t:
            return {"date": str(today), "phrase": "اليوم"}
        if "بكرا" in t or "غدا" in t or "غداً" in t:
            return {"date": str(today + timedelta(days=1)), "phrase": "بكرا"}
        if "أسبوع" in t or "اسبوع" in t or "الأسبوع" in t:
            return {"date": str(today + timedelta(days=7)), "phrase": "الأسبوع الجاي"}
        if "أي وقت" in t or "اي وقت" in t or "لا يهم" in t or "أي وقت متاح" in t:
            return {"date": None, "phrase": "أي وقت متاح"}
        return {"date": None, "phrase": text.strip()} if text.strip() else None

    def _parse_specialty_label(self, text: str) -> str | None:
        norm = normalize(text or "")
        for label_part, key in SPECIALTY_LABEL_TO_KEY.items():
            if normalize(label_part) in norm:
                return key
        if norm in SPECIALTY_NAMES_AR:
            return norm
        return None

    async def _try_ai_extraction(self, text: str):
        if not gemini._available:
            return

        if not self.data.get("name"):
            name = await gemini.extract_missing_field(text, "name")
            if name:
                self.data["name"] = name.strip()

        if not self.data.get("complaint"):
            complaint = await gemini.extract_missing_field(text, "complaint")
            if complaint:
                self.data["complaint"] = {
                    "raw": complaint.strip(),
                    "category": "general",
                    "urgency_score": 0.3,
                    "specialty": "general_practice",
                }

        if not self.data.get("urgency_score"):
            urgency = await gemini.extract_missing_field(text, "urgency")
            score = self._score_from_label(urgency)
            if score is not None:
                self.data["urgency_score"] = score

        if not self.data.get("time_pref"):
            time_pref = await gemini.extract_missing_field(text, "time_pref")
            if time_pref:
                self.data["time_pref"] = {"date": None, "phrase": time_pref.strip()}

    async def _reply_with_context(self, text: str) -> tuple[str, object | None]:
        if self.data:
            prompt = (
                f"المستخدم قال: {text}\n"
                f"البيانات المتوفرة عنه: {self.data}\n"
                f"الحالة الحالية: {self.state.name}\n"
                "اكتب ردًا عربيًا فلسطينيًا ودودًا ومباشرًا، مختصرًا، يوضح ما فهمته من المستخدم."
            )
            try:
                reply = await gemini.ask(prompt, max_tokens=120)
                if reply:
                    return self._reply(reply, None)
            except Exception:
                pass

        if self.data.get("name") and self.data.get("complaint"):
            complaint = self.data["complaint"].get("raw") if isinstance(self.data["complaint"], dict) else self.data["complaint"]
            return self._reply(f"فهمت إن اسمك {self.data['name']}، وسبب زيارتك هو {complaint}. أكمّل معك الحجز خطوة بخطوة.", None)
        if self.data.get("name"):
            return self._reply(f"فهمت إن اسمك {self.data['name']}. أقدر أكمّل معك الحجز خطوة بخطوة.", None)
        return self._reply("فهمت إنك بدك مساعدة، وأنا جاهز أكمّل معك خطوة بخطوة.", None)

    def _is_clarification_request(self, text: str) -> bool:
        lowered = (text or "").lower().strip()
        return bool(lowered) and any(word in lowered for word in [
            "أنت فهمت", "انت فهمت", "فهمت", "ماذا فهمت", "إيه اللي فهمته", "ايش فهمت",
            "أشرح", "اشرح", "قلت", "قلتلك", "what did you understand", "what do you know",
        ])

    def _is_new_booking_request(self, text: str) -> bool:
        lowered = (text or "").lower().strip()
        return any(token in lowered for token in ["حجز موعد", "موعد جديد", "ابدأ", "من جديد", "restart", "book"])

    def _reset(self) -> None:
        self.state = State.GREETING
        self.data.clear()
        self.slot = None
        self.priority = None
        self.finalized_appointment_id = None

    @property
    def is_done(self) -> bool:
        return self.state in (State.FINALIZED, State.WAITLISTED, State.CANCELLED)
