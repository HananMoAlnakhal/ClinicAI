"""Patient-facing Telegram handler with text/voice input and active TTS replies."""
from __future__ import annotations

import io
from datetime import datetime, timezone

from telegram import Update
from telegram.ext import ContextTypes

from bot.keyboards import main_menu_keyboard
from config import TTS_ENABLED, TTS_RESPONSE_MODE
from database import crud
from database.db import get_db
from fsm.patient_fsm import PatientFSM
from nlp.gemini_client import gemini
from voice.stt import transcribe_voice
from voice.tts import text_to_ogg

_sessions: dict[int, PatientFSM] = {}


def _is_repeat_request(text: str) -> bool:
    lowered = (text or "").lower().strip()
    return any(
        token in lowered
        for token in [
            "repeat",
            "كرر",
            "تكرار",
            "من جديد",
            "من اول",
            "restart",
            "ابدأ من جديد",
            "بدء جديد",
            "حجز موعد جديد",
        ]
    )


def _is_inquiry_request(text: str) -> bool:
    lowered = (text or "").lower().strip()
    return "استعلام" in lowered or "موعدي" in lowered or "🔍" in lowered


def _is_cancel_request(text: str) -> bool:
    lowered = (text or "").lower().strip()
    return "إلغاء موعد" in lowered or "الغاء موعد" in lowered or "❌" in lowered


def _is_contact_request(text: str) -> bool:
    lowered = (text or "").lower().strip()
    return "تواصل" in lowered or "📞" in lowered


def _get_fsm(user_id: int, reset: bool = False) -> PatientFSM:
    session = _sessions.get(user_id)
    if reset or session is None:
        session = PatientFSM(user_id=user_id)
        _sessions[user_id] = session
    return session


def _format_appointment(appt) -> str:
    if not appt:
        return "لا يوجد موعد مسجل باسمك حالياً. يمكنك اختيار 📅 حجز موعد جديد."
    date_text = (
        appt.appt_datetime.strftime("%A، %d/%m/%Y — %H:%M")
        if appt.appt_datetime
        else "قائمة الانتظار"
    )
    status_ar = {
        "confirmed": "مؤكد",
        "waitlisted": "قائمة انتظار",
        "completed": "مكتمل",
        "no_show": "غياب",
        "cancelled": "ملغي",
    }.get(appt.status, appt.status)
    patient_name = appt.patient.name if appt.patient else "—"
    specialty = appt.specialty_ar or appt.specialty or "—"
    doctor = appt.slot.doctor if appt.slot and appt.slot.doctor else None
    return (
        "📌 آخر موعد مسجل:\n"
        f"رقم الحجز: {appt.appt_id}\n"
        f"المريض: {patient_name}\n"
        f"الوقت: {date_text}\n"
        f"التخصص: {specialty}\n"
        f"الطبيب: {doctor.name if doctor else '—'}\n"
        f"العيادة: {doctor.clinic_name if doctor else '—'}\n"
        f"الأولوية: {appt.priority_class or '—'}\n"
        f"الحالة: {status_ar}"
    )


def _should_send_voice(incoming_was_voice: bool) -> bool:
    if not TTS_ENABLED or TTS_RESPONSE_MODE == "text":
        return False
    if TTS_RESPONSE_MODE in {"voice", "both"}:
        return True
    return TTS_RESPONSE_MODE == "auto" and incoming_was_voice


async def _send_patient_reply(
    message,
    reply: str,
    keyboard=None,
    incoming_was_voice: bool = False,
) -> bool:
    """
    Always send readable text (and any keyboard). When TTS is enabled, also send
    an OGG voice response according to TTS_RESPONSE_MODE. Returns True if voice sent.
    """
    await message.reply_text(reply, reply_markup=keyboard)
    if not _should_send_voice(incoming_was_voice):
        return False

    try:
        audio_bytes = await text_to_ogg(reply)
        voice_file = io.BytesIO(audio_bytes)
        voice_file.name = "clinicai_reply.ogg"
        voice_file.seek(0)
        await message.reply_voice(voice=voice_file)
        return True
    except Exception as exc:
        # TTS must never break the booking conversation; text remains the fallback.
        print(f"[ClinicAI TTS] Voice reply failed; text reply was sent: {exc}")
        return False


def _log_outbound(user_id: int, reply: str, voice_sent: bool = False) -> None:
    with get_db() as db:
        crud.log_bot_reply(db, user_id, reply)
        if voice_sent:
            crud.log_message(db, user_id, "outbound", "bot_voice", reply)


def _menu_response(user_id: int, text: str):
    """Return (reply, keyboard) for main-menu commands, or None."""
    if _is_repeat_request(text):
        _get_fsm(user_id, reset=True)
        return "📅 تمام، خلينا نبدأ حجز جديد. ما اسمك الكريم؟", None

    if _is_inquiry_request(text):
        with get_db() as db:
            appt = crud.get_latest_patient_appointment(db, user_id)
            reply = _format_appointment(appt)
        return reply, main_menu_keyboard()

    if _is_cancel_request(text):
        with get_db() as db:
            appt = crud.cancel_latest_patient_appointment(db, user_id)
        reply = (
            "تم إلغاء آخر موعد مؤكد/منتظر وإرجاع الـ slot كمتاح إذا كان محجوزاً. ✅"
            if appt
            else "لا يوجد موعد مؤكد أو منتظر لإلغائه حالياً."
        )
        return reply, main_menu_keyboard()

    if _is_contact_request(text):
        return (
            "📞 تواصلك وصل. يمكنك كتابة رسالتك هنا، وسيتم حفظها في سجل المحادثات للعيادة.",
            main_menu_keyboard(),
        )

    return None


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if user is None or update.message is None:
        return

    _get_fsm(user.id, reset=True)
    with get_db() as db:
        crud.get_or_create_conversation(db, user.id, user.username, user.first_name, user.last_name)
        crud.log_message(db, user.id, "inbound", "command", "/start")

    reply = "👋 مرحباً! أنا مساعد الحجز. اختاري/اختر من القائمة أو أرسل اسمك وسبب زيارتك لأساعدك بالحجز."
    voice_sent = await _send_patient_reply(update.message, reply, main_menu_keyboard())
    _log_outbound(user.id, reply, voice_sent)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message or update.edited_message or update.channel_post or update.effective_message
    if message is None:
        return
    user = message.from_user or update.effective_user
    if user is None:
        return

    text = message.text or ""
    with get_db() as db:
        crud.get_or_create_conversation(db, user.id, user.username, user.first_name, user.last_name)
        crud.log_message(db, user.id, "inbound", "text", text)

    if text.strip() == "/start":
        await handle_start(update, context)
        return

    menu = _menu_response(user.id, text)
    if menu:
        reply, keyboard = menu
        voice_sent = await _send_patient_reply(message, reply, keyboard, incoming_was_voice=False)
        _log_outbound(user.id, reply, voice_sent)
        return

    fsm = _get_fsm(user.id)
    reply, keyboard = await fsm.handle(text)
    if reply.strip() == "عفواً، ما فهمت. ممكن تعيد؟" and gemini._available:
        try:
            g_reply = await gemini.build_response(fsm.state.name, {**fsm.data, "last_user_message": text})
            if g_reply:
                reply = g_reply
        except Exception:
            pass

    voice_sent = await _send_patient_reply(message, reply, keyboard, incoming_was_voice=False)
    _log_outbound(user.id, reply, voice_sent)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message or update.effective_message
    user = update.effective_user
    if message is None or user is None or message.voice is None:
        return

    try:
        voice_file = await context.bot.get_file(message.voice.file_id)
        ogg_bytes = await voice_file.download_as_bytearray()
        uname = (user.username or user.first_name or str(user.id)).strip()
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        result = transcribe_voice(bytes(ogg_bytes), filename_prefix=f"{uname}_{ts}")
        text = result.get("text", "").strip()
    except Exception as exc:
        print(f"[ClinicAI STT] Voice transcription failed: {exc}")
        text = ""

    with get_db() as db:
        crud.get_or_create_conversation(db, user.id, user.username, user.first_name, user.last_name)
        crud.log_message(db, user.id, "inbound", "voice", text)

    if not text:
        reply = "🎙️ عذراً، لم أستطع التعرف على الصوت. هل يمكنك إعادة الإرسال كنص أو كرسالة صوتية أوضح؟"
        voice_sent = await _send_patient_reply(message, reply, incoming_was_voice=True)
        _log_outbound(user.id, reply, voice_sent)
        return

    # Show the recognized text so the patient can verify what the system understood.
    await message.reply_text(f"🎙️ فهمت رسالتك كالتالي:\n{text}")

    menu = _menu_response(user.id, text)
    if menu:
        reply, keyboard = menu
        voice_sent = await _send_patient_reply(message, reply, keyboard, incoming_was_voice=True)
        _log_outbound(user.id, reply, voice_sent)
        return

    fsm = _get_fsm(user.id)
    reply, keyboard = await fsm.handle(text)
    if reply.strip() == "عفواً، ما فهمت. ممكن تعيد؟" and gemini._available:
        try:
            g_reply = await gemini.build_response(fsm.state.name, {**fsm.data, "last_user_message": text})
            if g_reply:
                reply = g_reply
        except Exception:
            pass

    voice_sent = await _send_patient_reply(message, reply, keyboard, incoming_was_voice=True)
    _log_outbound(user.id, reply, voice_sent)


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if query is None:
        return
    data = query.data or ""
    user_id = query.from_user.id
    await query.answer()

    with get_db() as db:
        crud.get_or_create_conversation(
            db,
            user_id,
            query.from_user.username,
            query.from_user.first_name,
            query.from_user.last_name,
        )
        crud.log_message(db, user_id, "inbound", "callback", data)

    if data == "menu:book":
        _get_fsm(user_id, reset=True)
        reply, keyboard = "📅 جيد! ما اسمك الكريم؟", None
    elif data == "menu:inquiry":
        with get_db() as db:
            reply = _format_appointment(crud.get_latest_patient_appointment(db, user_id))
        keyboard = None
    elif data == "menu:cancel":
        with get_db() as db:
            appt = crud.cancel_latest_patient_appointment(db, user_id)
        reply = "تم إلغاء آخر موعد مؤكد/منتظر. ✅" if appt else "لا يوجد موعد لإلغائه حالياً."
        keyboard = None
    elif data == "menu:contact":
        reply, keyboard = "📞 اكتب رسالتك هنا، وسيتم حفظها في سجل محادثات العيادة.", None
    else:
        fsm = _get_fsm(user_id)
        reply, keyboard = await fsm.handle_callback(data)

    await query.edit_message_text(reply, reply_markup=keyboard)
    voice_sent = False
    if query.message is not None and _should_send_voice(incoming_was_voice=False):
        try:
            audio_bytes = await text_to_ogg(reply)
            voice_file = io.BytesIO(audio_bytes)
            voice_file.name = "clinicai_reply.ogg"
            voice_file.seek(0)
            await query.message.reply_voice(voice=voice_file)
            voice_sent = True
        except Exception as exc:
            print(f"[ClinicAI TTS] Callback voice reply failed; text reply was sent: {exc}")
    _log_outbound(user_id, reply, voice_sent=voice_sent)
