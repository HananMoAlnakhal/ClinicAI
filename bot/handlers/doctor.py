"""
bot/handlers/doctor.py — Doctor-facing handler for session data-entry.
"""
import io
from telegram import Update
from telegram.ext import ContextTypes
from fsm.doctor_fsm import DoctorFSM, DoctorState
from voice.stt import transcribe_voice
from voice.tts import text_to_ogg
from bot.keyboards import doctor_menu_keyboard, session_confirm_keyboard
from database.db import get_db
from database import crud

_sessions: dict[int, DoctorFSM] = {}


def _get_fsm(doctor: object) -> DoctorFSM:
    tid = doctor.telegram_id
    if tid not in _sessions or _sessions[tid].state == DoctorState.SAVED:
        _sessions[tid] = DoctorFSM(
            doctor_id=doctor.doctor_id,
            telegram_id=tid,
        )
    return _sessions[tid]


async def handle_doctor_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    with get_db() as db:
        crud.log_message(db, user_id, "inbound", "command", "/start", role="doctor")

    reply = "👨‍⚕️ مرحباً دكتور! ماذا تريد؟"
    await update.message.reply_text(reply, reply_markup=doctor_menu_keyboard())

    with get_db() as db:
        crud.log_bot_reply(db, user_id, reply, role="doctor")


async def handle_doctor_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text     = update.message.text

    with get_db() as db:
        crud.get_or_create_conversation(db, user_id, update.effective_user.username, update.effective_user.first_name, update.effective_user.last_name, role="doctor")
        crud.log_message(db, user_id, "inbound", "text", text, role="doctor")
        doctor = crud.get_doctor_by_telegram(db, user_id)

    fsm   = _get_fsm(doctor)
    reply = await fsm.handle(text)
    markup = session_confirm_keyboard() if fsm.state == DoctorState.REVIEW else doctor_menu_keyboard()
    await update.message.reply_text(reply, reply_markup=markup, parse_mode="Markdown")

    with get_db() as db:
        crud.log_bot_reply(db, user_id, reply, role="doctor")


async def handle_doctor_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    voice_file = await context.bot.get_file(update.message.voice.file_id)
    ogg_bytes   = await voice_file.download_as_bytearray()

    result = transcribe_voice(bytes(ogg_bytes))
    text   = result["text"]

    with get_db() as db:
        crud.get_or_create_conversation(db, user_id, update.effective_user.username, update.effective_user.first_name, update.effective_user.last_name, role="doctor")
        crud.log_message(db, user_id, "inbound", "voice", text, role="doctor")
        doctor = crud.get_doctor_by_telegram(db, user_id)

    await update.message.reply_text(f"🎙️ تم التعرف: _{text}_", parse_mode="Markdown")

    with get_db() as db:
        crud.log_bot_reply(db, user_id, f"🎙️ تم التعرف: {text}", role="doctor")

    fsm   = _get_fsm(doctor)
    reply = await fsm.handle(text, is_voice=True)
    markup = session_confirm_keyboard() if fsm.state == DoctorState.REVIEW else None
    await update.message.reply_text(reply, reply_markup=markup, parse_mode="Markdown")


async def handle_doctor_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query   = update.callback_query
    user_id = query.from_user.id
    data    = query.data
    await query.answer()

    with get_db() as db:
        crud.get_or_create_conversation(db, user_id, query.from_user.username, query.from_user.first_name, query.from_user.last_name, role="doctor")
        crud.log_message(db, user_id, "inbound", "callback", data or "", role="doctor")
        doctor = crud.get_doctor_by_telegram(db, user_id)

    fsm = _get_fsm(doctor)

    if data == "doc:session":
        reply = await fsm.handle("/session")
        await query.edit_message_text(reply)

    elif data == "session:confirm":
        reply = await fsm.handle("تأكيد")
        await query.edit_message_text(reply)

    elif data == "session:discard":
        _sessions.pop(user_id, None)
        await query.edit_message_text("🗑️ تم إلغاء الجلسة.", reply_markup=doctor_menu_keyboard())

    elif data == "doc:today":
        with get_db() as db:
            appts = crud.get_todays_queue(db)
        if not appts:
            await query.edit_message_text("لا توجد مواعيد اليوم.")
            return
        lines = [f"{a.appt_datetime.strftime('%H:%M')} — {a.patient.name or '؟'} — {a.priority_class}"
                 for a in appts]
        await query.edit_message_text("📋 مواعيد اليوم:\n" + "\n".join(lines))
