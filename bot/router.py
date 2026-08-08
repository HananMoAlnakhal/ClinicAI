"""
bot/router.py — Task: "Server connecting all components"
Routes every incoming Telegram update to the correct handler
based on whether the sender is a registered doctor or a patient.
"""
import logging

from telegram import Update
from telegram.ext import ContextTypes
from database.db import get_db
from database import crud
from bot.handlers import patient as patient_handler
from bot.handlers import doctor  as doctor_handler

logger = logging.getLogger(__name__)


async def route_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    preview = (update.message.text or "")[:80] if update.message else ""
    logger.info("Incoming text from %s: %r", user_id, preview)
    with get_db() as db:
        doctor = crud.get_doctor_by_telegram(db, user_id)

    if doctor:
        await doctor_handler.handle_doctor_text(update, context)
    else:
        await patient_handler.handle_text(update, context)


async def route_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    with get_db() as db:
        doctor = crud.get_doctor_by_telegram(db, user_id)

    if doctor:
        await doctor_handler.handle_doctor_voice(update, context)
    else:
        await patient_handler.handle_voice(update, context)


async def route_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.callback_query.from_user.id
    data    = update.callback_query.data
    with get_db() as db:
        doctor = crud.get_doctor_by_telegram(db, user_id)

    if doctor or data.startswith("doc:") or data.startswith("session:"):
        await doctor_handler.handle_doctor_callback(update, context)
    else:
        await patient_handler.handle_callback(update, context)


async def route_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    logger.info("Incoming /start from %s", user_id)
    with get_db() as db:
        doctor = crud.get_doctor_by_telegram(db, user_id)

    if doctor:
        await doctor_handler.handle_doctor_start(update, context)
    else:
        await patient_handler.handle_start(update, context)
