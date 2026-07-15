"""Scheduler package — Tier-1 real-time booking."""

import logging

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=_LOG_FORMAT,
    datefmt="%Y-%m-%d %H:%M:%S",
)

from scheduler.scheduler import (
    AppointmentSlot,
    ScheduleDecision,
    WaitlistEntry,
    book_slot,
    plan_appointment,
)

__all__ = [
    "plan_appointment",
    "AppointmentSlot",
    "ScheduleDecision",
    "WaitlistEntry",
    "book_slot",
]
