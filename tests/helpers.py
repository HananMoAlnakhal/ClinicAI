"""Shared helpers for ClinicAI unit tests."""
from __future__ import annotations

import asyncio
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Iterator

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from database.db import Base
from database import models  # noqa: F401
from database.models import Doctor, Patient, Slot


def infer_urgency_score(text: str) -> float:
    t = text
    high = ["الآن", "فجأة", "حاد", "شديد", "صعوبة نطق", "ضعف مفاجئ", "وقع", "أزمة", "هبوط سكر"]
    mid = ["دوخة", "تعب", "تنميل", "صفير", "تورم", "رجفة"]
    low = ["متابعة", "دوري", "روتيني", "مراجعة", "بدون أعراض جديدة"]

    if any(k in t for k in high):
        return 0.9
    if any(k in t for k in mid):
        return 0.55
    if any(k in t for k in low):
        return 0.25
    return 0.4


def run_async(coro):
    """Run an async coroutine in a fresh event loop (Python 3.7 compatible)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def make_test_engine():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        future=True,
    )

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(bind=engine)
    return engine


def make_test_session(engine=None) -> Session:
    engine = engine or make_test_engine()
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    return factory()


@contextmanager
def test_db_session(engine=None) -> Iterator[Session]:
    db = make_test_session(engine)
    try:
        yield db
    finally:
        db.close()


def seed_doctor(
    db: Session,
    specialty: str = "general_practice",
    *,
    name: str = "د. اختبار",
    clinic_code: str | None = None,
    clinic_name: str | None = None,
    is_active: bool = True,
    telegram_id: int | None = None,
) -> Doctor:
    doctor = Doctor(
        telegram_id=telegram_id,
        name=name,
        specialty=specialty,
        clinic_code=clinic_code or f"CLINIC-{specialty[:4].upper()}",
        clinic_name=clinic_name or f"عيادة {specialty}",
        is_active=is_active,
    )
    db.add(doctor)
    db.flush()
    db.refresh(doctor)
    return doctor


def seed_patient(db: Session, telegram_id: int = 900001, name: str = "مريض اختبار") -> Patient:
    patient = Patient(telegram_id=telegram_id, name=name, updated_at=datetime.utcnow())
    db.add(patient)
    db.flush()
    db.refresh(patient)
    return patient


def seed_slot(
    db: Session,
    doctor: Doctor,
    *,
    when: datetime | None = None,
    status: str = "available",
    priority_class: str | None = None,
) -> Slot:
    slot = Slot(
        doctor_id=doctor.doctor_id,
        slot_datetime=when or (datetime.utcnow() + timedelta(days=1)).replace(hour=10, minute=0, second=0, microsecond=0),
        specialty=doctor.specialty,
        priority_class=priority_class,
        status=status,
    )
    db.add(slot)
    db.flush()
    db.refresh(slot)
    return slot
