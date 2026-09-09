"""Sesion de base de datos relacional (SQLite via SQLAlchemy).

La propuesta (Informe, Sec. 3.5 "Factibilidad") delimita explicitamente
el trabajo restante a "la configuracion de la base de datos relacional,
la logica del backend en Python y la interfaz de reportes".
"""
from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import config
from app.models import Base

engine = create_engine(config.DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db() -> None:
    Base.metadata.create_all(engine)
