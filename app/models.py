"""Modelos ORM (SQLAlchemy) para persistencia de eventos de negocio.

Corresponde a los "Datos que gestiona" del Modulo 6.2 (Analytics Engine)
y 6.4 (Alert & Notification Service) del informe: catalogos de mesas,
contadores de interaccion por mesero, metricas de tiempo de espera y
registro historico de notificaciones.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class TableCatalog(Base):
    """Catalogo de zonas/mesas (Modulo 6.2)."""

    __tablename__ = "tables"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)


class WaiterRegistry(Base):
    """Tracks confirmados como mesero (ver app/vision/role_classifier.py).

    Se usa para filtrar el ranking de desempeno del Modulo 5.2 y no
    mezclar tracks aun indeterminados.
    """

    __tablename__ = "waiters"

    track_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    label: Mapped[str] = mapped_column(String, default="")
    confirmed_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)


class WaitRecord(Base):
    """Cronometro de espera por mesa (Funcionalidad 1 del Alcance inicial):
    desde que el cliente toma asiento hasta el primer abordaje del mesero.

    Sustentado por De Vries, Roy & De Koster (2018): el tiempo de espera
    es la variable con mayor evidencia de impacto en abandono, consumo y
    retorno del cliente.
    """

    __tablename__ = "wait_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    table_id: Mapped[str] = mapped_column(ForeignKey("tables.id"))
    customer_track_id: Mapped[int] = mapped_column(Integer)
    seated_at: Mapped[dt.datetime] = mapped_column(DateTime)
    attended_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    waited_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)


class Interaction(Base):
    """Interaccion sostenida mesero-mesa (Funcionalidad 2 del Alcance
    inicial): contabilizacion de abordajes para medir carga operativa.

    Sustentada por Akash et al. (2024): solo se registra tras un filtro
    de proximidad SOSTENIDA (ver PROXIMITY_SUSTAIN_FRAMES), no por mera
    cercania instantanea en 2D.
    """

    __tablename__ = "interactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    waiter_track_id: Mapped[int] = mapped_column(Integer)
    table_id: Mapped[str] = mapped_column(ForeignKey("tables.id"))
    timestamp: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)


class AlertLog(Base):
    """Registro historico de notificaciones emitidas (Modulo 6.4)."""

    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    table_id: Mapped[str] = mapped_column(ForeignKey("tables.id"))
    triggered_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    resolved_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    wait_seconds_at_trigger: Mapped[float] = mapped_column(Float)


class OccupancyEvent(Base):
    """Historial libre/ocupada por mesa, usado para construir el mapa de
    calor de uso (Funcionalidad 3 del Alcance inicial)."""

    __tablename__ = "occupancy_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    table_id: Mapped[str] = mapped_column(ForeignKey("tables.id"))
    status: Mapped[str] = mapped_column(String)  # "libre" | "ocupada"
    timestamp: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
