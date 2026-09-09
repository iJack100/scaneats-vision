"""Consultas de agregacion para el Dashboard UI (Modulo 6.3)."""
from __future__ import annotations

import datetime as dt

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models import AlertLog, Interaction, OccupancyEvent, TableCatalog, WaiterRegistry


def usage_heatmap(db: Session) -> dict[str, dict]:
    """Segundos ocupada y numero de ocupaciones por mesa: sustenta la
    Funcionalidad 3 (mapa de uso del restaurante) del Alcance inicial."""
    events = (
        db.query(OccupancyEvent)
        .order_by(OccupancyEvent.table_id, OccupancyEvent.timestamp)
        .all()
    )
    usage: dict[str, dict] = {}
    open_start: dict[str, dt.datetime] = {}

    for ev in events:
        entry = usage.setdefault(ev.table_id, {"occupied_seconds": 0.0, "times_occupied": 0})
        if ev.status == "ocupada":
            open_start[ev.table_id] = ev.timestamp
            entry["times_occupied"] += 1
        elif ev.status == "libre" and ev.table_id in open_start:
            delta = (ev.timestamp - open_start.pop(ev.table_id)).total_seconds()
            entry["occupied_seconds"] += max(delta, 0.0)

    now = dt.datetime.utcnow()
    for table_id, start in open_start.items():
        entry = usage.setdefault(table_id, {"occupied_seconds": 0.0, "times_occupied": 0})
        entry["occupied_seconds"] += max((now - start).total_seconds(), 0.0)

    return usage


def waiter_performance(db: Session) -> list[dict]:
    """Carga de trabajo por mesero confirmado: sustenta la Funcionalidad 2
    (Gestion de Desempeno Operativo)."""
    rows = (
        db.query(
            WaiterRegistry.track_id,
            func.count(Interaction.id).label("interacciones"),
            func.count(func.distinct(Interaction.table_id)).label("mesas_atendidas"),
        )
        .join(Interaction, Interaction.waiter_track_id == WaiterRegistry.track_id)
        .group_by(WaiterRegistry.track_id)
        .order_by(func.count(Interaction.id).desc())
        .all()
    )
    return [
        {"waiter_track_id": r.track_id, "interacciones": r.interacciones, "mesas_atendidas": r.mesas_atendidas}
        for r in rows
    ]


def recent_alerts(db: Session, limit: int = 20) -> list[dict]:
    rows = (
        db.query(AlertLog, TableCatalog.name)
        .join(TableCatalog, TableCatalog.id == AlertLog.table_id)
        .order_by(AlertLog.triggered_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "table_id": alert.table_id,
            "table_name": name,
            "triggered_at": alert.triggered_at.isoformat(),
            "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
            "wait_seconds_at_trigger": alert.wait_seconds_at_trigger,
            "active": alert.resolved_at is None,
        }
        for alert, name in rows
    ]
