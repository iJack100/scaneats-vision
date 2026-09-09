"""Motor de Analitica y Reglas de Negocio (Modulo 6.2 del informe).

Traduce las coordenadas crudas del Motor de Vision en eventos de negocio
concretos: estado de ocupacion de cada mesa (Risaldi et al. 2026),
interacciones mesero-mesa validadas por proximidad sostenida (Akash et
al. 2024) y cronometraje del tiempo de espera (De Vries, Roy & De
Koster, 2018). Centraliza la logica de las tres funcionalidades
principales del Alcance inicial.
"""
from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass, field

from app import config
from app.database import SessionLocal
from app.models import AlertLog, Interaction, OccupancyEvent, TableCatalog, WaitRecord, WaiterRegistry
from app.vision import geometry
from app.vision.role_classifier import Role, TrackState
from app.vision.table_zones import TableZone


@dataclass
class TableRuntimeState:
    """Estado en memoria de una mesa, expuesto al Dashboard UI."""

    table_id: str
    name: str
    status: str = "libre"  # "libre" | "ocupada" | "alerta"
    waiting_since: float | None = None  # time.monotonic(), None = ya atendida
    last_wait_seconds: float | None = None
    customer_track_id: int | None = None
    alert_active: bool = False
    wait_record_id: int | None = field(default=None, repr=False)
    alert_id: int | None = field(default=None, repr=False)
    empty_streak: int = field(default=0, repr=False)

    def waiting_seconds_now(self) -> float | None:
        if self.waiting_since is None:
            return None
        return time.monotonic() - self.waiting_since


class AnalyticsEngine:
    def __init__(self, table_zones: dict[str, TableZone]):
        self.table_zones = table_zones
        self.tables: dict[str, TableRuntimeState] = {
            tid: TableRuntimeState(table_id=tid, name=zone.name) for tid, zone in table_zones.items()
        }
        self.tracks: dict[int, TrackState] = {}
        self._sync_table_catalog()

    def _sync_table_catalog(self) -> None:
        with SessionLocal() as db:
            for tid, zone in self.table_zones.items():
                if not db.get(TableCatalog, tid):
                    db.add(TableCatalog(id=tid, name=zone.name))
            db.commit()

    def process_frame(
        self, tracked_boxes: list[tuple[int, geometry.BBox]], alive_ids: set[int] | None = None
    ) -> None:
        """`tracked_boxes`: [(track_id, caja_torso_superior), ...] del
        fotograma actual, ya asociadas por el SortTracker. `alive_ids`
        son todos los tracks que el tracker aun conserva (incluye
        ocluidos), para no perder el historial de rol de un track solo
        por 1-2 fotogramas sin deteccion."""
        now_wall = dt.datetime.utcnow()
        active_ids = alive_ids if alive_ids is not None else {tid for tid, _ in tracked_boxes}

        candidatos_por_mesa: dict[str, int] = {}

        for track_id, box in tracked_boxes:
            state = self.tracks.setdefault(track_id, TrackState(track_id=track_id))
            sentado, cerca_de, nuevas_interacciones = state.observe(box, self.table_zones)

            if sentado and cerca_de is not None:
                candidatos_por_mesa.setdefault(cerca_de, track_id)

            for table_id in nuevas_interacciones:
                self._registrar_interaccion(track_id, table_id, now_wall)

        for table_id, table_state in self.tables.items():
            ocupada = table_id in candidatos_por_mesa
            self._actualizar_ocupacion(table_state, ocupada, candidatos_por_mesa.get(table_id), now_wall)
            self._chequear_alerta(table_state, now_wall)

        self._purgar_tracks_perdidos(active_ids)

    # -- ocupacion (Risaldi et al. 2026) -------------------------------

    def _actualizar_ocupacion(
        self, table_state: TableRuntimeState, ocupada: bool, track_id: int | None, now_wall: dt.datetime
    ) -> None:
        if ocupada:
            table_state.empty_streak = 0
            if table_state.status == "libre":
                table_state.status = "ocupada"
                table_state.waiting_since = time.monotonic()
                table_state.customer_track_id = track_id
                table_state.alert_active = False
                with SessionLocal() as db:
                    db.add(OccupancyEvent(table_id=table_state.table_id, status="ocupada"))
                    record = WaitRecord(
                        table_id=table_state.table_id, customer_track_id=track_id, seated_at=now_wall
                    )
                    db.add(record)
                    db.commit()
                    table_state.wait_record_id = record.id
            return

        if table_state.status == "libre":
            return

        # con mesa aun marcada ocupada/alerta: tolerar una oclusion breve
        # del cliente (ver OCCUPANCY_GRACE_FRAMES) antes de liberar la mesa
        table_state.empty_streak += 1
        if table_state.empty_streak < config.OCCUPANCY_GRACE_FRAMES:
            return

        with SessionLocal() as db:
            db.add(OccupancyEvent(table_id=table_state.table_id, status="libre"))
            db.commit()
        table_state.status = "libre"
        table_state.waiting_since = None
        table_state.customer_track_id = None
        table_state.alert_active = False
        table_state.wait_record_id = None
        table_state.alert_id = None
        table_state.empty_streak = 0

    # -- interacciones (Akash et al. 2024) y cierre de espera (De Vries et al. 2018) --

    def _registrar_interaccion(self, waiter_track_id: int, table_id: str, now_wall: dt.datetime) -> None:
        state = self.tracks[waiter_track_id]
        with SessionLocal() as db:
            db.add(Interaction(waiter_track_id=waiter_track_id, table_id=table_id, timestamp=now_wall))
            if state.role == Role.MESERO and not db.get(WaiterRegistry, waiter_track_id):
                db.add(WaiterRegistry(track_id=waiter_track_id))
            db.commit()

        table_state = self.tables.get(table_id)
        if table_state and table_state.waiting_since is not None:
            waited = time.monotonic() - table_state.waiting_since
            table_state.last_wait_seconds = waited
            table_state.waiting_since = None
            was_alert = table_state.alert_active
            table_state.alert_active = False
            if table_state.status == "alerta":
                table_state.status = "ocupada"

            with SessionLocal() as db:
                if table_state.wait_record_id:
                    rec = db.get(WaitRecord, table_state.wait_record_id)
                    if rec and rec.attended_at is None:
                        rec.attended_at = now_wall
                        rec.waited_seconds = waited
                if was_alert and table_state.alert_id:
                    alert = db.get(AlertLog, table_state.alert_id)
                    if alert and alert.resolved_at is None:
                        alert.resolved_at = now_wall
                db.commit()
            table_state.alert_id = None

    # -- alertas visuales (De Vries et al. 2018 sustenta el umbral) ----

    def _chequear_alerta(self, table_state: TableRuntimeState, now_wall: dt.datetime) -> None:
        if table_state.waiting_since is None or table_state.alert_active:
            return
        elapsed = time.monotonic() - table_state.waiting_since
        if elapsed >= config.WAIT_ALERT_THRESHOLD_SECONDS:
            table_state.status = "alerta"
            table_state.alert_active = True
            with SessionLocal() as db:
                alert = AlertLog(
                    table_id=table_state.table_id,
                    triggered_at=now_wall,
                    wait_seconds_at_trigger=elapsed,
                )
                db.add(alert)
                db.commit()
                table_state.alert_id = alert.id

    def _purgar_tracks_perdidos(self, active_ids: set[int]) -> None:
        for tid in list(self.tracks.keys()):
            if tid not in active_ids:
                del self.tracks[tid]
