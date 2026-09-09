"""Inferencia de rol (cliente / mesero) por comportamiento del track.

Ningun articulo del sustento cientifico entrena una clase "mesero":
Risaldi et al. (2026) solo distinguen "customer" (sentado) / "non-customer"
(de pie) / "table", y entrenar una clase propia de personal exigiria un
dataset etiquetado que el equipo no tiene. La regla de negocio acordada
para el MVP es puramente conductual:

    "Un cliente no se va a estar moviendo por todo el local, y un mesero
    no se va a estar sentado tanto tiempo."

Esto se opera sobre el HISTORIAL de cada track (posible gracias al
tracking robusto a oclusion de Mamedov et al. 2021 - app/vision/tracker.py):

- CLIENTE: el track permanece sentado (heuristica geometrica, ver
  geometry.is_sitting) durante varios fotogramas seguidos cerca de una
  misma mesa.
- MESERO: el track permanece de pie y acumula proximidad SOSTENIDA (no
  instantanea, para evitar el falso positivo de "solo paso caminando"
  que describen Akash et al. 2024) con 2 o mas mesas distintas.

Mientras un track de pie aun no acumula evidencia suficiente para
confirmarse como mesero, se lo marca INDETERMINADO; sus interacciones
"candidatas" igual se registran (ver AnalyticsEngine), porque la primera
mesa que atiende un mesero nuevo no debe perderse solo por no haberse
confirmado todavia como personal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from app import config
from app.vision import geometry
from app.vision.geometry import BBox
from app.vision.table_zones import TableZone


class Role(str, Enum):
    INDETERMINADO = "indeterminado"
    CLIENTE = "cliente"
    MESERO = "mesero"


@dataclass
class TrackState:
    track_id: int
    role: Role = Role.INDETERMINADO
    seated_frames: int = 0
    tables_visited: set[str] = field(default_factory=set)
    bound_table_id: str | None = None
    proximity_counters: dict[str, int] = field(default_factory=dict)
    active_interaction_tables: set[str] = field(default_factory=set)

    def observe(
        self, box: BBox, table_zones: dict[str, TableZone]
    ) -> tuple[bool, str | None, list[str]]:
        """Procesa un fotograma para este track.

        Devuelve (sentado, mesa_cercana, nuevas_interacciones).
        """
        sentado = geometry.is_sitting(box)
        cerca_de = self._closest_table(box, table_zones)
        nuevas_interacciones: list[str] = []

        if sentado:
            self.seated_frames += 1
            if cerca_de is not None:
                self.bound_table_id = cerca_de
            if self.role != Role.MESERO and self.seated_frames >= config.CLIENTE_MIN_SEATED_FRAMES:
                self.role = Role.CLIENTE
            # al sentarse se limpia el rastro de desplazamiento "de pie"
            self.proximity_counters.clear()
            self.active_interaction_tables.clear()
        else:
            self.seated_frames = 0
            for tid in list(self.proximity_counters):
                if tid != cerca_de:
                    self.proximity_counters[tid] = 0
                    self.active_interaction_tables.discard(tid)

            if cerca_de is not None:
                self.tables_visited.add(cerca_de)
                self.proximity_counters[cerca_de] = self.proximity_counters.get(cerca_de, 0) + 1

                cruzo_umbral = self.proximity_counters[cerca_de] == config.PROXIMITY_SUSTAIN_FRAMES
                if (
                    cruzo_umbral
                    and cerca_de not in self.active_interaction_tables
                    and self.role != Role.CLIENTE
                ):
                    self.active_interaction_tables.add(cerca_de)
                    nuevas_interacciones.append(cerca_de)

            if len(self.tables_visited) >= config.MESERO_MIN_TABLES_VISITED:
                self.role = Role.MESERO

        return sentado, cerca_de, nuevas_interacciones

    def sustained_proximity(self, table_id: str) -> bool:
        return self.proximity_counters.get(table_id, 0) >= config.PROXIMITY_SUSTAIN_FRAMES

    @staticmethod
    def _closest_table(box: BBox, table_zones: dict[str, TableZone]) -> str | None:
        for table_id, zone in table_zones.items():
            if geometry.is_occupying(box, zone.polygon):
                return table_id
        return None
