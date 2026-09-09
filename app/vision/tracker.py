"""Tracker tipo SORT (Kalman filter + algoritmo hungaro).

Risaldi et al. (2026, Sec. 2.5 "Tracking with SORT") aplican SORT sobre
las detecciones de clientes para asignar un ID unico y estable a cada
persona a lo largo de los fotogramas, combinando un filtro de Kalman
(estima la posicion siguiente a partir de posicion/velocidad previas) con
el algoritmo hungaro (empareja detecciones nuevas con tracks existentes).

ScanEats sigue el mismo principio de diseno, pero -en linea con Mamedov,
Kuplyakov & Konushin (2021)- el tracker recibe las cajas de TORSO
SUPERIOR ya recortadas (ver app/vision/geometry.upper_body_box), no el
cuerpo completo, para mantener el ID de una persona sentada incluso
cuando sillas u otras personas ocultan sus piernas.

No se usan embeddings de reidentificacion profunda: la propuesta declara
explicitamente que se opta por "algoritmos de tracking" ya probados en
lugar de "construir redes neuronales desde cero" (Informe, Sec. 3.5).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from app import config
from app.vision.geometry import BBox, iou


class KalmanBoxTracker:
    """Filtro de Kalman de velocidad constante sobre (cx, cy, w, h)."""

    _next_id = 1

    def __init__(self, bbox: BBox):
        cx, cy, w, h = self._to_state(bbox)
        self.x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0])
        self.P = np.eye(8) * 10.0
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = 1.0
        self.H = np.zeros((4, 8))
        for i in range(4):
            self.H[i, i] = 1.0
        self.Q = np.eye(8) * 1.0
        self.R = np.eye(4) * 5.0

        self.id = KalmanBoxTracker._next_id
        KalmanBoxTracker._next_id += 1
        self.hits = 1
        self.age = 0
        self.time_since_update = 0

    @staticmethod
    def _to_state(bbox: BBox):
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1

    @staticmethod
    def _to_bbox(cx, cy, w, h) -> BBox:
        return cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0

    def predict(self) -> BBox:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        return self.get_state()

    def update(self, bbox: BBox) -> None:
        z = np.array(self._to_state(bbox))
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        self.time_since_update = 0
        self.hits += 1

    def get_state(self) -> BBox:
        cx, cy, w, h = self.x[:4]
        return self._to_bbox(cx, cy, max(w, 1.0), max(h, 1.0))


class SortTracker:
    """Asocia detecciones de torso superior entre fotogramas."""

    def __init__(self, max_age=None, min_hits=None, iou_threshold=None):
        self.max_age = config.TRACK_MAX_AGE if max_age is None else max_age
        self.min_hits = config.TRACK_MIN_HITS if min_hits is None else min_hits
        self.iou_threshold = config.TRACK_IOU_THRESHOLD if iou_threshold is None else iou_threshold
        self.trackers: list[KalmanBoxTracker] = []

    def update(self, detections: list[BBox]) -> list[tuple[int, BBox]]:
        """`detections`: cajas de torso superior del fotograma actual.

        Devuelve una lista de (track_id, caja_suavizada) por cada track
        confirmado y actualizado en este fotograma.
        """
        for t in self.trackers:
            t.predict()

        matches, unmatched_dets, _ = self._associate(detections)

        for det_idx, trk_idx in matches:
            self.trackers[trk_idx].update(detections[det_idx])

        for det_idx in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(detections[det_idx]))

        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        results: list[tuple[int, BBox]] = []
        for t in self.trackers:
            if t.time_since_update == 0 and (t.hits >= self.min_hits or t.age <= self.min_hits):
                results.append((t.id, t.get_state()))
        return results

    def alive_track_ids(self) -> set[int]:
        """IDs de tracks aun vivos (dentro de max_age), incluyendo los
        temporalmente ocluidos que no aparecen en el resultado de este
        fotograma. Se usa para NO purgar el historial de rol/mesas
        visitadas de un track solo porque fue ocluido 1-2 fotogramas -
        justamente el escenario que Mamedov et al. (2021) buscan resolver
        con el tracking por torso superior."""
        return {t.id for t in self.trackers}

    def _associate(self, detections: list[BBox]):
        if not detections or not self.trackers:
            return [], list(range(len(detections))), list(range(len(self.trackers)))

        iou_matrix = np.zeros((len(detections), len(self.trackers)))
        for d, det in enumerate(detections):
            for t, trk in enumerate(self.trackers):
                iou_matrix[d, t] = iou(det, trk.get_state())

        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        matches: list[tuple[int, int]] = []
        matched_dets, matched_trks = set(), set()
        for d, t in zip(row_ind, col_ind):
            if iou_matrix[d, t] >= self.iou_threshold:
                matches.append((d, t))
                matched_dets.add(d)
                matched_trks.add(t)

        unmatched_dets = [d for d in range(len(detections)) if d not in matched_dets]
        unmatched_trks = [t for t in range(len(self.trackers)) if t not in matched_trks]
        return matches, unmatched_dets, unmatched_trks
