"""Orquesta el bucle de video: Motor de Vision -> Motor Analitico.

Corre en un hilo de fondo (el procesamiento de video con OpenCV/YOLO es
bloqueante) mientras FastAPI atiende el Dashboard UI de forma asincrona.
Es, en esencia, la version productizada del prototipo original
`test_vision.py`, ahora conectada al tracker, al motor analitico y a la
base de datos.
"""
from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np

from app import config
from app.analytics.engine import AnalyticsEngine
from app.vision import geometry
from app.vision.detector import PersonDetector
from app.vision.role_classifier import Role
from app.vision.table_zones import load_table_zones
from app.vision.tracker import SortTracker

logger = logging.getLogger("scaneats.pipeline")

_STATUS_COLOR = {
    "libre": (80, 200, 80),
    "ocupada": (200, 160, 40),
    "alerta": (40, 40, 220),
}
_ROLE_COLOR = {
    Role.INDETERMINADO: (160, 160, 160),
    Role.CLIENTE: (200, 160, 40),
    Role.MESERO: (40, 200, 220),
}


class VideoPipeline:
    def __init__(self, source: str | None = None):
        self.source = source or config.VIDEO_SOURCE
        self.table_zones = load_table_zones()
        self.detector = PersonDetector()
        self.tracker = SortTracker()
        self.engine = AnalyticsEngine(self.table_zones)

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame_lock = threading.Lock()
        self._last_jpeg: bytes | None = None
        self.frames_processed = 0
        self.running = False

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="scaneats-pipeline")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.running = False

    def _run(self) -> None:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            logger.error("No se pudo abrir la fuente de video '%s'", self.source)
            return

        self.running = True
        frame_idx = 0
        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    # video de demostracion corto: se reproduce en bucle
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                frame_idx += 1
                if frame_idx % config.PROCESS_EVERY_N_FRAMES != 0:
                    continue

                self._process(frame)
                self.frames_processed += 1
        finally:
            cap.release()
            self.running = False

    def _process(self, frame: np.ndarray) -> None:
        full_boxes = self.detector.detect(frame)
        upper_boxes = [geometry.upper_body_box(b) for b in full_boxes]

        tracked = self.tracker.update(upper_boxes)
        self.engine.process_frame(tracked, alive_ids=self.tracker.alive_track_ids())

        annotated = self._annotate(frame, tracked)
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with self._frame_lock:
                self._last_jpeg = buf.tobytes()

    def _annotate(self, frame: np.ndarray, tracked: list[tuple[int, geometry.BBox]]) -> np.ndarray:
        out = frame.copy()

        for table_id, table_state in self.engine.tables.items():
            zone = self.table_zones[table_id]
            pts = np.array(zone.polygon.exterior.coords, dtype=np.int32).reshape((-1, 1, 2))
            color = _STATUS_COLOR.get(table_state.status, (200, 200, 200))
            cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)
            label = f"{zone.name}: {table_state.status.upper()}"
            waited = table_state.waiting_seconds_now()
            if waited is not None:
                label += f" ({int(waited)}s)"
            anchor = tuple(pts[0][0])
            cv2.putText(out, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        for track_id, box in tracked:
            x1, y1, x2, y2 = map(int, box)
            state = self.engine.tracks.get(track_id)
            role = state.role if state else Role.INDETERMINADO
            color = _ROLE_COLOR.get(role, (160, 160, 160))
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, f"#{track_id} {role.value}", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return out

    def get_frame_jpeg(self) -> bytes | None:
        with self._frame_lock:
            return self._last_jpeg

    def mjpeg_generator(self):
        boundary = b"--frame"
        while True:
            frame = self.get_frame_jpeg()
            if frame is not None:
                yield (
                    boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            time.sleep(1 / 20)
