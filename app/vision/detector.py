"""Wrapper del detector YOLOv8 (Modulo 6.1: Computer Vision Engine).

Risaldi et al. (2026) usan YOLOv8 por su diseno anchor-free, buena
representacion de caracteristicas e inferencia rapida, lo que lo hace
adecuado para reconocer clientes en un salon concurrido; Tercan et al.
(2023) confirman en un benchmark comparativo que la familia YOLO ofrece
el mejor equilibrio velocidad/precision frente a Cascade R-CNN, VFNet,
TOOD, etc. para deteccion en tiempo real por camara.

ScanEats usa los pesos pre-entrenados de COCO (clase 0 = "person") en
lugar de entrenar una red desde cero, tal como declara la propuesta en
"Factibilidad" (seccion 3.5): "Esto evita construir redes neuronales
desde cero, permitiendo que el equipo enfoque el tiempo restante en...
la logica del backend".
"""
from __future__ import annotations

from ultralytics import YOLO

from app import config
from app.vision.geometry import BBox


class PersonDetector:
    def __init__(self, model_path: str | None = None):
        self.model = YOLO(model_path or config.YOLO_MODEL_PATH)

    def detect(self, frame) -> list[BBox]:
        """Devuelve las cajas delimitadoras (x1, y1, x2, y2) de personas
        detectadas en el fotograma."""
        results = self.model(
            frame,
            classes=[config.PERSON_CLASS_ID],
            conf=config.YOLO_CONF_THRESHOLD,
            verbose=False,
        )[0]

        boxes: list[BBox] = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append((x1, y1, x2, y2))
        return boxes
