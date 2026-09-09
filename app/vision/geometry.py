"""Operaciones geometricas del Motor de Vision Artificial.

Implementa directamente las formulas 1 y 2 de Risaldi et al. (2026,
"Real-Time Table Availability Detection...", Sec. 2.6 "Table Status
Determination"):

    M(x, y) = {(x, y) | cx - m <= x <= cx + m, cy - m <= y <= cy + m}   (1)
    Occupied(T) = 1 si area(M inter T) > 0, 0 en otro caso              (2)

Es decir: se expande el centroide de la persona con un margen de
tolerancia `m` (por defecto 40 px, igual que en el articulo) y se evalua
si esa region interseca el poligono de la mesa mediante Shapely.
"""
from __future__ import annotations

from shapely.geometry import Polygon

from app import config

BBox = tuple[float, float, float, float]


def upper_body_box(full_box: BBox, ratio: float = None) -> BBox:
    """Recorta la mitad superior (torso) del bounding box de una persona.

    Justificacion cientifica: Mamedov, Kuplyakov & Konushin (2021),
    Sec. 3.2 "Upper Body Regression" - el torso superior sufre mucha
    menos oclusion por sillas/mesas que el cuerpo completo en un salon
    concurrido, por lo que produce un tracking (Sec. tracker.py) y un
    centroide de ocupacion (Risaldi et al. 2026) mas estables.
    """
    ratio = config.UPPER_BODY_HEIGHT_RATIO if ratio is None else ratio
    x1, y1, x2, y2 = full_box
    altura_total = y2 - y1
    y2_superior = y1 + altura_total * ratio
    return x1, y1, x2, y2_superior


def centroid_of(box: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def margin_polygon(cx: float, cy: float, margin: int = None) -> Polygon:
    """Formula (1) de Risaldi et al. (2026): region cuadrada de tolerancia
    alrededor del centroide del cliente."""
    m = config.CENTROID_MARGIN_PX if margin is None else margin
    return Polygon(
        [
            (cx - m, cy - m),
            (cx + m, cy - m),
            (cx + m, cy + m),
            (cx - m, cy + m),
        ]
    )

def is_occupying(box: BBox, table_polygon: Polygon, margin: int = None) -> bool:
    """Formula (2) de Risaldi et al. (2026): True si la region de margen
    del centroide interseca el poligono de la mesa."""
    cx, cy = centroid_of(box)
    region = margin_polygon(cx, cy, margin)
    return region.intersects(table_polygon)


def aspect_ratio(box: BBox) -> float:
    x1, y1, x2, y2 = box
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    return h / w


def is_sitting(upper_box: BBox) -> bool:
    """Heuristica geometrica que sustituye a las clases entrenadas
    "customer" (sentado) / "non-customer" (de pie) de Risaldi et al.
    (2026): ScanEats no dispone de un dataset propio para reentrenar
    YOLOv8 con esas clases, por lo que se aproxima la postura mediante la
    relacion alto/ancho del recorte de torso superior. Un torso "de pie"
    conserva una silueta mas alta y estrecha que uno "sentado", donde los
    hombros y brazos tienden a ensanchar la caja relativa a su altura.
    """
    return aspect_ratio(upper_box) < config.SITTING_ASPECT_RATIO_THRESHOLD


def iou(box_a: BBox, box_b: BBox) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
