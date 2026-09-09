"""Auto-calibracion de zonas de mesa usando la clase nativa 'dining table'
de COCO (con la que ya viene entrenado yolov8s.pt/yolov8n.pt - no hace
falta entrenar nada nuevo).

Por que no se usa esta deteccion en vivo, cuadro a cuadro, en el
pipeline principal: la clase "dining table" es bastante mas ruidosa que
"person" (mesas cubiertas por manteles/platos/comensales dan cajas
inconsistentes), y Risaldi et al. (2026, Sec. 2.6) fijan deliberadamente
los poligonos de mesa para que el calculo de ocupacion (centroide de
cliente + margen contra poligono de mesa) no herede ese ruido.

La solucion intermedia que implementa este script: correr el detector
varios fotogramas SOLO al calibrar una camara nueva, agrupar las cajas
de "dining table" detectadas en clusters estables (una mesa real produce
muchas detecciones ligeramente distintas alrededor del mismo lugar) y
promediarlas en un poligono fijo por mesa - el mismo resultado que
`calibrate_tables.py` (clic manual) mas rapido y sin intervencion, con
la opcion de revisar/ajustar antes de aplicarlo.

Uso:
    python tools/auto_calibrate_tables.py [fuente] [--seconds N]
        [--min-ratio 0.15] [--iou 0.35] [--conf 0.15] [--apply]

    fuente        ruta de video o URL rtsp:// (por defecto: config.VIDEO_SOURCE)
    --seconds     duracion de la observacion en segundos de video (default 20)
    --min-ratio   fraccion minima de fotogramas procesados en que debe
                  aparecer un cluster para aceptarse como mesa real (default 0.15)
    --iou         umbral de IoU para agrupar una deteccion nueva a un
                  cluster existente (default 0.35)
    --conf        umbral de confianza para 'dining table' (default 0.15,
                  mas bajo que YOLO_CONF_THRESHOLD: mesas cubiertas por
                  manteles/platos/comensales bajan mucho la confianza)
    --apply       escribe directamente en config/tables.json (con backup
                  del archivo previo). Sin este flag, escribe en
                  config/tables.auto.json para revision manual.

Salida:
    config/tables.auto.json (o config/tables.json con --apply)
    config/tables.auto.preview.jpg  (fotograma anotado con las zonas detectadas)
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app.vision.geometry import iou  # noqa: E402
from ultralytics import YOLO  # noqa: E402

DINING_TABLE_CLASS_NAME = "dining table"


class TableCluster:
    """Agrupa detecciones de 'dining table' que corresponden a la misma
    mesa fisica a lo largo del tiempo (misma idea que el tracking de
    personas, pero simplificada: aqui basta una media movil, no hace
    falta un filtro de Kalman porque una mesa no se desplaza)."""

    def __init__(self, box: tuple[float, float, float, float]):
        self.boxes: list[tuple[float, float, float, float]] = [box]
        self.running_box = box

    def add(self, box: tuple[float, float, float, float]) -> None:
        self.boxes.append(box)
        # media movil simple para decidir con que comparar la siguiente deteccion
        n = len(self.boxes)
        self.running_box = tuple(
            (self.running_box[i] * (n - 1) + box[i]) / n for i in range(4)
        )

    def final_polygon(self) -> list[list[int]]:
        arr = np.array(self.boxes)
        x1, y1, x2, y2 = np.median(arr, axis=0)
        return [[int(x1), int(y1)], [int(x2), int(y1)], [int(x2), int(y2)], [int(x1), int(y2)]]


def resolve_class_id(model: YOLO, class_name: str) -> int:
    for class_id, name in model.names.items():
        if name == class_name:
            return int(class_id)
    raise SystemExit(
        f"El modelo '{config.YOLO_MODEL_PATH}' no tiene una clase '{class_name}'. "
        "Verifica que sea un modelo entrenado sobre COCO."
    )


def cluster_detections(
    model: YOLO, class_id: int, source: str, seconds: float, iou_threshold: float, conf_threshold: float
) -> tuple[list[TableCluster], int, np.ndarray]:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la fuente de video '{source}'")

    clusters: list[TableCluster] = []
    frames_processed = 0
    last_frame: np.ndarray | None = None
    deadline = time.monotonic() + seconds
    frame_idx = 0

    print(f"Observando '{source}' durante {seconds:.0f}s buscando mesas...")
    while time.monotonic() < deadline:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_idx += 1
        if frame_idx % config.PROCESS_EVERY_N_FRAMES != 0:
            continue

        last_frame = frame
        frames_processed += 1

        results = model(frame, classes=[class_id], conf=conf_threshold, verbose=False)[0]
        for box in results.boxes:
            det = tuple(box.xyxy[0].tolist())
            best_cluster, best_iou = None, 0.0
            for cluster in clusters:
                score = iou(det, cluster.running_box)
                if score > best_iou:
                    best_cluster, best_iou = cluster, score
            if best_cluster is not None and best_iou >= iou_threshold:
                best_cluster.add(det)
            else:
                clusters.append(TableCluster(det))

    cap.release()
    if last_frame is None:
        raise SystemExit("No se pudo leer ningun fotograma de la fuente de video.")
    return clusters, frames_processed, last_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", default=config.VIDEO_SOURCE)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--min-ratio", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.35)
    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="Umbral de confianza para 'dining table' (mas bajo que el de personas: "
        "las mesas suelen quedar tapadas por platos/manteles/comensales, ver README).",
    )
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.01,
        help="Area minima de un cluster, como fraccion del area del frame, para "
        "aceptarse como mesa real (filtra falsos positivos diminutos, ej. un "
        "trozo de torso mal clasificado). Default 0.01 (1%%).",
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    model = YOLO(config.YOLO_MODEL_PATH)
    class_id = resolve_class_id(model, DINING_TABLE_CLASS_NAME)

    clusters, frames_processed, last_frame = cluster_detections(
        model, class_id, args.source, args.seconds, args.iou, args.conf
    )

    frame_h, frame_w = last_frame.shape[:2]
    min_area = args.min_area_ratio * frame_w * frame_h

    def area(box: tuple[float, float, float, float]) -> float:
        return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])

    min_observations = max(1, int(frames_processed * args.min_ratio))
    aceptadas = [
        c for c in clusters if len(c.boxes) >= min_observations and area(c.running_box) >= min_area
    ]

    # Fusionar clusters aceptados que en realidad son la misma mesa fisica
    # detectada con cajas de tamano inconsistente entre fotogramas (queda
    # por debajo del --iou frame-a-frame pero es un solapamiento evidente
    # una vez promediados). Sin esto, una sola mesa puede aparecer duplicada.
    merged = True
    while merged:
        merged = False
        for i in range(len(aceptadas)):
            for j in range(i + 1, len(aceptadas)):
                if iou(aceptadas[i].running_box, aceptadas[j].running_box) >= 0.2:
                    aceptadas[i].boxes.extend(aceptadas[j].boxes)
                    aceptadas[i].running_box = tuple(
                        float(np.median(np.array(aceptadas[i].boxes)[:, k])) for k in range(4)
                    )
                    del aceptadas[j]
                    merged = True
                    break
            if merged:
                break

    aceptadas.sort(key=lambda c: c.running_box[0])  # orden izquierda -> derecha

    print(f"\nFotogramas procesados: {frames_processed}")
    print(f"Candidatas detectadas: {len(clusters)}  |  aceptadas (>= {min_observations} observaciones): {len(aceptadas)}")

    if not aceptadas:
        print(
            "\nNo se confirmo ninguna mesa. Prueba con --seconds mas alto, "
            "--min-ratio mas bajo, o revisa que la camara vea las mesas sin "
            "demasiada oclusion."
        )
        return

    tables = []
    preview = last_frame.copy()
    for i, cluster in enumerate(aceptadas, start=1):
        table_id = f"mesa_{i}"
        polygon = cluster.final_polygon()
        tables.append({"id": table_id, "name": f"Mesa {i}", "polygon": polygon})
        print(f"  {table_id}: {len(cluster.boxes)} observaciones, poligono {polygon}")

        pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(preview, [pts], isClosed=True, color=(0, 200, 0), thickness=2)
        cv2.putText(preview, table_id, tuple(polygon[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

    preview_path = config.BASE_DIR / "config" / "tables.auto.preview.jpg"
    cv2.imwrite(str(preview_path), preview)
    print(f"\nVista previa guardada en: {preview_path}")

    payload = {
        "_comment": (
            f"Generado automaticamente por tools/auto_calibrate_tables.py "
            f"({DINING_TABLE_CLASS_NAME}, {frames_processed} fotogramas). "
            "Revisa config/tables.auto.preview.jpg antes de confiar en esto."
        ),
        "tables": tables,
    }

    import json

    if args.apply:
        target = config.TABLE_ZONES_FILE
        if target.exists():
            backup = target.with_suffix(".json.bak")
            shutil.copy2(target, backup)
            print(f"Respaldo del archivo anterior: {backup}")
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Aplicado directamente en: {target}")
    else:
        target = config.BASE_DIR / "config" / "tables.auto.json"
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Guardado para revision en: {target}")
        print("Si se ve bien, vuelve a correr este script con --apply, o renombralo a tables.json a mano.")


if __name__ == "__main__":
    main()
