"""Herramienta de calibracion de zonas de mesa.

Risaldi et al. (2026, Sec. 2.6) requieren una camara estatica con
poligonos de mesa anotados manualmente; esta herramienta permite marcar
esos poligonos haciendo clic sobre un fotograma real de la fuente de
video, sin necesidad de editar coordenadas a mano.

Uso:
    python tools/calibrate_tables.py [ruta_video] [numero_de_frame]

Controles:
    clic izquierdo   -> agrega un vertice al poligono de la mesa actual
    'n'               -> cierra la mesa actual y comienza una nueva
    'z'               -> deshace el ultimo vertice
    's'               -> guarda config/tables.json y termina
    'q' / ESC         -> termina sin guardar los cambios pendientes
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app.vision.table_zones import save_table_zones  # noqa: E402

WINDOW = "ScanEats - Calibracion de mesas"


def main() -> None:
    video_path = sys.argv[1] if len(sys.argv) > 1 else config.VIDEO_SOURCE
    frame_number = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir '{video_path}'")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"No se pudo leer el fotograma {frame_number} de '{video_path}'")

    tables: list[dict] = []
    current_points: list[list[int]] = []
    table_count = 0

    def redraw():
        canvas = frame.copy()
        for t in tables:
            pts = t["polygon"]
            for i in range(len(pts)):
                cv2.line(canvas, tuple(pts[i]), tuple(pts[(i + 1) % len(pts)]), (0, 200, 0), 2)
            cx = sum(p[0] for p in pts) // len(pts)
            cy = sum(p[1] for p in pts) // len(pts)
            cv2.putText(canvas, t["name"], (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

        for i, p in enumerate(current_points):
            cv2.circle(canvas, tuple(p), 4, (0, 165, 255), -1)
            if i > 0:
                cv2.line(canvas, tuple(current_points[i - 1]), tuple(p), (0, 165, 255), 2)
        if len(current_points) > 1:
            cv2.line(canvas, tuple(current_points[-1]), tuple(current_points[0]), (0, 165, 255), 1)

        cv2.imshow(WINDOW, canvas)

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            current_points.append([x, y])
            redraw()

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    redraw()

    print(__doc__)

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == ord("n"):
            if len(current_points) >= 3:
                table_count += 1
                table_id = f"mesa_{table_count}"
                tables.append({"id": table_id, "name": f"Mesa {table_count}", "polygon": current_points.copy()})
                print(f"Guardada {table_id} con {len(current_points)} vertices.")
            else:
                print("Se necesitan al menos 3 vertices para cerrar una mesa.")
            current_points.clear()
            redraw()

        elif key == ord("z"):
            if current_points:
                current_points.pop()
                redraw()

        elif key == ord("s"):
            if len(current_points) >= 3:
                table_count += 1
                table_id = f"mesa_{table_count}"
                tables.append({"id": table_id, "name": f"Mesa {table_count}", "polygon": current_points.copy()})
            if not tables:
                print("No hay mesas para guardar.")
            else:
                save_table_zones(tables)
                print(f"Guardado {config.TABLE_ZONES_FILE} con {len(tables)} mesas.")
            break

        elif key in (ord("q"), 27):
            print("Cancelado, no se guardaron cambios.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
