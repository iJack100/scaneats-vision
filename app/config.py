"""Configuracion central de ScanEats.

Cada umbral referencia el articulo cientifico que lo sustenta (ver
S2-TAREA_1/00_Informe_Propuesta.pdf, seccion 7 "Sustento cientifico").
Todos los valores marcados como override por variable de entorno pueden
ajustarse sin tocar codigo, por ejemplo para acortar el umbral de alerta
durante una demo en vivo.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Fuente de video ---
VIDEO_SOURCE = os.environ.get("SCANEATS_VIDEO", str(BASE_DIR / "video.mp4"))
TABLE_ZONES_FILE = Path(os.environ.get("SCANEATS_TABLES_FILE", str(BASE_DIR / "config" / "tables.json")))
DATABASE_URL = os.environ.get("SCANEATS_DB_URL", f"sqlite:///{BASE_DIR / 'scaneats.db'}")

# --- Motor de Vision Artificial (Computer Vision Engine, modulo 6.1) ---
# YOLOv8 como detector base: Risaldi et al. (2026) y Tercan et al. (2023)
# sustentan el uso de la familia YOLO por su balance velocidad/precision
# en tiempo real usando solo CPU.
YOLO_MODEL_PATH = os.environ.get("SCANEATS_YOLO_MODEL", str(BASE_DIR / "yolov8s.pt"))
YOLO_CONF_THRESHOLD = float(os.environ.get("SCANEATS_YOLO_CONF", "0.35"))
PERSON_CLASS_ID = 0  # clase 'person' de COCO (unica clase entrenada disponible sin dataset propio)
PROCESS_EVERY_N_FRAMES = int(os.environ.get("SCANEATS_FRAME_SKIP", "2"))

# --- Extraccion de torso superior (Mamedov, Kuplyakov & Konushin, 2021) ---
# El tracking y la geometria de ocupacion operan sobre la mitad superior
# del bounding box de cada persona: el torso sufre menos oclusion por
# sillas y mesas que el cuerpo completo (Mamedov et al. 2021, Sec. 3.2
# "Upper Body Regression"). Esto tambien sustenta el Modulo 6.1 del
# informe ("aislando caracteristicas fisicas como el torso superior").
UPPER_BODY_HEIGHT_RATIO = float(os.environ.get("SCANEATS_UPPER_BODY_RATIO", "0.5"))

# --- Ocupacion de mesas (Risaldi et al. 2026, Sec. 2.6 "Table Status Determination") ---
# Margen (px) aplicado al centroide de la persona antes de intersectar
# con el poligono de la mesa (Formulas 1 y 2 del articulo, m = 40 px).
CENTROID_MARGIN_PX = int(os.environ.get("SCANEATS_MARGIN_PX", "40"))

# Frames consecutivos sin overlap antes de declarar una mesa "libre".
# Evita que una oclusion momentanea del cliente (p. ej. un mesero
# pasando enfrente) reinicie el cronometro de espera de esa mesa; el
# propio SORT de Risaldi et al. (2026) usa max_age=50 con el mismo
# proposito de tolerar huecos breves de deteccion.
OCCUPANCY_GRACE_FRAMES = int(os.environ.get("SCANEATS_OCCUPANCY_GRACE", "5"))

# --- Clasificacion de postura (sentado / de pie) ---
# Heuristica geometrica (alto/ancho del recorte de torso superior) que
# sustituye a las clases entrenadas "customer"/"non-customer" de Risaldi
# et al. (2026), ya que ScanEats no cuenta con dataset propio para
# reentrenar YOLOv8 con esas clases.
SITTING_ASPECT_RATIO_THRESHOLD = float(os.environ.get("SCANEATS_SITTING_RATIO", "1.0"))

# --- Inferencia de rol por comportamiento (cliente vs. mesero) ---
# Regla de negocio acordada con el equipo: "un cliente no se va a estar
# moviendo por todo el local, y un mesero no se va a estar sentado tanto
# tiempo". Se opera sobre el historial del track (persistente gracias al
# tracking robusto a oclusion de Mamedov et al. 2021).
MESERO_MIN_TABLES_VISITED = int(os.environ.get("SCANEATS_MESERO_MIN_TABLES", "2"))
CLIENTE_MIN_SEATED_FRAMES = int(os.environ.get("SCANEATS_CLIENTE_MIN_FRAMES", "10"))

# --- Filtro de proximidad sostenida (Akash et al. 2024, filtro de colision 3D) ---
# Sin sensores de profundidad, la "distancia real" se sustituye por la
# exigencia de proximidad SOSTENIDA durante N frames procesados
# consecutivos, para no contar como interaccion a alguien que solo pasa
# caminando cerca de la mesa (el mismo problema que Akash et al. resuelven
# con su filtro de colision 3D para pares mesero-cliente).
PROXIMITY_SUSTAIN_FRAMES = int(os.environ.get("SCANEATS_PROXIMITY_FRAMES", "5"))

# --- Alertas de atencion al cliente (De Vries, Roy & De Koster, 2018) ---
# El articulo demuestra el fuerte impacto del tiempo de espera sobre el
# abandono, la duracion del consumo y el retorno del cliente, lo que
# sustenta priorizar esta metrica como eje central del sistema de alertas.
WAIT_ALERT_THRESHOLD_SECONDS = int(os.environ.get("SCANEATS_ALERT_THRESHOLD", "600"))

# --- Tracking (SORT: Kalman filter + algoritmo hungaro) ---
# Combinacion explicitamente citada por Risaldi et al. (2026, Sec. 2.5)
# para mantener IDs consistentes de clientes entre fotogramas.
TRACK_MAX_AGE = int(os.environ.get("SCANEATS_TRACK_MAX_AGE", "30"))
TRACK_MIN_HITS = int(os.environ.get("SCANEATS_TRACK_MIN_HITS", "3"))
TRACK_IOU_THRESHOLD = float(os.environ.get("SCANEATS_TRACK_IOU", "0.3"))

# --- CORS ---
# El proyecto Django (app principal, repo aparte) sirve su dashboard en
# otro origen/puerto y hace fetch() en el navegador directamente contra
# esta API; sin CORS el navegador bloquearia esas llamadas.
CORS_ALLOWED_ORIGINS = os.environ.get(
    "SCANEATS_CORS_ORIGINS", "http://127.0.0.1:8001,http://localhost:8001"
).split(",")
