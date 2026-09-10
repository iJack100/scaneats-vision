# ScanEats Vision — Microservicio ajaja

Sistema de videovigilancia analitica para gestion operativa de restaurantes,
implementado segun el alcance inicial del informe de propuesta
(`S2-TAREA_1/00_Informe_Propuesta.pdf`): un microservicio en **FastAPI** que
procesa video con **YOLOv8 + OpenCV** para medir tiempos de espera, contar
interacciones del personal de salon y mapear el uso de las mesas.

Este repo es **solo** el microservicio de vision (Modulos 6.1 y 6.2 del
informe). La app principal (autenticacion, catalogo de mesas/meseros,
reportes consolidados — Modulo 6.3) vive en un repo aparte: **`scaneats`**
(Django + DRF), que consume la API de este microservicio por HTTP. Ver el
README de ese repo para levantar ambos juntos. El middleware CORS
(`app/main.py`) ya permite el origen `http://127.0.0.1:8001` (Django) por
defecto; ajustalo con `SCANEATS_CORS_ORIGINS` si Django corre en otro
puerto/host.

## Como ejecutarlo (standalone, con su propio dashboard)

```bash
# Windows (PowerShell), desde la raiz del proyecto
.\venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Abrir `http://127.0.0.1:8000` para el dashboard. El video de prueba
(`video.mp4`) se reproduce en bucle automaticamente en un hilo de fondo
apenas arranca el servidor.

### Recalibrar las mesas (recomendado)

`config/tables.json` trae 3 zonas ubicadas *a ojo* solo para que el MVP
corra de inmediato. Hay dos formas de recalibrar para una camara real:

**Manual (clic sobre un fotograma):**

```bash
.\venv\Scripts\python.exe tools\calibrate_tables.py video.mp4 0
```

**Automatica (sin clics, usando la clase nativa "dining table" de COCO):**

```bash
.\venv\Scripts\python.exe tools\auto_calibrate_tables.py video.mp4 --seconds 40 --conf 0.15
```

Observa la fuente de video un rato, agrupa las detecciones de "dining
table" en zonas estables y escribe `config/tables.auto.json` +
`config/tables.auto.preview.jpg` para que revises el resultado antes de
aplicarlo (`--apply` lo escribe directo en `config/tables.json`, con
backup del archivo anterior). Es genuinamente automatico, pero la clase
"dining table" es mas ruidosa que "person" -mesas tapadas por
manteles/platos/comensales bajan mucho la confianza-, asi que:

- Puede que necesites bajar `--conf` (default 0.15) o alargar
  `--seconds` si no detecta ninguna mesa.
- Es normal que en escenas muy concurridas/con obstrucciones en primer
  plano se pierdan algunas mesas reales (el filtro es conservador a
  proposito: prefiere omitir una mesa real a inventar una falsa). Usa
  `calibrate_tables.py` para completar a mano las que falten.
- Con una camara CCTV fija bien ubicada (el escenario que asumen Risaldi
  et al. y Mamedov et al.) deberia funcionar mejor que con este video de
  stock, grabado a mano y con mucha obstruccion en primer plano.

### Ajustar umbrales sin tocar codigo

Todos los parametros de `app/config.py` aceptan variables de entorno,
por ejemplo para ver una alerta en segundos durante una demo en vivo:

```bash
$env:SCANEATS_ALERT_THRESHOLD = "15"
.\venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

## Arquitectura (Modulo 6 del informe)

```
app/vision/       Motor de Vision Artificial (6.1)
  detector.py        YOLOv8 (deteccion de personas)
  geometry.py         torso superior, centroide+margen, IoU
  tracker.py           SORT: Kalman filter + algoritmo hungaro
  role_classifier.py   inferencia de rol cliente/mesero por comportamiento
  table_zones.py       poligonos de mesa (config/tables.json)

app/analytics/     Motor de Analitica y Reglas de Negocio (6.2)
  engine.py            ocupacion, cronometro de espera, interacciones, alertas

app/pipeline.py    conecta ambos motores en un hilo de video en vivo

app/api/           Interfaz de Administracion / Dashboard UI (6.3)
templates/ static/    dashboard servido con Jinja2 + JS plano

app/models.py      Gestion de Alertas y persistencia (6.2 / 6.4), SQLite
```

## Trazabilidad cientifica (que decision de codigo sustenta cada articulo)

| Articulo | Decision sustentada en el codigo |
|---|---|
| **Risaldi et al. (2026)** — overlap geometrico centroide+margen | `app/vision/geometry.py::is_occupying` implementa literalmente las formulas 1 y 2 del articulo (margen de 40 px, Shapely) para decidir si una mesa esta ocupada, en vez de clasificar visualmente "se ve ocupada". |
| **Tercan et al. (2023)** — YOLO como mejor tradeoff velocidad/precision | Sustenta usar YOLOv8 preentrenado (`app/vision/detector.py`) en vez de arquitecturas mas pesadas (Cascade R-CNN) para mantener tiempo real en CPU. |
| **Akash et al. (2024)** — filtro de colision 3D (no solo 2D) | Sin sensores de profundidad, `app/vision/role_classifier.py` exige **proximidad sostenida durante N frames** (`PROXIMITY_SUSTAIN_FRAMES`) antes de contar una interaccion mesero-mesa, evitando el falso positivo de "solo paso caminando" que describe el articulo. |
| **Mamedov, Kuplyakov & Konushin (2021)** — reidentificacion por torso superior | `app/vision/geometry.py::upper_body_box` recorta el 50% superior de cada deteccion; el tracker SORT (`app/vision/tracker.py`) opera sobre esas cajas, no el cuerpo completo, para no perder el ID de un cliente ocluido por sillas/mesas. |
| **De Vries, Roy & De Koster (2018)** — impacto del tiempo de espera en ingresos/abandono | Justifica que el tiempo de espera sea la metrica central: `WaitRecord` y `AlertLog` en `app/models.py`, y el umbral configurable `WAIT_ALERT_THRESHOLD_SECONDS`. |

### Decision no cubierta por ningun articulo: cliente vs. mesero

Ningun articulo entrena una clase "mesero" (Risaldi solo distingue
customer/non-customer/table). Se resolvio con la regla de comportamiento
acordada con el equipo: *"un cliente no se va a estar moviendo por todo
el local, y un mesero no se va a estar sentado tanto tiempo"*. Ver
`app/vision/role_classifier.py::TrackState` — un track se confirma
**MESERO** al acumular proximidad sostenida con `MESERO_MIN_TABLES_VISITED`
(2 por defecto) mesas distintas estando de pie, y **CLIENTE** al
permanecer sentado `CLIENTE_MIN_SEATED_FRAMES` fotogramas cerca de una
sola mesa.

## Limitaciones conocidas del MVP (heredadas de los propios articulos)

- Camara estatica obligatoria; cualquier cambio de angulo exige
  recalibrar `config/tables.json` (limitacion explicita de Risaldi et al.).
- La postura sentado/de pie es una heuristica geometrica (relacion
  alto/ancho), no un clasificador entrenado: puede fallar con angulos de
  camara muy cenitales o muy laterales.
- No se distinguen objetos abandonados sobre la mesa (celular, bolso)
  como senal de ocupacion — limitacion que el propio articulo de Risaldi
  reconoce como trabajo futuro.
