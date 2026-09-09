"""Punto de entrada de ScanEats (FastAPI).

Arranca el pipeline de video (Motor de Vision + Motor Analitico) como
tarea de fondo al iniciar el servidor y expone el Dashboard UI (Modulo
6.3) junto con el Alert & Notification Service (Modulo 6.4), que en este
MVP se resuelve consultando el estado en memoria de cada mesa: cuando
`status == "alerta"` el frontend pinta la mesa de rojo y muestra un
icono de advertencia, sin ningun sonido (Funcionalidad 3 del informe:
"notificaciones... estrictamente visuales y silenciosas").
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app import config
from app.api.routes_dashboard import router as dashboard_router
from app.database import init_db
from app.pipeline import VideoPipeline

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    pipeline = VideoPipeline()
    app.state.pipeline = pipeline
    pipeline.start()
    try:
        yield
    finally:
        pipeline.stop()


app = FastAPI(title="ScanEats", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(config.BASE_DIR / "static")), name="static")
app.include_router(dashboard_router)
