"""Rutas del Dashboard UI (Modulo 6.3) y API REST consumida por el frontend."""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates

from app import config
from app.api import stats
from app.database import SessionLocal
from app.vision.role_classifier import Role

router = APIRouter()
templates = Jinja2Templates(directory=str(config.BASE_DIR / "templates"))


def _pipeline(request: Request):
    return request.app.state.pipeline


@router.get("/")
def dashboard_page(request: Request):
    return templates.TemplateResponse(request, "dashboard.html", {"alert_threshold": config.WAIT_ALERT_THRESHOLD_SECONDS})


@router.get("/api/status")
def api_status(request: Request):
    pipeline = _pipeline(request)
    return {
        "running": pipeline.running,
        "frames_processed": pipeline.frames_processed,
        "tables_configured": len(pipeline.table_zones),
        "alert_threshold_seconds": config.WAIT_ALERT_THRESHOLD_SECONDS,
    }


@router.get("/api/tables")
def api_tables(request: Request):
    pipeline = _pipeline(request)
    result = []
    for table_id, state in pipeline.engine.tables.items():
        result.append(
            {
                "table_id": table_id,
                "name": state.name,
                "status": state.status,
                "waiting_seconds": state.waiting_seconds_now(),
                "last_wait_seconds": state.last_wait_seconds,
                "customer_track_id": state.customer_track_id,
            }
        )
    result.sort(key=lambda t: t["table_id"])
    return result


@router.get("/api/waiters")
def api_waiters(request: Request):
    pipeline = _pipeline(request)
    with SessionLocal() as db:
        performance = stats.waiter_performance(db)

    live_meseros = {
        tid for tid, t in pipeline.engine.tracks.items() if t.role == Role.MESERO
    }
    for row in performance:
        row["en_pantalla"] = row["waiter_track_id"] in live_meseros
    return performance


@router.get("/api/alerts")
def api_alerts(request: Request):
    with SessionLocal() as db:
        return stats.recent_alerts(db)


@router.get("/api/heatmap")
def api_heatmap(request: Request):
    pipeline = _pipeline(request)
    with SessionLocal() as db:
        usage = stats.usage_heatmap(db)
    result = []
    for table_id, zone in pipeline.table_zones.items():
        entry = usage.get(table_id, {"occupied_seconds": 0.0, "times_occupied": 0})
        result.append(
            {
                "table_id": table_id,
                "name": zone.name,
                "occupied_seconds": round(entry["occupied_seconds"], 1),
                "times_occupied": entry["times_occupied"],
            }
        )
    result.sort(key=lambda t: t["occupied_seconds"], reverse=True)
    return result


@router.get("/stream/video")
def stream_video(request: Request):
    pipeline = _pipeline(request)
    return StreamingResponse(
        pipeline.mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame"
    )
