"""Catalogo de zonas de mesa (poligonos fijos sobre la imagen de la camara).

Risaldi et al. (2026) senalan explicitamente como limitacion que su
metodo "depende de una configuracion estatica del restaurante y de la
perspectiva de la camara. Por lo tanto, alteraciones significativas en
la disposicion de las mesas o en la posicion de la camara requieren
re-anotacion de los poligonos de las mesas" (Sec. 2.6). ScanEats asume lo
mismo para su MVP: las mesas se calibran una vez por camara con
`tools/calibrate_tables.py` y se guardan en `config/tables.json`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from shapely.geometry import Polygon

from app import config


@dataclass
class TableZone:
    id: str
    name: str
    polygon: Polygon


def load_table_zones(path: Path | None = None) -> dict[str, TableZone]:
    path = path or config.TABLE_ZONES_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"No existe '{path}'. Ejecuta 'python tools/calibrate_tables.py' "
            "para marcar los poligonos de las mesas sobre un fotograma del "
            "video, o usa el archivo de ejemplo en config/tables.json."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    zones: dict[str, TableZone] = {}
    for entry in data["tables"]:
        zones[entry["id"]] = TableZone(
            id=entry["id"],
            name=entry.get("name", entry["id"]),
            polygon=Polygon(entry["polygon"]),
        )
    return zones


def save_table_zones(zones: list[dict], path: Path | None = None) -> None:
    path = path or config.TABLE_ZONES_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"tables": zones}, indent=2, ensure_ascii=False), encoding="utf-8")
