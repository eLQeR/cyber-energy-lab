"""Сервер тривог: приймає alerts від edge-analyzer та віддає їх інженерові у веб-UI.

Архітектура:
  edge-analyzer (Raspberry Pi) ── HTTP POST /api/alerts ──► alerts_server (FastAPI)
                                                            │
                                                            ├─ SQLite (alerts.db)
                                                            └─ Web Dashboard /
                                                               (read-only для інженера)

Запуск:
  uvicorn alerts_server.alerts_server:app --host 0.0.0.0 --port 5003 --reload
  # або:
  python alerts_server/alerts_server.py            # підніме uvicorn локально

REST API:
  POST   /api/alerts                    — створити нову тривогу
  GET    /api/alerts                    — список (?status=active|acknowledged|resolved&device_id=...)
  GET    /api/alerts/<id>               — деталі
  POST   /api/alerts/<id>/acknowledge   — інженер прийняв до уваги
  POST   /api/alerts/<id>/resolve       — інженер усунув причину
  GET    /api/devices                   — поточний стан кожного пристрою (з онтології)
  GET    /api/devices/<id>              — деталі пристрою + остання тривога
  GET    /api/stats                     — зведена статистика (для cards у dashboard)
  GET    /docs                          — OpenAPI / Swagger UI (FastAPI built-in)

Web UI:
  GET    /                              — dashboard
  GET    /device/<device_id>            — деталі обладнання
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import requests
from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from influxdb_client import InfluxDBClient
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.schemas import AlertPayload, utcnow_iso

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("alerts_server")

# ─── Конфігурація ─────────────────────────────────────────────────────────────
DB_PATH      = Path(os.getenv("ALERTS_DB", str(Path(__file__).parent / "alerts.db")))
ONTOLOGY_API = os.getenv("ONTOLOGY_API", "http://localhost:5000")
PORT         = int(os.getenv("ALERTS_PORT", "5003"))

# Скільки секунд тривога вважається "active", якщо не було оновлень
STALE_AFTER_SEC = int(os.getenv("STALE_AFTER_SEC", "600"))

# InfluxDB — джерело історії метрик для графіків на сторінці пристрою
INFLUX_URL    = os.getenv("INFLUX_URL",    "http://localhost:8086")
INFLUX_TOKEN  = os.getenv("INFLUX_TOKEN",  "lab-dev-token")
INFLUX_ORG    = os.getenv("INFLUX_ORG",    "lab")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "metrics")

HERE      = Path(__file__).parent
templates = Jinja2Templates(directory=str(HERE / "templates"))

app = FastAPI(
    title="Engineer Alerts Dashboard",
    description="Edge-analyzer → центральний сервер тривог для лабораторії кіберенергетичних систем.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")

_db_lock = threading.Lock()


# ─── База даних ───────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id        TEXT    NOT NULL,
    severity         TEXT    NOT NULL CHECK(severity IN ('warning','anomaly')),
    anomaly_codes    TEXT    NOT NULL,                -- JSON array
    explanation      TEXT    NOT NULL DEFAULT '',
    confidence       REAL    NOT NULL DEFAULT 1.0,
    metrics_snapshot TEXT    NOT NULL DEFAULT '{}',   -- JSON
    bounds_snapshot  TEXT    NOT NULL DEFAULT '{}',   -- JSON
    raised_at        TEXT    NOT NULL,
    acknowledged_at  TEXT,
    acknowledged_by  TEXT,
    resolved_at      TEXT
);
CREATE INDEX IF NOT EXISTS idx_alerts_device   ON alerts(device_id);
CREATE INDEX IF NOT EXISTS idx_alerts_raised   ON alerts(raised_at DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_resolved ON alerts(resolved_at);

CREATE TABLE IF NOT EXISTS device_state (
    device_id     TEXT    PRIMARY KEY,
    last_seen     TEXT    NOT NULL,
    state         TEXT    NOT NULL,
    metrics       TEXT    NOT NULL DEFAULT '{}',
    bounds        TEXT    NOT NULL DEFAULT '{}'
);
"""


@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _db_lock, db() as conn:
        conn.executescript(SCHEMA)
    log.info("DB ready at %s", DB_PATH)


# ─── Pydantic-моделі для UI-only ендпоінтів ──────────────────────────────────

class AcknowledgeBody(BaseModel):
    user: str = "engineer"


# ─── Допоміжні ────────────────────────────────────────────────────────────────

def _row_to_alert(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["anomaly_codes"]    = json.loads(d["anomaly_codes"])
    d["metrics_snapshot"] = json.loads(d["metrics_snapshot"])
    d["bounds_snapshot"]  = json.loads(d["bounds_snapshot"])
    if d["resolved_at"]:
        d["status"] = "resolved"
    elif d["acknowledged_at"]:
        d["status"] = "acknowledged"
    else:
        d["status"] = "active"
    return d


def _is_stale(raised_at: str | None) -> bool:
    if not raised_at:
        return True
    try:
        dt = datetime.strptime(raised_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds() > STALE_AFTER_SEC
    except Exception:
        return False


def fetch_ontology_devices() -> list[dict]:
    try:
        r = requests.get(f"{ONTOLOGY_API}/devices", timeout=3)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        log.warning("Ontology API unavailable: %s", exc)
        return []


def fetch_ontology_bounds(device_id: str) -> dict:
    try:
        r = requests.get(f"{ONTOLOGY_API}/device/{device_id}/expected-bounds", timeout=3)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


# ─── REST: тривоги ────────────────────────────────────────────────────────────

@app.post("/api/alerts", status_code=201, summary="Створити нову тривогу (від edge-analyzer)")
def create_alert(payload: AlertPayload):
    """analyzer публікує сюди при переході стану в warning / anomaly."""
    with _db_lock, db() as conn:
        cur = conn.execute(
            """INSERT INTO alerts
               (device_id, severity, anomaly_codes, explanation, confidence,
                metrics_snapshot, bounds_snapshot, raised_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                payload.device_id, payload.severity,
                json.dumps(payload.anomaly_codes, ensure_ascii=False),
                payload.explanation, payload.confidence,
                json.dumps(payload.metrics_snapshot, ensure_ascii=False),
                json.dumps(payload.bounds_snapshot, ensure_ascii=False),
                payload.timestamp,
            ),
        )
        alert_id = cur.lastrowid

        conn.execute(
            """INSERT INTO device_state (device_id, last_seen, state, metrics, bounds)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(device_id) DO UPDATE SET
                 last_seen=excluded.last_seen,
                 state=excluded.state,
                 metrics=excluded.metrics,
                 bounds=excluded.bounds""",
            (
                payload.device_id, payload.timestamp, payload.severity,
                json.dumps(payload.metrics_snapshot, ensure_ascii=False),
                json.dumps(payload.bounds_snapshot, ensure_ascii=False),
            ),
        )

    log.info("ALERT [%d] %s severity=%s codes=%s",
             alert_id, payload.device_id, payload.severity, payload.anomaly_codes)
    return {"id": alert_id, "status": "created"}


@app.get("/api/alerts", summary="Список тривог")
def list_alerts(
    status: Literal["active", "acknowledged", "resolved"] | None = None,
    device_id: str | None = None,
    limit: int = Query(200, ge=1, le=1000),
):
    sql, params = "SELECT * FROM alerts WHERE 1=1", []
    if status == "active":
        sql += " AND acknowledged_at IS NULL AND resolved_at IS NULL"
    elif status == "acknowledged":
        sql += " AND acknowledged_at IS NOT NULL AND resolved_at IS NULL"
    elif status == "resolved":
        sql += " AND resolved_at IS NOT NULL"
    if device_id:
        sql += " AND device_id = ?"
        params.append(device_id)
    sql += " ORDER BY raised_at DESC LIMIT ?"
    params.append(limit)

    with db() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_alert(r) for r in rows]


@app.get("/api/alerts/{alert_id}", summary="Деталі тривоги")
def get_alert(alert_id: int):
    with db() as conn:
        row = conn.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Alert not found")
    return _row_to_alert(row)


@app.post("/api/alerts/{alert_id}/acknowledge", summary="Підтвердити тривогу")
def acknowledge_alert(alert_id: int, body: AcknowledgeBody = Body(default=AcknowledgeBody())):
    with _db_lock, db() as conn:
        cur = conn.execute(
            """UPDATE alerts SET acknowledged_at = ?, acknowledged_by = ?
               WHERE id = ? AND acknowledged_at IS NULL""",
            (utcnow_iso(), body.user, alert_id),
        )
        if cur.rowcount == 0:
            raise HTTPException(404, "Alert not found or already acknowledged")
    return {"id": alert_id, "status": "acknowledged"}


@app.post("/api/alerts/{alert_id}/resolve", summary="Закрити тривогу (причину усунено)")
def resolve_alert(alert_id: int):
    with _db_lock, db() as conn:
        cur = conn.execute(
            "UPDATE alerts SET resolved_at = ? WHERE id = ? AND resolved_at IS NULL",
            (utcnow_iso(), alert_id),
        )
        if cur.rowcount == 0:
            raise HTTPException(404, "Alert not found or already resolved")
    return {"id": alert_id, "status": "resolved"}


# ─── REST: пристрої ──────────────────────────────────────────────────────────

@app.get("/api/devices", summary="Список пристроїв з онтології + поточний стан")
def list_devices():
    devices = fetch_ontology_devices()

    with db() as conn:
        states = {r["device_id"]: dict(r) for r in
                  conn.execute("SELECT * FROM device_state").fetchall()}
        active_counts = {r["device_id"]: r["c"] for r in conn.execute(
            """SELECT device_id, COUNT(*) AS c FROM alerts
               WHERE acknowledged_at IS NULL AND resolved_at IS NULL
               GROUP BY device_id"""
        ).fetchall()}
        last_alerts = {}
        for r in conn.execute(
            """SELECT device_id, MAX(raised_at) AS latest, severity, anomaly_codes
               FROM alerts GROUP BY device_id"""
        ).fetchall():
            last_alerts[r["device_id"]] = {
                "latest": r["latest"], "severity": r["severity"],
                "anomaly_codes": json.loads(r["anomaly_codes"]),
            }

    out = []
    for d in devices:
        did = d["id"]
        st  = states.get(did, {})
        metrics = json.loads(st.get("metrics", "{}")) if st else {}
        bounds  = json.loads(st.get("bounds",  "{}")) if st else {}
        if not bounds:
            bounds = fetch_ontology_bounds(did)
        out.append({
            **d,
            "current_state": st.get("state", "unknown"),
            "last_seen":     st.get("last_seen"),
            "stale":         _is_stale(st.get("last_seen")),
            "metrics":       metrics,
            "bounds":        bounds,
            "active_alerts": active_counts.get(did, 0),
            "last_alert":    last_alerts.get(did),
        })
    return out


@app.get("/api/devices/{device_id}", summary="Деталі пристрою + історія тривог")
def get_device(device_id: str):
    bounds = fetch_ontology_bounds(device_id)

    specs = {}
    try:
        r = requests.get(f"{ONTOLOGY_API}/device/{device_id}/specs", timeout=3)
        if r.ok:
            specs = r.json().get("specs", {})
    except Exception:
        pass

    with db() as conn:
        st = conn.execute("SELECT * FROM device_state WHERE device_id = ?",
                          (device_id,)).fetchone()
        alerts = conn.execute(
            "SELECT * FROM alerts WHERE device_id = ? ORDER BY raised_at DESC LIMIT 100",
            (device_id,),
        ).fetchall()

    return {
        "device_id": device_id,
        "specs":     specs,
        "bounds":    bounds,
        "current_state": st["state"]   if st else "unknown",
        "metrics":       json.loads(st["metrics"])   if st else {},
        "last_seen":     st["last_seen"]             if st else None,
        "stale":         _is_stale(st["last_seen"])  if st else True,
        "alerts":  [_row_to_alert(r) for r in alerts],
    }


@app.get("/api/devices/{device_id}/history",
         summary="Історія метрик з InfluxDB для графіків (за замовчуванням 60 хв)")
def device_history(device_id: str, minutes: int = Query(60, ge=1, le=1440)):
    """Повертає рівномірно дискретизовані точки для графіків:
    [{ timestamp, power_kw, cop, flow_temp_c, return_temp_c, outdoor_temp_c }, …]"""
    flux = f"""
        from(bucket: "{INFLUX_BUCKET}")
          |> range(start: -{minutes}m)
          |> filter(fn: (r) => r["_measurement"] == "equipment_metrics")
          |> filter(fn: (r) => r["device_id"] == "{device_id}")
          |> filter(fn: (r) => r["_field"] == "power_kw" or r["_field"] == "cop"
                            or r["_field"] == "flow_temp_c" or r["_field"] == "return_temp_c"
                            or r["_field"] == "outdoor_temp_c")
          |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
          |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> sort(columns: ["_time"])
    """
    try:
        with InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG) as client:
            tables = client.query_api().query(flux)
    except Exception as exc:
        log.warning("InfluxDB query failed: %s", exc)
        return {"device_id": device_id, "minutes": minutes, "points": []}

    points = []
    for table in tables:
        for rec in table.records:
            v = rec.values
            points.append({
                "timestamp":      v["_time"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                "power_kw":       v.get("power_kw"),
                "cop":            v.get("cop"),
                "flow_temp_c":    v.get("flow_temp_c"),
                "return_temp_c":  v.get("return_temp_c"),
                "outdoor_temp_c": v.get("outdoor_temp_c"),
            })
    return {"device_id": device_id, "minutes": minutes, "points": points}


@app.get("/api/stats", summary="Зведена статистика для dashboard")
def stats():
    with db() as conn:
        c_active = conn.execute(
            "SELECT COUNT(*) FROM alerts WHERE acknowledged_at IS NULL AND resolved_at IS NULL"
        ).fetchone()[0]
        c_ack = conn.execute(
            "SELECT COUNT(*) FROM alerts WHERE acknowledged_at IS NOT NULL AND resolved_at IS NULL"
        ).fetchone()[0]
        c_resolved_24h = conn.execute(
            "SELECT COUNT(*) FROM alerts WHERE resolved_at >= ?",
            ((datetime.now(timezone.utc) - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ"),),
        ).fetchone()[0]
        by_severity = {r["severity"]: r["c"] for r in conn.execute(
            """SELECT severity, COUNT(*) AS c FROM alerts
               WHERE acknowledged_at IS NULL AND resolved_at IS NULL
               GROUP BY severity"""
        ).fetchall()}
    devices = fetch_ontology_devices()
    return {
        "active":        c_active,
        "acknowledged":  c_ack,
        "resolved_24h":  c_resolved_24h,
        "devices_total": len(devices),
        "by_severity":   by_severity,
    }


# ─── Web UI ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def dashboard(request: Request):
    return templates.TemplateResponse(request, "dashboard.html")


@app.get("/device/{device_id}", response_class=HTMLResponse, include_in_schema=False)
def device_detail(request: Request, device_id: str):
    return templates.TemplateResponse(
        request, "device_detail.html", {"device_id": device_id},
    )


@app.get("/health", summary="Liveness-перевірка")
def health():
    return {"status": "ok", "db": str(DB_PATH)}


# ─── Startup hook + entrypoint ────────────────────────────────────────────────

@app.on_event("startup")
def _startup():
    init_db()
    log.info("Alerts server ready (ontology API: %s)", ONTOLOGY_API)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
