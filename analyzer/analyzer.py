"""ДИПЛОМ 3 — Edge-analyzer.

Крутиться на Raspberry Pi поруч з обладнанням:
  1. Читає метрики з MQTT (Диплом 1 publisher).
  2. Запитує очікувані межі з онтологічного API (Диплом 2).
  3. Пропускає через IsolationForest.
  4. Поєднує ML-вердикт з правилами на межах з онтології.
  5. Публікує стан у MQTT і тримає кеш останнього стану (для веб-панелі).

Аргумент «edge»: усе це робиться локально, без інтернету, затримка <100мс.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import paho.mqtt.client as mqtt
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.schemas import (
    MetricsMessage,
    StateMessage,
    TOPIC_METRICS_WILDCARD,
    TOPIC_STATE,
    utcnow_iso,
)

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("analyzer")

MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
ONTOLOGY_API = os.getenv("ONTOLOGY_API", "http://localhost:5000")

HERE = Path(__file__).parent
bundle = joblib.load(HERE / "anomaly_model.pkl")
MODEL = bundle["model"]
FEATURES = bundle["features"]


# ─── Кеш останніх станів для веб-панелі ────────────────────────────────
@dataclass
class DeviceStatus:
    device_id: str
    last_metrics: dict = field(default_factory=dict)
    last_state: dict = field(default_factory=dict)
    updated_at: str = ""


STATE_CACHE: dict[str, DeviceStatus] = {}
STATE_LOCK = threading.Lock()

# ─── Кеш очікуваних меж з онтології (щоб не бити API на кожне повідомлення)
BOUNDS_CACHE: dict[str, dict] = {}
BOUNDS_TTL_SEC = 300
BOUNDS_FETCHED: dict[str, float] = {}


def get_bounds(device_id: str) -> dict:
    now = time.time()
    if device_id in BOUNDS_CACHE and now - BOUNDS_FETCHED.get(device_id, 0) < BOUNDS_TTL_SEC:
        return BOUNDS_CACHE[device_id]
    try:
        r = requests.get(f"{ONTOLOGY_API}/device/{device_id}/expected-bounds", timeout=3)
        r.raise_for_status()
        BOUNDS_CACHE[device_id] = r.json()
        BOUNDS_FETCHED[device_id] = now
    except Exception as exc:
        log.warning("Ontology API unavailable for %s (%s) — fallback bounds", device_id, exc)
        BOUNDS_CACHE[device_id] = {}
        BOUNDS_FETCHED[device_id] = now
    return BOUNDS_CACHE[device_id]


def rule_based_checks(metrics: dict, bounds: dict) -> list[str]:
    """Явні правила з онтології — дають інтерпретовані аномалії
    поруч зі статистичним вердиктом моделі."""
    anomalies = []
    if bounds.get("min_cop") is not None and metrics.get("cop") is not None:
        if metrics["cop"] < bounds["min_cop"]:
            anomalies.append(f"cop_below_nominal({metrics['cop']:.2f}<{bounds['min_cop']:.2f})")
    if bounds.get("max_power_kw") is not None and metrics["power_kw"] > bounds["max_power_kw"]:
        anomalies.append(
            f"power_over_limit({metrics['power_kw']:.2f}>{bounds['max_power_kw']:.2f})"
        )
    if bounds.get("max_flow_c") is not None and metrics["flow_temp_c"] > bounds["max_flow_c"]:
        anomalies.append(
            f"flow_temp_over_limit({metrics['flow_temp_c']:.1f}>{bounds['max_flow_c']:.1f})"
        )
    return anomalies


def analyze(msg: MetricsMessage) -> StateMessage:
    m = msg.metrics
    feat = np.array([[
        m.power_kw,
        m.flow_temp_c - m.return_temp_c,
        m.flow_temp_c,
        m.outdoor_temp_c if m.outdoor_temp_c is not None else 0.0,
        m.cop if m.cop is not None else 3.5,
    ]])
    prediction = int(MODEL.predict(feat)[0])      # 1 = норма, -1 = викид
    score = float(MODEL.decision_function(feat)[0])  # більше = нормальніше

    bounds = get_bounds(msg.device_id)
    rule_anomalies = rule_based_checks(m.model_dump(), bounds)

    if prediction == -1 and rule_anomalies:
        state, anomalies = "anomaly", ["ml_outlier"] + rule_anomalies
    elif rule_anomalies:
        state, anomalies = "warning", rule_anomalies
    elif prediction == -1:
        state, anomalies = "warning", ["ml_outlier"]
    else:
        state, anomalies = "normal", []

    explanation = (
        f"ML score={score:.3f}; "
        f"rules matched: {len(rule_anomalies)}"
    )

    return StateMessage(
        device_id=msg.device_id,
        timestamp=utcnow_iso(),
        state=state,
        anomalies=anomalies,
        confidence=min(1.0, abs(score) / 0.5),
        explanation=explanation,
    )


# ─── MQTT glue ─────────────────────────────────────────────────────────
def on_message(client: mqtt.Client, _userdata, mqtt_msg: mqtt.MQTTMessage) -> None:
    try:
        incoming = MetricsMessage.model_validate_json(mqtt_msg.payload)
    except Exception:
        log.exception("Bad metrics payload on %s", mqtt_msg.topic)
        return

    state = analyze(incoming)
    out_topic = TOPIC_STATE.format(device_id=state.device_id)
    client.publish(out_topic, state.model_dump_json(), qos=0)

    with STATE_LOCK:
        status = STATE_CACHE.setdefault(state.device_id, DeviceStatus(state.device_id))
        status.last_metrics = incoming.metrics.model_dump()
        status.last_state = state.model_dump()
        status.updated_at = state.timestamp

    log.info(
        "%s → %s (%s) anomalies=%s",
        state.device_id, state.state, state.confidence, state.anomalies,
    )


def on_connect(client, *_):
    log.info("MQTT connected — subscribing %s", TOPIC_METRICS_WILDCARD)
    client.subscribe(TOPIC_METRICS_WILDCARD, qos=0)


def start(block: bool = True) -> mqtt.Client:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="edge-analyzer")
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    if block:
        client.loop_forever()
    else:
        client.loop_start()
    return client


if __name__ == "__main__":
    start()
