"""ДИПЛОМ 2 — ШІ-компонент: парсить PDF-паспорт обладнання через LLM
і витягує технічні характеристики у структурованому форматі.

Це саме те, що робить тему «засобами ШІ». Можна показувати в захисті:
на вхід — «сирий» паспорт у PDF, на виході — готові тріпли для онтології.

Підтримує два бекенди:
  * OpenAI  (якщо є OPENAI_API_KEY у .env)
  * Ollama локальна LLM (за замовчуванням http://localhost:11434, модель llama3)
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("llm_parser")

OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

SYSTEM_PROMPT = (
    "You are an information extractor for heat-pump datasheets. "
    "From the given datasheet text, extract ONLY these fields into JSON: "
    "manufacturer (string), model (string), nominal_power_kw (float), "
    "max_power_kw (float), max_flow_temp_c (float), min_flow_temp_c (float), "
    "min_cop (float). Use null for anything not explicitly stated. "
    "Respond with JSON only, no prose."
)


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join((p.extract_text() or "") for p in reader.pages)


def call_openai(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_KEY)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content


def call_ollama(prompt: str) -> str:
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def extract_specs(pdf_path: Path) -> dict:
    text = extract_text(pdf_path)[:12000]
    raw = call_openai(text) if OPENAI_KEY else call_ollama(text)
    return json.loads(raw)


def to_turtle(device_id: str, specs: dict) -> str:
    """Перетворює витягнуті характеристики на фрагмент Turtle для онтології."""
    lines = [
        "@prefix lab: <http://lab.example/ontology#> .",
        "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
        "",
        f"lab:{device_id} a lab:AirToWaterHP ;",
    ]
    mapping = {
        "manufacturer":     ("lab:manufacturer",   "xsd:string"),
        "model":            ("lab:model",          "xsd:string"),
        "nominal_power_kw": ("lab:nominalPowerKw", "xsd:float"),
        "max_power_kw":     ("lab:maxPowerKw",     "xsd:float"),
        "max_flow_temp_c":  ("lab:maxFlowTempC",   "xsd:float"),
        "min_flow_temp_c":  ("lab:minFlowTempC",   "xsd:float"),
        "min_cop":          ("lab:minCOP",         "xsd:float"),
    }
    triples = []
    for key, (pred, dtype) in mapping.items():
        val = specs.get(key)
        if val is None:
            continue
        literal = f'"{val}"' if dtype == "xsd:string" else f'"{val}"^^{dtype}'
        triples.append(f"    {pred} {literal}")
    return "\n".join(lines) + " ;\n" + " ;\n".join(triples) + " .\n"


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python llm_parser.py <device_id> <datasheet.pdf>")
        sys.exit(1)
    device_id, pdf = sys.argv[1], Path(sys.argv[2])
    specs = extract_specs(pdf)
    log.info("Extracted: %s", specs)
    print(to_turtle(device_id, specs))
