"""Stage 4 — single LLM call with structured output.

Uses Anthropic's messages.parse() so the response is validated against
HeatPumpProfile automatically. One call per PDF — no chunking needed
because we already filtered to ≤30 spec-dense pages.
"""
from __future__ import annotations

import logging

import anthropic

from .schema import HeatPumpProfile

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You extract structured technical specifications from
heat-pump manufacturer manuals.

Read the provided manual excerpts (already filtered to specification-rich
pages) and emit a single HeatPumpProfile JSON object.

Rules:
1. Cite ONLY values stated explicitly in the text. Use null when a value
   is not stated. Do not infer or estimate from related values.
2. Prefer English text. If a value appears only in a non-English section,
   extract it but use the English unit names.
3. If multiple variants are described in one table, pick the most
   representative numbers (typically the middle or default variant) and
   list ALL variant SKUs in model_variants.
4. Convert units to the schema's units: °C, kW, kg, L, V.
5. operating_modes: include a mode only if the manual states the unit
   supports it (heating / cooling / dhw / standby).
"""


def extract_profile(text: str, model: str = "claude-opus-4-7") -> HeatPumpProfile:
    """Send filtered manual text to Claude, return validated profile."""
    client = anthropic.Anthropic()
    log.info("Calling %s on %d chars of input", model, len(text))

    response = client.messages.parse(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text}],
        output_format=HeatPumpProfile,
    )

    log.info(
        "Tokens — input=%d output=%d (cache_read=%d)",
        response.usage.input_tokens,
        response.usage.output_tokens,
        getattr(response.usage, "cache_read_input_tokens", 0) or 0,
    )
    return response.parsed_output
