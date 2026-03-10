from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


API_BASE = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com").rstrip("/")
DEFAULT_TEXT_MODELS = ["gemini-2.5-flash"]
DEFAULT_VISION_MODELS = ["gemini-2.5-flash"]


def _get_models(env_name: str, defaults: list[str]) -> list[str]:
    raw = (os.getenv(env_name) or "").strip()
    if not raw:
        return defaults
    models = [m.strip() for m in raw.split(",") if m.strip()]
    return models or defaults


def _extract_text(payload: dict[str, Any]) -> str:
    texts: list[str] = []
    candidates = payload.get("candidates") or []
    for candidate in candidates:
        content = candidate.get("content") or {}
        parts = content.get("parts") or []
        for part in parts:
            text = part.get("text")
            if text:
                texts.append(text)
    return "\n".join(texts).strip()


def _extract_error_message(payload: dict[str, Any]) -> str:
    err = payload.get("error")
    if isinstance(err, dict):
        msg = err.get("message")
        if msg:
            return str(msg)
        return str(err)
    return str(payload)


def _generate_content(
    model: str,
    contents: list[dict[str, Any]],
    max_output_tokens: int = 800,
    temperature: float = 0.2,
) -> dict[str, Any]:
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    model_path = urllib.parse.quote(model, safe="")
    key_param = urllib.parse.quote(api_key, safe="")
    url = f"{API_BASE}/v1beta/models/{model_path}:generateContent?key={key_param}"

    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")
            parsed = json.loads(body)
            message = _extract_error_message(parsed)
        except Exception:
            message = str(exc)
        raise RuntimeError(f"Gemini HTTP {exc.code}: {message}") from exc
    except Exception as exc:
        raise RuntimeError(f"Gemini request failed: {exc}") from exc


def analyze_text(prompt: str, max_output_tokens: int = 800) -> str:
    errors: list[str] = []
    for model in _get_models("GEMINI_TEXT_MODELS", DEFAULT_TEXT_MODELS):
        try:
            payload = _generate_content(
                model=model,
                contents=[{"parts": [{"text": prompt}]}],
                max_output_tokens=max_output_tokens,
                temperature=0.2,
            )
            text = _extract_text(payload)
            if text:
                return text
            errors.append(f"{model}: empty response")
        except Exception as exc:
            errors.append(f"{model}: {exc}")
    return "AI analysis failed (Gemini): " + " | ".join(errors)


def analyze_image(prompt: str, image_bytes: bytes, mime_type: str, max_output_tokens: int = 500) -> str:
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    errors: list[str] = []
    for model in _get_models("GEMINI_VISION_MODELS", DEFAULT_VISION_MODELS):
        try:
            payload = _generate_content(
                model=model,
                contents=[
                    {
                        "parts": [
                            {"text": prompt},
                            {"inline_data": {"mime_type": mime_type, "data": b64}},
                        ]
                    }
                ],
                max_output_tokens=max_output_tokens,
                temperature=0.1,
            )
            text = _extract_text(payload)
            if text:
                return text
            errors.append(f"{model}: empty response")
        except Exception as exc:
            errors.append(f"{model}: {exc}")
    return "AI analysis failed (Gemini Vision): " + " | ".join(errors)
