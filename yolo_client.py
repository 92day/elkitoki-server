from __future__ import annotations

import glob
import os
import tempfile
from typing import Any

import torch

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    try:
        from ultralytics import YOLO

        patterns = [
            "/Users/joha-eun/Projects/elkitoki-yolo/runs/**/weights/best.pt",
            "/Users/joha-eun/Projects/elkitoki-yolo/runs/**/weights/last.pt",
        ]
        candidates = []
        for p in patterns:
            candidates.extend(glob.glob(p, recursive=True))
        if not candidates:
            return None
        model_path = max(candidates, key=os.path.getmtime)
        _MODEL = YOLO(model_path)
        print(f"[YOLO] 모델 로드: {model_path}")
        return _MODEL
    except Exception as e:
        print(f"[YOLO] 모델 로드 실패: {e}")
        return None


def _device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def analyze_with_yolo(image_bytes: bytes, ext: str) -> dict[str, Any]:
    model = _get_model()
    if model is None:
        return {"available": False, "detections": [], "summary": "YOLO 모델 없음"}

    suffix = ext if ext.startswith(".") else f".{ext}"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        result = model.predict(
            source=tmp_path,
            conf=0.25,
            imgsz=320,
            device=_device(),
            verbose=False,
        )[0]

        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                names = result.names
                cls_name = (
                    names[cls_id]
                    if isinstance(names, (list, dict)) and cls_id in (names if isinstance(names, dict) else range(len(names)))
                    else str(cls_id)
                )
                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(float(box.conf.item()), 3),
                    "xyxy": [round(float(v), 1) for v in box.xyxy[0].tolist()],
                })

        RISK_CLASSES = {"fall_down", "no_vest", "no_helmet"}
        risk_detected = any(d["class_name"] in RISK_CLASSES for d in detections)

        return {
            "available": True,
            "detections": detections,
            "count": len(detections),
            "risk_detected": risk_detected,
            "summary": _build_summary(detections),
        }
    except Exception as e:
        return {"available": False, "detections": [], "summary": f"YOLO 오류: {e}"}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def _build_summary(detections: list[dict]) -> str:
    if not detections:
        return "No objects detected."
    counter: dict[str, int] = {}
    for d in detections:
        counter[d["class_name"]] = counter.get(d["class_name"], 0) + 1
    return "Detected: " + ", ".join(f"{v}x {k}" for k, v in counter.items())
