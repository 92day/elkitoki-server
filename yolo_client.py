from __future__ import annotations

import glob
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

try:
    import torch
except Exception:
    torch = None

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

_MODEL = None

RISK_LEVEL_HIGH = "high"
RISK_LEVEL_MEDIUM = "medium"
RISK_LEVEL_LOW = "low"
RISK_LEVEL_SAFE = "safe"

PPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "helmet": ("helmet", "hardhat", "hard_hat", "safety_helmet", "hard_hat"),
    "vest": ("vest", "safety_vest", "hi_vis", "reflective_vest"),
    "harness": ("harness", "safety_belt", "safety_harness"),
    "gloves": ("glove", "gloves"),
    "boots": ("boot", "boots", "safety_boot"),
    "goggles": ("goggle", "goggles", "eye_protection"),
    "mask": ("mask", "respirator"),
}

PPE_LABELS = {
    "helmet": "Helmet",
    "vest": "Safety vest",
    "harness": "Safety harness",
    "gloves": "Gloves",
    "boots": "Safety boots",
    "goggles": "Safety goggles",
    "mask": "Mask/respirator",
}

PERSON_KEYWORDS = (
    "person",
    "worker",
    "human",
    "man",
    "woman",
    "people",
)

HEAVY_EQUIPMENT_KEYWORDS = (
    "excavator",
    "forklift",
    "crane",
    "truck",
    "dump",
    "bulldozer",
    "loader",
    "mixer",
    "roller",
    "backhoe",
    "lift",
    "drill",
)

FALL_HAZARD_KEYWORDS = (
    "fall",
    "edge",
    "ladder",
    "scaffold",
    "unprotected",
    "opening",
)

FIRE_ELECTRIC_KEYWORDS = (
    "fire",
    "smoke",
    "spark",
    "flame",
    "electric",
    "cable",
    "wire",
    "short",
)


def _normalize_label(raw: str) -> str:
    return (
        str(raw)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def _class_name(names: Any, cls_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(cls_id, cls_id))
    if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
        return str(names[cls_id])
    return str(cls_id)


def _candidate_model_paths() -> list[str]:
    candidates: list[str] = []

    explicit = (os.getenv("YOLO_MODEL_PATH") or "").strip()
    if explicit:
        explicit_path = Path(explicit).expanduser()
        if explicit_path.exists():
            candidates.append(str(explicit_path.resolve()))

    server_root = Path(__file__).resolve().parent
    workspace_root = server_root.parent
    legacy_root = Path.home() / "Projects" / "elkitoki-yolo"

    search_roots = [server_root, workspace_root, legacy_root]
    search_patterns = [
        "best.pt",
        "runs/**/weights/best.pt",
        "runs/**/weights/last.pt",
    ]

    for root in search_roots:
        if not root.exists():
            continue
        for pattern in search_patterns:
            found = glob.glob(str(root / pattern), recursive=True)
            candidates.extend(found)

    unique_paths: list[str] = []
    seen: set[str] = set()
    for path in candidates:
        abs_path = str(Path(path).resolve())
        if abs_path in seen:
            continue
        if not Path(abs_path).exists():
            continue
        seen.add(abs_path)
        unique_paths.append(abs_path)
    return unique_paths


def _get_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    try:
        from ultralytics import YOLO

        candidates = _candidate_model_paths()
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
    if torch is None:
        return "cpu"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _has_any_keyword(label: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in label for keyword in keywords)


def _missing_ppe_item(label: str) -> Optional[str]:
    missing_tokens = ("no_", "without_", "missing_", "not_wearing_")
    if not (label.startswith(missing_tokens) or "_no_" in label):
        return None

    for item, keywords in PPE_KEYWORDS.items():
        if _has_any_keyword(label, keywords):
            return item
    return None


def _assess_safety_risk(detections: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [_normalize_label(str(d.get("class_name", ""))) for d in detections]

    ppe_missing_items: set[str] = set()
    fall_hazards: list[str] = []
    fire_or_electric_hazards: list[str] = []
    heavy_equipment: list[str] = []

    person_count = 0

    for label in labels:
        if _has_any_keyword(label, PERSON_KEYWORDS):
            person_count += 1

        missing_item = _missing_ppe_item(label)
        if missing_item:
            ppe_missing_items.add(missing_item)

        if _has_any_keyword(label, FALL_HAZARD_KEYWORDS):
            fall_hazards.append(label)

        if _has_any_keyword(label, FIRE_ELECTRIC_KEYWORDS):
            fire_or_electric_hazards.append(label)

        if _has_any_keyword(label, HEAVY_EQUIPMENT_KEYWORDS):
            heavy_equipment.append(label)

    warnings: list[str] = []
    recommended_actions: list[str] = []
    risk_score = 0

    if ppe_missing_items:
        readable_items = [PPE_LABELS[item] for item in sorted(ppe_missing_items)]
        warnings.append("Missing PPE detected: " + ", ".join(readable_items))
        recommended_actions.append(
            "Stop work for workers without PPE and provide the required protective gear immediately."
        )
        risk_score += 3 + len(ppe_missing_items)

    if fall_hazards:
        warnings.append("Fall-related hazard pattern detected in the scene.")
        recommended_actions.append(
            "Inspect work-at-height controls: guardrails, lifelines, and mandatory harness use."
        )
        risk_score += 4

    if fire_or_electric_hazards:
        warnings.append("Potential fire/electrical hazard pattern detected.")
        recommended_actions.append(
            "Isolate power if needed, remove ignition sources, and assign a fire watch for the area."
        )
        risk_score += 4

    if heavy_equipment:
        if person_count > 0:
            warnings.append("Workers and heavy equipment are present in the same scene.")
            recommended_actions.append(
                "Set an exclusion zone around heavy equipment and assign a spotter for safe movement control."
            )
            risk_score += 3
        else:
            warnings.append("Heavy equipment activity detected.")
            recommended_actions.append(
                "Verify traffic routes, blind-spot controls, and signaler guidance for equipment operations."
            )
            risk_score += 1

    if risk_score >= 7:
        risk_level = RISK_LEVEL_HIGH
    elif risk_score >= 4:
        risk_level = RISK_LEVEL_MEDIUM
    elif risk_score >= 1:
        risk_level = RISK_LEVEL_LOW
    else:
        risk_level = RISK_LEVEL_SAFE

    risk_detected = risk_level in {RISK_LEVEL_HIGH, RISK_LEVEL_MEDIUM}
    if ppe_missing_items:
        risk_detected = True

    if not detections:
        warnings.append("No objects detected. Confirm camera angle and lighting quality.")
        if not recommended_actions:
            recommended_actions.append(
                "Retake the photo from a wider angle to improve safety inspection coverage."
            )

    if not warnings:
        warnings.append("No critical safety pattern detected from current YOLO classes.")

    if not recommended_actions:
        recommended_actions.append("Maintain routine patrol and continue periodic PPE checks.")

    return {
        "risk_level": risk_level,
        "risk_detected": risk_detected,
        "warnings": warnings,
        "ppe_missing": [PPE_LABELS[item] for item in sorted(ppe_missing_items)],
        "recommended_actions": recommended_actions,
    }


def analyze_with_yolo(image_bytes: bytes, ext: str) -> dict[str, Any]:
    model = _get_model()
    if model is None:
        return {
            "available": False,
            "detections": [],
            "summary": "YOLO model is unavailable.",
            "risk_detected": False,
            "risk_level": RISK_LEVEL_SAFE,
            "warnings": [],
            "ppe_missing": [],
            "recommended_actions": [],
        }

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
                cls_name = _class_name(names, cls_id)
                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(float(box.conf.item()), 3),
                    "xyxy": [round(float(v), 1) for v in box.xyxy[0].tolist()],
                })

        assessment = _assess_safety_risk(detections)

        return {
            "available": True,
            "detections": detections,
            "count": len(detections),
            "risk_detected": assessment["risk_detected"],
            "risk_level": assessment["risk_level"],
            "warnings": assessment["warnings"],
            "ppe_missing": assessment["ppe_missing"],
            "recommended_actions": assessment["recommended_actions"],
            "summary": _build_summary(detections, assessment),
        }
    except Exception as e:
        return {
            "available": False,
            "detections": [],
            "summary": f"YOLO error: {e}",
            "risk_detected": False,
            "risk_level": RISK_LEVEL_SAFE,
            "warnings": [f"YOLO inference failed: {e}"],
            "ppe_missing": [],
            "recommended_actions": [],
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def _build_summary(detections: list[dict[str, Any]], assessment: dict[str, Any]) -> str:
    if not detections:
        return "No objects detected. Risk level: SAFE."

    counter: dict[str, int] = {}
    for d in detections:
        counter[d["class_name"]] = counter.get(d["class_name"], 0) + 1

    detected_text = "Detected: " + ", ".join(f"{v}x {k}" for k, v in counter.items())
    risk_text = f"Risk level: {str(assessment.get('risk_level', RISK_LEVEL_SAFE)).upper()}"
    ppe_missing = assessment.get("ppe_missing") or []
    if ppe_missing:
        return detected_text + " | " + risk_text + " | Missing PPE: " + ", ".join(ppe_missing)
    return detected_text + " | " + risk_text
