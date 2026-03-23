from __future__ import annotations

import glob
import os
import tempfile
import importlib
from pathlib import Path
from typing import Any

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

_MODEL = None
_TORCH = None
_TORCH_LOADED = False

RISK_LEVEL_HIGH = 'high'
RISK_LEVEL_MEDIUM = 'medium'
RISK_LEVEL_LOW = 'low'
RISK_LEVEL_SAFE = 'safe'

PPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    'helmet': ('helmet', 'hardhat', 'hard_hat', 'safety_helmet'),
    'vest': ('vest', 'safety_vest', 'hi_vis', 'reflective_vest'),
    'harness': ('harness', 'safety_belt', 'safety_harness'),
    'gloves': ('glove', 'gloves'),
    'boots': ('boot', 'boots', 'safety_boot'),
    'goggles': ('goggle', 'goggles', 'eye_protection'),
    'mask': ('mask', 'respirator'),
}

PPE_LABELS = {
    'helmet': '\uC548\uC804\uBAA8',
    'vest': '\uC548\uC804\uC870\uB07C',
    'harness': '\uC548\uC804\uB300',
    'gloves': '\uC7A5\uAC11',
    'boots': '\uC548\uC804\uD654',
    'goggles': '\uBCF4\uC548\uACBD',
    'mask': '\uB9C8\uC2A4\uD06C/\uD638\uD761\uBCF4\uD638\uAD6C',
}

PERSON_KEYWORDS = ('person', 'worker', 'human', 'man', 'woman', 'people')
HEAVY_EQUIPMENT_KEYWORDS = (
    'excavator', 'forklift', 'crane', 'truck', 'dump', 'bulldozer', 'loader', 'mixer', 'roller', 'backhoe', 'lift', 'drill'
)
FALL_HAZARD_KEYWORDS = ('fall', 'edge', 'ladder', 'scaffold', 'unprotected', 'opening')
FIRE_ELECTRIC_KEYWORDS = ('fire', 'smoke', 'spark', 'flame', 'electric', 'cable', 'wire', 'short')


def _yolo_enabled() -> bool:
    return os.getenv('ENABLE_YOLO', '1').strip().lower() not in {'0', 'false', 'no', 'off'}


def _get_torch_module():
    global _TORCH, _TORCH_LOADED
    if _TORCH_LOADED:
        return _TORCH

    _TORCH_LOADED = True
    if not _yolo_enabled():
        _TORCH = None
        return _TORCH

    try:
        _TORCH = importlib.import_module('torch')
    except Exception:  # pragma: no cover
        _TORCH = None
    return _TORCH


def _normalize_label(raw: str) -> str:
    return str(raw).strip().lower().replace('-', '_').replace(' ', '_')


def _class_name(names: Any, cls_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(cls_id, cls_id))
    if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
        return str(names[cls_id])
    return str(cls_id)


def _resolve_model_path() -> str | None:
    if not _yolo_enabled():
        return None

    explicit_path = (os.getenv('YOLO_MODEL_PATH') or '').strip()
    if explicit_path:
        candidate = Path(explicit_path).expanduser()
        if candidate.exists():
            return str(candidate)

    raw_globs = (os.getenv('YOLO_MODEL_GLOBS') or '').strip()
    patterns = [pattern.strip() for pattern in raw_globs.split(';') if pattern.strip()]
    if patterns:
        candidates: list[str] = []
        for pattern in patterns:
            candidates.extend(glob.glob(pattern, recursive=True))
        if candidates:
            return max(candidates, key=os.path.getmtime)

    server_root = Path(__file__).resolve().parent
    fallback_patterns = [
        server_root / 'best.pt',
        server_root / 'runs' / '**' / 'weights' / 'best.pt',
        server_root.parent / 'elkitoki-yolo' / 'runs' / '**' / 'weights' / 'best.pt',
    ]
    found: list[str] = []
    for pattern in fallback_patterns:
        found.extend(glob.glob(str(pattern), recursive=True))
    if found:
        return max(found, key=os.path.getmtime)
    return None


def _torch_device() -> str:
    torch = _get_torch_module()
    try:
        if torch is not None and getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            return 'mps'
        if torch is not None and torch.cuda.is_available():
            return 'cuda'
    except Exception:
        pass
    return 'cpu'


def _get_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    if not _yolo_enabled():
        return None

    model_path = _resolve_model_path()
    if not model_path:
        return None

    try:
        from ultralytics import YOLO

        _MODEL = YOLO(model_path)
        print(f'[YOLO] model loaded: {model_path}')
        return _MODEL
    except Exception as error:
        print(f'[YOLO] model load failed: {error}')
        return None


def _has_any_keyword(label: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in label for keyword in keywords)


def _missing_ppe_item(label: str) -> str | None:
    missing_tokens = ('no_', 'without_', 'missing_', 'not_wearing_')
    if not (label.startswith(missing_tokens) or '_no_' in label):
        return None

    for item, keywords in PPE_KEYWORDS.items():
        if _has_any_keyword(label, keywords):
            return item
    return None


def _assess_safety_risk(detections: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [_normalize_label(str(d.get('class_name', ''))) for d in detections]

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
        warnings.append('\uBCF4\uD638\uAD6C \uBBF8\uCC29\uC6A9\uC774 \uAC10\uC9C0\uB418\uC5C8\uC2B5\uB2C8\uB2E4: ' + ', '.join(readable_items))
        recommended_actions.append('\uBCF4\uD638\uAD6C \uBBF8\uCC29\uC6A9 \uC791\uC5C5\uC790\uB97C \uC989\uC2DC \uD655\uC778\uD558\uACE0 \uD544\uC694\uD55C \uBCF4\uD638\uAD6C\uB97C \uCC29\uC6A9\uC2DC\uCF1C \uC8FC\uC138\uC694.')
        risk_score += 3 + len(ppe_missing_items)

    if fall_hazards:
        warnings.append('\uB099\uC0C1 \uC704\uD5D8 \uC694\uC18C\uAC00 \uAC10\uC9C0\uB418\uC5C8\uC2B5\uB2C8\uB2E4.')
        recommended_actions.append('\uACE0\uC18C \uC791\uC5C5 \uAD6C\uAC04\uC758 \uB09C\uAC04, \uC548\uC804\uB300, \uC791\uC5C5 \uBC1C\uD310 \uC0C1\uD0DC\uB97C \uC989\uC2DC \uC810\uAC80\uD574 \uC8FC\uC138\uC694.')
        risk_score += 4

    if fire_or_electric_hazards:
        warnings.append('\uC804\uAE30 \uB610\uB294 \uD654\uC7AC \uAD00\uB828 \uC704\uD5D8 \uC9D5\uD6C4\uAC00 \uAC10\uC9C0\uB418\uC5C8\uC2B5\uB2C8\uB2E4.')
        recommended_actions.append('\uC804\uC6D0, \uCF00\uC774\uBE14, \uC778\uD654\uBB3C \uC8FC\uBCC0\uC744 \uC989\uC2DC \uC810\uAC80\uD558\uACE0 \uD544\uC694 \uC2DC \uC791\uC5C5\uC744 \uC911\uB2E8\uD574 \uC8FC\uC138\uC694.')
        risk_score += 4

    if heavy_equipment:
        if person_count > 0:
            warnings.append('\uC911\uC7A5\uBE44\uC640 \uC791\uC5C5\uC790\uAC00 \uAC19\uC740 \uAD6C\uC5ED\uC5D0\uC11C \uD568\uAED8 \uAC10\uC9C0\uB418\uC5C8\uC2B5\uB2C8\uB2E4.')
            recommended_actions.append('\uC911\uC7A5\uBE44 \uC8FC\uBCC0 \uC811\uADFC \uAE08\uC9C0 \uAD6C\uC5ED\uC744 \uD45C\uC2DC\uD558\uACE0 \uC720\uB3C4 \uC778\uB825\uC744 \uBC30\uCE58\uD574 \uC8FC\uC138\uC694.')
            risk_score += 3
        else:
            warnings.append('\uC911\uC7A5\uBE44 \uC791\uC5C5\uC774 \uAC10\uC9C0\uB418\uC5C8\uC2B5\uB2C8\uB2E4.')
            recommended_actions.append('\uC911\uC7A5\uBE44 \uC774\uB3D9 \uB3D9\uC120\uACFC \uC791\uC5C5 \uBC18\uACBD\uC744 \uB2E4\uC2DC \uD655\uC778\uD574 \uC8FC\uC138\uC694.')
            risk_score += 1

    if risk_score >= 7:
        risk_level = RISK_LEVEL_HIGH
    elif risk_score >= 4:
        risk_level = RISK_LEVEL_MEDIUM
    elif risk_score >= 1:
        risk_level = RISK_LEVEL_LOW
    else:
        risk_level = RISK_LEVEL_SAFE

    risk_detected = risk_level in {RISK_LEVEL_HIGH, RISK_LEVEL_MEDIUM} or bool(ppe_missing_items)

    if not detections:
        warnings.append('\uAC10\uC9C0\uB41C \uAC1D\uCCB4\uAC00 \uC5C6\uC2B5\uB2C8\uB2E4. \uCE74\uBA54\uB77C \uAC01\uB3C4\uC640 \uC870\uBA85\uC744 \uD655\uC778\uD574 \uC8FC\uC138\uC694.')
        recommended_actions.append('\uC0AC\uC9C4\uC744 \uB354 \uB113\uC740 \uAC01\uB3C4\uC5D0\uC11C \uB2E4\uC2DC \uCD2C\uC601\uD574 \uC8FC\uC138\uC694.')

    if not warnings:
        warnings.append('\uD604\uC7AC YOLO \uBD84\uC11D \uAE30\uC900\uC73C\uB85C \uC989\uC2DC \uC704\uD5D8\uC694\uC18C\uB294 \uAC10\uC9C0\uB418\uC9C0 \uC54A\uC558\uC2B5\uB2C8\uB2E4.')

    if not recommended_actions:
        recommended_actions.append('\uD604\uC7A5 \uC8FC\uBCC0\uC744 \uC7AC\uC810\uAC80\uD558\uACE0 \uBCF4\uD638\uAD6C \uCC29\uC6A9 \uC5EC\uBD80\uB97C \uC8FC\uAE30\uC801\uC73C\uB85C \uD655\uC778\uD574 \uC8FC\uC138\uC694.')

    return {
        'risk_level': risk_level,
        'risk_detected': risk_detected,
        'warnings': warnings,
        'ppe_missing': [PPE_LABELS[item] for item in sorted(ppe_missing_items)],
        'recommended_actions': recommended_actions,
    }


def analyze_with_yolo(image_bytes: bytes, ext: str) -> dict[str, Any]:
    model = _get_model()
    if model is None:
        return {
            'available': False,
            'detections': [],
            'summary': 'YOLO model unavailable',
            'risk_detected': False,
            'risk_level': RISK_LEVEL_SAFE,
            'warnings': [],
            'ppe_missing': [],
            'recommended_actions': [],
        }

    suffix = ext if ext.startswith('.') else f'.{ext}'
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(image_bytes)
            temp_path = temp_file.name

        result = model.predict(
            source=temp_path,
            conf=0.25,
            imgsz=640,
            device=_torch_device(),
            verbose=False,
        )[0]

        detections = []
        names = getattr(result, 'names', {})
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                class_name = _class_name(names, class_id)
                detections.append(
                    {
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': round(float(box.conf.item()), 3),
                        'xyxy': [round(float(value), 1) for value in box.xyxy[0].tolist()],
                    }
                )

        assessment = _assess_safety_risk(detections)

        return {
            'available': True,
            'detections': detections,
            'count': len(detections),
            'risk_detected': assessment['risk_detected'],
            'risk_level': assessment['risk_level'],
            'warnings': assessment['warnings'],
            'ppe_missing': assessment['ppe_missing'],
            'recommended_actions': assessment['recommended_actions'],
            'summary': _build_summary(detections, assessment),
        }
    except Exception as error:
        return {
            'available': False,
            'detections': [],
            'summary': f'YOLO error: {error}',
            'risk_detected': False,
            'risk_level': RISK_LEVEL_SAFE,
            'warnings': [f'YOLO inference failed: {error}'],
            'ppe_missing': [],
            'recommended_actions': [],
        }
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def _build_summary(detections: list[dict[str, Any]], assessment: dict[str, Any]) -> str:
    if not detections:
        return 'No objects detected. Risk level: SAFE.'

    counter: dict[str, int] = {}
    for detection in detections:
        class_name = detection['class_name']
        counter[class_name] = counter.get(class_name, 0) + 1

    detected_text = 'Detected: ' + ', '.join(f'{count}x {class_name}' for class_name, count in counter.items())
    risk_text = f'Risk level: {str(assessment.get("risk_level", RISK_LEVEL_SAFE)).upper()}'
    ppe_missing = assessment.get('ppe_missing') or []
    if ppe_missing:
        return detected_text + ' | ' + risk_text + ' | Missing PPE: ' + ', '.join(ppe_missing)
    return detected_text + ' | ' + risk_text
