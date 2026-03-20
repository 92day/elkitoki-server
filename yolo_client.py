from __future__ import annotations

import glob
import os
import tempfile
from pathlib import Path
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

_MODEL = None

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
    'helmet': '안전모',
    'vest': '안전조끼',
    'harness': '안전대',
    'gloves': '장갑',
    'boots': '안전화',
    'goggles': '보안경',
    'mask': '마스크/호흡보호구',
}

PERSON_KEYWORDS = ('person', 'worker', 'human', 'man', 'woman', 'people')
HEAVY_EQUIPMENT_KEYWORDS = (
    'excavator', 'forklift', 'crane', 'truck', 'dump', 'bulldozer', 'loader', 'mixer', 'roller', 'backhoe', 'lift', 'drill'
)
FALL_HAZARD_KEYWORDS = ('fall', 'edge', 'ladder', 'scaffold', 'unprotected', 'opening')
FIRE_ELECTRIC_KEYWORDS = ('fire', 'smoke', 'spark', 'flame', 'electric', 'cable', 'wire', 'short')


def _normalize_label(raw: str) -> str:
    return str(raw).strip().lower().replace('-', '_').replace(' ', '_')


def _class_name(names: Any, cls_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(cls_id, cls_id))
    if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
        return str(names[cls_id])
    return str(cls_id)


def _resolve_model_path() -> str | None:
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
        warnings.append('보호구 미착용이 감지되었습니다: ' + ', '.join(readable_items))
        recommended_actions.append('보호구 미착용 작업자를 즉시 확인하고 필요한 보호구를 착용시켜 주세요.')
        risk_score += 3 + len(ppe_missing_items)

    if fall_hazards:
        warnings.append('낙상 위험 요소가 감지되었습니다.')
        recommended_actions.append('고소 작업 구간의 난간, 안전대, 작업 발판 상태를 즉시 점검해 주세요.')
        risk_score += 4

    if fire_or_electric_hazards:
        warnings.append('전기 또는 화재 관련 위험 패턴이 감지되었습니다.')
        recommended_actions.append('전원, 케이블, 점화원 주변을 즉시 점검하고 필요 시 작업을 중지해 주세요.')
        risk_score += 4

    if heavy_equipment:
        if person_count > 0:
            warnings.append('중장비와 작업자가 같은 구역에 함께 감지되었습니다.')
            recommended_actions.append('중장비 주변 접근 금지 구역을 확보하고 유도 인력을 배치해 주세요.')
            risk_score += 3
        else:
            warnings.append('중장비 작업이 감지되었습니다.')
            recommended_actions.append('중장비 이동 동선과 작업 반경을 다시 확인해 주세요.')
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
        warnings.append('탐지된 객체가 없습니다. 카메라 각도와 조명을 확인해 주세요.')
        recommended_actions.append('사진을 더 넓은 각도에서 다시 촬영해 주세요.')

    if not warnings:
        warnings.append('현재 YOLO 클래스 기준으로 즉시 위험요소는 감지되지 않았습니다.')

    if not recommended_actions:
        recommended_actions.append('현장 순찰을 유지하고 보호구 착용 여부를 주기적으로 확인해 주세요.')

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
