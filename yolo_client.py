from __future__ import annotations

import glob
import os
import tempfile
from pathlib import Path
from typing import Any

_MODEL = None


def _resolve_model_path() -> str | None:
    explicit_path = (os.getenv('YOLO_MODEL_PATH') or '').strip()
    if explicit_path:
        candidate = Path(explicit_path).expanduser()
        if candidate.exists():
            return str(candidate)

    raw_globs = (os.getenv('YOLO_MODEL_GLOBS') or '').strip()
    patterns = [pattern.strip() for pattern in raw_globs.split(';') if pattern.strip()]
    if not patterns:
        return None

    candidates: list[str] = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern, recursive=True))

    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _torch_device() -> str:
    try:
        import torch

        if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            return 'mps'
        if torch.cuda.is_available():
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


def analyze_with_yolo(image_bytes: bytes, ext: str) -> dict[str, Any]:
    model = _get_model()
    if model is None:
        return {'available': False, 'detections': [], 'summary': 'YOLO model unavailable', 'risk_detected': False}

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
                if isinstance(names, dict):
                    class_name = names.get(class_id, str(class_id))
                elif isinstance(names, list) and 0 <= class_id < len(names):
                    class_name = names[class_id]
                else:
                    class_name = str(class_id)

                detections.append(
                    {
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': round(float(box.conf.item()), 3),
                        'xyxy': [round(float(value), 1) for value in box.xyxy[0].tolist()],
                    }
                )

        risk_classes = {'fall_down', 'no_vest', 'no_helmet', 'fall', 'without_helmet', 'without_vest'}
        risk_detected = any(detection['class_name'] in risk_classes for detection in detections)

        return {
            'available': True,
            'detections': detections,
            'count': len(detections),
            'risk_detected': risk_detected,
            'summary': _build_summary(detections),
        }
    except Exception as error:
        return {'available': False, 'detections': [], 'summary': f'YOLO error: {error}', 'risk_detected': False}
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def _build_summary(detections: list[dict[str, Any]]) -> str:
    if not detections:
        return 'No objects detected.'

    counter: dict[str, int] = {}
    for detection in detections:
        class_name = detection['class_name']
        counter[class_name] = counter.get(class_name, 0) + 1

    return 'Detected: ' + ', '.join(f'{count}x {class_name}' for class_name, count in counter.items())
