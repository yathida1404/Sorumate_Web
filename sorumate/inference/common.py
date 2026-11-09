"""Engine-independent part of detection: turns raw engine output into the JSON
the frontend expects (same shape as the original /predict)."""
import re
from typing import NamedTuple

from .. import config
from ..catalog import by_name, normal


class RawDet(NamedTuple):
    """One detection from any engine. Box corners are NORMALIZED 0-1 relative to
    the ORIGINAL uploaded image."""
    name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


# Legacy aliases kept from the original main.py (used only if a class is not in the catalog).
LEGACY_LABELS = {
    'pet plastic bottle': ('pet_plastic_bottle', 'ขวดพลาสติก PET', 'PETボトル', 'PET plastic bottle'),
    'pet bottle': ('pet_plastic_bottle', 'ขวดพลาสติก PET', 'PETボトル', 'PET plastic bottle'),
    'pet bottle cap': ('pet_bottle_cap', 'ฝาขวด PET', 'ペットボトルのキャップ', 'PET bottle cap'),
    'aluminum can': ('aluminum_can', 'กระป๋องอะลูมิเนียม', 'アルミ缶', 'Aluminum can'),
    'aluminium can': ('aluminum_can', 'กระป๋องอะลูมิเนียม', 'アルミ缶', 'Aluminum can'),
}


def resolve_label(cls: str):
    cls = str(cls or 'Unknown').strip() or 'Unknown'
    item = by_name().get(normal(cls))
    if item:
        return item['id'], item['th'], item['ja'], item['en']
    key = re.sub(r'[_-]+', ' ', cls).casefold()
    return LEGACY_LABELS.get(key, (re.sub(r'[^a-z0-9]+', '_', cls.lower()).strip('_') or 'unknown', cls, cls, cls))


def _clamp(v: float) -> float:
    return max(0., min(1., v))


def build_response(raw, engine_name: str, threshold: float = None) -> dict:
    threshold = config.THRESHOLD if threshold is None else threshold
    detections = []
    for d in raw:
        if d.confidence < threshold:
            continue
        label, th, ja, en = resolve_label(d.name)
        x1, y1, x2, y2 = _clamp(d.x1), _clamp(d.y1), _clamp(d.x2), _clamp(d.y2)
        detections.append({
            'label': label, 'label_th': th, 'label_ja': ja, 'label_en': en,
            'model_label': d.name, 'confidence': round(float(d.confidence), 4),
            'bbox': [round(x1, 5), round(y1, 5), round(max(0., x2 - x1), 5), round(max(0., y2 - y1), 5)],
        })
    return {'mode': 'real', 'engine': engine_name,
            'detections': sorted(detections, key=lambda x: x['confidence'], reverse=True),
            'confidence_threshold': threshold}
