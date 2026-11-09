"""Roboflow hosted inference. Needs `inference-sdk` (NOT in requirements.txt by default)."""
from .common import RawDet


class RoboflowEngine:
    name = 'roboflow'
    error_detail = 'เชื่อมต่อ AI ไม่สำเร็จ โปรดตรวจ MODEL_ID, API Key และสถานะโมเดลบน Roboflow'

    def __init__(self, api_key: str, model_id: str, server_url: str):
        from inference_sdk import InferenceHTTPClient  # lazy: only if this engine is used
        self.model_id = model_id
        self.client = InferenceHTTPClient(api_url=server_url, api_key=api_key)

    def detect(self, image):
        raw = self.client.infer(image, model_id=self.model_id)
        meta = raw.get('image') or {}
        width = float(meta.get('width') or image.width)
        height = float(meta.get('height') or image.height)
        if width <= 0 or height <= 0:
            raise ValueError('Invalid inference dimensions')
        out = []
        for p in raw.get('predictions', []):
            xc, yc = float(p.get('x', 0)), float(p.get('y', 0))
            bw, bh = float(p.get('width', 0)), float(p.get('height', 0))
            out.append(RawDet(str(p.get('class') or 'Unknown').strip(), float(p.get('confidence', 0)),
                              (xc - bw / 2) / width, (yc - bh / 2) / height,
                              (xc + bw / 2) / width, (yc + bh / 2) / height))
        return out
