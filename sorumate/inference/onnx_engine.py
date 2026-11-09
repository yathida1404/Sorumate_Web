"""Local ONNX detection (Ultralytics YOLO export). Runs on CPU, no external API.

Output formats handled automatically:
  * end-to-end (NMS inside the model), shape (1, N, 6): x1, y1, x2, y2, score, class
  * classic, shape (1, 4+nc, N):  cx, cy, w, h, class scores...  (NMS done here)
Class names are read from the ONNX metadata that Ultralytics embeds at export.
"""
import ast
import logging

import numpy as np
from PIL import Image

from .. import config
from .common import RawDet

log = logging.getLogger('sorumate')


def parse_names(raw) -> list:
    """Ultralytics stores names as a string like "{0: 'Battery', 1: 'Beverage Can'}"."""
    if not raw:
        return []
    try:
        d = ast.literal_eval(raw) if isinstance(raw, str) else raw
    except (ValueError, SyntaxError):
        return []
    if isinstance(d, dict):
        return [str(d[k]) for k in sorted(d, key=int)]
    if isinstance(d, (list, tuple)):
        return [str(x) for x in d]
    return []


def letterbox(image: Image.Image, size):
    """Resize keeping aspect ratio, pad with gray 114 (Ultralytics default).
    size = (height, width). Returns tensor (1,3,H,W) float32, scale, pad_x, pad_y."""
    sh, sw = size
    w, h = image.size
    r = min(sw / w, sh / h)
    nw, nh = max(1, int(round(w * r))), max(1, int(round(h * r)))
    resized = image.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new('RGB', (sw, sh), (114, 114, 114))
    px, py = (sw - nw) // 2, (sh - nh) // 2
    canvas.paste(resized, (px, py))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None]), r, px, py


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list:
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return keep


def agnostic_nms(boxes, scores, cls, iou_thr: float):
    """Drop lower-confidence boxes that overlap a better box, whatever their class.
    (Fixes the same object being boxed twice under two different class names.)"""
    if iou_thr >= 1 or len(boxes) < 2:
        return boxes, scores, cls
    keep = nms(boxes, scores, iou_thr)
    return boxes[keep], scores[keep], cls[keep]


def decode(output, conf_min: float, iou_thr: float):
    """Return (boxes_xyxy_in_input_pixels, scores, class_ids) from either format."""
    a = np.asarray(output, dtype=np.float32)
    if a.ndim == 3:
        a = a[0]
    if a.ndim != 2:
        raise ValueError(f'Unexpected ONNX output shape {output.shape}')
    if a.shape[1] == 6:                      # end-to-end: x1,y1,x2,y2,score,class
        m = a[:, 4] >= conf_min
        a = a[m]
        return a[:, :4], a[:, 4], a[:, 5].astype(int)
    if a.shape[0] < a.shape[1]:              # classic (4+nc, N) -> (N, 4+nc)
        a = a.T
    scores_all = a[:, 4:]
    cls = scores_all.argmax(1)
    conf = scores_all.max(1)
    m = conf >= conf_min
    a, cls, conf = a[m], cls[m], conf[m]
    if not len(a):
        return np.zeros((0, 4), np.float32), conf, cls
    cx, cy, w, h = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    keep = nms(boxes + cls[:, None] * 8192.0, conf, iou_thr)   # class-aware NMS
    return boxes[keep], conf[keep], cls[keep]


class OnnxEngine:
    name = 'onnx'
    error_detail = 'ประมวลผลภาพด้วยโมเดล ONNX ไม่สำเร็จ'

    def __init__(self, path):
        import onnxruntime as ort  # lazy
        so = ort.SessionOptions()
        so.intra_op_num_threads = config.ORT_THREADS
        self.sess = ort.InferenceSession(str(path), sess_options=so, providers=['CPUExecutionProvider'])
        inp = self.sess.get_inputs()[0]
        meta = self.sess.get_modelmeta().custom_metadata_map or {}
        self.input_name = inp.name
        self.size = self._input_size(inp.shape, meta.get('imgsz'))
        self.names = parse_names(meta.get('names'))
        if not self.names:
            from ..catalog import catalog
            self.names = list(catalog().get('model_classes', []))
            log.warning('ONNX has no class names in metadata; using catalog order (verify with scripts/check_classes.py)')
        self._warmup()

    @staticmethod
    def _input_size(shape, imgsz_meta):
        h, w = shape[2], shape[3]
        if isinstance(h, int) and isinstance(w, int):
            return h, w
        try:
            v = ast.literal_eval(imgsz_meta) if imgsz_meta else 640
            return (v[0], v[1]) if isinstance(v, (list, tuple)) else (int(v), int(v))
        except (ValueError, SyntaxError):
            return 640, 640

    def _warmup(self):
        try:
            self.detect(Image.new('RGB', (64, 64), (128, 128, 128)))
        except Exception:
            log.exception('ONNX warm-up failed')
            raise

    def detect(self, image):
        W, H = image.size
        tensor, r, px, py = letterbox(image, self.size)
        out = self.sess.run(None, {self.input_name: tensor})[0]
        boxes, scores, cls = decode(out, conf_min=config.THRESHOLD, iou_thr=config.NMS_IOU)
        boxes, scores, cls = agnostic_nms(boxes, scores, cls, config.AGNOSTIC_NMS_IOU)
        dets = []
        for (x1, y1, x2, y2), s, c in zip(boxes, scores, cls):
            name = self.names[c] if 0 <= c < len(self.names) else f'class_{c}'
            dets.append(RawDet(name, float(s),
                               float((x1 - px) / r / W), float((y1 - py) / r / H),
                               float((x2 - px) / r / W), float((y2 - py) / r / H)))
        return dets
