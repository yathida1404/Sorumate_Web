"""All settings in one place. Everything is read from environment variables
(Render > Environment). Never put secrets in static/ files."""
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE / 'static'
DATA_DIR = BASE / 'data'


def _s(name: str, default: str = '') -> str:
    return os.environ.get(name, default).strip()


def _num(name: str, default, cast):
    try:
        return cast(_s(name) or default)
    except ValueError:
        return default


# ---- Which AI engine runs detection -------------------------------------
# auto     : use ONNX if models/best.onnx exists, else Roboflow if keys are set
# onnx     : local ONNX model only (models/best.onnx or ONNX_MODEL_PATH)
# roboflow : Roboflow hosted inference only
ENGINE = _s('SORUMATE_ENGINE', 'auto').lower()
ONNX_MODEL_PATH = Path(_s('ONNX_MODEL_PATH') or (BASE / 'models' / 'best.onnx'))
ORT_THREADS = _num('ORT_THREADS', 1, int)
NMS_IOU = _num('NMS_IOU', 0.7, float)
# Boxes of DIFFERENT classes overlapping more than this are treated as one object (keep the most confident).
# Set to 1 to turn off.
AGNOSTIC_NMS_IOU = _num('AGNOSTIC_NMS_IOU', 0.6, float)

ROBOFLOW_MODEL_ID = _s('ROBOFLOW_MODEL_ID')
ROBOFLOW_API_KEY = _s('ROBOFLOW_API_KEY')
ROBOFLOW_SERVER_URL = _s('ROBOFLOW_SERVER_URL', 'https://serverless.roboflow.com').rstrip('/')

# ---- Detection / upload limits ------------------------------------------
THRESHOLD = _num('CONFIDENCE_THRESHOLD', 0.30, float)
MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000

# ---- Access control / rate limit ----------------------------------------
DEMO_ACCESS_CODE = _s('DEMO_ACCESS_CODE')
RATE_LIMIT = _num('RATE_LIMIT_PER_10MIN', 60, int)   # per IP (original value was 20)
RATE_WINDOW_SECONDS = 600

UI_RELEASE = '2026-11-06-r1'
