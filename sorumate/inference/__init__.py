"""Chooses the detection engine once per process (see SORUMATE_ENGINE in config.py)."""
import logging
from functools import lru_cache

from .. import config

log = logging.getLogger('sorumate')


@lru_cache(maxsize=1)
def _build():
    """Returns (engine_or_None, error_text_or_None)."""
    choice = config.ENGINE
    onnx_file = config.ONNX_MODEL_PATH
    if choice == 'onnx' and not onnx_file.is_file():
        return None, f'ไม่พบไฟล์โมเดล ONNX: {onnx_file}'
    if choice in ('onnx', 'auto') and onnx_file.is_file():
        try:
            from .onnx_engine import OnnxEngine
            return OnnxEngine(onnx_file), None
        except Exception as e:  # visible in /health instead of silently falling back
            log.exception('Failed to load ONNX model')
            return None, f'โหลดโมเดล ONNX ไม่สำเร็จ: {e}'
    if choice in ('roboflow', 'auto') and config.ROBOFLOW_MODEL_ID and config.ROBOFLOW_API_KEY:
        try:
            from .roboflow_engine import RoboflowEngine
            return RoboflowEngine(config.ROBOFLOW_API_KEY, config.ROBOFLOW_MODEL_ID, config.ROBOFLOW_SERVER_URL), None
        except Exception as e:
            log.exception('Failed to start Roboflow engine')
            return None, f'เริ่มระบบ Roboflow ไม่สำเร็จ (ตรวจว่าติดตั้ง inference-sdk แล้ว): {e}'
    return None, None


def get_engine():
    return _build()[0]


def engine_error():
    return _build()[1]


def status() -> dict:
    engine, err = _build()
    info = {'ai_ready': engine is not None,
            'engine': engine.name if engine else None,
            'mode': 'real' if engine else 'not_configured'}
    if err:
        info['engine_error'] = err
    if engine is not None and engine.name == 'onnx':
        from ..catalog import by_name, normal
        info['model_classes'] = len(engine.names)
        info['unmapped_classes'] = [n for n in engine.names if normal(n) not in by_name()]
    return info
