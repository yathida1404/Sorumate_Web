import logging

from fastapi import APIRouter, File, Header, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from .. import config, inference
from ..errors import ApiError
from ..imaging import load_image
from ..inference.common import build_response
from ..security import check_access_code, rate_limit

log = logging.getLogger('sorumate')
router = APIRouter()


def _detect(engine, image):
    return build_response(engine.detect(image), engine.name)


# /predict is what the frontend calls; /api/predict is kept as an alias.
@router.post('/api/predict')
@router.post('/predict', include_in_schema=False)
async def predict(request: Request, file: UploadFile = File(...), x_demo_code: str | None = Header(default=None)):
    engine = inference.get_engine()
    if engine is None:
        raise ApiError(503, 'ยังไม่เชื่อมต่อโมเดล AI กรุณาวางไฟล์ models/best.onnx หรือตั้งค่า Roboflow บน Render')
    check_access_code(x_demo_code)
    rate_limit(request.client.host if request.client else 'unknown')
    data = await file.read(config.MAX_BYTES + 1)
    await file.close()
    image = load_image(data)
    try:
        return await run_in_threadpool(_detect, engine, image)
    except Exception:
        log.exception('%s inference failed', engine.name)
        raise ApiError(503, engine.error_detail)
