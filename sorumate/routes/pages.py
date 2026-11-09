from fastapi import APIRouter, Request, Response
from fastapi.responses import FileResponse

from .. import config, inference

router = APIRouter()


@router.get('/')
def index():
    return FileResponse(config.STATIC_DIR / 'index.html', headers={'Cache-Control': 'no-store, max-age=0'})


@router.post('/event', status_code=204, include_in_schema=False)
async def accept_legacy_event(request: Request):
    """Compatibility with the frontend's best-effort anonymous events. Nothing is stored."""
    await request.body()
    return Response(status_code=204)


@router.get('/health')
def health():
    return {
        **inference.status(),
        'model_id': config.ROBOFLOW_MODEL_ID if config.ROBOFLOW_MODEL_ID and config.ROBOFLOW_API_KEY else None,
        'requires_access_code': bool(config.DEMO_ACCESS_CODE),
        'ui_release': config.UI_RELEASE,
    }
