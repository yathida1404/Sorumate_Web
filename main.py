"""SoruMate entry point (kept at repo root so `uvicorn main:app` keeps working).

All logic lives in the `sorumate/` package:
  routes/    thin HTTP layer      catalog.py    rules from data/*.json
  inference/ ONNX or Roboflow     config.py     environment settings
Images are handled in memory only. Do not put secrets in static files.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from sorumate import config, inference
from sorumate.errors import ApiError
from sorumate.routes import pages, predict, rules

log = logging.getLogger('sorumate')


@asynccontextmanager
async def lifespan(app: FastAPI):
    inference.get_engine()          # load the model once at startup, not on the first user's request
    st = inference.status()
    log.warning('SoruMate engine: %s', st)
    yield


app = FastAPI(title='SoruMate – Ebetsu Waste Sorting', version='0.2.0', lifespan=lifespan)
app.mount('/static', StaticFiles(directory=config.STATIC_DIR), name='static')


@app.exception_handler(ApiError)
async def api_error_handler(_, exc: ApiError):
    return JSONResponse({'detail': exc.detail}, status_code=exc.status)


app.include_router(pages.router)
app.include_router(predict.router)
app.include_router(rules.router)
