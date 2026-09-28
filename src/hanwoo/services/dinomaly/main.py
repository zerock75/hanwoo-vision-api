from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from hanwoo.core.auth import APIKeyAuthMiddleware, get_required_api_key
from hanwoo.core.config import DEVICE
from hanwoo.services.dinomaly.pipeline import DinomalyService
from hanwoo.services.dinomaly.routes import router, set_dinomaly_service

import hanwoo.services.dinomaly.db as db
import aiomysql


# uvicorn only configures its own loggers, so without this the app's records
# reach the root logger's lastResort handler and anything below WARNING is dropped.
logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(message)s")

dinomaly_service = DinomalyService(device_name=DEVICE)
set_dinomaly_service(dinomaly_service)
logger = logging.getLogger(__name__)


async def create_db_pool():
    db.pool = await aiomysql.create_pool(
        host="192.168.0.233",
        port=6306,
        user="dev-igogi",
        password="igogi",
        db="igogi_dev",
        autocommit=True,
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_required_api_key()
    await create_db_pool()
    try:
        dinomaly_service.load()
    except Exception as exc:
        logger.warning("Dinomaly service not loaded: %s", exc)
    yield


app = FastAPI(
    title="Hanwoo Dinomaly API",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(APIKeyAuthMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
