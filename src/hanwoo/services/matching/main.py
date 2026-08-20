from __future__ import annotations

from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from hanwoo.core.auth import APIKeyAuthMiddleware, get_required_api_key
from hanwoo.core.config import DEVICE
from hanwoo.services.matching.pipeline import MatchingService
from hanwoo.services.matching.routes import router, set_matching_service


matching_service = MatchingService(device_name=DEVICE)
set_matching_service(matching_service)
VALIDATOR_URL = os.getenv("VALIDATOR_URL", "http://localhost:8501/validator")


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_required_api_key()
    matching_service.load()
    yield


app = FastAPI(
    title="Hanwoo Matching API",
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


@app.get("/validator", include_in_schema=False)
def validator_ui():
    return RedirectResponse(VALIDATOR_URL)
