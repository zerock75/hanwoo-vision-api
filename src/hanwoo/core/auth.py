from __future__ import annotations

import secrets

from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from hanwoo.core.config import HANWOO_API_KEY


API_KEY_HEADER = "X-API-Key"
PUBLIC_PATHS = {"/health", "/docs", "/redoc", "/openapi.json", "/validator", "/validator/"}


def get_required_api_key() -> str:
    if not HANWOO_API_KEY:
        raise RuntimeError("HANWOO_API_KEY must be set")
    return HANWOO_API_KEY


def verify_api_key(api_key: str | None) -> None:
    expected = get_required_api_key()
    if api_key is None or not secrets.compare_digest(api_key, expected):
        raise ValueError("Invalid or missing API key")


class APIKeyAuthMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] in PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return
        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope["headers"]
        }
        try:
            verify_api_key(headers.get(API_KEY_HEADER.lower()))
        except ValueError:
            response = JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)
