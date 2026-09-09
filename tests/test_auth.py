from __future__ import annotations

import asyncio
import json

import pytest
from starlette.types import Message

import hanwoo.core.auth as auth
from hanwoo.core.auth import APIKeyAuthMiddleware


async def ok_app(scope, receive, send) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send({"type": "http.response.body", "body": b'{"ok":true}'})


def call_auth_app(
    path: str,
    headers: list[tuple[bytes, bytes]] | None = None,
) -> tuple[int, dict]:
    messages: list[Message] = []
    app = APIKeyAuthMiddleware(ok_app)

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        messages.append(message)

    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "headers": headers or [],
    }
    asyncio.run(app(scope, receive, send))

    status = next(
        message["status"]
        for message in messages
        if message["type"] == "http.response.start"
    )
    body = b"".join(
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
    )
    return status, json.loads(body)


def test_api_key_config_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth, "HANWOO_API_KEY", None)

    with pytest.raises(RuntimeError, match="HANWOO_API_KEY must be set"):
        auth.get_required_api_key()


def test_health_is_public_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth, "HANWOO_API_KEY", "secret")

    status, body = call_auth_app("/health")

    assert status == 200
    assert body == {"ok": True}


def test_validator_is_public_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth, "HANWOO_API_KEY", "secret")

    status, body = call_auth_app("/validator")

    assert status == 200
    assert body == {"ok": True}


def test_validator_slash_is_public_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(auth, "HANWOO_API_KEY", "secret")

    status, body = call_auth_app("/validator/")

    assert status == 200
    assert body == {"ok": True}


def test_protected_endpoint_rejects_missing_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(auth, "HANWOO_API_KEY", "secret")

    status, body = call_auth_app("/metadata")

    assert status == 401
    assert body == {"detail": "Invalid or missing API key"}


def test_protected_endpoint_rejects_wrong_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(auth, "HANWOO_API_KEY", "secret")

    status, body = call_auth_app("/metadata", headers=[(b"x-api-key", b"wrong")])

    assert status == 401
    assert body == {"detail": "Invalid or missing API key"}


def test_protected_endpoint_accepts_valid_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(auth, "HANWOO_API_KEY", "secret")

    status, body = call_auth_app("/metadata", headers=[(b"x-api-key", b"secret")])

    assert status == 200
    assert body == {"ok": True}
