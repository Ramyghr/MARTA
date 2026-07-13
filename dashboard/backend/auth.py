"""API key authentication for the MARTA dashboard backend.

Set MARTA_API_KEY in the environment to enable authentication.
When the variable is unset (default) all requests are allowed, which is safe
for localhost-only deployment. For any non-localhost deployment, export a
random secret before starting the server:

    export MARTA_API_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"
    uvicorn main:app --host 0.0.0.0

Clients must then pass the header:   X-API-Key: <secret>
"""

from __future__ import annotations

import os
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

_CONFIGURED_KEY: str | None = os.environ.get("MARTA_API_KEY")


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Starlette middleware — rejects HTTP requests that omit/mismatch the key.

    WebSocket upgrade requests are passed through unconditionally; the WS
    handshake begins as an HTTP GET but BaseHTTPMiddleware does not wrap it,
    so WebSocket connections bypass this check entirely (correct for a
    localhost-only dashboard).
    """

    async def dispatch(self, request: Request, call_next: Callable):
        if _CONFIGURED_KEY:
            key = request.headers.get("X-API-Key") or request.headers.get("x-api-key")
            if key != _CONFIGURED_KEY:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing X-API-Key header"},
                    headers={"WWW-Authenticate": "ApiKey"},
                )
        return await call_next(request)


def is_auth_enabled() -> bool:
    return bool(_CONFIGURED_KEY)
