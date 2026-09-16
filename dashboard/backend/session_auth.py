"""Session-cookie auth gate for the dashboard backend.

Distinct from auth.py's APIKeyMiddleware (a single shared service-to-service key,
still optional/orthogonal) — this is real per-user login, added because the
dashboard now has signup/login/profile. Every HTTP request needs a valid
`marta_session` cookie except the handful of endpoints that must be reachable
pre-login (signup/login/logout) and anything outside `/api/` (FastAPI's own
/docs, /openapi.json, /redoc).

WebSocket routes bypass BaseHTTPMiddleware entirely (same caveat documented in
auth.py) — they authenticate separately via a `session` query param, checked
inside each `@app.websocket` handler (see main.py).
"""

from __future__ import annotations

from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

import auth_db

# Reachable with no session — everything else under /api/ requires one.
_PUBLIC_API_PATHS = {
    "/api/auth/signup",
    "/api/auth/login",
    "/api/auth/logout",
    "/api/auth/providers",
    "/api/auth/google/login",
    "/api/auth/google/callback",
}


class SessionAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        path = request.url.path
        if not path.startswith("/api/") or path in _PUBLIC_API_PATHS:
            return await call_next(request)

        token = request.cookies.get(auth_db.SESSION_COOKIE_NAME)
        user = auth_db.get_session_user(token) if token else None
        if user is None:
            return JSONResponse(status_code=401, content={"detail": "Not authenticated"})

        request.state.user = user
        return await call_next(request)
