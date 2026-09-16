"""Google OAuth login. Plugs into the existing session-cookie auth (auth_db.py)
rather than introducing a separate mechanism — the callback below ends the same
way password login does: auth_db.create_session() + the marta_session cookie.

Requires GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET in dashboard/backend/.env (see
.env.example) and that client's "Authorized redirect URI" in the Google Cloud
Console set to exactly {backend origin}/api/auth/google/callback — e.g.
http://localhost:8001/api/auth/google/callback for the docker-compose deployment.
"""

from __future__ import annotations

import os

from authlib.integrations.starlette_client import OAuth
from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse

import auth_db

router = APIRouter()

_FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:3001")

oauth = OAuth()
oauth.register(
    name="google",
    client_id=os.environ.get("GOOGLE_CLIENT_ID"),
    client_secret=os.environ.get("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)


def google_oauth_enabled() -> bool:
    return bool(os.environ.get("GOOGLE_CLIENT_ID")) and bool(os.environ.get("GOOGLE_CLIENT_SECRET"))


@router.get("/api/auth/google/login")
async def google_login(request: Request):
    redirect_uri = request.url_for("google_callback")
    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get("/api/auth/google/callback", name="google_callback")
async def google_callback(request: Request):
    token = await oauth.google.authorize_access_token(request)
    userinfo = token.get("userinfo") or await oauth.google.userinfo(token=token)
    user = auth_db.get_or_create_oauth_user(
        provider="google",
        oauth_id=userinfo["sub"],
        email=userinfo["email"],
        display_name=userinfo.get("name", ""),
        avatar_url=userinfo.get("picture", ""),
    )
    session_token = auth_db.create_session(user["id"])
    response = RedirectResponse(url=f"{_FRONTEND_URL}/dashboard")
    auth_db.set_session_cookie(response, session_token)
    return response
