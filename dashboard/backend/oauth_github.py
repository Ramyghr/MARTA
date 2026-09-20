"""GitHub OAuth login. Mirrors oauth_google.py — same session-cookie auth
(auth_db.py), same find-or-create-by-email flow, different provider quirks:
GitHub has no OIDC discovery document (plain OAuth2 authorize/token URLs) and
doesn't always return an email on /user (private-by-default), so a fallback
call to /user/emails is needed.

Requires GITHUB_CLIENT_ID / GITHUB_CLIENT_SECRET in dashboard/backend/.env
(see .env.example) and that OAuth App's "Authorization callback URL" in
https://github.com/settings/developers set to exactly
{backend origin}/api/auth/github/callback — e.g.
http://localhost:8001/api/auth/github/callback for the docker-compose deployment.
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
    name="github",
    client_id=os.environ.get("GITHUB_CLIENT_ID"),
    client_secret=os.environ.get("GITHUB_CLIENT_SECRET"),
    access_token_url="https://github.com/login/oauth/access_token",
    authorize_url="https://github.com/login/oauth/authorize",
    api_base_url="https://api.github.com/",
    client_kwargs={"scope": "read:user user:email"},
)


def github_oauth_enabled() -> bool:
    return bool(os.environ.get("GITHUB_CLIENT_ID")) and bool(os.environ.get("GITHUB_CLIENT_SECRET"))


@router.get("/api/auth/github/login")
async def github_login(request: Request):
    redirect_uri = request.url_for("github_callback")
    return await oauth.github.authorize_redirect(request, redirect_uri)


@router.get("/api/auth/github/callback", name="github_callback")
async def github_callback(request: Request):
    token = await oauth.github.authorize_access_token(request)
    profile = (await oauth.github.get("user", token=token)).json()

    email = profile.get("email")
    if not email:
        # Private-by-default on GitHub — /user omits it; /user/emails has it.
        emails = (await oauth.github.get("user/emails", token=token)).json()
        primary = next((e for e in emails if e.get("primary")), emails[0] if emails else None)
        email = (primary or {}).get("email") or f"{profile['id']}+{profile['login']}@users.noreply.github.com"

    user = auth_db.get_or_create_oauth_user(
        provider="github",
        oauth_id=str(profile["id"]),
        email=email,
        display_name=profile.get("name") or profile.get("login", ""),
        avatar_url=profile.get("avatar_url", ""),
    )
    session_token = auth_db.create_session(user["id"])
    response = RedirectResponse(url=f"{_FRONTEND_URL}/dashboard")
    auth_db.set_session_cookie(response, session_token)
    return response
