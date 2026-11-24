# Signing in and out with the company's OpenID Connect provider: the login redirect and its
# callback, the logout and the provider's back-channel logout, and the signed-in employee's /me.

import hmac
import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Form, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from starlette.concurrency import run_in_threadpool

from deskclerk import cookies, oidc
from deskclerk.auth import AuthError, AuthUnavailable, identify, logout_target
from deskclerk.routes.services import (
    SESSION_COOKIE,
    Caller,
    Services,
    clip,
    link_failure,
)
from deskclerk.spending import month_of

LOGIN_COOKIE = "deskclerk_login"
LOGIN_LIFETIME_SECONDS = 600

_log = logging.getLogger(__name__)


def _sign_in_failed(code: str, reason: str) -> RedirectResponse:
    _log.warning("sign-in failed: %s", reason)
    return link_failure(code)


def router(services: Services) -> APIRouter:
    routes = APIRouter()
    settings = services.settings

    @routes.get("/login")
    async def login() -> Response:
        try:
            found = await services.endpoints()
        except AuthUnavailable as unreachable:
            return _sign_in_failed("login_unreachable", str(unreachable))
        begun = oidc.start(settings, found)
        pending = {"state": begun.state, "verifier": begun.verifier, "nonce": begun.nonce}
        value = cookies.sign(services.session_key, "login", pending, LOGIN_LIFETIME_SECONDS)
        response = RedirectResponse(begun.url, status_code=302)
        response.set_cookie(
            LOGIN_COOKIE,
            value,
            LOGIN_LIFETIME_SECONDS,
            path="/login",
            httponly=True,
            samesite="lax",
            secure=services.secure_cookies,
        )
        return response

    @routes.get("/login/callback")
    async def login_callback(
        request: Request,
        code: str | None = None,
        state: str | None = None,
        error: str | None = None,
        error_description: str | None = None,
    ) -> Response:
        if error is not None:
            reason = f"the login server returned error={clip(error)} "
            return _sign_in_failed(
                "sign_in_refused", reason + f"error_description={clip(error_description)}"
            )
        pending = cookies.unsign(
            services.session_key, "login", request.cookies.get(LOGIN_COOKIE, "")
        )
        if pending is None:
            return _sign_in_failed(
                "sign_in_refused", "the login cookie is missing, expired or forged"
            )
        if code is None or state is None:
            return _sign_in_failed("sign_in_refused", "the callback carries no code or no state")
        if not hmac.compare_digest(str(pending.get("state", "")).encode(), state.encode()):
            return _sign_in_failed(
                "sign_in_refused", "the callback state does not match the login cookie"
            )
        try:
            found = await services.endpoints()
        except AuthUnavailable as unreachable:
            return _sign_in_failed("login_unreachable", str(unreachable))
        async with services.idp_client() as client:
            try:
                verifier = str(pending.get("verifier", ""))
                token = await oidc.exchange_code(settings, found, client, code, verifier)
            except oidc.SignInRefused as refused:
                return _sign_in_failed("sign_in_refused", str(refused))
            except oidc.NoIdToken as missing:
                return _sign_in_failed("sign_in_refused", str(missing))
            except AuthUnavailable as unreachable:
                return _sign_in_failed("login_unreachable", str(unreachable))
        try:
            identity = await run_in_threadpool(
                identify, token, settings, services.signing_key, str(pending.get("nonce", ""))
            )
        except AuthError as refused:
            return _sign_in_failed("sign_in_refused", f"the ID token was refused: {refused}")
        except AuthUnavailable as unreachable:
            return _sign_in_failed("login_unreachable", str(unreachable))
        await run_in_threadpool(services.employees.record, identity)
        session = await run_in_threadpool(
            services.sessions.create, identity, services.session_seconds
        )
        response = RedirectResponse("/", status_code=303)
        response.delete_cookie(LOGIN_COOKIE, path="/login")
        response.set_cookie(
            SESSION_COOKIE,
            session,
            services.session_seconds,
            path="/",
            httponly=True,
            samesite="lax",
            secure=services.secure_cookies,
        )
        return response

    @routes.post("/logout", status_code=204)
    async def logout(request: Request, _: Caller = Depends(services.posting_caller)) -> Response:
        await run_in_threadpool(services.sessions.end, request.cookies.get(SESSION_COOKIE, ""))
        response = Response(status_code=204)
        response.delete_cookie(SESSION_COOKIE, path="/")
        return response

    @routes.post("/logout/backchannel")
    async def backchannel_logout(logout_token: str = Form()) -> Response:
        try:
            target = await run_in_threadpool(
                logout_target, logout_token, settings, services.signing_key
            )
        except AuthError:
            raise HTTPException(400, "the logout token was refused") from None
        except AuthUnavailable:
            raise HTTPException(503, "the login server cannot be reached") from None
        if target.sid is not None:
            await run_in_threadpool(services.sessions.end_by_sid, target.sid)
        elif target.sub is not None:
            await run_in_threadpool(services.sessions.end_by_subject, target.sub)
        return Response(status_code=200, headers={"Cache-Control": "no-store"})

    @routes.get("/me")
    async def me(who: Caller = Depends(services.caller)) -> dict[str, str | float | bool]:
        month = month_of(datetime.now(UTC))
        spent = await run_in_threadpool(services.spending_store.load, who.identity.user_id, month)
        return {
            "user": who.identity.user_id,
            "name": who.identity.display_name,
            "spent_usd": spent,
            "budget_usd": settings.budget_usd,
            "admin": services.is_admin(who.identity),
            "unscreened": services.jev is None,
        }

    return routes
