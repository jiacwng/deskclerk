# The routes that need no sign-in: liveness, readiness, the metrics scrape, the chat page and the
# vendored scripts it loads, plus the content security policy that pins the page to those files.

import base64
import hashlib
import re
from importlib import resources

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST
from starlette.concurrency import run_in_threadpool

from deskclerk.auth import AuthUnavailable
from deskclerk.routes.services import Caller, Services

VENDOR_SCRIPTS = ("marked-18.1.0.umd.js", "purify-3.4.16.min.js")
SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


def _sha256_source(data: bytes) -> str:
    return "sha256-" + base64.b64encode(hashlib.sha256(data).digest()).decode()


# The page has one inline script, one style block and two vendored scripts, so the policy allows
# exactly those by hash and nothing else. Each vendored script carries an integrity attribute that
# must equal the hash of the file the app serves, which browsers need to match a hash to a file.
def content_security_policy(page: str, vendor: dict[str, bytes]) -> str:
    blocks = {}
    for tag in ("script", "style"):
        match = re.search(rf"<{tag}>(.*?)</{tag}>", page, re.DOTALL)
        if match is None:
            raise ValueError(f"index.html has no <{tag}> block")
        blocks[tag] = [_sha256_source(match.group(1).encode())]
    for name, data in vendor.items():
        source = _sha256_source(data)
        if f'<script src="/vendor/{name}" integrity="{source}"></script>' not in page:
            raise ValueError(f"index.html does not load vendor/{name} with integrity {source}")
        blocks["script"].append(source)
    fixed = ["default-src 'none'", "connect-src 'self'", "base-uri 'none'"]
    fixed += ["form-action 'self'", "frame-ancestors 'none'"]
    allowed = [
        f"{tag}-src " + " ".join(f"'{source}'" for source in found) for tag, found in blocks.items()
    ]
    return "; ".join([*fixed, *allowed])


def page_headers(page: str, vendor: dict[str, bytes]) -> dict[str, str]:
    return {"Content-Security-Policy": content_security_policy(page, vendor), **SECURITY_HEADERS}


def router(services: Services) -> APIRouter:
    routes = APIRouter()
    page = resources.files("deskclerk").joinpath("static/index.html").read_text(encoding="utf-8")
    vendor = {
        name: resources.files("deskclerk").joinpath(f"static/vendor/{name}").read_bytes()
        for name in VENDOR_SCRIPTS
    }
    headers = page_headers(page, vendor)

    @routes.get("/livez")
    async def livez() -> dict[str, str]:
        return {"status": "ok"}

    @routes.get("/metrics")
    async def prometheus_metrics(_: Caller = Depends(services.admin_caller)) -> Response:
        return Response(services.metrics.render(), media_type=CONTENT_TYPE_LATEST)

    @routes.get("/readyz")
    async def readyz(request: Request) -> JSONResponse:
        failed = []
        try:
            await run_in_threadpool(services.tokens.check)
        except psycopg.Error:
            failed.append("token store")
        if not hasattr(request.app.state, "checkpointer"):
            failed.append("checkpointer")
        try:
            await services.endpoints()
        except AuthUnavailable:
            failed.append("login server discovery")
        if failed:
            return JSONResponse({"not_ready": failed}, status_code=503)
        return JSONResponse({"status": "ready"})

    @routes.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(page, headers=headers)

    @routes.get("/vendor/{name}")
    async def vendor_script(name: str) -> Response:
        script = vendor.get(name)
        if script is None:
            raise HTTPException(404, f"no script named {name}")
        script_headers = {
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "public, max-age=31536000, immutable",
        }
        return Response(script, media_type="text/javascript", headers=script_headers)

    return routes
