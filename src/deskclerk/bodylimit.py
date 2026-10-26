# A pure ASGI middleware that answers 413 to a request whose body is over a byte cap, judged by
# Content-Length first and by the bytes actually received for chunked bodies, before any route or
# sign-in check reads the body.

from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# The largest legal body is a SCIM group PATCH or PUT from the identity provider. At about 100
# bytes per member, 256 KiB holds some 2,500 members. An ask is 4,000 characters, at most 24 KB
# as JSON with every character escaped, and a confirm payload is far smaller.
MAX_BODY_BYTES = 256 * 1024


class BodyTooLarge(Exception):
    pass


class LimitRequestSize:
    def __init__(self, app: ASGIApp, max_bytes: int = MAX_BODY_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        refusal = PlainTextResponse("request body too large", status_code=413)
        declared = dict(scope["headers"]).get(b"content-length", b"")
        if declared.isdigit() and int(declared) > self.max_bytes:
            await refusal(scope, receive, send)
            return
        received = 0
        started = False

        async def counted_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise BodyTooLarge
            return message

        async def tracked_send(message: Message) -> None:
            nonlocal started
            started = started or message["type"] == "http.response.start"
            await send(message)

        try:
            await self.app(scope, counted_receive, tracked_send)
        except BodyTooLarge:
            if started:
                raise
            await refusal(scope, receive, send)
