# The request size cap answers 413 for a body over the cap, declared or streamed, and lets a
# smaller one through.

from collections.abc import Iterator

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from deskclerk.bodylimit import MAX_BODY_BYTES, LimitRequestSize


def make_client() -> TestClient:
    app = FastAPI()
    app.add_middleware(LimitRequestSize)

    @app.post("/echo")
    async def echo(request: Request) -> dict[str, int]:
        return {"bytes": len(await request.body())}

    return TestClient(app)


def test_a_body_at_the_cap_passes() -> None:
    response = make_client().post("/echo", content=b"x" * MAX_BODY_BYTES)
    assert (response.status_code, response.json()) == (200, {"bytes": MAX_BODY_BYTES})


def test_a_declared_body_over_the_cap_is_refused() -> None:
    response = make_client().post("/echo", content=b"x" * (MAX_BODY_BYTES + 1))
    assert response.status_code == 413


def test_a_streamed_body_over_the_cap_is_refused() -> None:
    def chunks() -> Iterator[bytes]:
        for _ in range(3):
            yield b"x" * (MAX_BODY_BYTES // 2)

    response = make_client().post("/echo", content=chunks())
    assert response.status_code == 413
