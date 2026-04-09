import json
from contextlib import asynccontextmanager
from typing import AsyncIterator, Mapping, Union

from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from config import AppConfig, LlamaConfig
from llama_proxy import LlamaProxy


HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


def _filter_request_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS | {"host"}
    }


def _filter_response_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }


def _accepts_event_stream(accept_header: str) -> bool:
    media_ranges = [part.strip() for part in accept_header.split(",") if part.strip()]
    return any(
        range_part.split(";", 1)[0].strip() == "text/event-stream"
        for range_part in media_ranges
    )


def _body_requests_stream(content_type: str, content: bytes) -> bool:
    if not content or not content_type.lower().startswith("application/json"):
        return False

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return False

    return isinstance(payload, dict) and payload.get("stream") is True


def _is_streaming_request(request: Request, content: bytes) -> bool:
    accept_header = request.headers.get("accept", "")
    content_type = request.headers.get("content-type", "")
    return _accepts_event_stream(accept_header) or _body_requests_stream(
        content_type, content
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.llama = LlamaProxy(LlamaConfig.from_env(), AppConfig.from_env())
    await app.state.llama.start()
    yield
    await app.state.llama.close()
    await app.state.llama.stop()


app = FastAPI(lifespan=lifespan)


@app.get("/ping", response_model=None)
async def health_check() -> Union[dict, Response]:
    llama = app.state.llama
    if await llama.health_check():
        return {"status": "healthy"}
    return Response(
        content='{"status": "unhealthy"}',
        status_code=503,
        media_type="application/json",
    )


@app.api_route(
    "/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
)
async def proxy(path: str, request: Request) -> Response:
    llama = app.state.llama

    headers = _filter_request_headers(request.headers)
    content = await request.body()

    is_streaming = _is_streaming_request(request, content)

    if is_streaming:
        response = await llama.proxy_stream_response(
            method=request.method,
            path=f"/v1/{path}",
            headers=headers,
            content=content if content else None,
        )
        return StreamingResponse(
            response.aiter_raw(),
            status_code=response.status_code,
            headers=_filter_response_headers(response.headers),
            media_type=response.headers.get("content-type"),
            background=BackgroundTask(response.aclose),
        )

    response = await llama.proxy_request(
        method=request.method,
        path=f"/v1/{path}",
        headers=headers,
        content=content if content else None,
    )
    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=_filter_response_headers(response.headers),
        media_type=response.headers.get("content-type"),
    )


if __name__ == "__main__":
    import uvicorn

    config = AppConfig.from_env()
    uvicorn.run(app, host="0.0.0.0", port=config.port)
