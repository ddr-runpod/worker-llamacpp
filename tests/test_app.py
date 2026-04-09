import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def mock_llama_proxy():
    with patch("app.LlamaProxy") as mock:
        proxy_instance = MagicMock()
        proxy_instance.start = AsyncMock()
        proxy_instance.stop = AsyncMock()
        proxy_instance.close = AsyncMock()
        proxy_instance.health_check = AsyncMock(return_value=True)
        proxy_instance.proxy_request = AsyncMock()
        proxy_instance.proxy_stream_response = AsyncMock()
        mock.return_value = proxy_instance
        yield proxy_instance


@pytest.fixture
def client(mock_llama_proxy):
    with patch("app.LlamaConfig") as mock_config:
        mock_config.from_env.return_value = MagicMock(
            model="/models/test.gguf",
            ctx_size=4096,
            n_gpu_layers=99,
            threads=None,
            temperature=0.8,
            top_p=0.95,
            top_k=40,
            mmpproj=None,
            port=8080,
            n_parallel=1,
            extra_args=None,
            to_args=MagicMock(return_value=["-m", "/models/test.gguf"]),
        )
        with patch("app.AppConfig") as mock_app_config:
            mock_app_config.from_env.return_value = MagicMock(
                port=80,
                llama_host="127.0.0.1",
                llama_connect_host="127.0.0.1",
            )
            from app import app

            with TestClient(app) as test_client:
                yield test_client


class TestHealthEndpoint:
    def test_ping_healthy(self, client, mock_llama_proxy):
        mock_llama_proxy.health_check.return_value = True
        response = client.get("/ping")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    def test_ping_unhealthy(self, client, mock_llama_proxy):
        mock_llama_proxy.health_check.return_value = False
        response = client.get("/ping")
        assert response.status_code == 503
        assert response.json() == {"status": "unhealthy"}


class TestProxyEndpoints:
    def test_proxy_chat_completions(self, client, mock_llama_proxy):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{"id":"chatcmpl-123","choices":[{"message":{"role":"assistant","content":"Hello"}}]}'
        mock_response.headers = {"content-type": "application/json"}
        mock_llama_proxy.proxy_request.return_value = mock_response

        response = client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "Hi"}]},
        )

        assert response.status_code == 200
        mock_llama_proxy.proxy_request.assert_called_once()
        _, kwargs = mock_llama_proxy.proxy_request.call_args
        assert kwargs["path"] == "/v1/chat/completions"
        assert kwargs["headers"]["content-type"].startswith("application/json")
        assert "host" not in kwargs["headers"]

    def test_proxy_completions(self, client, mock_llama_proxy):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{"id":"cmpl-123","choices":[{"text":"Hello"}]}'
        mock_response.headers = {"content-type": "application/json"}
        mock_llama_proxy.proxy_request.return_value = mock_response

        response = client.post(
            "/v1/completions",
            json={"model": "test", "prompt": "Hi"},
        )

        assert response.status_code == 200
        mock_llama_proxy.proxy_request.assert_called_once()

    def test_proxy_get_models(self, client, mock_llama_proxy):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{"object":"list","data":[{"id":"test-model"}]}'
        mock_response.headers = {"content-type": "application/json"}
        mock_llama_proxy.proxy_request.return_value = mock_response

        response = client.get("/v1/models")

        assert response.status_code == 200
        mock_llama_proxy.proxy_request.assert_called_once()

    def test_proxy_filters_hop_by_hop_response_headers(self, client, mock_llama_proxy):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{"ok":true}'
        mock_response.headers = {
            "content-type": "application/json",
            "connection": "keep-alive",
            "x-test-header": "kept",
        }
        mock_llama_proxy.proxy_request.return_value = mock_response

        response = client.get("/v1/models")

        assert response.status_code == 200
        assert response.headers["x-test-header"] == "kept"
        assert "connection" not in response.headers

    def test_proxy_streaming_when_accept_includes_event_stream(
        self, client, mock_llama_proxy
    ):
        async def stream_body():
            yield b"data: hello\n\n"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/event-stream; charset=utf-8"}
        mock_response.aiter_raw.return_value = stream_body()
        mock_response.aclose = AsyncMock()
        mock_llama_proxy.proxy_stream_response.return_value = mock_response

        response = client.post(
            "/v1/chat/completions",
            headers={"accept": "application/json, text/event-stream"},
            json={"model": "test", "messages": [{"role": "user", "content": "Hi"}]},
        )

        assert response.status_code == 200
        assert response.text == "data: hello\n\n"
        mock_llama_proxy.proxy_stream_response.assert_called_once()
        mock_response.aclose.assert_awaited_once()

    def test_proxy_streaming_when_body_requests_stream(self, client, mock_llama_proxy):
        async def stream_body():
            yield b'{"error":"backend unavailable"}'

        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.headers = {"content-type": "application/json"}
        mock_response.aiter_raw.return_value = stream_body()
        mock_response.aclose = AsyncMock()
        mock_llama_proxy.proxy_stream_response.return_value = mock_response

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "stream": True,
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

        assert response.status_code == 503
        assert response.json() == {"error": "backend unavailable"}
        mock_llama_proxy.proxy_stream_response.assert_called_once()
        mock_llama_proxy.proxy_request.assert_not_called()
