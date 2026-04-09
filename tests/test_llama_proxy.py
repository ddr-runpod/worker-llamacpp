from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from config import AppConfig, LlamaConfig
from llama_proxy import LlamaProxy


@pytest.fixture
def proxy():
    with patch.object(LlamaProxy, "_find_llama_server", return_value="llama-server"):
        yield LlamaProxy(LlamaConfig(model="/models/test.gguf"), AppConfig())


@pytest.mark.asyncio
async def test_start_fails_fast_when_process_exits(proxy):
    process = MagicMock()
    process.poll.return_value = 1
    process.wait.return_value = 1

    client = AsyncMock()
    client.get.side_effect = httpx.ConnectError("not ready")

    async_client = MagicMock()
    async_client.__aenter__ = AsyncMock(return_value=client)
    async_client.__aexit__ = AsyncMock(return_value=None)

    def popen(*args, **kwargs):
        kwargs["stdout"].write(b"failed to load model\n")
        kwargs["stdout"].flush()
        return process

    with (
        patch("llama_proxy.subprocess.Popen", side_effect=popen),
        patch("llama_proxy.httpx.AsyncClient", return_value=async_client),
    ):
        with pytest.raises(RuntimeError, match="exit code 1") as exc_info:
            await proxy.start()

    assert "failed to load model" in str(exc_info.value)
    process.send_signal.assert_called_once()


@pytest.mark.asyncio
async def test_wait_for_server_timeout_includes_recent_output(proxy):
    process = MagicMock()
    process.poll.return_value = None
    proxy.process = process

    client = AsyncMock()
    client.get.side_effect = httpx.ConnectError("not ready")

    async_client = MagicMock()
    async_client.__aenter__ = AsyncMock(return_value=client)
    async_client.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("llama_proxy.httpx.AsyncClient", return_value=async_client),
        patch("llama_proxy.asyncio.sleep", new=AsyncMock()),
    ):
        log_file = MagicMock()
        log_file.tell.side_effect = [12]
        log_file.read.return_value = b"still starting"
        proxy._process_log = log_file

        with pytest.raises(
            RuntimeError, match="timed out waiting for /health"
        ) as exc_info:
            await proxy._wait_for_server(timeout=1)

    assert "still starting" in str(exc_info.value)
