from unittest.mock import patch

import pytest

from config import AppConfig, LlamaConfig
from llama_proxy import LlamaProxy


@pytest.fixture
def proxy():
    with patch.object(LlamaProxy, "_find_llama_server", return_value="llama-server"):
        yield LlamaProxy(LlamaConfig(model="/models/test.gguf"), AppConfig())


class TestStart:
    @pytest.mark.asyncio
    async def test_calls_validate_files(self, proxy):
        with (
            patch.object(proxy.llama_config, "validate_files") as mock_validate,
            patch.object(proxy, "_wait_for_server"),
            patch("llama_proxy.subprocess.Popen"),
        ):
            await proxy.start()
            mock_validate.assert_called_once()
