from unittest.mock import patch

import pytest

from config import AppConfig, LlamaConfig
from llama_proxy import LlamaProxy


@pytest.fixture
def proxy():
    with patch.object(LlamaProxy, "_find_llama_server", return_value="llama-server"):
        yield LlamaProxy(LlamaConfig(model="/models/test.gguf"), AppConfig())
