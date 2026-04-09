import pytest
from config import AppConfig, LlamaConfig


class TestLlamaConfig:
    def test_from_env_requires_model(self, monkeypatch):
        monkeypatch.delenv("LLAMA_MODEL", raising=False)

        with pytest.raises(ValueError, match="LLAMA_MODEL is required"):
            LlamaConfig.from_env()

    def test_from_env_with_defaults(self, monkeypatch):
        monkeypatch.setenv("LLAMA_MODEL", "/models/default.gguf")
        monkeypatch.delenv("LLAMA_THREADS", raising=False)
        monkeypatch.delenv("LLAMA_MMPROJ", raising=False)
        monkeypatch.delenv("LLAMA_EXTRA_ARGS", raising=False)

        config = LlamaConfig.from_env()

        assert config.model == "/models/default.gguf"
        assert config.ctx_size == 4096
        assert config.n_gpu_layers == 99
        assert config.threads is None
        assert config.temperature == 0.8
        assert config.top_p == 0.95
        assert config.top_k == 40
        assert config.mmpproj is None
        assert config.port == 8080
        assert config.n_parallel == 1
        assert config.extra_args is None

    def test_from_env_with_custom_values(self, monkeypatch):
        monkeypatch.setenv("LLAMA_MODEL", "/models/custom.gguf")
        monkeypatch.setenv("LLAMA_CONTEXT_SIZE", "8192")
        monkeypatch.setenv("LLAMA_N_GPU_LAYERS", "50")
        monkeypatch.setenv("LLAMA_THREADS", "16")
        monkeypatch.setenv("LLAMA_TEMPERATURE", "0.5")
        monkeypatch.setenv("LLAMA_TOP_P", "0.9")
        monkeypatch.setenv("LLAMA_TOP_K", "100")
        monkeypatch.setenv("LLAMA_MMPROJ", "/models/mmpproj.bin")
        monkeypatch.setenv("LLAMA_PORT", "9090")
        monkeypatch.setenv("LLAMA_N_PARALLEL", "4")
        monkeypatch.setenv("LLAMA_EXTRA_ARGS", "--flash-attn on --embedding")

        config = LlamaConfig.from_env()

        assert config.model == "/models/custom.gguf"
        assert config.ctx_size == 8192
        assert config.n_gpu_layers == 50
        assert config.threads == 16
        assert config.temperature == 0.5
        assert config.top_p == 0.9
        assert config.top_k == 100
        assert config.mmpproj == "/models/mmpproj.bin"
        assert config.port == 9090
        assert config.n_parallel == 4
        assert config.extra_args == "--flash-attn on --embedding"

    def test_to_args_basic(self):
        config = LlamaConfig(
            model="/models/test.gguf",
            ctx_size=2048,
            n_gpu_layers=32,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            port=8080,
            n_parallel=2,
        )

        args = config.to_args()

        assert "-m" in args and "/models/test.gguf" in args
        assert "-c" in args and "2048" in args
        assert "-ngl" in args and "32" in args
        assert "--temp" in args and "0.7" in args
        assert "--top-p" in args and "0.9" in args
        assert "--top-k" in args and "50" in args
        assert "--port" in args and "8080" in args
        assert "-np" in args and "2" in args

    def test_to_args_with_threads(self):
        config = LlamaConfig(model="/models/test.gguf", threads=8)
        args = config.to_args()
        assert "-t" in args and "8" in args

    def test_to_args_without_threads(self):
        config = LlamaConfig(model="/models/test.gguf", threads=None)
        args = config.to_args()
        assert "-t" not in args

    def test_to_args_with_mmpproj(self):
        config = LlamaConfig(model="/models/test.gguf", mmpproj="/models/mmpproj.bin")
        args = config.to_args()
        assert "-mm" in args and "/models/mmpproj.bin" in args

    def test_to_args_with_extra_args(self):
        config = LlamaConfig(
            model="/models/test.gguf", extra_args="--flash-attn on --embedding"
        )
        args = config.to_args()
        assert "--flash-attn" in args
        assert "on" in args
        assert "--embedding" in args

    def test_to_args_with_quoted_extra_args(self):
        config = LlamaConfig(
            model="/models/test.gguf",
            extra_args='--log-file "/tmp/path with spaces/server.log"',
        )

        args = config.to_args()

        assert "--log-file" in args
        assert "/tmp/path with spaces/server.log" in args

    def test_from_env_rejects_invalid_integer(self, monkeypatch):
        monkeypatch.setenv("LLAMA_MODEL", "/models/test.gguf")
        monkeypatch.setenv("LLAMA_PORT", "nope")

        with pytest.raises(ValueError, match="LLAMA_PORT must be an integer"):
            LlamaConfig.from_env()

    def test_from_env_rejects_invalid_top_p(self, monkeypatch):
        monkeypatch.setenv("LLAMA_MODEL", "/models/test.gguf")
        monkeypatch.setenv("LLAMA_TOP_P", "1.5")

        with pytest.raises(ValueError, match="top_p must be between 0 and 1"):
            LlamaConfig.from_env()


class TestAppConfig:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("PORT", raising=False)
        monkeypatch.delenv("LLAMA_HOST", raising=False)
        monkeypatch.delenv("LLAMA_CONNECT_HOST", raising=False)

        config = AppConfig.from_env()

        assert config.port == 80
        assert config.llama_host == "127.0.0.1"
        assert config.llama_connect_host == "127.0.0.1"

    def test_custom_values(self, monkeypatch):
        monkeypatch.setenv("PORT", "8000")
        monkeypatch.setenv("LLAMA_HOST", "10.0.0.5")

        config = AppConfig.from_env()

        assert config.port == 8000
        assert config.llama_host == "10.0.0.5"
        assert config.llama_connect_host == "10.0.0.5"

    def test_wildcard_llama_host_uses_loopback_for_connect(self, monkeypatch):
        monkeypatch.setenv("LLAMA_HOST", "0.0.0.0")
        monkeypatch.delenv("LLAMA_CONNECT_HOST", raising=False)

        config = AppConfig.from_env()

        assert config.llama_host == "0.0.0.0"
        assert config.llama_connect_host == "127.0.0.1"

    def test_connect_host_override(self, monkeypatch):
        monkeypatch.setenv("LLAMA_HOST", "0.0.0.0")
        monkeypatch.setenv("LLAMA_CONNECT_HOST", "172.17.0.2")

        config = AppConfig.from_env()

        assert config.llama_host == "0.0.0.0"
        assert config.llama_connect_host == "172.17.0.2"
