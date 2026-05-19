import os
import pytest
from config import AppConfig, LlamaConfig, resolve_runpod_cache_path


class TestLlamaConfig:
    def test_from_env_requires_model(self, monkeypatch):
        monkeypatch.delenv("LLAMA_HF_MODEL", raising=False)
        monkeypatch.delenv("LLAMA_MODEL", raising=False)

        with pytest.raises(ValueError, match="One of LLAMA_HF_MODEL, LLAMA_MODEL, or LLAMA_MODEL_RUNPOD_CACHE is required"):
            LlamaConfig.from_env()

    def test_from_env_rejects_both_set(self, monkeypatch):
        monkeypatch.setenv("LLAMA_HF_MODEL", "philipsorst/gemma-4")
        monkeypatch.setenv("LLAMA_MODEL", "/models/test.gguf")

        with pytest.raises(ValueError, match="Only one of LLAMA_HF_MODEL, LLAMA_MODEL, or LLAMA_MODEL_RUNPOD_CACHE may be set"):
            LlamaConfig.from_env()

    def test_from_env_with_local_model(self, monkeypatch):
        monkeypatch.setenv("LLAMA_MODEL", "/models/test.gguf")
        monkeypatch.delenv("LLAMA_HF_MODEL", raising=False)
        monkeypatch.delenv("HF_HOME", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("LLAMA_CHAT_TEMPLATE_KWARGS", raising=False)
        monkeypatch.delenv("LLAMA_REASONING", raising=False)

        config = LlamaConfig.from_env()

        assert config.model == "/models/test.gguf"
        assert config.hf_model is None
        assert config.ctx_size is None
        assert config.n_gpu_layers is None
        assert config.threads is None
        assert config.temperature is None
        assert config.top_p is None
        assert config.top_k is None
        assert config.port == 8080
        assert config.n_parallel is None
        assert config.extra_args is None
        assert config.hf_home == "/runpod-volume/huggingface-cache"
        assert config.hf_token is None
        assert config.chat_template_kwargs is None
        assert config.reasoning is None

    def test_from_env_with_no_extra_env_vars(self, monkeypatch):
        monkeypatch.setenv("LLAMA_HF_MODEL", "unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q6_K_XL")
        monkeypatch.delenv("LLAMA_MODEL", raising=False)
        monkeypatch.delenv("HF_HOME", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("LLAMA_CHAT_TEMPLATE_KWARGS", raising=False)
        monkeypatch.delenv("LLAMA_REASONING", raising=False)

        config = LlamaConfig.from_env()

        assert config.hf_model == "unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q6_K_XL"
        assert config.model is None
        assert config.ctx_size is None
        assert config.n_gpu_layers is None
        assert config.threads is None
        assert config.temperature is None
        assert config.top_p is None
        assert config.top_k is None
        assert config.port == 8080
        assert config.n_parallel is None
        assert config.extra_args is None
        assert config.hf_home == "/runpod-volume/huggingface-cache"
        assert config.hf_token is None
        assert config.chat_template_kwargs is None
        assert config.reasoning is None

    def test_from_env_with_custom_values(self, monkeypatch):
        monkeypatch.setenv("LLAMA_HF_MODEL", "unsloth/gemma-4-26B-A4B-it-GGUF")
        monkeypatch.setenv("LLAMA_MMPROJ", "/models/mmproj.gguf")
        monkeypatch.setenv("LLAMA_CONTEXT_SIZE", "8192")
        monkeypatch.setenv("LLAMA_N_GPU_LAYERS", "50")
        monkeypatch.setenv("LLAMA_THREADS", "16")
        monkeypatch.setenv("LLAMA_TEMPERATURE", "0.5")
        monkeypatch.setenv("LLAMA_TOP_P", "0.9")
        monkeypatch.setenv("LLAMA_TOP_K", "100")
        monkeypatch.setenv("LLAMA_PORT", "9090")
        monkeypatch.setenv("LLAMA_N_PARALLEL", "4")
        monkeypatch.setenv("LLAMA_EXTRA_ARGS", "--flash-attn on --embedding")
        monkeypatch.setenv("HF_HOME", "/runpod-volume/huggingface-cache")
        monkeypatch.setenv("HF_TOKEN", "hf_token123")
        monkeypatch.setenv("LLAMA_CHAT_TEMPLATE_KWARGS", '{"enable_thinking":true}')
        monkeypatch.setenv("LLAMA_REASONING", "on")

        config = LlamaConfig.from_env()

        assert config.hf_model == "unsloth/gemma-4-26B-A4B-it-GGUF"
        assert config.model is None
        assert config.mmproj == "/models/mmproj.gguf"
        assert config.ctx_size == 8192
        assert config.n_gpu_layers == 50
        assert config.threads == 16
        assert config.temperature == 0.5
        assert config.top_p == 0.9
        assert config.top_k == 100
        assert config.port == 9090
        assert config.n_parallel == 4
        assert config.extra_args == "--flash-attn on --embedding"
        assert config.hf_home == "/runpod-volume/huggingface-cache"
        assert config.hf_token == "hf_token123"
        assert config.chat_template_kwargs == '{"enable_thinking":true}'
        assert config.reasoning == "on"

    def test_reasoning_parses_variations(self, monkeypatch):
        monkeypatch.setenv("LLAMA_HF_MODEL", "test")

        for val in ("on", "ON", "On", "1", "yes", "YES"):
            monkeypatch.setenv("LLAMA_REASONING", val)
            assert LlamaConfig.from_env().reasoning == "on", f"failed for {val!r}"

        for val in ("off", "OFF", "Off", "0", "no", "NO"):
            monkeypatch.setenv("LLAMA_REASONING", val)
            assert LlamaConfig.from_env().reasoning == "off", f"failed for {val!r}"

    def test_reasoning_invalid_value_raises(self, monkeypatch):
        monkeypatch.setenv("LLAMA_HF_MODEL", "test")
        monkeypatch.setenv("LLAMA_REASONING", "maybe")

        with pytest.raises(ValueError, match="LLAMA_REASONING must be one of"):
            LlamaConfig.from_env()

    def test_to_args_uses_hf_flag(self):
        config = LlamaConfig(
            hf_model="unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q6_K_XL",
            ctx_size=2048,
            n_gpu_layers=32,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            port=8080,
            n_parallel=2,
        )

        args = config.to_args()

        assert "-hf" in args and "unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q6_K_XL" in args
        assert "--model" not in args
        assert "-c" in args and "2048" in args
        assert "-ngl" in args and "32" in args
        assert "--temp" in args and "0.7" in args
        assert "--top-p" in args and "0.9" in args
        assert "--top-k" in args and "50" in args
        assert "--port" in args and "8080" in args
        assert "-np" in args and "2" in args

    def test_to_args_uses_model_flag_for_local_path(self):
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

        assert "--model" in args and "/models/test.gguf" in args
        assert "-hf" not in args
        assert "-c" in args and "2048" in args
        assert "-ngl" in args and "32" in args
        assert "--temp" in args and "0.7" in args
        assert "--top-p" in args and "0.9" in args
        assert "--top-k" in args and "50" in args
        assert "--port" in args and "8080" in args
        assert "-np" in args and "2" in args

    def test_to_args_with_mmproj(self):
        config = LlamaConfig(
            hf_model="test",
            mmproj="/models/mmproj.gguf",
        )
        args = config.to_args()
        assert "-hf" in args and "--mmproj" in args and "/models/mmproj.gguf" in args

    def test_to_args_with_only_hf_model(self):
        config = LlamaConfig(hf_model="unsloth/gemma-4-26B-A4B-it-GGUF")
        args = config.to_args()
        assert args == ["-hf", "unsloth/gemma-4-26B-A4B-it-GGUF"]

    def test_to_args_with_only_local_model(self):
        config = LlamaConfig(model="/models/test.gguf")
        args = config.to_args()
        assert args == ["--model", "/models/test.gguf"]

    def test_to_args_with_reasoning_on(self):
        config = LlamaConfig(hf_model="test", reasoning="on")
        args = config.to_args()
        assert "--reasoning" in args and "on" in args

    def test_to_args_with_reasoning_off(self):
        config = LlamaConfig(hf_model="test", reasoning="off")
        args = config.to_args()
        assert "--reasoning" in args and "off" in args

    def test_to_args_with_threads(self):
        config = LlamaConfig(hf_model="test", threads=8)
        args = config.to_args()
        assert "-t" in args and "8" in args

    def test_to_args_without_threads(self):
        config = LlamaConfig(hf_model="test", threads=None)
        args = config.to_args()
        assert "-t" not in args
        assert "--reasoning" not in args

    def test_to_args_with_chat_template_kwargs(self):
        config = LlamaConfig(
            hf_model="test", chat_template_kwargs='{"enable_thinking":true}'
        )
        args = config.to_args()
        assert "--chat-template-kwargs" in args
        assert '{"enable_thinking":true}' in args

    def test_to_args_with_extra_args(self):
        config = LlamaConfig(hf_model="test", extra_args="--flash-attn on --embedding")
        args = config.to_args()
        assert "--flash-attn" in args
        assert "on" in args
        assert "--embedding" in args

    def test_to_args_with_quoted_extra_args(self):
        config = LlamaConfig(
            hf_model="test",
            extra_args='--log-file "/tmp/path with spaces/server.log"',
        )

        args = config.to_args()

        assert "--log-file" in args
        assert "/tmp/path with spaces/server.log" in args

    def test_from_env_rejects_invalid_integer(self, monkeypatch):
        monkeypatch.setenv("LLAMA_HF_MODEL", "test")
        monkeypatch.setenv("LLAMA_PORT", "nope")

        with pytest.raises(ValueError, match="LLAMA_PORT must be an integer"):
            LlamaConfig.from_env()

    def test_get_env_returns_hf_vars(self):
        config = LlamaConfig(
            hf_model="test",
            hf_home="/runpod-volume/huggingface-cache",
            hf_token="hf_123",
        )
        env = config.get_env()
        assert env["HF_HOME"] == "/runpod-volume/huggingface-cache"
        assert env["HF_TOKEN"] == "hf_123"

    def test_get_env_returns_empty_when_no_hf_vars(self):
        config = LlamaConfig(hf_model="test")
        env = config.get_env()
        assert env == {}

    def test_validate_files_raises_for_missing_model(self):
        config = LlamaConfig(model="/nonexistent/model.gguf")
        with pytest.raises(FileNotFoundError, match="LLAMA_MODEL file not found"):
            config.validate_files()

    def test_validate_files_raises_for_missing_mmproj(self):
        config = LlamaConfig(hf_model="test", mmproj="/nonexistent/mmproj.gguf")
        with pytest.raises(FileNotFoundError, match="LLAMA_MMPROJ file not found"):
            config.validate_files()

    def test_validate_files_passes_for_existing_files(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.write_bytes(b"dummy")
        mmproj = tmp_path / "mmproj.gguf"
        mmproj.write_bytes(b"dummy")
        config = LlamaConfig(model=str(model), mmproj=str(mmproj))
        config.validate_files()

    def test_validate_files_skips_check_when_using_hf_model(self):
        config = LlamaConfig(hf_model="test")
        config.validate_files()

    def test_from_env_with_runpod_cache_model(self, monkeypatch):
        monkeypatch.setenv("LLAMA_MODEL_RUNPOD_CACHE", "org/test/model.gguf")
        monkeypatch.delenv("LLAMA_HF_MODEL", raising=False)
        monkeypatch.delenv("LLAMA_MODEL", raising=False)

        config = LlamaConfig.from_env()

        assert config.model_runpod_cache == "org/test/model.gguf"
        assert config.model is None
        assert config.hf_model is None

    def test_from_env_rejects_both_runpod_cache_and_other_source(self, monkeypatch):
        monkeypatch.setenv("LLAMA_HF_MODEL", "org/test")
        monkeypatch.setenv("LLAMA_MODEL_RUNPOD_CACHE", "org/test/model.gguf")

        with pytest.raises(ValueError, match="Only one of"):
            LlamaConfig.from_env()

    def test_from_env_rejects_runpod_cache_with_local_model(self, monkeypatch):
        monkeypatch.setenv("LLAMA_MODEL", "/models/test.gguf")
        monkeypatch.setenv("LLAMA_MODEL_RUNPOD_CACHE", "org/test/model.gguf")

        with pytest.raises(ValueError, match="Only one of"):
            LlamaConfig.from_env()

    def test_from_env_rejects_both_mmproj_options(self, monkeypatch):
        monkeypatch.setenv("LLAMA_HF_MODEL", "test")
        monkeypatch.setenv("LLAMA_MMPROJ", "/models/mmproj.gguf")
        monkeypatch.setenv("LLAMA_MMPROJ_RUNPOD_CACHE", "org/test/mmproj.gguf")

        with pytest.raises(ValueError, match="Only one of LLAMA_MMPROJ or LLAMA_MMPROJ_RUNPOD_CACHE"):
            LlamaConfig.from_env()

    def _make_cache(self, tmp_path, org="org", name="test", hash="abc123", files=None):
        model_dir = tmp_path / f"models--{org}--{name}"
        snap_dir = model_dir / "snapshots" / hash
        snap_dir.mkdir(parents=True)
        if files:
            for f in files:
                (snap_dir / f).write_bytes(b"dummy")
        refs_dir = model_dir / "refs"
        refs_dir.mkdir()
        (refs_dir / "main").write_text(hash + "\n")
        return model_dir

    def test_resolve_clears_model_runpod_cache_and_sets_model(
        self, tmp_path, monkeypatch
    ):
        cache = self._make_cache(tmp_path, files=["test.gguf"])
        model_file = cache / "snapshots" / "abc123" / "test.gguf"
        monkeypatch.setattr("config.RUNPOD_CACHE_HUB", str(tmp_path))

        config = LlamaConfig(model_runpod_cache="org/test/test.gguf")
        config.resolve()

        assert config.model == str(model_file)
        assert config.model_runpod_cache is None

    def test_resolve_with_mmproj(self, tmp_path, monkeypatch):
        cache = self._make_cache(tmp_path, hash="def456", files=["mmproj.gguf"])
        mmproj_file = cache / "snapshots" / "def456" / "mmproj.gguf"
        monkeypatch.setattr("config.RUNPOD_CACHE_HUB", str(tmp_path))

        config = LlamaConfig(
            hf_model="test",
            mmproj_runpod_cache="org/test/mmproj.gguf",
        )
        config.resolve()

        assert config.mmproj == str(mmproj_file)
        assert config.mmproj_runpod_cache is None

    def test_constructor_rejects_both_mmproj_options(self):
        with pytest.raises(ValueError, match="Only one of LLAMA_MMPROJ or LLAMA_MMPROJ_RUNPOD_CACHE"):
            LlamaConfig(
                hf_model="test",
                mmproj="/models/mmproj.gguf",
                mmproj_runpod_cache="org/test/mmproj.gguf",
            )

    def test_resolve_skipped_when_no_runpod_cache_set(self):
        config = LlamaConfig(hf_model="test")
        config.resolve()
        assert config.model is None
        assert config.mmproj is None

    def test_to_args_with_runpod_cache_after_resolve(self, tmp_path, monkeypatch):
        cache = self._make_cache(tmp_path, hash="xyz789", files=["test.gguf"])
        model_file = cache / "snapshots" / "xyz789" / "test.gguf"
        monkeypatch.setattr("config.RUNPOD_CACHE_HUB", str(tmp_path))

        config = LlamaConfig(
            model_runpod_cache="org/test/test.gguf",
            ctx_size=4096,
        )
        config.resolve()
        args = config.to_args()

        assert "--model" in args and str(model_file) in args
        assert "-hf" not in args


class TestResolveRunpodCachePath:
    def _make_cache(self, tmp_path, org="org", name="test", hash="abc123", files=None):
        model_dir = tmp_path / f"models--{org}--{name}"
        snap_dir = model_dir / "snapshots" / hash
        snap_dir.mkdir(parents=True)
        if files:
            for f in files:
                (snap_dir / f).write_bytes(b"dummy")
        refs_dir = model_dir / "refs"
        refs_dir.mkdir()
        (refs_dir / "main").write_text(hash + "\n")
        return model_dir

    def test_resolves_from_refs_main(self, tmp_path):
        self._make_cache(tmp_path, files=["model.gguf"])
        result = resolve_runpod_cache_path("org/test/model.gguf", hub_root=str(tmp_path))
        assert result == str(tmp_path / "models--org--test" / "snapshots" / "abc123" / "model.gguf")

    def test_falls_back_to_sorted_snapshots(self, tmp_path):
        model_dir = tmp_path / "models--org--test"
        snap_dir = model_dir / "snapshots"
        snap_dir.mkdir(parents=True)
        (snap_dir / "002").mkdir()
        (snap_dir / "001").mkdir()
        (snap_dir / "001" / "model.gguf").write_bytes(b"dummy")

        result = resolve_runpod_cache_path("org/test/model.gguf", hub_root=str(tmp_path))
        assert result == str(snap_dir / "001" / "model.gguf")

    def test_raises_when_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="RunPod cached model not found"):
            resolve_runpod_cache_path("org/test/model.gguf", hub_root=str(tmp_path))

    def test_resolves_with_mixed_case(self, tmp_path):
        self._make_cache(tmp_path, files=["model.gguf"])
        result = resolve_runpod_cache_path(
            "ORG/Test/model.gguf", hub_root=str(tmp_path)
        )
        assert result == str(
            tmp_path / "models--org--test" / "snapshots" / "abc123" / "model.gguf"
        )

    def test_raises_on_invalid_format(self):
        with pytest.raises(ValueError, match="must be in 'org/name/filename' format"):
            resolve_runpod_cache_path("invalid", hub_root="/ignored")

    def test_raises_on_too_many_parts(self):
        with pytest.raises(ValueError, match="must be in 'org/name/filename' format"):
            resolve_runpod_cache_path("a/b/c/d.gguf", hub_root="/ignored")


class TestAppConfig:
    def test_port_required(self, monkeypatch):
        monkeypatch.delenv("PORT", raising=False)
        monkeypatch.delenv("LLAMA_HOST", raising=False)
        monkeypatch.delenv("LLAMA_CONNECT_HOST", raising=False)

        with pytest.raises(ValueError, match="PORT is required"):
            AppConfig.from_env()

    def test_defaults(self, monkeypatch):
        monkeypatch.setenv("PORT", "80")
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
        monkeypatch.setenv("PORT", "80")
        monkeypatch.setenv("LLAMA_HOST", "0.0.0.0")
        monkeypatch.delenv("LLAMA_CONNECT_HOST", raising=False)

        config = AppConfig.from_env()

        assert config.llama_host == "0.0.0.0"
        assert config.llama_connect_host == "127.0.0.1"

    def test_connect_host_override(self, monkeypatch):
        monkeypatch.setenv("PORT", "80")
        monkeypatch.setenv("LLAMA_HOST", "0.0.0.0")
        monkeypatch.setenv("LLAMA_CONNECT_HOST", "172.17.0.2")

        config = AppConfig.from_env()

        assert config.llama_host == "0.0.0.0"
        assert config.llama_connect_host == "172.17.0.2"
