import os
import shlex
import sys
from dataclasses import dataclass
from typing import Optional


def _dump_volume_tree() -> None:
    root = "/runpod-volume"
    if not os.path.isdir(root):
        sys.stderr.write(f"[ERROR] {root} does not exist or is not a directory\n")
        return
    sys.stderr.write(f"[ERROR] Contents of {root}:\n")
    for dirpath, dirnames, filenames in os.walk(root):
        level = dirpath.replace(root, "").count(os.sep)
        if level > 7:
            continue
        indent = "  " * level
        sys.stderr.write(f"{indent}{os.path.basename(dirpath) or 'runpod-volume'}/\n")
        for name in sorted(filenames):
            sys.stderr.write(f"{indent}  {name}\n")


def _required_str(name: str) -> str:
    val = os.getenv(name)
    if val and val.strip():
        return val
    raise ValueError(f"{name} is required")


def _optional_str(name: str) -> Optional[str]:
    val = os.getenv(name)
    return val if val else None


def _optional_int(name: str) -> Optional[int]:
    val = os.getenv(name)
    if not val:
        return None
    try:
        return int(val)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _optional_float(name: str) -> Optional[float]:
    val = os.getenv(name)
    if not val:
        return None
    try:
        return float(val)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


def _optional_bool(name: str) -> Optional[str]:
    val = os.getenv(name)
    if not val:
        return None
    lower = val.lower()
    if lower in ("on", "1", "yes"):
        return "on"
    if lower in ("off", "0", "no"):
        return "off"
    raise ValueError(f"{name} must be one of: on, 1, yes, off, 0, no")


RUNPOD_CACHE_HUB = "/runpod-volume/huggingface-cache/hub"


def resolve_runpod_cache_path(path: str, hub_root: Optional[str] = None) -> str:
    if hub_root is None:
        hub_root = RUNPOD_CACHE_HUB
    parts = path.split("/")
    if len(parts) != 3:
        raise ValueError(
            f"RunPod cache path must be in 'org/name/filename' format, got: {path}"
        )
    org, name, filename = parts
    cache_dir = os.path.join(hub_root, f"models--{org}--{name}")
    refs_file = os.path.join(cache_dir, "refs", "main")
    snapshots_dir = os.path.join(cache_dir, "snapshots")

    snapshot_hash = None
    if os.path.isfile(refs_file):
        with open(refs_file) as f:
            snapshot_hash = f.read().strip()

    if not snapshot_hash and os.path.isdir(snapshots_dir):
        versions = sorted(
            d for d in os.listdir(snapshots_dir)
            if os.path.isdir(os.path.join(snapshots_dir, d))
        )
        if versions:
            snapshot_hash = versions[0]

    if not snapshot_hash:
        _dump_volume_tree()
        raise FileNotFoundError(
            f"RunPod cached model not found for {path}. "
            f"Checked {cache_dir}. "
            "Make sure the Model field is set in the RunPod endpoint configuration."
        )

    return os.path.join(snapshots_dir, snapshot_hash, filename)


@dataclass
class LlamaConfig:
    hf_model: Optional[str] = None
    model: Optional[str] = None
    mmproj: Optional[str] = None
    model_runpod_cache: Optional[str] = None
    mmproj_runpod_cache: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    ctx_size: Optional[int] = None
    n_gpu_layers: Optional[int] = None
    threads: Optional[int] = None
    port: Optional[int] = None
    n_parallel: Optional[int] = None
    hf_home: Optional[str] = None
    hf_token: Optional[str] = None
    chat_template_kwargs: Optional[str] = None
    reasoning: Optional[str] = None
    extra_args: Optional[str] = None

    def __post_init__(self) -> None:
        sources = [self.hf_model, self.model, self.model_runpod_cache]
        set_count = sum(1 for s in sources if s is not None)
        if set_count == 0:
            raise ValueError(
                "One of LLAMA_HF_MODEL, LLAMA_MODEL, or LLAMA_MODEL_RUNPOD_CACHE is required"
            )
        if set_count > 1:
            raise ValueError(
                "Only one of LLAMA_HF_MODEL, LLAMA_MODEL, or LLAMA_MODEL_RUNPOD_CACHE may be set"
            )
        if self.mmproj and self.mmproj_runpod_cache:
            raise ValueError(
                "Only one of LLAMA_MMPROJ or LLAMA_MMPROJ_RUNPOD_CACHE may be set"
            )

    def resolve(self) -> None:
        if self.model_runpod_cache:
            self.model = resolve_runpod_cache_path(self.model_runpod_cache)
            self.model_runpod_cache = None
        if self.mmproj_runpod_cache:
            self.mmproj = resolve_runpod_cache_path(self.mmproj_runpod_cache)
            self.mmproj_runpod_cache = None

    def to_args(self) -> list[str]:
        args = []
        if self.hf_model:
            args.extend(["-hf", self.hf_model])
        if self.model:
            args.extend(["--model", self.model])
        if self.mmproj:
            args.extend(["--mmproj", self.mmproj])

        if self.ctx_size is not None:
            args.extend(["-c", str(self.ctx_size)])
        if self.n_gpu_layers is not None:
            args.extend(["-ngl", str(self.n_gpu_layers)])
        if self.threads is not None:
            args.extend(["-t", str(self.threads)])
        if self.temperature is not None:
            args.extend(["--temp", str(self.temperature)])
        if self.top_p is not None:
            args.extend(["--top-p", str(self.top_p)])
        if self.top_k is not None:
            args.extend(["--top-k", str(self.top_k)])
        if self.port is not None:
            args.extend(["--port", str(self.port)])
        if self.n_parallel is not None:
            args.extend(["-np", str(self.n_parallel)])
        if self.chat_template_kwargs is not None:
            args.extend(["--chat-template-kwargs", self.chat_template_kwargs])
        if self.reasoning is not None:
            args.extend(["--reasoning", self.reasoning])
        if self.extra_args is not None:
            args.extend(shlex.split(self.extra_args))

        return args

    def validate_files(self) -> None:
        if self.model and not os.path.isfile(self.model):
            _dump_volume_tree()
            raise FileNotFoundError(
                f"LLAMA_MODEL file not found: {self.model}"
            )
        if self.mmproj and not os.path.isfile(self.mmproj):
            _dump_volume_tree()
            raise FileNotFoundError(
                f"LLAMA_MMPROJ file not found: {self.mmproj}"
            )

    def get_env(self) -> dict:
        env = {}
        if self.hf_home is not None:
            env["HF_HOME"] = self.hf_home
        if self.hf_token is not None:
            env["HF_TOKEN"] = self.hf_token
        return env

    @classmethod
    def from_env(cls) -> "LlamaConfig":
        return cls(
            hf_model=_optional_str("LLAMA_HF_MODEL"),
            model=_optional_str("LLAMA_MODEL"),
            mmproj=_optional_str("LLAMA_MMPROJ"),
            model_runpod_cache=_optional_str("LLAMA_MODEL_RUNPOD_CACHE"),
            mmproj_runpod_cache=_optional_str("LLAMA_MMPROJ_RUNPOD_CACHE"),
            temperature=_optional_float("LLAMA_TEMPERATURE"),
            top_p=_optional_float("LLAMA_TOP_P"),
            top_k=_optional_int("LLAMA_TOP_K"),
            ctx_size=_optional_int("LLAMA_CONTEXT_SIZE"),
            n_gpu_layers=_optional_int("LLAMA_N_GPU_LAYERS"),
            threads=_optional_int("LLAMA_THREADS"),
            port=_optional_int("LLAMA_PORT") or 8080,
            n_parallel=_optional_int("LLAMA_N_PARALLEL"),
            hf_home=os.getenv("HF_HOME") or "/runpod-volume/huggingface-cache",
            hf_token=_optional_str("HF_TOKEN"),
            chat_template_kwargs=_optional_str("LLAMA_CHAT_TEMPLATE_KWARGS"),
            reasoning=_optional_bool("LLAMA_REASONING"),
            extra_args=_optional_str("LLAMA_EXTRA_ARGS"),
        )


@dataclass
class AppConfig:
    port: int = 80
    llama_host: str = "127.0.0.1"
    llama_connect_host: str = "127.0.0.1"

    def __post_init__(self):
        if self.port <= 0:
            raise ValueError("PORT must be greater than 0")
        if not self.llama_host:
            raise ValueError("LLAMA_HOST must not be empty")
        if not self.llama_connect_host:
            raise ValueError("LLAMA_CONNECT_HOST must not be empty")

    @classmethod
    def from_env(cls) -> "AppConfig":
        llama_host = os.getenv("LLAMA_HOST", "127.0.0.1")
        llama_connect_host = os.getenv("LLAMA_CONNECT_HOST")
        if not llama_connect_host:
            if llama_host in {"0.0.0.0", "::"}:
                llama_connect_host = "127.0.0.1"
            else:
                llama_connect_host = llama_host

        port = os.getenv("PORT")
        if not port:
            raise ValueError("PORT is required")
        try:
            port = int(port)
        except ValueError as exc:
            raise ValueError("PORT must be an integer") from exc

        return cls(
            port=port,
            llama_host=llama_host,
            llama_connect_host=llama_connect_host,
        )
