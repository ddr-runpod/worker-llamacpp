import os
import shlex
from dataclasses import dataclass
from typing import Optional


def _required_str(name: str) -> str:
    val = os.getenv(name)
    if val:
        return val
    raise ValueError(f"{name} is required")


def _optional_str(name: str) -> Optional[str]:
    val = os.getenv(name)
    return val if val else None


def _int_env(name: str, default: str) -> int:
    val = os.getenv(name, default)
    try:
        return int(val)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _float_env(name: str, default: str) -> float:
    val = os.getenv(name, default)
    try:
        return float(val)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


def _optional_int(name: str) -> Optional[int]:
    val = os.getenv(name)
    if not val:
        return None
    try:
        return int(val)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


@dataclass
class LlamaConfig:
    model: str = ""
    ctx_size: int = 4096
    n_gpu_layers: int = 99
    threads: Optional[int] = None
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 40
    mmpproj: Optional[str] = None
    port: int = 8080
    n_parallel: int = 1
    extra_args: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("LLAMA_MODEL is required")

        for field_name, value in {
            "ctx_size": self.ctx_size,
            "port": self.port,
            "n_parallel": self.n_parallel,
        }.items():
            if value <= 0:
                raise ValueError(f"{field_name} must be greater than 0")

        if self.n_gpu_layers < 0:
            raise ValueError("n_gpu_layers must be greater than or equal to 0")

        if self.threads is not None and self.threads <= 0:
            raise ValueError("threads must be greater than 0")

        if self.top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if self.temperature < 0:
            raise ValueError("temperature must be greater than or equal to 0")

        if not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")

    @classmethod
    def from_env(cls) -> "LlamaConfig":
        return cls(
            model=_required_str("LLAMA_MODEL"),
            ctx_size=_int_env("LLAMA_CONTEXT_SIZE", "4096"),
            n_gpu_layers=_int_env("LLAMA_N_GPU_LAYERS", "99"),
            threads=_optional_int("LLAMA_THREADS"),
            temperature=_float_env("LLAMA_TEMPERATURE", "0.8"),
            top_p=_float_env("LLAMA_TOP_P", "0.95"),
            top_k=_int_env("LLAMA_TOP_K", "40"),
            mmpproj=_optional_str("LLAMA_MMPROJ"),
            port=_int_env("LLAMA_PORT", "8080"),
            n_parallel=_int_env("LLAMA_N_PARALLEL", "1"),
            extra_args=_optional_str("LLAMA_EXTRA_ARGS"),
        )

    def to_args(self) -> list[str]:
        args = [
            "-m",
            self.model,
            "-c",
            str(self.ctx_size),
            "-ngl",
            str(self.n_gpu_layers),
            "--temp",
            str(self.temperature),
            "--top-p",
            str(self.top_p),
            "--top-k",
            str(self.top_k),
            "--port",
            str(self.port),
            "-np",
            str(self.n_parallel),
        ]
        if self.threads is not None:
            args.extend(["-t", str(self.threads)])
        if self.mmpproj:
            args.extend(["-mm", self.mmpproj])
        if self.extra_args:
            args.extend(shlex.split(self.extra_args))
        return args


@dataclass
class AppConfig:
    port: int = 80
    llama_host: str = "127.0.0.1"
    llama_connect_host: str = "127.0.0.1"

    def __post_init__(self) -> None:
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

        return cls(
            port=_int_env("PORT", "80"),
            llama_host=llama_host,
            llama_connect_host=llama_connect_host,
        )
