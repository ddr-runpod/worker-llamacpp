import os
import shlex
from dataclasses import dataclass
from typing import Optional


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


@dataclass
class LlamaConfig:
    model: str = ""
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
        if not self.model:
            raise ValueError("LLAMA_MODEL is required")

    def to_args(self) -> list[str]:
        args = ["-hf", self.model]

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
            model=_required_str("LLAMA_MODEL"),
            temperature=_optional_float("LLAMA_TEMPERATURE"),
            top_p=_optional_float("LLAMA_TOP_P"),
            top_k=_optional_int("LLAMA_TOP_K"),
            ctx_size=_optional_int("LLAMA_CONTEXT_SIZE"),
            n_gpu_layers=_optional_int("LLAMA_N_GPU_LAYERS"),
            threads=_optional_int("LLAMA_THREADS"),
            port=_optional_int("LLAMA_PORT") or 8080,
            n_parallel=_optional_int("LLAMA_N_PARALLEL"),
            hf_home=_optional_str("HF_HOME"),
            hf_token=_optional_str("HF_TOKEN"),
            chat_template_kwargs=_optional_str("LLAMA_CHAT_TEMPLATE_KWARGS"),
            reasoning=_optional_bool("LLAMA_REASONING"),
            extra_args=_optional_str("LLAMA_EXTRA_ARGS"),
        )


@dataclass
class AppConfig:
    port: int = 5000
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
        if port:
            try:
                port = int(port)
            except ValueError as exc:
                raise ValueError("PORT must be an integer") from exc
        else:
            port = 5000

        return cls(
            port=port,
            llama_host=llama_host,
            llama_connect_host=llama_connect_host,
        )
