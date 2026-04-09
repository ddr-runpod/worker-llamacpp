import asyncio
import os
import shutil
import signal
import subprocess
import tempfile
from typing import BinaryIO, Optional

import httpx

from config import AppConfig, LlamaConfig


class LlamaProxy:
    def __init__(self, llama_config: LlamaConfig, app_config: AppConfig):
        self.llama_config = llama_config
        self.app_config = app_config
        self.process: Optional[subprocess.Popen] = None
        self.base_url = f"http://{app_config.llama_connect_host}:{llama_config.port}"
        self._client: Optional[httpx.AsyncClient] = None
        self._llama_server_path = self._find_llama_server()
        self._process_log: Optional[BinaryIO] = None

    def _find_llama_server(self) -> str:
        custom_path = os.getenv("LLAMA_SERVER_PATH")
        if custom_path:
            return custom_path

        if shutil.which("llama-server"):
            return "llama-server"

        common_paths = [
            "/usr/local/bin/llama-server",
            "/usr/bin/llama-server",
            "/opt/llama-server/llama-server",
            "./llama-server",
        ]
        for path in common_paths:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path

        return "llama-server"

    async def start(self) -> None:
        if self.process is not None:
            return

        args = [
            self._llama_server_path,
            "--host",
            self.app_config.llama_host,
        ] + self.llama_config.to_args()

        env = os.environ.copy()
        llama_env = self.llama_config.get_env()
        env.update(llama_env)

        self._process_log = tempfile.TemporaryFile()

        try:
            self.process = subprocess.Popen(
                args,
                stdout=self._process_log,
                stderr=subprocess.STDOUT,
                env=env,
            )
        except OSError as exc:
            self._close_process_log()
            raise RuntimeError(
                f"failed to start llama-server using {self._llama_server_path}: {exc}"
            ) from exc

        try:
            await self._wait_for_server(timeout=300)
        except Exception:
            await self.stop()
            raise

    async def _wait_for_server(self, timeout: int = 300) -> None:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=10.0) as client:
            for _ in range(timeout):
                if self.process is None:
                    raise RuntimeError("llama-server process is not running")

                exit_code = self.process.poll()
                if exit_code is not None:
                    raise RuntimeError(self._format_startup_failure(exit_code))

                try:
                    response = await client.get("/health")
                    if response.status_code == 200:
                        return
                except (httpx.ConnectError, httpx.TimeoutException):
                    pass
                await asyncio.sleep(1)
        raise RuntimeError(
            self._format_startup_failure(reason="timed out waiting for /health")
        )

    async def stop(self) -> None:
        if self.process is None:
            self._close_process_log()
            return

        self.process.send_signal(signal.SIGTERM)
        try:
            await asyncio.to_thread(self.process.wait, timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            await asyncio.to_thread(self.process.wait)

        self.process = None
        self._close_process_log()

    async def health_check(self) -> bool:
        if self.process is None or self.process.poll() is not None:
            return False
        try:
            response = await self.client.get("/health", timeout=10.0)
            return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=600.0)
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def proxy_request(
        self,
        method: str,
        path: str,
        headers: Optional[dict] = None,
        content: Optional[bytes] = None,
    ) -> httpx.Response:
        return await self.client.request(
            method=method,
            url=path,
            headers=headers,
            content=content,
            timeout=600.0,
        )

    async def proxy_stream_response(
        self,
        method: str,
        path: str,
        headers: Optional[dict] = None,
        content: Optional[bytes] = None,
    ) -> httpx.Response:
        request = self.client.build_request(
            method=method,
            url=path,
            headers=headers,
            content=content,
        )
        return await self.client.send(request, stream=True)

    def _close_process_log(self) -> None:
        if self._process_log is not None:
            self._process_log.close()
            self._process_log = None

    def _read_process_log_tail(self, max_bytes: int = 4096) -> str:
        if self._process_log is None:
            return ""

        self._process_log.flush()
        self._process_log.seek(0, os.SEEK_END)
        size = self._process_log.tell()
        self._process_log.seek(max(size - max_bytes, 0))
        output = self._process_log.read().decode("utf-8", errors="replace").strip()
        return output

    def _format_startup_failure(
        self, exit_code: Optional[int] = None, reason: Optional[str] = None
    ) -> str:
        message = "llama-server failed to start"
        if exit_code is not None:
            message = (
                f"llama-server exited before becoming healthy (exit code {exit_code})"
            )
        elif reason:
            message = f"llama-server failed to start: {reason}"

        output = self._read_process_log_tail()
        if output:
            return f"{message}\nRecent llama-server output:\n{output}"
        return message
