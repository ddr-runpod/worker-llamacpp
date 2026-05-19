import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

LOG_LEVELS = {"DEBUG": 0, "INFO": 1, "WARN": 2, "ERROR": 3}
DEFAULT_SERVICE_NAME = "worker-llamacpp"
DEFAULT_SERVICE_VERSION = "unknown"
DEFAULT_ENV = "unknown"


class Logger:
    def __init__(
        self,
        service_name: Optional[str] = None,
        service_version: Optional[str] = None,
    ):
        self.service_name = service_name or os.getenv(
            "RUNPOD_SERVICE_NAME", DEFAULT_SERVICE_NAME
        )
        self.service_version = service_version or os.getenv(
            "RUNPOD_SERVICE_VERSION", DEFAULT_SERVICE_VERSION
        )
        raw_level = os.getenv("LOG_LEVEL", "INFO").upper()
        self.log_level = raw_level if raw_level in LOG_LEVELS else "INFO"
        self._level_value = LOG_LEVELS[self.log_level]
        self.env = os.getenv("ENV", DEFAULT_ENV)
        self._instance_id = uuid4().hex[:8]
        self._service_start = datetime.now(timezone.utc).isoformat()

    def _log(self, level: str, message: str, extra: Optional[dict] = None) -> None:
        level_value = LOG_LEVELS.get(level, 0)
        if level_value < self._level_value:
            return

        log_obj: dict[str, Any] = {
            "message": message,
            "level": level,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                "service_name": self.service_name,
                "service_version": self.service_version,
                "env": self.env,
                "instance_id": self._instance_id,
                "service_start": self._service_start,
            },
        }
        if extra:
            log_obj["metadata"].update(extra)

        sys.stderr.write(json.dumps(log_obj) + "\n")
        sys.stderr.flush()

    def debug(self, message: str, extra: Optional[dict] = None) -> None:
        self._log("DEBUG", message, extra)

    def info(self, message: str, extra: Optional[dict] = None) -> None:
        self._log("INFO", message, extra)

    def warn(self, message: str, extra: Optional[dict] = None) -> None:
        self._log("WARN", message, extra)

    def error(self, message: str, extra: Optional[dict] = None) -> None:
        self._log("ERROR", message, extra)


log = Logger()
