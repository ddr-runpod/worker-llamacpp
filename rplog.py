import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4


class Logger:
    def __init__(
        self,
        service_name: Optional[str] = None,
        service_version: Optional[str] = None,
    ):
        self.service_name = service_name or os.getenv(
            "RUNPOD_SERVICE_NAME", "worker-llamacpp"
        )
        self.service_version = service_version or os.getenv(
            "RUNPOD_SERVICE_VERSION", "unknown"
        )
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.env = os.getenv("ENV", "unknown")
        self._instance_id = uuid4().hex[:8]
        self._service_start = datetime.now(timezone.utc).isoformat()

    def _log(self, level: str, message: str, extra: Optional[dict] = None) -> None:
        if level == "DEBUG" and self.log_level != "DEBUG":
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
