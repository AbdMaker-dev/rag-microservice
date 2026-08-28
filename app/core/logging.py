"""Structured logging with redaction.

Questions, prompts and chunk contents never reach the logs. ADR-010 in the
monorepo excludes minors' data, credentials and tokens from journals; the same
rule applies here, and the filter below is the enforcement rather than a
convention.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from typing import Any, Dict

from app.config import Settings

_FORBIDDEN_KEYS = {
    "query",
    "question",
    "content",
    "prompt",
    "excerpt",
    "text",
    "embedding",
    "password",
    "token",
    "authorization",
    "x_service_token",
}
_BEARER = re.compile(r"(?i)\b(bearer|session)\s+[A-Za-z0-9._~+/=-]+")

_RESERVED = {
    "args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName",
    "levelname", "levelno", "lineno", "module", "msecs", "message", "msg", "name",
    "pathname", "process", "processName", "relativeCreated", "stack_info",
    "thread", "threadName", "taskName",
}


class RedactingJsonFormatter(logging.Formatter):
    def __init__(self, redact: bool) -> None:
        super().__init__()
        self._redact = redact

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": _BEARER.sub("[redacted]", record.getMessage()),
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED or key.startswith("_"):
                continue
            if self._redact and key.lower() in _FORBIDDEN_KEYS:
                payload[key] = "[redacted]"
                continue
            payload[key] = value
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(settings: Settings) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(RedactingJsonFormatter(settings.log_redaction_enabled))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(settings.log_level.upper())
