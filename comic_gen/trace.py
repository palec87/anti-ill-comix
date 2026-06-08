from __future__ import annotations

from typing import Any

from .models import utc_now_iso


def add_trace(
    document: dict[str, Any],
    step: str,
    status: str,
    message: str,
) -> None:
    document.setdefault("trace", []).append(
        {
            "ts": utc_now_iso(),
            "step": step,
            "status": status,
            "message": message,
        }
    )
