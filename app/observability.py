from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class RunLogEvent:
    run_id: str
    request_id: str
    event: str
    timestamp: str
    node: str | None = None
    duration_ms: int | None = None
    details: dict[str, Any] | None = None

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        request_id: str,
        event: str,
        node: str | None = None,
        duration_ms: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> "RunLogEvent":
        return cls(
            run_id=run_id,
            request_id=request_id,
            event=event,
            timestamp=datetime.now(UTC).isoformat(),
            node=node,
            duration_ms=duration_ms,
            details=details,
        )


class RunLogger(Protocol):
    async def log(self, event: RunLogEvent) -> None: ...


class JsonlRunLogger:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = asyncio.Lock()

    async def log(self, event: RunLogEvent) -> None:
        async with self._lock:
            await asyncio.to_thread(self._append, event)

    def _append(self, event: RunLogEvent) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(event), separators=(",", ":")) + "\n")
