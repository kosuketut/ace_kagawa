from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID


@dataclass
class TurnTimer:
    started_at: float = field(default_factory=time.perf_counter)
    marks: dict[str, float] = field(default_factory=dict)

    def mark(self, name: str) -> None:
        self.marks.setdefault(name, time.perf_counter())

    def as_latency_map(self) -> dict[str, int | None]:
        result: dict[str, int | None] = {}
        for name in (
            "vad_start",
            "eou_detected",
            "asr_final",
            "llm_first_token",
            "tts_first_audio",
            "a2f_start",
        ):
            value = self.marks.get(name)
            result[f"{name}_ms"] = None if value is None else int((value - self.started_at) * 1000)
        result["turn_total_ms"] = int((time.perf_counter() - self.started_at) * 1000)
        return result


class JsonlTurnLogger:
    def __init__(self, log_dir: Path) -> None:
        self._log_dir = log_dir
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def write_turn(
        self,
        *,
        session_id: UUID,
        turn_id: UUID,
        user_text: str,
        assistant_text: str,
        timer: TurnTimer,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        now = datetime.now(timezone.utc)
        path = self._log_dir / f"turns-{now:%Y%m%d}.jsonl"
        payload = {
            "timestamp": now.isoformat(),
            "session_id": str(session_id),
            "turn_id": str(turn_id),
            "user_text": user_text,
            "assistant_text": assistant_text,
            **timer.as_latency_map(),
        }
        if metadata:
            payload["metadata"] = metadata
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return path

