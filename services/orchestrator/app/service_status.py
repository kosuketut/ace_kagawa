from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class ServiceStatus:
    name: str
    ok: bool
    detail: str
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "detail": self.detail,
            "meta": self.meta,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

