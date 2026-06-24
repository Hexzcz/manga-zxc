"""Shared helpers for optional per-stage wall timing (multiprocessing-safe)."""

from __future__ import annotations

import time
from typing import Any, Optional


def append_stage_timing(
    timing_log: Optional[Any],
    stage: str,
    page_index: Optional[int],
    t0: float,
    *,
    pages: Optional[int] = None,
) -> None:
    if timing_log is None:
        return
    row: dict[str, Any] = {"stage": stage, "dt_s": time.perf_counter() - t0}
    if page_index is not None:
        row["page"] = page_index
    if pages is not None:
        row["pages"] = pages
    timing_log.append(row)
