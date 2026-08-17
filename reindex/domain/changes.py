"""Filesystem change records coalesced by the watcher."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ChangeOp = Literal["upsert", "delete"]


@dataclass(frozen=True, slots=True)
class FsChange:
    """One document path to upsert into or delete from the index."""

    op: ChangeOp
    path: str
    is_prefix: bool = False
