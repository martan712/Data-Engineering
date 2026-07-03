"""RunConfig: configuration for a single testbed Runner experiment run."""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RunConfig:
    """Configuration for a single Runner experiment run."""

    dataset: str
    method: str
    dimension_order: str
    threshold_policy: str
    k: int = 10
    shrink: float = 1.0
    """Recall knob: 1.0 = exact-safe (shrink=1), <1.0 = approximate."""
    checkpoints: Optional[tuple[int, ...]] = None
    """Fused-BOND bound-checkpoint dims (R3 e08 ablation); None = kernel
    default {32, 64}.  Ignored by the wide-block and oracle kernels."""
    candidate_budget: Optional[int] = None
    thread_count: int = 1
    machine: str = field(default_factory=platform.node)
    """Hostname for fair-comparison tracking (override for cross-machine runs)."""
    os: str = "linux"
    notes: Optional[str] = None
