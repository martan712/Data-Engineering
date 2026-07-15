"""Declarative experiment arms and workload-level experiment specifications."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
EXACTNESS_CLASSES = {"exact_safe", "approximate", "reference"}
COMPARISON_SCOPES = {
    "system_cap",
    "equal_work_reranking",
    "probe_frontier",
    "uncapped_reference",
}


@dataclass(frozen=True)
class ArmSpec:
    arm_id: str
    method_family: str
    operation: Callable[[], Any] = field(repr=False, compare=False)
    parameters: Mapping[str, Any] = field(default_factory=dict)
    exactness: str = "approximate"
    comparison_scope: str = "system_cap"
    validator: Callable[[Any], Any] | None = field(default=None, repr=False, compare=False)
    timer_scope_id: str = "scoring"

    def __post_init__(self) -> None:
        if not _ID.fullmatch(self.arm_id):
            raise ValueError("arm_id must be a stable lowercase ID")
        if not _ID.fullmatch(self.method_family):
            raise ValueError("method_family must be a stable lowercase ID")
        if not callable(self.operation):
            raise TypeError("operation must be callable")
        if self.validator is not None and not callable(self.validator):
            raise TypeError("validator must be callable")
        if self.exactness not in EXACTNESS_CLASSES:
            raise ValueError(f"invalid exactness class {self.exactness!r}")
        if self.comparison_scope not in COMPARISON_SCOPES:
            raise ValueError(f"invalid comparison scope {self.comparison_scope!r}")
        if not isinstance(self.timer_scope_id, str) or not self.timer_scope_id:
            raise ValueError("timer_scope_id must be non-empty")

    def configuration(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "method_family": self.method_family,
            "parameters": dict(self.parameters),
            "exactness": self.exactness,
            "comparison_scope": self.comparison_scope,
            "timer_scope_id": self.timer_scope_id,
        }


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    artifact_id: str
    dataset_id: str
    workload_id: str
    arms: tuple[ArmSpec, ...]
    command: str
    baseline_arm_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    workload_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (
            ("experiment_id", self.experiment_id),
            ("artifact_id", self.artifact_id),
            ("dataset_id", self.dataset_id),
            ("workload_id", self.workload_id),
        ):
            if not _ID.fullmatch(value):
                raise ValueError(f"{name} must be a stable lowercase ID")
        if not self.arms:
            raise ValueError("an experiment needs at least one arm")
        arm_ids = [arm.arm_id for arm in self.arms]
        if len(set(arm_ids)) != len(arm_ids):
            raise ValueError("arm IDs must be unique")
        if self.baseline_arm_id is not None and self.baseline_arm_id not in arm_ids:
            raise ValueError("baseline_arm_id does not name an arm")
        if not self.command:
            raise ValueError("command must be non-empty")
        if self.workload_metadata:
            if self.workload_metadata.get("workload_id") != self.workload_id:
                raise ValueError("workload metadata ID does not match ExperimentSpec")
            if self.workload_metadata.get("dataset") != self.dataset_id:
                raise ValueError("workload metadata dataset does not match ExperimentSpec")

    def method_configuration(self) -> dict[str, Any]:
        return {
            "arms": [arm.configuration() for arm in self.arms],
            "baseline_arm_id": self.baseline_arm_id,
            **dict(self.metadata),
        }
