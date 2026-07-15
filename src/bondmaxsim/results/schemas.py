"""Current schema registry and historical reader names."""

from __future__ import annotations

from dataclasses import dataclass

from bondmaxsim.results.models import CURRENT_SCHEMA_NAME, CURRENT_SCHEMA_VERSION, PAYLOAD_KINDS


@dataclass(frozen=True)
class SchemaRegistration:
    name: str
    version: str
    resource: str


CURRENT_SCHEMAS = {
    CURRENT_SCHEMA_NAME: SchemaRegistration(
        CURRENT_SCHEMA_NAME,
        CURRENT_SCHEMA_VERSION,
        "artifacts/schemas/result-envelope-1.0.0.json",
    ),
    **{
        f"bondmaxsim.payload.{kind}": SchemaRegistration(
            f"bondmaxsim.payload.{kind}",
            CURRENT_SCHEMA_VERSION,
            "artifacts/schemas/payloads-1.0.0.json",
        )
        for kind in PAYLOAD_KINDS
    },
}

HISTORICAL_READERS = frozenset(
    {
        "historical.result-record.v1",
        "historical.stage3.bound-slack.v1",
        "historical.stage3.e02_pruning_rate.v1",
        "historical.stage3.e03_order_ablation.v1",
        "historical.stage3.e04_exact_safe_pruning.v1",
        "historical.stage3.e05_approximate_recall_sweep.v1",
        "historical.stage3.e06_threshold_policy_ablation.v1",
        "historical.stage3.e07_cache_layout_sensitivity.v1",
        "historical.stage3.e08_checkpoint_ablation.v1",
        "historical.stage3.e09_bound_tightness_ablation.v1",
        "historical.stage3.r12c_interleaved_exact_safe.v1",
        "historical.stage4.fixed-candidate-arms.v1",
        "historical.stage4.seeded-tau-recovery.v1",
        "historical.stage4.partitioned-frontier.v1",
        "historical.stage5.ir-evaluation.v1",
        "historical.stage5.paired-significance.v1",
        "historical.stage5.bm25.v1",
    }
)
