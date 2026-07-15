"""Versioned result envelopes, compatibility readers, and artifact validation."""

from bondmaxsim.results.io import atomic_write_envelope, deterministic_result_name
from bondmaxsim.results.frozen_inputs import (
    FROZEN_INPUT_HASH_FIELDS,
    validate_frozen_inputs,
)
from bondmaxsim.results.migrations import load_result
from bondmaxsim.results.models import (
    CURRENT_SCHEMA_NAME,
    CURRENT_SCHEMA_VERSION,
    ExperimentResultEnvelope,
    ResultValidationError,
)

__all__ = [
    "CURRENT_SCHEMA_NAME",
    "CURRENT_SCHEMA_VERSION",
    "ExperimentResultEnvelope",
    "FROZEN_INPUT_HASH_FIELDS",
    "ResultValidationError",
    "atomic_write_envelope",
    "deterministic_result_name",
    "load_result",
    "validate_frozen_inputs",
]
