"""Tests for bondmaxsim.oracle.normalization."""

import numpy as np
import pytest

from bondmaxsim.oracle.normalization import assert_unit_norm, check_unit_norm


def make_unit_matrix(n: int = 20, D: int = 16, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, D)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def test_assert_unit_norm_passes_for_unit_matrix():
    tokens = make_unit_matrix()
    # Should not raise
    assert_unit_norm(tokens)


def test_assert_unit_norm_raises_for_scaled_row():
    tokens = make_unit_matrix()
    tokens[5] *= 1.5  # norm becomes 1.5
    with pytest.raises(AssertionError) as exc_info:
        assert_unit_norm(tokens)
    msg = str(exc_info.value)
    assert "n_violating" in msg
    assert "min_norm" in msg or "max_norm" in msg


def test_assert_unit_norm_raises_message_contains_count():
    tokens = make_unit_matrix(n=10)
    # Scale two rows to have norm != 1
    tokens[0] *= 2.0
    tokens[3] *= 0.5
    with pytest.raises(AssertionError) as exc_info:
        assert_unit_norm(tokens)
    msg = str(exc_info.value)
    # Should mention that 2 tokens violate
    assert "2" in msg


def test_check_unit_norm_stats_correct():
    tokens = make_unit_matrix(n=10)
    stats = check_unit_norm(tokens)
    assert set(stats.keys()) == {"min_norm", "max_norm", "mean_norm", "n_violating"}
    assert stats["n_violating"] == 0
    assert abs(stats["min_norm"] - 1.0) < 1e-3
    assert abs(stats["max_norm"] - 1.0) < 1e-3
    assert abs(stats["mean_norm"] - 1.0) < 1e-3


def test_check_unit_norm_detects_violation():
    tokens = make_unit_matrix(n=10)
    tokens[2] *= 1.5
    stats = check_unit_norm(tokens)
    assert stats["n_violating"] == 1
    assert stats["max_norm"] > 1.0 + 1e-4


def test_check_unit_norm_custom_tol():
    tokens = make_unit_matrix(n=5)
    # Scale by 1.05 — within default tol=1e-4? No (1.05 > 1+1e-4).
    tokens[0] *= 1.05
    stats_tight = check_unit_norm(tokens, tol=1e-4)
    assert stats_tight["n_violating"] == 1
    # With tol=0.1 it should pass (|1.05 - 1| = 0.05 < 0.1)
    stats_loose = check_unit_norm(tokens, tol=0.1)
    assert stats_loose["n_violating"] == 0
