"""Tests for bondmaxsim.ordering.orders."""

import numpy as np
import pytest

from bondmaxsim.ordering.orders import (
    natural_order,
    bond_order,
    pca_order,
    bond_q2_var_order,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

D = 32  # dimension for all tests


def is_valid_permutation(order: np.ndarray, D: int) -> bool:
    """Check that order is a permutation of 0..D-1."""
    return (np.sort(order) == np.arange(D)).all()


def make_query(m: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((m, D)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q


def make_mu_var(seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    mu = rng.standard_normal(D).astype(np.float32)
    var = np.abs(rng.standard_normal(D)).astype(np.float32)
    return mu, var


def orthogonal_rotation(seed: int = 123) -> np.ndarray:
    """Build an orthogonal D×D matrix via QR of a random Gaussian."""
    g = np.random.default_rng(seed).standard_normal((D, D))
    Q, _ = np.linalg.qr(g)
    return Q.astype(np.float32)


# ---------------------------------------------------------------------------
# natural_order
# ---------------------------------------------------------------------------

def test_natural_order_is_identity():
    q = make_query()
    order = natural_order(q)
    assert order.dtype == np.int64
    assert is_valid_permutation(order, D)
    np.testing.assert_array_equal(order, np.arange(D))


# ---------------------------------------------------------------------------
# bond_order
# ---------------------------------------------------------------------------

def test_bond_order_is_valid_permutation():
    q = make_query()
    mu, _ = make_mu_var()
    order = bond_order(q, mu)
    assert order.dtype == np.int64
    assert is_valid_permutation(order, D)


def test_bond_order_top_partition_is_top_importance_dims():
    """The first floor(D*top_frac) entries must be exactly the top-importance dims (as a set)."""
    q = make_query()
    mu, _ = make_mu_var()
    top_frac = 0.25
    tp = int(np.floor(D * top_frac))

    imp = ((q - mu) ** 2).sum(axis=0)
    expected_hot = set(np.argsort(imp)[::-1][:tp].tolist())

    order = bond_order(q, mu, top_frac=top_frac)
    got_hot = set(order[:tp].tolist())
    assert got_hot == expected_hot, (
        f"hot partition mismatch: got {sorted(got_hot)}, expected {sorted(expected_hot)}"
    )


def test_bond_order_top_fraction_sweep():
    """bond_order is valid for top_frac = 0, 0.25, 0.5, 1.0."""
    q = make_query()
    mu, _ = make_mu_var()
    for frac in [0.0, 0.125, 0.25, 0.5, 1.0]:
        order = bond_order(q, mu, top_frac=frac)
        assert is_valid_permutation(order, D), f"Invalid permutation for top_frac={frac}"


# ---------------------------------------------------------------------------
# pca_order
# ---------------------------------------------------------------------------

def test_pca_order_returns_valid_permutation():
    q = make_query()
    R = orthogonal_rotation()
    rotated_q, order = pca_order(q, R)
    assert order.dtype == np.int64
    assert is_valid_permutation(order, D)
    np.testing.assert_array_equal(order, np.arange(D))


def test_pca_order_rotated_query_dtype():
    q = make_query()
    R = orthogonal_rotation()
    rotated_q, order = pca_order(q, R)
    assert rotated_q.dtype == np.float32
    assert rotated_q.shape == q.shape


def test_pca_order_preserves_inner_products():
    """For an orthogonal R: (q @ R) @ (d @ R).T ≈ q @ d.T."""
    rng = np.random.default_rng(5)
    q = make_query(m=4)
    d = rng.standard_normal((6, D)).astype(np.float32)
    R = orthogonal_rotation()

    rotated_q, _ = pca_order(q, R)
    rotated_d = (d @ R).astype(np.float32)

    orig_sims = q @ d.T           # [4, 6]
    rot_sims = rotated_q @ rotated_d.T  # [4, 6]
    np.testing.assert_allclose(rot_sims, orig_sims, rtol=1e-4, atol=1e-5,
                               err_msg="pca_order does not preserve inner products")


def test_pca_order_preserves_unit_norms():
    """An orthogonal rotation preserves L2 norms."""
    q = make_query()
    q /= np.linalg.norm(q, axis=1, keepdims=True)  # ensure unit norm
    R = orthogonal_rotation()
    rotated_q, _ = pca_order(q, R)
    orig_norms = np.linalg.norm(q, axis=1)
    rot_norms = np.linalg.norm(rotated_q, axis=1)
    np.testing.assert_allclose(rot_norms, orig_norms, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# bond_q2_var_order
# ---------------------------------------------------------------------------

def test_bond_q2_var_order_is_valid_permutation():
    q = make_query()
    mu, var = make_mu_var()
    order = bond_q2_var_order(q, mu, var)
    assert order.dtype == np.int64
    assert is_valid_permutation(order, D)


def test_bond_q2_var_order_top_partition_correct():
    """The hot partition must match the top-importance dims for the q²·(mu²+var) signal."""
    q = make_query()
    mu, var = make_mu_var()
    top_frac = 0.25
    tp = int(np.floor(D * top_frac))

    imp = (q ** 2).sum(axis=0) * (mu ** 2 + var)
    expected_hot = set(np.argsort(imp)[::-1][:tp].tolist())

    order = bond_q2_var_order(q, mu, var, top_frac=top_frac)
    got_hot = set(order[:tp].tolist())
    assert got_hot == expected_hot, (
        f"hot partition mismatch: got {sorted(got_hot)}, expected {sorted(expected_hot)}"
    )


def test_bond_q2_var_order_various_top_fracs():
    q = make_query()
    mu, var = make_mu_var()
    for frac in [0.0, 0.25, 0.5, 1.0]:
        order = bond_q2_var_order(q, mu, var, top_frac=frac)
        assert is_valid_permutation(order, D), f"Invalid permutation for top_frac={frac}"
