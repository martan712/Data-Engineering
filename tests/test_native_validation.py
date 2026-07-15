from __future__ import annotations

import ctypes

import numpy as np
import pytest

from bondmaxsim.data.packing import (
    DEFAULT_FETCH,
    build_qcum,
    pack_corpus,
    pack_corpus_panels,
    pack_corpus_wide,
)
from bondmaxsim.kernels._ctypes_util import (
    NativeInputError,
    PackedCorpusDimMajor,
    PackedCorpusPanels,
    PackedCorpusWide,
    csz,
    fp,
    lp,
    up,
)
from bondmaxsim.kernels.fused_panel import (
    load_fused_panel_kernel,
    run_fused_panel_bond,
    run_fused_panel_brute,
    run_fused_panel_brute_validated,
)
from bondmaxsim.kernels.per_document import (
    load_per_document_oracle,
    run_accounting,
    run_full,
    run_full_validated,
)
from bondmaxsim.kernels.wide_block import (
    load_wide_block_kernel,
    run_wide_block_accounting,
    run_wide_block_accounting_validated,
)
from bondmaxsim.testbed.packing_cache import PackingCache

NATIVE_ERROR = np.iinfo(np.uint64).max


@pytest.fixture(scope="module")
def inputs():
    rng = np.random.default_rng(2025)
    docs = []
    for length in (2, 5, 3):
        values = rng.standard_normal((length, 4)).astype(np.float32)
        values /= np.linalg.norm(values, axis=1, keepdims=True)
        docs.append(values)
    tokens = np.concatenate(docs)
    starts = np.array([0, 2, 7], dtype=np.int64)
    query = rng.standard_normal((3, 4)).astype(np.float32)
    query /= np.linalg.norm(query, axis=1, keepdims=True)
    order = np.array([2, 0, 3, 1], dtype=np.uint32)
    qcum = build_qcum(query, order)
    return {
        "query": query,
        "order": order,
        "qcum": qcum,
        "per": pack_corpus(docs),
        "wide": pack_corpus_wide(tokens, starts, target_group_tokens=6),
        "panels": pack_corpus_panels(tokens, starts, target_group_tokens=32)[:4],
    }


def test_validated_corpus_objects_normalize_and_derive_counts(inputs):
    per = PackedCorpusDimMajor(*inputs["per"], 4)
    wide = PackedCorpusWide(*inputs["wide"], 4)
    panels = PackedCorpusPanels(*inputs["panels"], 4)

    assert (per.n_documents, wide.n_documents, wide.n_groups) == (3, 3, 3)
    assert panels.n_documents == 3
    assert panels.panel_tokens == 16
    assert per.data.dtype == np.float32 and per.doc_offsets.dtype == np.uint64
    assert not per.data.flags.writeable and not wide.group_offsets.flags.writeable


@pytest.mark.parametrize(
    "offsets, message",
    [
        (np.array([1, 2, 5, 10]), "start at zero"),
        (np.array([0, 5, 4, 10]), "monotone"),
        (np.array([0, 2, 5, 11]), "size disagrees"),
    ],
)
def test_dim_major_rejects_invalid_offsets(inputs, offsets, message):
    with pytest.raises(NativeInputError, match=message):
        PackedCorpusDimMajor(inputs["per"][0], offsets, 4)


def test_grouped_corpus_rejects_inconsistent_metadata(inputs):
    data, groups, docs, group_docs = inputs["wide"]
    bad_group_docs = group_docs.copy()
    bad_group_docs[1] = 0
    with pytest.raises(NativeInputError, match="boundaries"):
        PackedCorpusWide(data, groups, docs, bad_group_docs, 4)

    panel_data, panel_groups, panel_docs, panel_group_docs = inputs["panels"]
    bad_panel_docs = panel_docs.copy()
    bad_panel_docs[1] += 1
    with pytest.raises(NativeInputError, match="aligned"):
        PackedCorpusPanels(
            panel_data, panel_groups, bad_panel_docs, panel_group_docs, 4
        )


class _NeverNative:
    def __getattr__(self, name):
        def fail(*args, **kwargs):
            raise AssertionError(f"unsafe native call reached: {name}")

        return fail


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda q, o, c: (q[:, :3], o, c), "size disagrees"),
        (lambda q, o, c: (np.where(np.eye(*q.shape), np.nan, q), o, c), "non-finite"),
        (lambda q, o, c: (q, np.array([0, 0, 2, 3]), c), "permutation"),
        (lambda q, o, c: (q, o, c[:, :-1]), "Qcum must have shape"),
        (lambda q, o, c: (q, o, c + np.float32(0.25)), "start with a zero"),
    ],
)
def test_per_document_rejects_query_metadata_before_native(inputs, mutation, message):
    flat, offsets = inputs["per"]
    query, order, qcum = mutation(
        inputs["query"].copy(), inputs["order"].copy(), inputs["qcum"].copy()
    )
    with pytest.raises(NativeInputError, match=message):
        run_accounting(_NeverNative(), flat, offsets, query, order, qcum, 1.0, 2)


def test_qcum_values_must_match_query_and_order(inputs):
    flat, offsets = inputs["per"]
    bad = inputs["qcum"].copy()
    bad[:, 2:] += np.float32(0.01)
    with pytest.raises(NativeInputError, match="inconsistent"):
        run_accounting(
            _NeverNative(), flat, offsets, inputs["query"], inputs["order"], bad, 1.0, 2
        )


@pytest.mark.parametrize("k", [0, 4, True])
def test_all_wrapper_families_reject_infeasible_k(inputs, k):
    query, order, qcum = inputs["query"], inputs["order"], inputs["qcum"]
    flat, offsets = inputs["per"]
    with pytest.raises(NativeInputError, match="K"):
        run_full(_NeverNative(), flat, offsets, query, k)

    with pytest.raises(NativeInputError, match="K"):
        run_wide_block_accounting(
            _NeverNative(), *inputs["wide"], query, order, qcum, 1.0, -np.inf, k
        )

    with pytest.raises(NativeInputError, match="K"):
        run_fused_panel_brute(_NeverNative(), *inputs["panels"], query, k)


def test_fused_controls_are_checked_before_native(inputs):
    query, order, qcum = inputs["query"], inputs["order"], inputs["qcum"]
    with pytest.raises(NativeInputError, match="eight"):
        run_fused_panel_bond(
            _NeverNative(), *inputs["panels"], query, order, qcum,
            1.0, -np.inf, 2, checkpoints=np.arange(9),
        )
    with pytest.raises(NativeInputError, match="C int"):
        run_fused_panel_brute(
            _NeverNative(), *inputs["panels"], query, 2, n_threads=2**31
        )


def _load_or_skip(loader, name):
    try:
        return loader()
    except FileNotFoundError:
        pytest.skip(f"{name} native library is not built")


def test_reusable_validated_corpora_match_raw_wrappers(inputs):
    query, order, qcum = inputs["query"], inputs["order"], inputs["qcum"]

    per_lib = _load_or_skip(load_per_document_oracle, "per-document")
    per = PackedCorpusDimMajor(*inputs["per"], 4)
    raw = run_full(per_lib, *inputs["per"], query, 2)
    reused = run_full_validated(per_lib, per, query, 2)
    np.testing.assert_array_equal(raw[0], reused[0])
    np.testing.assert_allclose(raw[1], reused[1])

    wide_lib = _load_or_skip(load_wide_block_kernel, "wide-block")
    wide = PackedCorpusWide(*inputs["wide"], 4)
    raw = run_wide_block_accounting(
        wide_lib, *inputs["wide"], query, order, qcum, 1.0, -np.inf, 2
    )
    reused = run_wide_block_accounting_validated(
        wide_lib, wide, query, order, qcum, 1.0, -np.inf, 2
    )
    np.testing.assert_array_equal(raw[0], reused[0])
    np.testing.assert_allclose(raw[1], reused[1])

    fused_lib = _load_or_skip(load_fused_panel_kernel, "fused-panel")
    panels = PackedCorpusPanels(*inputs["panels"], 4)
    raw = run_fused_panel_brute(fused_lib, *inputs["panels"], query, 2)
    reused = run_fused_panel_brute_validated(fused_lib, panels, query, 2)
    np.testing.assert_array_equal(raw[0], reused[0])
    np.testing.assert_allclose(raw[1], reused[1])


def test_packing_cache_reuses_validated_corpora(inputs):
    per_data, per_offsets = inputs["per"]
    lengths = np.diff(per_offsets).astype(np.int64)
    docs = []
    cursor = 0
    for length in lengths:
        block = per_data[cursor * 4:(cursor + length) * 4].reshape(4, length).T
        docs.append(block)
        cursor += length
    tokens = np.ascontiguousarray(np.concatenate(docs), dtype=np.float32)
    starts = np.array([0, lengths[0], lengths[0] + lengths[1]], dtype=np.int64)
    cache = PackingCache(tokens, starts)

    per_a, _, _ = cache.dispatch_order_corpus(inputs["query"], "natural")
    per_b, _, _ = cache.dispatch_order_corpus(inputs["query"], "bond")
    wide_a, _, _ = cache.dispatch_order_wide_corpus(inputs["query"], "natural")
    wide_b, _, _ = cache.dispatch_order_wide_corpus(inputs["query"], "bond")
    panel_a, _, _ = cache.dispatch_order_panel_corpus(inputs["query"], "natural")
    panel_b, _, _ = cache.dispatch_order_panel_corpus(inputs["query"], "bond")

    assert per_a is per_b
    assert wide_a is wide_b
    assert panel_a is panel_b

def test_cpp_entry_points_return_error_for_malformed_metadata(inputs):
    query, order, qcum = inputs["query"], inputs["order"], inputs["qcum"]
    ids = np.empty(2, dtype=np.uint32)
    scores = np.empty(2, dtype=np.float32)
    stats = np.zeros(3, dtype=np.uint64)
    fetch = np.ascontiguousarray(DEFAULT_FETCH, dtype=np.uint32)

    per_lib = _load_or_skip(load_per_document_oracle, "per-document")
    per_data, per_offsets = inputs["per"]
    bad_order = np.array([0, 0, 2, 3], dtype=np.uint32)
    status = per_lib.maxsim_knn_accounting(
        fp(per_data), lp(per_offsets), csz(3), fp(query), csz(3), csz(4),
        up(bad_order), up(fetch), csz(len(fetch)), fp(qcum), ctypes.c_float(1.0),
        csz(2), up(ids), fp(scores), lp(stats),
    )
    assert status == NATIVE_ERROR

    wide_lib = _load_or_skip(load_wide_block_kernel, "wide-block")
    wide_data, wide_groups, wide_docs, wide_group_docs = inputs["wide"]
    bad_group_docs = wide_group_docs.copy()
    bad_group_docs[-1] -= 1
    status = wide_lib.wide_block_maxsim_accounting(
        fp(wide_data), lp(wide_groups), csz(len(wide_groups) - 1),
        lp(wide_docs), csz(3), lp(bad_group_docs), fp(query), csz(3), csz(4),
        up(order), up(fetch), csz(len(fetch)), fp(qcum), ctypes.c_float(1.0),
        ctypes.c_float(-np.inf), csz(2), up(ids), fp(scores), lp(stats), None, None,
    )
    assert status == NATIVE_ERROR

    fused_lib = _load_or_skip(load_fused_panel_kernel, "fused-panel")
    panel_data, panel_groups, panel_docs, panel_group_docs = inputs["panels"]
    bad_panel_docs = panel_docs.copy()
    bad_panel_docs[1] += 1
    status = fused_lib.fused_panel_maxsim_brute(
        fp(panel_data), lp(panel_groups), csz(len(panel_groups) - 1),
        lp(bad_panel_docs), lp(panel_group_docs), fp(query), csz(3), csz(4),
        csz(2), ctypes.c_int(1), up(ids), fp(scores),
    )
    assert status == NATIVE_ERROR
