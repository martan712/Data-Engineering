"""Generate and validate the deterministic offline reproduction fixture."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


FIXTURE_SCHEMA = "bondmaxsim.fixture-manifest"
FIXTURE_VERSION = "1.0.0"
FIXTURE_SEED = 20260715
DIMENSION = 8
DOCUMENT_LENGTHS = (2, 3, 1, 4, 2, 3)
QUERY_LENGTHS = (1, 2, 3, 2)


def _normalized_random(rng: np.random.Generator, rows: int) -> np.ndarray:
    values = rng.standard_normal((rows, DIMENSION), dtype=np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    return np.ascontiguousarray(values, dtype=np.float32)


def _starts(lengths: tuple[int, ...]) -> np.ndarray:
    return np.asarray([0, *np.cumsum(lengths[:-1])], dtype=np.int64)


def fixture_arrays() -> dict[str, np.ndarray]:
    """Return the canonical fixture arrays."""
    rng = np.random.default_rng(FIXTURE_SEED)
    return {
        "doc_values": _normalized_random(rng, sum(DOCUMENT_LENGTHS)),
        "doc_starts": _starts(DOCUMENT_LENGTHS),
        "query_values": _normalized_random(rng, sum(QUERY_LENGTHS)),
        "query_starts": _starts(QUERY_LENGTHS),
    }


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, array in sorted(arrays.items()):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, _npy_bytes(array))
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_bytes(path: Path, contents: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(contents)
    temporary.replace(path)


def generate_fixture(output: Path) -> dict[str, Any]:
    """Create the fixture and return its manifest."""
    output.mkdir(parents=True, exist_ok=True)
    arrays = fixture_arrays()
    npz_path = output / "small.npz"
    _write_deterministic_npz(npz_path, arrays)

    ids = {
        "corpus_ids": [f"d{i}" for i in range(len(DOCUMENT_LENGTHS))],
        "dataset": "synthetic-small-v1",
        "query_ids": [f"q{i}" for i in range(len(QUERY_LENGTHS))],
    }
    ids_path = output / "ids.json"
    _atomic_bytes(ids_path, _json_bytes(ids))

    qrels_lines = ["query-id\tcorpus-id\tscore", "q0\td0\t2", "q1\td2\t1", "q2\td4\t1", "q3\td5\t2"]
    qrels_path = output / "qrels.tsv"
    _atomic_bytes(qrels_path, ("\n".join(qrels_lines) + "\n").encode("utf-8"))

    manifest = {
        "schema_name": FIXTURE_SCHEMA,
        "schema_version": FIXTURE_VERSION,
        "generator": "numpy-pcg64",
        "seed": FIXTURE_SEED,
        "dimension": DIMENSION,
        "document_lengths": list(DOCUMENT_LENGTHS),
        "query_lengths": list(QUERY_LENGTHS),
        "counts": {
            "documents": len(DOCUMENT_LENGTHS),
            "document_tokens": sum(DOCUMENT_LENGTHS),
            "queries": len(QUERY_LENGTHS),
            "query_tokens": sum(QUERY_LENGTHS),
        },
        "files": {
            "small.npz": _sha256(npz_path),
            "ids.json": _sha256(ids_path),
            "qrels.tsv": _sha256(qrels_path),
        },
    }
    _atomic_bytes(output / "manifest.json", _json_bytes(manifest))
    return manifest


def validate_fixture(output: Path) -> dict[str, Any]:
    """Validate file hashes, arrays, offsets, finiteness, and unit norms."""
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_name") != FIXTURE_SCHEMA:
        raise ValueError("unknown fixture schema")
    for filename, expected in manifest["files"].items():
        actual = _sha256(output / filename)
        if actual != expected:
            raise ValueError(f"fixture checksum mismatch for {filename}")

    with np.load(output / "small.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    expected = fixture_arrays()
    if set(arrays) != set(expected):
        raise ValueError("fixture array set is incomplete")
    for name in expected:
        if not np.array_equal(arrays[name], expected[name]):
            raise ValueError(f"fixture array differs: {name}")
    for values_name in ("doc_values", "query_values"):
        values = arrays[values_name]
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite fixture values: {values_name}")
        if not np.allclose(np.linalg.norm(values, axis=1), 1.0, atol=2e-6, rtol=0):
            raise ValueError(f"non-unit fixture values: {values_name}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    arguments = parser.parse_args()
    if (arguments.output is None) == (arguments.verify is None):
        parser.error("choose exactly one of --output or --verify")
    if arguments.output is not None:
        manifest = generate_fixture(arguments.output)
        print(json.dumps(manifest["counts"], sort_keys=True))
    else:
        manifest = validate_fixture(arguments.verify)
        print(json.dumps(manifest["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
