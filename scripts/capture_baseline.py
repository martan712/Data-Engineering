"""Capture the pre-finalization result and paper inventory.

This is intentionally a development snapshot, not the final artifact catalog.
It preserves historical schema examples without modifying source result files.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO_ROOT / "results"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_label(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _run(*command: str) -> str:
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout.strip()


def _stage(path: Path) -> str:
    match = re.search(r"stage\d+_[a-z0-9]+", path.name)
    if match:
        return match.group(0)
    parts = path.relative_to(RESULT_ROOT).parts
    return parts[1] if len(parts) > 2 else "unclassified"


def _result_inventory() -> tuple[list[dict[str, Any]], dict[tuple[str, ...], list[Path]]]:
    records: list[dict[str, Any]] = []
    shapes: dict[tuple[str, ...], list[Path]] = defaultdict(list)
    for path in sorted((RESULT_ROOT / "json").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        keys = tuple(sorted(payload))
        shapes[keys].append(path)
        records.append(
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
                "stage": _stage(path),
                "top_level_keys": list(keys),
            }
        )
    return records, shapes


def _figure_inventory() -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
            "stage": _stage(path),
        }
        for path in sorted((RESULT_ROOT / "figures").rglob("*"))
        if path.is_file()
    ]


def _copy_shape_fixtures(
    shapes: dict[tuple[str, ...], list[Path]], fixture_dir: Path
) -> list[dict[str, Any]]:
    if fixture_dir.exists():
        shutil.rmtree(fixture_dir)
    fixture_dir.mkdir(parents=True)
    fixtures = []
    for number, (keys, paths) in enumerate(
        sorted(shapes.items(), key=lambda item: item[0]), start=1
    ):
        source = min(paths, key=lambda path: (path.stat().st_size, path.name))
        destination = fixture_dir / f"shape_{number:02d}__{source.name}"
        shutil.copyfile(source, destination)
        fixtures.append(
            {
                "shape_id": f"historical-{number:02d}",
                "source": str(source.relative_to(REPO_ROOT)),
                "fixture": _path_label(destination),
                "source_sha256": _sha256(source),
                "top_level_keys": list(keys),
                "matching_file_count": len(paths),
            }
        )
    return fixtures


def _paper_inventory() -> list[dict[str, Any]]:
    source = (REPO_ROOT / "report" / "main.tex").read_text(encoding="utf-8")
    assets = []
    pattern = re.compile(
        r"\\begin\{(?P<kind>table\*?|figure\*?)\}.*?"
        r"\\end\{(?P=kind)\}",
        re.DOTALL,
    )
    for block in pattern.finditer(source):
        text = block.group(0)
        labels = re.findall(r"\\label\{([^}]+)\}", text)
        images = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text)
        inputs = re.findall(r"\\input\{([^}]+)\}", text)
        line = source.count("\n", 0, block.start()) + 1
        assets.append(
            {
                "kind": block.group("kind"),
                "line": line,
                "labels": labels,
                "image_sources": images,
                "generated_inputs": inputs,
                "numeric_body": "generated" if inputs else "handwritten_or_figure_only",
            }
        )
    return assets


def _driver_inventory() -> list[dict[str, Any]]:
    drivers = []
    for path in sorted((REPO_ROOT / "experiments").rglob("*.py")):
        if path.name == "__init__.py":
            continue
        source = path.read_text(encoding="utf-8")
        if "results/json" not in source and "RESULTS_JSON" not in source:
            continue
        inputs = []
        if "load_dataset" in source:
            inputs.append("data/embeddings/<dataset>.npz")
        if "load_eval_queries" in source:
            inputs.append("data/embeddings/<dataset>_test_queries.npz where required")
        if "load_qrels" in source:
            inputs.append("data/qrels/<dataset>.tsv")
        if "FaissIVF" in source or "FAISS_CACHE" in source:
            inputs.append("data/faiss_indexes/<dataset>.faiss")
        if "PLAID" in source or "plaid" in path.name:
            inputs.append("data/plaid_indexes/<dataset>/")
        if "PDX" in source or "pdx" in source:
            inputs.append("extern/PDX or extern/PDX-sigmod")
        drivers.append(
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "expected_inputs": sorted(set(inputs)),
            }
        )
    return drivers


def _environment(revision: str, worktree_state: str) -> dict[str, Any]:
    packages = sorted(
        {
            (distribution.metadata.get("Name") or "unknown", distribution.version)
            for distribution in importlib.metadata.distributions()
        },
        key=lambda row: row[0].lower(),
    )
    return {
        "captured_baseline_revision": revision,
        "captured_baseline_worktree_state": worktree_state,
        "python": _run(str(Path(sys.executable)), "--version"),
        "compiler": _run("g++", "--version").splitlines()[0],
        "kernel": _run("uname", "-a"),
        "cpu_summary": {
            line.split(":", 1)[0].strip(): line.split(":", 1)[1].strip()
            for line in _run("lscpu").splitlines()
            if ":" in line
            and line.split(":", 1)[0].strip()
            in {"Architecture", "CPU(s)", "Model name", "Thread(s) per core", "Core(s) per socket"}
        },
        "submodules": _run("git", "submodule", "status", "--recursive").splitlines(),
        "installed_packages": [f"{name}=={version}" for name, version in packages],
        "native_flags": {
            "per_document_oracle": "-O3 -march=native -DNDEBUG -std=c++20 -shared -fPIC",
            "wide_block_maxsim_bond": "-O3 -march=native -DNDEBUG -std=c++20 -shared -fPIC",
            "fused_panel_maxsim": "-O3 -march=native -DNDEBUG -std=c++20 -fPIC -fopenmp",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--worktree-state", required=True)
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "artifacts" / "baseline"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    results, shapes = _result_inventory()
    fixtures = _copy_shape_fixtures(shapes, output / "fixtures")
    inventory = {
        "snapshot_kind": "pre_contract_development_inventory",
        "results": results,
        "figures": _figure_inventory(),
        "historical_shapes": fixtures,
        "paper_assets": _paper_inventory(),
        "result_drivers": _driver_inventory(),
    }
    (output / "inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "environment.json").write_text(
        json.dumps(_environment(args.revision, args.worktree_state), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
