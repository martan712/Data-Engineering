"""Render the report's IR-evaluation table (tab:ir) from the e01/e03 JSONs.

Single responsibility: turn the mt runs of stage5 e01 (vector arms) and e03
(BM25 reference) into the LaTeX rows of report/main.tex Table \ref{tab:ir},
so the table is regenerated rather than hand-edited after a re-run.  Prints
the tabular body to stdout; the surrounding table environment and caption
stay in main.tex.

Arms shown per dataset (the report's best-of-family selection):
  dense_fused, openblas, bond_exact_safe, partitioned@{16,32},
  faiss@{100 on scidocs only, 1000}, plaid@1000, bm25.

Usage:
    uv run python -m experiments.stage5_corect.render_ir_table
"""
from __future__ import annotations

import json

from bondmaxsim.config import REPO_ROOT

RESULTS_JSON = REPO_ROOT / "results" / "json"
DATASETS = ["scifact", "nfcorpus", "arguana", "scidocs"]
PRETTY = {"scifact": "SciFact", "nfcorpus": "NFCorpus",
          "arguana": "ArguAna", "scidocs": "SciDocs"}
LABELS = {
    "dense_fused": "dense fused (exact)",
    "openblas": "OpenBLAS (exact)",
    "bond_exact_safe": "exact-safe BOND",
    "partitioned@16": "partitioned, nprobe 16",
    "partitioned@32": "partitioned, nprobe 32",
    "faiss@100": "FAISS-IVF ($B{=}100$)",
    "faiss@1000": "FAISS-IVF ($B{=}1000$)",
    "plaid@1000": "PLAID ($B{=}1000$)",
}


def rows_for(dataset: str) -> list[str]:
    e01 = json.loads((RESULTS_JSON /
                      f"stage5_corect_e01_ir_evaluation_{dataset}.json"
                      ).read_text())
    arms = {a["arm"]: a for a in e01["runs"]["mt"]["arms"]}
    order = ["dense_fused", "openblas", "bond_exact_safe",
             "partitioned@16", "partitioned@32"]
    if dataset == "scidocs":
        order += ["faiss@100"]
    order += ["faiss@1000", "plaid@1000"]

    fastest = min(arms[k]["ms_per_query"] for k in order)
    out = []
    for key in order:
        a = arms[key]
        ms = a["ms_per_query"]
        ms_tex = f"\\textbf{{{ms:.1f}}}" if ms == fastest else f"{ms:.1f}"
        out.append(f" & {LABELS[key]:<27} & {a['nDCG_at_10']:.3f} & "
                   f"{a['recall_at_100']:.3f} & {a['MRR_at_10']:.3f} & "
                   f"{a['recall_vs_exact_at_10']:.3f} & {ms_tex} \\\\")

    bm25 = json.loads((RESULTS_JSON /
                       "stage5_corect_e03_bm25_baseline.json"
                       ).read_text())["datasets"][dataset]
    out.append(f" & BM25                        & {bm25['nDCG_at_10']:.3f} & "
               f"{bm25['recall_at_100']:.3f} & {bm25['MRR_at_10']:.3f} & "
               f"--    & {min(bm25['ms_per_query_reps']):.1f} \\\\")
    return out


def main():
    blocks = []
    for ds in DATASETS:
        rows = rows_for(ds)
        blocks.append(f"\\multirow{{{len(rows)}}}{{*}}{{{PRETTY[ds]}}}\n"
                      + "\n".join(rows))
    print("\n\\midrule\n".join(blocks))


if __name__ == "__main__":
    main()
