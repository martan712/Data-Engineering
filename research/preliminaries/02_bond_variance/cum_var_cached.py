"""Compute cumulative explained variance (cum_var, global variance ordering)
for the datasets already cached in 02_bond_variance/.cache, without re-encoding.

Mirrors the cum_var computation in bond_variance_analysis.py:
    dim_var    = all_vecs.var(axis=0)
    global_idx = argsort(dim_var)[::-1]
    cum_var    = cumsum(dim_var[global_idx]) / dim_var.sum()
"""
import pathlib, gc
import numpy as np

CACHE_DIR = pathlib.Path(__file__).parent / ".cache"
BUDGETS = [1, 2, 4, 8, 16, 32, 64, 128]

names = sorted(p.name[:-len("_doc_embs.npy")]
               for p in CACHE_DIR.glob("*_doc_embs.npy"))

print(f"Cached datasets: {names}\n")

for name in names:
    doc_embs = np.load(CACHE_DIR / f"{name}_doc_embs.npy", allow_pickle=True)
    all_vecs = np.vstack(doc_embs)
    dim_var    = all_vecs.var(axis=0)
    global_idx = np.argsort(dim_var)[::-1]
    cum_var    = np.cumsum(dim_var[global_idx]) / dim_var.sum()

    print(f"[{name}]  ({len(doc_embs)} docs, {all_vecs.shape[0]} tokens, {all_vecs.shape[1]} dims)")
    for k in BUDGETS:
        print(f"  {k:3d} dims — cum.var {cum_var[min(k,128)-1]*100:5.1f}%")
    print()

    del doc_embs, all_vecs; gc.collect()
