"""PLAID (ColBERT candidate generation) baseline wrapper.

Single responsibility: wrap PyLate / PLAID retrieval as a baseline so it can be
compared on the same result schema as bondmaxsim methods.

Ported artifact: PLAID wrapper from
  research/colbert/02_corect_quick.py and research/colbert/03_beir_comparison.py.
Stage 1 reference: docs/stage1_bond_maxsim_formalization.md §5.4 (PLAID is the
  canonical candidate-generation baseline for ColBERT; it uses single-vector
  centroid search, not MaxSim dimension pruning, and must be kept as a separate
  arm), docs/project_b_analysis_and_research_plan.md Stage 4 (PLAID with tuned
  settings is a required baseline for the integration comparison).
"""

from __future__ import annotations

from bondmaxsim.schema import ResultRecord


class PLAIDBaseline:
    """PLAID retrieval baseline via PyLate.

    Parameters
    ----------
    index_path : str — path to a pre-built PyLate PLAID index
    dataset    : str — BEIR corpus name for result record labeling
    """

    def __init__(self, index_path: str, dataset: str) -> None:
        raise NotImplementedError

    def search(
        self,
        queries: list,
        k: int = 10,
        ncells: int = 1,
        centroid_score_threshold: float = 0.5,
    ) -> list[ResultRecord]:
        """Run PLAID retrieval and return ResultRecords."""
        raise NotImplementedError
