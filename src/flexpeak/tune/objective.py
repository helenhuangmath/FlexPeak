"""The tuning objective -- making "looks right in IGV" explicit.

DESIGN.md section 3.4.  Expert tuning by browser inspection optimises something
real but unstated.  Automation requires naming it.  The terms:

  stability      calls agree across independent depth-subsamples of the sample
  reproducibility calls agree across true replicates, when supplied
  territory      penalty on the fraction of the genome called

The territory penalty is **not optional**.  Stability and reproducibility are
both maximised by over-merging -- one giant peak per chromosome is perfectly
stable and perfectly reproducible.  Without the penalty the objective drives
straight into the failure mode this tool exists to fix (DESIGN.md 8.3), so
``TuningObjective`` refuses to be constructed with ``territory_weight <= 0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..peaks import NestedPeakSet, interval_jaccard

__all__ = ["TuningObjective", "ObjectiveScore"]


@dataclass
class ObjectiveScore:
    total: float
    stability: float
    reproducibility: float
    territory: float
    penalty: float
    n_regions: int
    n_subpeaks: int
    median_width: float
    detail: Dict = None

    def __repr__(self) -> str:
        return (
            f"<score={self.total:.4f} stab={self.stability:.3f} "
            f"repro={self.reproducibility:.3f} territory={self.territory:.4f} "
            f"n={self.n_regions}>"
        )


@dataclass
class TuningObjective:
    """Weighted objective over a parameter setting's calls."""

    stability_weight: float = 1.0
    reproducibility_weight: float = 1.0
    territory_weight: float = 1.0
    target_territory: float = 0.05
    """Expected genome fraction covered.  Broad marks legitimately reach 0.2-0.4;
    the penalty is one-sided above ``max_territory``, not a hard target."""
    max_territory: float = 0.35
    min_regions: int = 10

    def __post_init__(self):
        if self.territory_weight <= 0:
            raise ValueError(
                "territory_weight must be > 0: stability and reproducibility are "
                "maximised by over-merging, so removing the territory penalty makes "
                "the objective reward exactly the failure mode FlexPeak exists to fix "
                "(see DESIGN.md 3.4 / 8.3)"
            )

    def score(
        self,
        calls: NestedPeakSet,
        subsample_calls: Sequence[NestedPeakSet] = (),
        replicate_calls: Sequence[NestedPeakSet] = (),
        genome_size: Optional[int] = None,
    ) -> ObjectiveScore:
        n = len(calls)
        genome_size = genome_size or sum(calls.chrom_sizes.values()) or 1

        stability = _mean_jaccard(calls, subsample_calls)
        repro = _mean_jaccard(calls, replicate_calls)
        territory = calls.total_bp / float(genome_size)

        penalty = 0.0
        if territory > self.max_territory:
            # Quadratic beyond the cap: cheap near the boundary, brutal far past it.
            over = (territory - self.max_territory) / max(self.max_territory, 1e-9)
            penalty += self.territory_weight * over * over
        if n < self.min_regions:
            penalty += 1.0  # degenerate call sets are never the answer

        widths = calls.widths()
        median_width = float(np.median(widths)) if widths else 0.0

        terms, weights = [], []
        if subsample_calls:
            terms.append(stability)
            weights.append(self.stability_weight)
        if replicate_calls:
            terms.append(repro)
            weights.append(self.reproducibility_weight)
        base = float(np.average(terms, weights=weights)) if terms else 0.0

        return ObjectiveScore(
            total=base - penalty,
            stability=stability,
            reproducibility=repro,
            territory=territory,
            penalty=penalty,
            n_regions=n,
            n_subpeaks=calls.n_subpeaks,
            median_width=median_width,
            detail={"genome_size": genome_size},
        )


def _mean_jaccard(ref: NestedPeakSet, others: Sequence[NestedPeakSet]) -> float:
    if not others:
        return 0.0
    ri = ref.intervals()
    vals = [interval_jaccard(ri, o.intervals()) for o in others]
    return float(np.mean(vals)) if vals else 0.0


def select_plateau(scores: List[float], tolerance: float = 0.02) -> int:
    """Index of the plateau, not the argmax (DESIGN.md 3.4).

    Every setting within ``tolerance`` of the best is considered equivalent;
    among those we take the median position, which lands in the middle of the
    flat region rather than on a noise spike at its edge.
    """
    if not scores:
        raise ValueError("no scores to select from")
    arr = np.asarray(scores, dtype=float)
    best = float(np.nanmax(arr))
    near = np.flatnonzero(arr >= best - tolerance * max(abs(best), 1e-9) - tolerance)
    if near.size == 0:
        return int(np.nanargmax(arr))
    return int(near[near.size // 2])
