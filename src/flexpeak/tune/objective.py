"""The tuning objective -- making "looks right in IGV" explicit.

DESIGN.md section 3.4.  Expert tuning by browser inspection optimises something
real but unstated.  Automation requires naming it.  The terms:

  held-out fit    how well calls made on one half of the reads predict the
                  OTHER half (cross-validated likelihood, below)
  stability       calls agree between the two independent read halves
  reproducibility calls agree across true replicates, when supplied
  territory       penalty on the fraction of the genome called

**Why the held-out term exists.**  The objective used to be stability and
reproducibility minus a territory penalty.  That is one-sided: the penalty
stops over-merging, but nothing stops the symmetric failure of calling almost
nothing -- a very strict setting makes few, strong, perfectly reproducible
calls, and stability rewards it.  On the benchmark that is exactly what the
tuner picked: F1 99% -> 7% on mixed narrow+broad data, with a stability score
of 0.98 for the broken setting.

The held-out term is two-sided by construction.  Split the reads at random
into halves A and B.  Call on A; from A, read off a fold for each called piece
(each sub-peak, and the rest of its region).  Then score how much better that
piecewise-constant enrichment explains B's reads than the background alone
does -- a Poisson log-likelihood ratio on held-out data::

    gain = sum over pieces  B * log(f_A) - E_B * (f_A - 1)

* **Missing real enrichment** forfeits its gain, so under-calling loses.
* **Calling noise** costs: noise that pushed A up does not push B up, and the
  term is negative whenever B sits at background (log f < f - 1).
* **Over-merging** dilutes f_A over background bins, which then pay the same
  penalty, and under-fits the enriched part.
* **Narrow and broad compete on equal terms**, in units of evidence: a 600 bp
  peak at 8x and a 60 kb domain at 2.5x contribute gains of the same order, so
  neither scale can be sacrificed for the other without the score noticing.

The territory penalty stays, and stays **non-optional**.  Stability and
reproducibility are maximised by over-merging, so ``TuningObjective`` refuses
to be constructed with ``territory_weight <= 0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..peaks import NestedPeakSet, interval_jaccard
from ..signal.coverage import Coverage

__all__ = ["TuningObjective", "ObjectiveScore", "HeldOut", "heldout_gain"]


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
    heldout_gain: Optional[float] = None
    """Raw held-out log-likelihood gain (nats), when the split was scored."""
    heldout: float = 0.0
    """``heldout_gain`` as a fraction of the best gain on the grid."""

    def __repr__(self) -> str:
        return (
            f"<score={self.total:.4f} heldout={self.heldout:.3f} "
            f"stab={self.stability:.3f} repro={self.reproducibility:.3f} "
            f"territory={self.territory:.4f} n={self.n_regions}>"
        )


class HeldOut:
    """Prefix sums of one read half and its background, for O(1) range sums."""

    def __init__(self, cov: Coverage, mu: Dict[str, np.ndarray]):
        self.bin_size = cov.bin_size
        self._c = {c: _prefix(cov.bins[c]) for c in cov.bins}
        self._m = {c: _prefix(mu[c]) for c in cov.bins if c in mu}

    def sums(self, chrom: str, b0: np.ndarray, b1: np.ndarray):
        pc, pm = self._c[chrom], self._m[chrom]
        n = min(pc.size, pm.size) - 1
        b0 = np.clip(b0, 0, n)
        b1 = np.clip(b1, 0, n)
        return pc[b1] - pc[b0], pm[b1] - pm[b0]


def _prefix(a: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(a, dtype=np.float64)])


def _pieces(region, bin_size: int):
    """Bin intervals of a region's sub-peaks and of the region outside them."""
    b0, b1 = region.start // bin_size, -(-region.end // bin_size)
    out, cur = [], b0
    for s in sorted(region.subpeaks, key=lambda s: s.start):
        s0 = max(s.start // bin_size, cur)
        s1 = min(-(-s.end // bin_size), b1)
        if s1 <= s0:
            continue
        if s0 > cur:
            out.append((cur, s0))
        out.append((s0, s1))
        cur = s1
    if cur < b1:
        out.append((cur, b1))
    return out


def heldout_gain(calls: NestedPeakSet, fit: HeldOut, held: HeldOut) -> float:
    """Held-out Poisson log-likelihood gain of ``calls`` (see module docstring).

    ``fit`` is the half the calls were made on, and supplies each piece's fold;
    ``held`` is the other half, which the folds must predict.
    """
    by_chrom: Dict[str, List] = {}
    for r in calls.regions:
        if r.chrom in fit._c and r.chrom in held._c:
            by_chrom.setdefault(r.chrom, []).extend(_pieces(r, fit.bin_size))
    gain = 0.0
    for chrom, pieces in by_chrom.items():
        b = np.array(pieces, dtype=np.int64).reshape(-1, 2)
        a_obs, a_exp = fit.sums(chrom, b[:, 0], b[:, 1])
        h_obs, h_exp = held.sums(chrom, b[:, 0], b[:, 1])
        ok = (a_exp > 0) & (h_exp > 0)
        # Depletion is not something a peak caller claims, so a piece that
        # came out below background on its own half contributes nothing rather
        # than being rewarded for predicting a dip.
        f = np.maximum(a_obs[ok] / a_exp[ok], 1.0)
        gain += float(np.sum(h_obs[ok] * np.log(f) - h_exp[ok] * (f - 1.0)))
    return gain


@dataclass
class TuningObjective:
    """Weighted objective over a parameter setting's calls."""

    heldout_weight: float = 1.0
    stability_weight: float = 0.25
    reproducibility_weight: float = 0.25
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
        heldout_gain: Optional[float] = None,
        stability: Optional[float] = None,
        reproducibility: Optional[float] = None,
    ) -> ObjectiveScore:
        """Score one setting.

        With ``heldout_gain`` the total is provisional: the gain only means
        something relative to the other settings, so :meth:`rank` fills in
        ``heldout`` and ``total`` once the whole grid has been scored.

        ``stability`` / ``reproducibility`` take precomputed agreement values in
        place of ``subsample_calls`` / ``replicate_calls`` -- the sweep measures
        stability between read halves, neither of which is ``calls``.
        """
        n = len(calls)
        genome_size = genome_size or sum(calls.chrom_sizes.values()) or 1

        has_sub = bool(subsample_calls) or stability is not None
        has_rep = bool(replicate_calls) or reproducibility is not None
        if stability is None:
            stability = _mean_jaccard(calls, subsample_calls)
        repro = (reproducibility if reproducibility is not None
                 else _mean_jaccard(calls, replicate_calls))
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

        sc = ObjectiveScore(
            total=0.0,
            stability=stability,
            reproducibility=repro,
            territory=territory,
            penalty=penalty,
            n_regions=n,
            n_subpeaks=calls.n_subpeaks,
            median_width=median_width,
            detail={"genome_size": genome_size,
                    "has_subsamples": has_sub,
                    "has_replicates": has_rep},
            heldout_gain=heldout_gain,
        )
        sc.total = self._combine(sc)
        return sc

    def rank(self, scores: Sequence[ObjectiveScore]) -> None:
        """Normalise held-out gains across a grid and recompute totals, in place.

        The gain is scaled by the best gain on the grid, so 1.0 is "explains the
        held-out reads as well as any setting tried" -- the same units as the
        Jaccard terms it is averaged with.  A sample with no positive gain
        anywhere (nothing enriched) gets 0 throughout, which leaves the choice
        to the other terms rather than to noise in a ratio of tiny numbers.
        """
        gains = [s.heldout_gain for s in scores if s.heldout_gain is not None]
        best = max(gains) if gains else 0.0
        for s in scores:
            if s.heldout_gain is not None:
                s.heldout = s.heldout_gain / best if best > 0 else 0.0
            s.total = self._combine(s)

    def _combine(self, s: ObjectiveScore) -> float:
        terms, weights = [], []
        if s.heldout_gain is not None and self.heldout_weight > 0:
            terms.append(s.heldout)
            weights.append(self.heldout_weight)
        if (s.detail or {}).get("has_subsamples"):
            terms.append(s.stability)
            weights.append(self.stability_weight)
        if (s.detail or {}).get("has_replicates"):
            terms.append(s.reproducibility)
            weights.append(self.reproducibility_weight)
        base = (float(np.average(terms, weights=weights))
                if terms and sum(weights) > 0 else 0.0)
        return base - s.penalty


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
