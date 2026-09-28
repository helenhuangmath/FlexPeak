"""Runtime self-tuning: parameter sweep on a stratified chromosome subset.

DESIGN.md section 8.  There is no pretrained model and no training corpus.  The
tuner learns from the sample in front of it:

  1. build coverage once on a stratified chromosome subset, cached
  2. make depth-degraded copies by binomial thinning (Coverage.thin)
  3. sweep the parameter grid, calling on full depth and on each degraded copy
  4. score with the section 3.4 objective
  5. take the plateau, not the argmax

Because coverage is parsed once and every grid point is an array operation over
in-memory data, the whole sweep costs seconds to low minutes.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..caller import CallParams, call_peaks
from ..peaks import NestedPeakSet
from ..signal.coverage import Coverage
from .objective import ObjectiveScore, TuningObjective, select_plateau

__all__ = ["TuneResult", "tune", "default_grid", "pick_tuning_chroms"]


def pick_tuning_chroms(cov: Coverage, n: int = 3) -> List[str]:
    """Stratified chromosome subset (DESIGN.md 7.5).

    One large gene-dense, one mid/gene-poor, one small.  A single chromosome is
    not representative for broad marks, and chr1 alone biases toward gene-dense
    regions.
    """
    chroms = [c for c in cov.chroms if cov.bins[c].size > 10]
    if len(chroms) <= n:
        return chroms
    ordered = sorted(chroms, key=lambda c: cov.chrom_sizes.get(c, 0), reverse=True)
    if n == 1:
        return [ordered[0]]
    picks = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    return list(dict.fromkeys(picks))[:n]


def default_grid(broad_hint: bool = False) -> Dict[str, Sequence]:
    """A deliberately small grid.  Each axis is a decision the user would
    otherwise have made by eye."""
    if broad_hint:
        # segment_bin_size rather than min_width: for a broad mark the scale the
        # HMM segments at decides whether domains come out as domains at all,
        # while min_width only trims the short tail.  Measured on CUT&RUN
        # H3K27me3, moving it 900 bp -> 10 kb took median called width from
        # 26 kb to 40 kb and territory from 17.2% to 19.3% against a 20.2%
        # envelope; min_width moved neither.  Whether a given library supports
        # the coarse end is a property of that library, so it is swept.
        return {
            "posterior_cutoff": (0.4, 0.5, 0.6),
            "min_fold": (1.2, 1.5, 2.0),
            "segment_bin_size": (1_000, 5_000, 20_000),
        }
    return {
        "posterior_cutoff": (0.4, 0.5, 0.6, 0.7),
        "min_fold": (1.5, 2.0, 3.0),
        "min_width": (100, 200, 500),
    }


@dataclass
class TuneResult:
    params: CallParams
    score: ObjectiveScore
    curve: List[Tuple[Dict, ObjectiveScore]] = field(default_factory=list)
    chroms: List[str] = field(default_factory=list)
    stable: bool = True
    notes: List[str] = field(default_factory=list)

    def report(self) -> str:
        """Human-readable summary -- this is what makes the tuner auditable."""
        lines = [
            "FlexPeak self-tuning report",
            f"  tuning chromosomes : {', '.join(self.chroms)}",
            f"  grid points        : {len(self.curve)}",
            f"  selected           : {self._selected_str()}",
            f"  objective          : {self.score.total:.4f} "
            f"(stability {self.score.stability:.3f}, "
            f"reproducibility {self.score.reproducibility:.3f}, "
            f"territory {self.score.territory:.4f}, penalty {self.score.penalty:.3f})",
            f"  regions            : {self.score.n_regions} "
            f"(median width {self.score.median_width:.0f} bp, "
            f"{self.score.n_subpeaks} subpeaks)",
            f"  stable across grid : {'yes' if self.stable else 'NO -- see notes'}",
        ]
        for note in self.notes:
            lines.append(f"  ! {note}")
        lines.append("")
        lines.append("  top settings by objective:")
        for pt, sc in sorted(self.curve, key=lambda x: -x[1].total)[:5]:
            desc = ", ".join(f"{k}={v}" for k, v in sorted(pt.items()))
            lines.append(f"    {sc.total:+.4f}  {desc}  (n={sc.n_regions})")
        return "\n".join(lines)

    def _selected_str(self) -> str:
        keys = sorted({k for pt, _ in self.curve for k in pt})
        d = self.params.to_dict()
        return ", ".join(f"{k}={d[k]}" for k in keys if k in d)


def tune(
    treatment: Coverage,
    control: Optional[Coverage] = None,
    base: Optional[CallParams] = None,
    grid: Optional[Dict[str, Sequence]] = None,
    chroms: Optional[Sequence[str]] = None,
    n_chroms: int = 3,
    thin_fractions: Sequence[float] = (0.5, 0.25),
    seeds: Sequence[int] = (0, 1),
    replicates: Optional[Sequence[Coverage]] = None,
    objective: Optional[TuningObjective] = None,
    tolerance: float = 0.02,
    max_points: Optional[int] = None,
    verbose: bool = False,
) -> TuneResult:
    """Sweep parameters on a chromosome subset and return the plateau setting."""
    base = base or CallParams()
    objective = objective or TuningObjective()
    grid = grid or default_grid(broad_hint=base.bin_size >= 100)

    chroms = list(chroms) if chroms else pick_tuning_chroms(treatment, n_chroms)
    sub = treatment.subset(chroms)
    sub_ctrl = control.subset([c for c in chroms if c in control.bins]) if control else None
    genome_size = sum(sub.chrom_sizes.values())

    # Depth-degraded copies of *this* sample -- the corpus-free stand-in for a
    # deeper reference library.
    thinned = [
        sub.thin(f, seed=s) for f in thin_fractions for s in seeds
    ]
    rep_subsets = [
        r.subset([c for c in chroms if c in r.bins]) for r in (replicates or [])
    ]

    keys = sorted(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    notes: List[str] = []
    if max_points and len(combos) > max_points:
        step = len(combos) / float(max_points)
        combos = [combos[int(i * step)] for i in range(max_points)]
        notes.append(
            f"grid reduced from {len(list(itertools.product(*(grid[k] for k in keys))))} "
            f"to {len(combos)} points by max_points={max_points}"
        )

    curve: List[Tuple[Dict, ObjectiveScore]] = []
    for combo in combos:
        point = dict(zip(keys, combo))
        p = base.replace(**point)
        try:
            full = call_peaks(sub, sub_ctrl, p)
            subs = [call_peaks(t, sub_ctrl, p) for t in thinned]
            reps = [call_peaks(r, sub_ctrl, p) for r in rep_subsets]
        except Exception as exc:  # a bad grid point must not kill the sweep
            notes.append(f"grid point {point} failed: {type(exc).__name__}: {exc}")
            continue
        sc = objective.score(full, subs, reps, genome_size=genome_size)
        curve.append((point, sc))
        if verbose:
            print(f"  {point} -> {sc}")

    if not curve:
        notes.append("every grid point failed; falling back to base parameters")
        return TuneResult(base, ObjectiveScore(0, 0, 0, 0, 0, 0, 0, 0.0),
                          [], chroms, False, notes)

    idx = select_plateau([s.total for _, s in curve], tolerance=tolerance)
    best_point, best_score = curve[idx]
    stable, stability_note = _assess_stability(curve, tolerance)
    if stability_note:
        notes.append(stability_note)

    return TuneResult(
        params=base.replace(**best_point),
        score=best_score,
        curve=curve,
        chroms=chroms,
        stable=stable,
        notes=notes,
    )


def _assess_stability(curve, tolerance: float) -> Tuple[bool, Optional[str]]:
    """Flat objective across the whole grid means the data cannot distinguish
    settings -- report it rather than pretending a choice was made."""
    totals = np.array([s.total for _, s in curve])
    spread = float(np.nanmax(totals) - np.nanmin(totals))
    if spread < tolerance:
        return False, (
            f"objective varies by only {spread:.4f} across the grid: the sample "
            "cannot distinguish these settings, so the selection is arbitrary. "
            "Consider a preset (--preset) or more tuning chromosomes."
        )
    near_best = int((totals >= totals.max() - tolerance).sum())
    if near_best == totals.size:
        return False, "all grid points score within tolerance of the best"
    return True, None
