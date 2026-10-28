"""Runtime self-tuning: parameter sweep on a stratified chromosome subset.

DESIGN.md section 8.  There is no pretrained model and no training corpus.  The
tuner learns from the sample in front of it:

  1. build coverage once on a stratified chromosome subset, cached
  2. split the reads into two random halves (Coverage.split)
  3. sweep the parameter grid, calling on each half
  4. score each half's calls against the *other* half's reads, plus the
     agreement between the halves (section 3.4 objective, tune.objective)
  5. take the plateau, not the argmax

Splitting replaced depth-thinning (calls at full depth compared with calls on
a 50% subsample).  Thinning is biased toward conservative settings: the
subsample always loses the weakest calls, so the fewer weak calls a setting
makes, the better it agrees with itself -- and on the mixed and low-quality
benchmarks the tuner duly chose settings that kept only the strongest peaks.
Two halves are symmetric, and calls on one can be tested against reads the
caller never saw.

Because coverage is parsed once, the background is fitted once per half, and
the HMM is fitted once per half per segmentation setting (the posterior cutoff
and fold floor only threshold its output), a 27-point sweep costs a few
seconds on the benchmark genome.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..caller import BACKGROUND_FIELDS, CallParams, call_peaks, fit_background
from ..peaks import NestedPeakSet, interval_jaccard
from ..signal.coverage import Coverage
from .objective import (HeldOut, ObjectiveScore, TuningObjective, heldout_gain,
                        select_plateau)

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
    otherwise have made by eye.

    Both grids sweep the same three questions -- how confident the HMM must be
    (``posterior_cutoff``), how weak an enrichment still counts (``min_fold``),
    and at what scale to segment -- so a sample containing narrow peaks *and*
    broad domains can be tuned toward keeping both.  ``min_fold`` reaches down
    to 1.2 in both: a fold floor above a weak mark's measured fold erases every
    domain of that mark however good the rest of the setting is, and the
    held-out objective is what now stops the low end from admitting noise.
    """
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
    # Scale as a multiple of the sample's own autocorrelation length rather
    # than in bp, so the same axis means "finer / as measured / coarser" for a
    # TF and for H3K9me3 alike.
    return {
        "posterior_cutoff": (0.4, 0.5, 0.6),
        "min_fold": (1.2, 1.5, 2.0),
        "segment_bins_per_scale": (4, 8, 16),
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
        gain = self.score.heldout_gain
        lines = [
            "FlexPeak self-tuning report",
            f"  tuning chromosomes : {', '.join(self.chroms)}",
            f"  grid points        : {len(self.curve)}",
            f"  selected           : {self._selected_str()}",
            f"  objective          : {self.score.total:.4f} "
            f"(held-out fit {self.score.heldout:.3f}"
            + (f" [{gain:,.0f} nats]" if gain is not None else "")
            + f", stability {self.score.stability:.3f}, "
            f"reproducibility {self.score.reproducibility:.3f}, "
            f"territory {self.score.territory:.4f}, penalty {self.score.penalty:.3f})",
            f"  regions (half depth): {self.score.n_regions} "
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


class _Half:
    """One read half: its coverage, and what can be reused across grid points."""

    def __init__(self, cov: Coverage, control: Optional[Coverage],
                 base: CallParams, reuse_background: bool):
        self.cov = cov
        self.control = control
        self.background = (fit_background(cov, control, base)
                           if reuse_background else None)
        self.heldout = (HeldOut(cov, self.background.mu)
                        if self.background is not None else None)
        self.segmentation: Dict = {}

    def call(self, params: CallParams) -> Tuple[NestedPeakSet, HeldOut]:
        if self.background is not None:
            peaks = call_peaks(self.cov, self.control, params,
                               background=self.background,
                               segmentation_cache=self.segmentation)
            return peaks, self.heldout
        # The grid varies the background itself: refit, and score against it.
        res = call_peaks(self.cov, self.control, params, return_details=True)
        return res.peaks, HeldOut(self.cov, res.background.mu)


def tune(
    treatment: Coverage,
    control: Optional[Coverage] = None,
    base: Optional[CallParams] = None,
    grid: Optional[Dict[str, Sequence]] = None,
    chroms: Optional[Sequence[str]] = None,
    n_chroms: int = 3,
    seeds: Sequence[int] = (0,),
    replicates: Optional[Sequence[Coverage]] = None,
    objective: Optional[TuningObjective] = None,
    tolerance: float = 0.02,
    max_points: Optional[int] = None,
    verbose: bool = False,
) -> TuneResult:
    """Sweep parameters on a chromosome subset and return the plateau setting.

    ``seeds`` gives one random read split per seed; more splits average out the
    split's own noise at a cost of two calls per split per grid point.
    """
    base = base or CallParams()
    objective = objective or TuningObjective()
    grid = grid or default_grid(broad_hint=base.bin_size >= 100)

    chroms = list(chroms) if chroms else pick_tuning_chroms(treatment, n_chroms)
    sub = treatment.subset(chroms)
    sub_ctrl = control.subset([c for c in chroms if c in control.bins]) if control else None
    genome_size = sum(sub.chrom_sizes.values())

    # Bin once, here, so every call below sees coverage already at the
    # reporting resolution and a reused background lines up with it.
    if base.bin_size < sub.bin_size:
        base = base.replace(bin_size=sub.bin_size)
    sub = sub.rebin(base.bin_size)
    if sub_ctrl is not None and sub_ctrl.bin_size < base.bin_size:
        sub_ctrl = sub_ctrl.rebin(base.bin_size)
    rep_subsets = [
        r.subset([c for c in chroms if c in r.bins]) for r in (replicates or [])
    ]

    reuse_bg = not (set(grid) & BACKGROUND_FIELDS)
    splits = []
    for s in seeds:
        a, b = sub.split(seed=s)
        splits.append((_Half(a, sub_ctrl, base, reuse_bg),
                       _Half(b, sub_ctrl, base, reuse_bg)))
    full_seg: Dict = {}

    keys = sorted(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    notes: List[str] = []
    if max_points and len(combos) > max_points:
        step = len(combos) / float(max_points)
        n_all = len(combos)
        combos = [combos[int(i * step)] for i in range(max_points)]
        notes.append(f"grid reduced from {n_all} to {len(combos)} points by "
                     f"max_points={max_points}")

    curve: List[Tuple[Dict, ObjectiveScore]] = []
    for combo in combos:
        point = dict(zip(keys, combo))
        p = base.replace(**point)
        try:
            gains, jacc, first = [], [], None
            for ha, hb in splits:
                ca, ho_a = ha.call(p)
                cb, ho_b = hb.call(p)
                gains.append(0.5 * (heldout_gain(ca, ho_a, ho_b)
                                    + heldout_gain(cb, ho_b, ho_a)))
                jacc.append(interval_jaccard(ca.intervals(), cb.intervals()))
                first = first or ca
            repro = None
            if rep_subsets:
                full = call_peaks(sub, sub_ctrl, p, segmentation_cache=full_seg)
                ri = full.intervals()
                repro = float(np.mean([
                    interval_jaccard(ri, call_peaks(r, sub_ctrl, p).intervals())
                    for r in rep_subsets
                ]))
        except Exception as exc:  # a bad grid point must not kill the sweep
            notes.append(f"grid point {point} failed: {type(exc).__name__}: {exc}")
            continue
        sc = objective.score(first, genome_size=genome_size,
                             heldout_gain=float(np.mean(gains)),
                             stability=float(np.mean(jacc)),
                             reproducibility=repro)
        curve.append((point, sc))

    if not curve:
        notes.append("every grid point failed; falling back to base parameters")
        return TuneResult(base, ObjectiveScore(0, 0, 0, 0, 0, 0, 0, 0.0),
                          [], chroms, False, notes)

    objective.rank([s for _, s in curve])
    if verbose:
        for point, sc in curve:
            print(f"  {point} -> {sc}")
    if max(s.heldout_gain or 0.0 for _, s in curve) <= 0:
        notes.append(
            "no setting explains held-out reads better than background alone: "
            "the tuning chromosomes show no reproducible enrichment, so the "
            "selection rests on stability only")

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
