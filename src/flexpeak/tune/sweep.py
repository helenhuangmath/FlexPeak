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

__all__ = ["TuneResult", "tune", "default_grid", "pick_tuning_chroms", "tuning_windows",
           "window_coverage", "replicate_halves"]


def pick_tuning_chroms(cov: Coverage, n: int = 3, skip_largest: bool = False) -> List[str]:
    """Stratified chromosome subset (DESIGN.md 7.5).

    One large gene-dense, one mid/gene-poor, one small.  A single chromosome is
    not representative for broad marks, and chr1 alone biases toward gene-dense
    regions.

    ``skip_largest`` (the ``--quick`` default, with ``n=2``) takes the
    median-sized and the smallest primary chromosome instead: together about
    the size of the largest one, so a sweep costs what a chr1 sweep did, but
    two chromosomes of different character are seen -- and the largest
    chromosome, where people tend to look first, stays out of the tuning.
    """
    from ..scan import _is_primary

    chroms = [c for c in cov.chroms if cov.bins[c].size > 10]
    prim = [c for c in chroms if _is_primary(c) and c not in ("chrX", "X")] or chroms
    if len(prim) <= n:
        return prim
    ordered = sorted(prim, key=lambda c: cov.chrom_sizes.get(c, 0), reverse=True)
    if skip_largest:
        picks = [ordered[len(ordered) // 2], ordered[-1], ordered[len(ordered) // 4]]
        return list(dict.fromkeys(picks))[:n]
    if n == 1:
        return [ordered[0]]
    picks = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    return list(dict.fromkeys(picks))[:n]


def window_coverage(cov: Coverage, plan: Sequence[Tuple[str, int, int]]) -> Coverage:
    """A Coverage of ``(chrom, start, end)`` windows, each a pseudo-chromosome
    named ``chrom:start-end``.  Lets the sweep run on stratified pieces of
    several chromosomes for the cost of one (``tune(windows=...)``)."""
    bs = cov.bin_size
    bins, sizes = {}, {}
    for c, s, e in plan:
        if c not in cov.bins:
            continue
        k = f"{c}:{s}-{e}"
        bins[k] = cov.bins[c][s // bs: e // bs]
        sizes[k] = int(bins[k].size * bs)
    return Coverage(bins=bins, bin_size=bs, chrom_sizes=sizes, source=cov.source,
                    is_count=cov.is_count, n_fragments=None,
                    fragment_length=cov.fragment_length, meta=dict(cov.meta))


def tuning_windows(cov: Coverage, total_bp: int = 100_000_000,
                   n: int = 4) -> List[Tuple[str, int, int]]:
    """Stratified tuning windows: ``n`` primary autosomes spread over the
    size ranking, ``total_bp / n`` from each (``scan.plan_scan_windows``).
    Measured on pooled H3K9me3 (six chromosomes, same node): tuning on
    chr10 + chr15 whole took the sweep from 141 s (chr1, 4 points) to 266 s
    (6 points with replicate halves); 4 x 25 Mb is half of chr1."""
    from ..scan import _is_primary, plan_scan_windows

    sizes = {c: cov.chrom_sizes.get(c, a.size * cov.bin_size) for c, a in cov.bins.items()
             if _is_primary(c) and c not in ("chrX", "X")}
    return plan_scan_windows(sizes or dict(cov.chrom_sizes), total_bp, n)


class ReplicateSupport:
    """Share of calls the replicates support: a region counts when its mean
    is >= ``min_fold`` x the replicate's own level (median of the 5 kb-
    smoothed track of that chromosome) in at least ``min_share`` of the
    replicates.  Prefix sums, so scoring a call set is O(regions).

    This is the reproducibility term when replicates are given.  The held-
    out likelihood saturates on domain marks -- every setting explains ~99%
    of the best gain (GSE285245 H3K27me3: spread 0.005 over the grid) -- and
    then picks noise: its best point had 0.91 of its calls supported by the
    replicates on held-out chromosomes, against 0.99 for a setting it scored
    0.01 lower.
    """

    def __init__(self, replicates: Sequence[Coverage], min_fold: float = 1.5,
                 min_share: float = 0.75):
        from scipy.ndimage import uniform_filter1d

        self.min_fold, self.n = float(min_fold), len(replicates)
        self.need = int(np.ceil(min_share * self.n))
        self.bin_size = replicates[0].bin_size
        self.pref, self.level = {}, {}
        for r in replicates:
            k = max(1, int(5_000 // r.bin_size))
            for c, a in r.bins.items():
                a = a.astype(np.float64)
                self.pref.setdefault(c, []).append(np.concatenate([[0.0], np.cumsum(a)]))
                sm = uniform_filter1d(a, k, mode="nearest") if a.size > k else a
                self.level.setdefault(c, []).append(max(float(np.median(sm)), 1e-6))

    def __call__(self, calls: NestedPeakSet) -> float:
        bs = self.bin_size
        ok, tot = 0, 0
        by: Dict[str, List[Tuple[int, int]]] = {}
        for r in calls.regions:
            if r.chrom in self.pref:
                by.setdefault(r.chrom, []).append((r.start // bs, max(r.start // bs + 1, -(-r.end // bs))))
        for c, ivs in by.items():
            iv = np.array(ivs, dtype=np.int64)
            votes = np.zeros(len(iv), dtype=np.int64)
            for pc, lv in zip(self.pref[c], self.level[c]):
                b1 = np.minimum(iv[:, 1], pc.size - 1)
                b0 = np.minimum(iv[:, 0], b1)
                m = (pc[b1] - pc[b0]) / np.maximum(b1 - b0, 1)
                votes += m >= self.min_fold * lv
            ok += int((votes >= self.need).sum())
            tot += len(iv)
        return ok / tot if tot else 0.0


def replicate_halves(replicates: Sequence[Coverage], chroms: Sequence[str],
                     bin_size: int, total: float) -> Tuple[Coverage, Coverage, str]:
    """Two halves made of whole replicates (1, 3, ... | 2, 4, ...), each
    rescaled to half of ``total``.

    The read split assumes the counts are reads.  For a bigWig they are
    pseudo-counts, and a binomial split of them invents a sampling noise the
    data never had while missing the replicate-to-replicate variation it
    does have -- so the two halves agree almost perfectly at every setting
    and the held-out score cannot tell settings apart (GSE285245 H3K27me3:
    objective spread 0.005 over the grid).  Replicate groups are real
    independent measurements, and calls on one group are tested against
    the other group's signal.
    """
    def total_of(c):
        return sum(float(a.sum()) for a in c.bins.values())

    reps = []
    for r in replicates:
        r = r.subset([c for c in chroms if c in r.bins])
        if r.bin_size < bin_size:
            r = r.rebin(bin_size)
        reps.append(r)
    groups = [reps[0::2], reps[1::2]]
    halves = []
    for g in groups:
        bins = {}
        for c in chroms:
            arrs = [r.bins[c] for r in g if c in r.bins]
            if not arrs:
                continue
            m = min(a.size for a in arrs)
            bins[c] = np.sum([a[:m].astype(np.float64) for a in arrs], axis=0)
        h = Coverage(bins=bins, bin_size=bin_size,
                     chrom_sizes={c: g[0].chrom_sizes.get(c, bins[c].size * bin_size)
                                  for c in bins},
                     source="+".join(r.source for r in g), is_count=g[0].is_count)
        t = total_of(h)
        scale = 0.5 * total / t if t > 0 else 1.0
        h.bins = {c: (a * scale).astype(np.float32) for c, a in h.bins.items()}
        halves.append(h)
    ids = lambda k: ",".join(str(i + 1) for i in range(k, len(reps), 2))  # noqa: E731
    return halves[0], halves[1], f"replicates {ids(0)} | {ids(1)}"


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
    search: str = "auto",
    replicate_split: bool = True,
    skip_largest: bool = False,
    windows: Optional[Sequence[Tuple[str, int, int]]] = None,
) -> TuneResult:
    """Sweep parameters on a chromosome subset and return the plateau setting.

    ``seeds`` gives one random read split per seed; more splits average out the
    split's own noise at a cost of two calls per split per grid point.

    With two or more ``replicates`` and ``replicate_split`` the halves are
    replicate groups instead of read splits (see ``replicate_halves``); the
    stability term is then agreement between independent replicates.

    ``search``: ``"grid"`` evaluates every combination; ``"coordinate"`` sweeps
    one axis at a time with the others held at their current best (two
    passes), which costs the sum of the axis lengths rather than their
    product and is what lets the grid include the structural settings --
    gap merging, the focal floor, the weak-domain pass -- at the price of
    the old 4-point grid.  ``"auto"``: grid up to 8 points, else coordinate.
    """
    base = base or CallParams()
    objective = objective or TuningObjective()
    grid = grid or default_grid(broad_hint=base.bin_size >= 100)
    # The sweep scores regions; sub-peaks are called once, on the final call.
    # Scale-space detection inside every region at every grid point on both
    # halves was most of the sweep's cost on punctate marks (tens of
    # thousands of regions per chromosome), and the grid does not vary
    # anything a sub-peak depends on.
    final_subpeaks = base.call_subpeaks
    base = base.replace(call_subpeaks=False)

    if windows:
        # Stratified windows as pseudo-chromosomes (see tuning_windows).
        treatment = window_coverage(treatment, windows)
        control = window_coverage(control, windows) if control is not None else None
        replicates = [window_coverage(r, windows) for r in (replicates or [])]
        chroms = list(treatment.bins)
    chroms = list(chroms) if chroms else pick_tuning_chroms(treatment, n_chroms,
                                                            skip_largest=skip_largest)
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
    notes: List[str] = []
    splits = []
    split_desc = "read halves"
    use_reps = bool(replicate_split and len(rep_subsets) >= 2)
    support = ReplicateSupport(rep_subsets) if len(rep_subsets) >= 2 else None
    if use_reps:
        total = sum(float(a.sum()) for a in sub.bins.values())
        a, b, split_desc = replicate_halves(rep_subsets, list(sub.bins), base.bin_size, total)
        splits.append((_Half(a, sub_ctrl, base, reuse_bg), _Half(b, sub_ctrl, base, reuse_bg)))
        rep_subsets = []  # agreement between the groups is the stability term
        notes.append("reproducibility = share of calls >= 1.5x in >= 3/4 of the replicates")
    else:
        for s in seeds:
            a, b = sub.split(seed=s)
            splits.append((_Half(a, sub_ctrl, base, reuse_bg),
                           _Half(b, sub_ctrl, base, reuse_bg)))
    full_seg: Dict = {}

    keys = list(grid)
    n_product = int(np.prod([len(grid[k]) for k in keys])) if keys else 1
    if search == "auto":
        search = "grid" if n_product <= 8 else "coordinate"
    scored: Dict[Tuple, ObjectiveScore] = {}
    curve: List[Tuple[Dict, ObjectiveScore]] = []

    def evaluate(point: Dict) -> Optional[ObjectiveScore]:
        key = tuple(point[k] for k in keys)
        if key in scored:
            return scored[key]
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
            if support is not None:
                repro = float(np.mean([support(ca), support(cb)]))
            elif rep_subsets:
                full = call_peaks(sub, sub_ctrl, p, segmentation_cache=full_seg)
                ri = full.intervals()
                repro = float(np.mean([
                    interval_jaccard(ri, call_peaks(r, sub_ctrl, p).intervals())
                    for r in rep_subsets
                ]))
        except Exception as exc:  # a bad grid point must not kill the sweep
            notes.append(f"grid point {point} failed: {type(exc).__name__}: {exc}")
            scored[key] = None
            return None
        sc = objective.score(first, genome_size=genome_size,
                             heldout_gain=float(np.mean(gains)),
                             stability=float(np.mean(jacc)),
                             reproducibility=repro)
        scored[key] = sc
        curve.append((dict(point), sc))
        return sc

    if search == "grid":
        combos = list(itertools.product(*(grid[k] for k in keys)))
        if max_points and len(combos) > max_points:
            step = len(combos) / float(max_points)
            n_all = len(combos)
            combos = [combos[int(i * step)] for i in range(max_points)]
            notes.append(f"grid reduced from {n_all} to {len(combos)} points by "
                         f"max_points={max_points}")
        for combo in combos:
            evaluate(dict(zip(keys, combo)))
    elif search == "coordinate":
        bd = base.to_dict()
        cur = {}
        for k in keys:
            vals = list(grid[k])
            bv = bd.get(k)
            if isinstance(bv, list):
                bv = tuple(bv)
            cur[k] = bv if bv in vals else vals[len(vals) // 2]
        for _pass in range(2):
            changed = False
            for k in keys:
                pts = [dict(cur, **{k: v}) for v in grid[k]]
                scs = [evaluate(pt) for pt in pts]
                objective.rank([sc for sc in scored.values() if sc is not None])
                ok = [(pt, sc) for pt, sc in zip(pts, scs) if sc is not None]
                if not ok:
                    continue
                # Best value on this axis; the current one is kept on a
                # near-tie (1e-3) so the search does not wander on noise.
                # Not select_plateau: with two values its tolerance holds
                # both, and the median position is always the second
                # (measured: focal floor 1.0 at 0.929 chosen over 0.5 at 0.953).
                tot = [sc.total for _, sc in ok]
                j = int(np.argmax(tot))
                here = [i for i, (pt, _) in enumerate(ok) if pt[k] == cur[k]]
                if here and tot[here[0]] >= tot[j] - 1e-3:
                    j = here[0]
                if ok[j][0][k] != cur[k]:
                    cur[k] = ok[j][0][k]
                    changed = True
            if not changed:
                break
        notes.append(f"coordinate search: {len(curve)} of {n_product} grid points evaluated")
    else:
        raise ValueError(f"search must be 'grid', 'coordinate' or 'auto', got {search!r}")
    notes.insert(0, f"halves: {split_desc}")

    if not curve:
        notes.append("every grid point failed; falling back to base parameters")
        return TuneResult(base.replace(call_subpeaks=final_subpeaks),
                          ObjectiveScore(0, 0, 0, 0, 0, 0, 0, 0.0), [], chroms, False, notes)

    objective.rank([s for _, s in curve])
    if verbose:
        for point, sc in curve:
            print(f"  {point} -> {sc}")
    if max(s.heldout_gain or 0.0 for _, s in curve) <= 0:
        notes.append(
            "no setting explains held-out reads better than background alone: "
            "the tuning chromosomes show no reproducible enrichment, so the "
            "selection rests on stability only")

    if search == "coordinate":
        # The best point evaluated, not necessarily where the walk stopped:
        # held-out gains are re-normalised to the grid's best as points are
        # added, so an axis decided early can be beaten by a point seen later.
        best_point, best_score = max(curve, key=lambda x: x[1].total)
        best_point = dict(best_point)
    else:
        idx = select_plateau([s.total for _, s in curve], tolerance=tolerance)
        best_point, best_score = curve[idx]
    stable, stability_note = _assess_stability(curve, tolerance)
    if stability_note:
        notes.append(stability_note)

    return TuneResult(
        params=base.replace(call_subpeaks=final_subpeaks, **best_point),
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
