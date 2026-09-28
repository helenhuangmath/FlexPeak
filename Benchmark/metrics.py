"""Evaluation metrics for the method comparison.

Every method is scored by *this* code, from a plain list of ``(chrom, start,
end)`` intervals.  No method is scored by its own output conventions, and no
metric reads anything a method emits beyond its call intervals -- otherwise the
comparison would measure score-column semantics rather than calls.

Metric definitions follow docs/DESIGN.md section 9.4.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from flexpeak.peaks import interval_overlap_bp, interval_total_bp

Interval = Tuple[str, int, int]

MIN_OVERLAP_BP = 200
"""Overlap below this is treated as incidental and ignored by the
fragmentation and fusion counts, so that a 1 bp touch is not a 'match'."""

REGION_HIT_FRACTION = 0.4
"""A truth region counts as recovered when this fraction of its base pairs is
covered by calls.  Fixed across all methods."""


def _by_chrom(ivs: Sequence[Interval]) -> Dict[str, List[Tuple[int, int]]]:
    out: Dict[str, List[Tuple[int, int]]] = {}
    for c, s, e in ivs:
        out.setdefault(c, []).append((s, e))
    for c in out:
        out[c].sort()
    return out


def _overlap(a: Sequence[Interval], b: Sequence[Interval]) -> int:
    return interval_overlap_bp(list(a), list(b))


def evaluate(calls: Sequence[Interval], truth: Sequence[Interval],
             genome_size: int) -> Dict[str, float]:
    """Score one method's calls against known truth on one dataset."""
    calls = [(c, int(s), int(e)) for c, s, e in calls if e > s]
    truth = [(c, int(s), int(e)) for c, s, e in truth if e > s]

    called_bp = interval_total_bp(calls)
    truth_bp = interval_total_bp(truth)
    inter = _overlap(calls, truth)

    bp_precision = inter / called_bp if called_bp else 0.0
    bp_recall = inter / truth_bp if truth_bp else float("nan")
    bp_f1 = (2 * bp_precision * bp_recall / (bp_precision + bp_recall)
             if (bp_precision + bp_recall) > 0 else 0.0)
    union = called_bp + truth_bp - inter
    jaccard = inter / union if union else 0.0

    # Region-level sensitivity: a truth region is recovered when enough of it
    # is covered, regardless of how many calls do the covering.
    hits = sum(1 for t in truth
               if _overlap([t], calls) >= REGION_HIT_FRACTION * (t[2] - t[1]))
    region_recall = hits / len(truth) if truth else float("nan")

    # Fragmentation: truth regions split across more than one call.
    # Fusion: calls spanning more than one truth region.  These are the two
    # over-merge / over-split failure modes and must be reported together.
    frag = sum(1 for t in truth
               if sum(1 for c in calls
                      if c[0] == t[0] and _overlap([t], [c]) >= MIN_OVERLAP_BP) > 1)
    fused = sum(1 for c in calls
                if sum(1 for t in truth
                       if t[0] == c[0] and _overlap([t], [c]) >= MIN_OVERLAP_BP) > 1)
    fragmentation = frag / len(truth) if truth else float("nan")
    fusion = fused / len(calls) if calls else 0.0

    # Boundary error against the single best-overlapping call per truth region.
    d_start, d_end = [], []
    calls_by_chrom = _by_chrom(calls)
    for chrom, ts, te in truth:
        best, best_ov = None, 0
        for cs, ce in calls_by_chrom.get(chrom, ()):
            ov = min(te, ce) - max(ts, cs)
            if ov > best_ov:
                best, best_ov = (cs, ce), ov
        if best is not None and best_ov >= MIN_OVERLAP_BP:
            d_start.append(abs(best[0] - ts))
            d_end.append(abs(best[1] - te))

    return {
        "n_calls": len(calls),
        "called_bp": called_bp,
        "territory": called_bp / genome_size if genome_size else float("nan"),
        "bp_precision": bp_precision,
        "bp_recall": bp_recall,
        "bp_f1": bp_f1,
        "jaccard": jaccard,
        "region_recall": region_recall,
        "fragmentation": fragmentation,
        "fusion": fusion,
        "median_dstart": float(np.median(d_start)) if d_start else float("nan"),
        "median_dend": float(np.median(d_end)) if d_end else float("nan"),
        "n_boundary_matched": len(d_start),
    }


def evaluate_null(calls: Sequence[Interval], genome_size: int) -> Dict[str, float]:
    """Specificity on a dataset containing no enrichment: every call is false."""
    calls = [(c, int(s), int(e)) for c, s, e in calls if e > s]
    called_bp = interval_total_bp(calls)
    return {
        "n_calls": len(calls),
        "called_bp": called_bp,
        "territory": called_bp / genome_size if genome_size else float("nan"),
        "bp_precision": 0.0 if calls else float("nan"),
        "bp_recall": float("nan"),
        "bp_f1": float("nan"),
        "jaccard": 0.0,
        "region_recall": float("nan"),
        "fragmentation": float("nan"),
        "fusion": float("nan"),
        "median_dstart": float("nan"),
        "median_dend": float("nan"),
        "n_boundary_matched": 0,
    }


METRIC_LABELS = {
    "n_calls": "calls",
    "territory": "genome fraction called",
    "bp_precision": "base-pair precision",
    "bp_recall": "base-pair recall",
    "bp_f1": "base-pair F1",
    "jaccard": "Jaccard (territory)",
    "region_recall": "region recall",
    "fragmentation": "fragmentation index",
    "fusion": "fusion index",
    "median_dstart": "median |Δstart| (bp)",
    "median_dend": "median |Δend| (bp)",
    "runtime_s": "wall-clock (s)",
}
