"""Scale-space sub-peak detection.

DESIGN.md section 3.3, layer 2.  Smooth the fine-binned signal at a ladder of
bandwidths, find local maxima at every scale, and link maxima across scales.
A maximum that persists across many scales is real structure; one that appears
at a single scale is noise.  This is standard practice in astronomical source
detection, and it is the right tool for "sharp and broad in the same track" --
it needs no prior declaration of peak width.

Persistence is carried through to the output (``SubPeak.persistence``) so users
can filter on it directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d

__all__ = ["DEFAULT_BANDWIDTHS", "ScaleSpaceMaximum", "find_subpeaks", "scale_space"]

DEFAULT_BANDWIDTHS = (50, 150, 500, 1500, 5000)


@dataclass
class ScaleSpaceMaximum:
    """One focal maximum, with the evidence that it is not noise."""

    index: int
    start: int
    end: int
    height: float
    persistence: int
    scale_bp: int


def scale_space(signal: np.ndarray, bin_size: int,
                bandwidths: Sequence[int] = DEFAULT_BANDWIDTHS) -> List[np.ndarray]:
    """Gaussian smoothing ladder.  Bandwidths below one bin are skipped."""
    out: List[np.ndarray] = []
    for bw in bandwidths:
        sigma = bw / float(bin_size) / 2.0
        if sigma < 0.5:
            out.append(np.asarray(signal, dtype=np.float64))
            continue
        out.append(gaussian_filter1d(np.asarray(signal, dtype=np.float64),
                                     sigma, mode="nearest"))
    return out


def _local_maxima(arr: np.ndarray, min_distance: int) -> np.ndarray:
    """Indices of local maxima, at least ``min_distance`` bins apart."""
    if arr.size < 3:
        return np.empty(0, dtype=int)
    size = 2 * max(1, min_distance) + 1
    mx = maximum_filter1d(arr, size=size, mode="nearest")
    idx = np.flatnonzero((arr == mx) & (arr > 0))
    if idx.size == 0:
        return idx
    keep = [idx[0]]
    for i in idx[1:]:
        if i - keep[-1] < max(1, min_distance):
            # Too close to the previous maximum: keep whichever is taller.
            if arr[i] > arr[keep[-1]]:
                keep[-1] = i
            continue
        keep.append(i)
    return np.array(keep, dtype=int)


def find_subpeaks(
    signal: np.ndarray,
    bin_size: int,
    background: Optional[np.ndarray] = None,
    bandwidths: Sequence[int] = DEFAULT_BANDWIDTHS,
    min_persistence: int = 2,
    min_fold: float = 1.5,
    boundary_fraction: float = 0.5,
    offset: int = 0,
) -> List[ScaleSpaceMaximum]:
    """Detect sub-peaks in one signal array.

    Parameters
    ----------
    background : per-bin expected background, used for the fold cutoff and for
        the descent that sets sub-peak boundaries.
    min_persistence : minimum number of scales a maximum must survive.  1 keeps
        everything (useful for the section 9.5 single-scale ablation).
    boundary_fraction : descend from each summit until the smoothed signal drops
        below this fraction of the summit height above background -- a
        half-maximum watershed.
    offset : added to every returned index, so callers working on a slice get
        coordinates in the parent array.
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = signal.size
    if n < 3:
        return []

    if background is None:
        bg = np.full(n, max(signal.mean(), 1e-9))
    else:
        bg = np.maximum(np.asarray(background, dtype=np.float64), 1e-9)

    pyramid = scale_space(signal, bin_size, bandwidths)
    maxima_per_scale = []
    for level, sm in enumerate(pyramid):
        min_dist = max(1, int(bandwidths[level] / bin_size / 2))
        maxima_per_scale.append(_local_maxima(sm, min_distance=min_dist))

    finest = maxima_per_scale[0]
    results: List[ScaleSpaceMaximum] = []
    claimed = np.zeros(n, dtype=bool)

    for idx in finest:
        # Persistence: how many coarser scales still show a maximum nearby.
        persistence = 1
        for level in range(1, len(pyramid)):
            others = maxima_per_scale[level]
            if others.size == 0:
                continue
            window = max(1, int(bandwidths[level] / bin_size))
            if np.min(np.abs(others - idx)) <= window:
                persistence += 1

        if persistence < min_persistence:
            continue
        height = float(signal[idx])
        if height < min_fold * bg[idx]:
            continue

        start, end = _descend(pyramid[0], bg, idx, boundary_fraction)
        # A summit whose whole footprint is already inside another sub-peak is
        # the same feature seen twice, not a second one.
        if claimed[start:end].all():
            continue
        claimed[start:end] = True
        results.append(
            ScaleSpaceMaximum(
                index=idx + offset,
                start=start + offset,
                end=end + offset,
                height=height,
                persistence=persistence,
                scale_bp=int(bandwidths[0]),
            )
        )

    results.sort(key=lambda m: m.start)
    return results


def _descend(smooth: np.ndarray, bg: np.ndarray, peak: int,
             fraction: float) -> Tuple[int, int]:
    """Walk outward from a summit to the half-maximum-above-background point."""
    n = smooth.size
    base = bg[peak]
    cutoff = base + fraction * max(smooth[peak] - base, 0.0)

    i = peak
    while i > 0 and smooth[i - 1] >= cutoff and smooth[i - 1] <= smooth[i] + 1e-12:
        i -= 1
    start = i

    j = peak
    while j < n - 1 and smooth[j + 1] >= cutoff and smooth[j + 1] <= smooth[j] + 1e-12:
        j += 1
    end = j + 1

    if end <= start:
        start = peak
        end = peak + 1
    return int(start), int(min(end, n))
