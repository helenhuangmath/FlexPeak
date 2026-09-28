"""Method wrappers for the comparison.

Two classes of comparator, kept strictly separate because they support
different claims:

**External tools** (``kind="tool"``) -- the published software, executed as a
subprocess.  The exact version and the full command line are recorded for every
run.  Results are directly comparable to published behaviour.

**Re-implemented baselines** (``kind="baseline"``) -- the *algorithm* of a
published method, implemented here because the tool itself could not be
installed in this environment (see Benchmark/README.md).  These support claims
about the algorithm class, NOT about the published software: they are neither
version-matched nor author-tuned, and any difference may be an artefact of this
implementation.  They must never be reported as "we benchmarked against X".

The baselines deliberately share FlexPeak's negative-binomial numerics
(``flexpeak.stats.nbtest``) where NB testing is involved.  That is the point:
holding the arithmetic fixed isolates the contribution of the *structural*
difference (HMM segmentation, local-lambda background, scale-space sub-peaks)
rather than confounding it with a second implementation of the same maths.
"""

from __future__ import annotations

import os
import subprocess
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from flexpeak import CallParams, call_peaks
from flexpeak.background.nb import estimate_dispersion
from flexpeak.stats.nbtest import nb_neglog10_sf, qvalue_from_neglog10p

Interval = Tuple[str, int, int]


# -- output parsing ---------------------------------------------------------

def read_bed_like(path: str) -> List[Interval]:
    """First three columns of any BED-derived format; comments skipped."""
    out: List[Interval] = []
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            f = line.split("\t")
            if len(f) < 3:
                continue
            try:
                out.append((f[0], int(f[1]), int(f[2])))
            except ValueError:
                continue
    return out


# -- FlexPeak ---------------------------------------------------------------

def run_flexpeak(treat, control, params: Optional[CallParams] = None,
                 tune_first: bool = False) -> Dict:
    """FlexPeak on in-memory coverage.  Returns intervals plus timing.

    ``tune_first`` runs the runtime parameter sweep; timing then includes it,
    since it is part of what the user pays for.
    """
    params = params or CallParams()
    note = ""
    t0 = time.perf_counter()
    if tune_first:
        from flexpeak import tune as tune_fn

        res = tune_fn(treat, control, base=params, n_chroms=2,
                      thin_fractions=(0.5,), seeds=(0,))
        params = res.params
        note = "tuned: " + ", ".join(
            f"{k}={v}" for k, v in sorted(res.params.to_dict().items())
            if k in {"posterior_cutoff", "min_fold", "min_width"})
    peaks = call_peaks(treat, control, params)
    runtime = time.perf_counter() - t0
    return {
        "intervals": [(r.chrom, r.start, r.end) for r in peaks],
        "runtime_s": runtime,
        "command": f"flexpeak.call_peaks(params={params.to_dict()})",
        "note": note,
    }


# -- MACS2 / MACS3 ----------------------------------------------------------

def run_macs(exe: str, treat_bam: str, control_bam: Optional[str],
             genome_size: int, outdir: str, name: str,
             broad: bool = False, extsize: int = 50,
             qvalue: float = 0.05, broad_cutoff: float = 0.1) -> Dict:
    """MACS2 or MACS3 as a subprocess.

    ``--nomodel --extsize`` is used because the simulated reads carry no strand
    asymmetry, so MACS's cross-correlation fragment-size model has nothing to
    estimate from.  ``extsize`` is set to the true simulated fragment length,
    which is the information FlexPeak also derives from the same file -- neither
    method is given an advantage here.
    """
    os.makedirs(outdir, exist_ok=True)
    cmd = [exe, "callpeak", "-t", treat_bam]
    if control_bam:
        cmd += ["-c", control_bam]
    cmd += ["-g", str(int(genome_size)), "-n", name, "--outdir", outdir,
            "--nomodel", "--extsize", str(extsize), "-q", str(qvalue)]
    if broad:
        cmd += ["--broad", "--broad-cutoff", str(broad_cutoff)]

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    runtime = time.perf_counter() - t0
    if proc.returncode != 0:
        return {"intervals": [], "runtime_s": runtime, "command": " ".join(cmd),
                "note": f"FAILED rc={proc.returncode}: {proc.stderr.strip()[-200:]}"}

    suffix = "_peaks.broadPeak" if broad else "_peaks.narrowPeak"
    return {
        "intervals": read_bed_like(os.path.join(outdir, name + suffix)),
        "runtime_s": runtime,
        "command": " ".join(cmd),
        "note": "",
    }


def tool_version(exe: str) -> str:
    try:
        p = subprocess.run([exe, "--version"], capture_output=True, text=True)
        return (p.stdout + p.stderr).strip().splitlines()[0]
    except Exception as e:  # pragma: no cover
        return f"unavailable ({e})"


# -- SICER / epic2 style: fixed window, fixed gap ---------------------------

def run_sicer_like(treat, control, window: int = 200, gap_windows: int = 3,
                   p_eligible: float = 0.20, qvalue: float = 0.05,
                   min_eligible: int = 1, lambda_floor: float = 0.25) -> Dict:
    """Re-implementation of the SICER / epic2 island approach.

    The published algorithm: score fixed-width windows against a Poisson
    background, mark windows individually 'eligible' at a permissive threshold,
    link eligible windows separated by no more than ``gap_windows`` ineligible
    windows into islands, then test islands and FDR-control them.

    The fixed gap parameter is the mechanism this benchmark is testing: it is
    what produces the documented over-merge behaviour, and it is exactly what
    FlexPeak replaces with an HMM posterior.

    NOT epic2 and NOT SICER2 -- see the module docstring.
    """
    t0 = time.perf_counter()
    tw = treat.rebin(window) if treat.bin_size != window else treat
    cw = control.rebin(window) if control is not None and control.bin_size != window else control

    scale = 1.0
    if cw is not None:
        ct, cc = tw.total_signal, cw.total_signal
        scale = (ct / cc) if cc > 0 else 1.0

    # SICER forms and scores islands against a GENOME-WIDE Poisson background,
    # and only then uses the control to assign island p-values.  Scoring
    # eligibility against raw per-window control counts instead would let every
    # window where the control happens to dip become eligible, chaining
    # background into enormous islands -- a failure of the wrapper, not of the
    # published method.
    lam0 = max(tw.mean_signal(), 1e-9)

    islands: List[Interval] = []
    scores: List[float] = []
    for chrom in sorted(tw.bins):
        counts = tw.bins[chrom].astype(np.float64)

        # Eligibility against the genome background at SICER's permissive p0.
        p = stats.poisson.sf(np.maximum(counts - 1, 0), lam0)
        eligible = p <= p_eligible
        if not eligible.any():
            continue

        # Link eligible windows across gaps of at most gap_windows.
        idx = np.flatnonzero(eligible)
        starts = [idx[0]]
        ends = [idx[0]]
        n_elig = [1]
        for i in idx[1:]:
            if i - ends[-1] - 1 <= gap_windows:
                ends[-1] = i
                n_elig[-1] += 1
            else:
                starts.append(i)
                ends.append(i)
                n_elig.append(1)

        # Island p-value from the control, depth-scaled, as SICER does when a
        # control is supplied.
        if cw is not None and chrom in cw.bins:
            ctrl = cw.bins[chrom].astype(np.float64)[: counts.size] * scale
        else:
            ctrl = np.full(counts.size, lam0)

        chrom_len = tw.chrom_sizes.get(chrom, counts.size * window)
        for s, e, k in zip(starts, ends, n_elig):
            if k < min_eligible:
                continue
            obs = float(counts[s:e + 1].sum())
            exp = max(float(ctrl[s:e + 1].sum()),
                      lam0 * (e - s + 1) * lambda_floor)
            nlp = -stats.poisson.logsf(max(obs - 1, 0), exp) / np.log(10.0)
            islands.append((chrom, min(s * window, chrom_len),
                            min((e + 1) * window, chrom_len)))
            scores.append(float(np.nan_to_num(nlp, posinf=323.0)))

    kept: List[Interval] = []
    if islands:
        q = qvalue_from_neglog10p(np.array(scores))
        cut = -np.log10(qvalue)
        kept = [iv for iv, qq in zip(islands, q) if qq >= cut and iv[2] > iv[1]]

    return {
        "intervals": kept,
        "runtime_s": time.perf_counter() - t0,
        "command": (f"sicer_like(window={window}, gap_windows={gap_windows}, "
                    f"p_eligible={p_eligible}, min_eligible={min_eligible}, "
                    f"lambda_floor={lambda_floor}, q={qvalue})"),
        "note": "re-implementation of the SICER/epic2 algorithm, not the tool",
    }


# -- DESeq2 / csaw style: windowed NB test, no segmentation -----------------

def run_deseq2_like(treat, control, window: int = 200,
                    qvalue: float = 0.05, min_width: int = 200) -> Dict:
    """Re-implementation of windowed negative-binomial testing.

    This is the csaw / DESeq2 workflow applied to enrichment calling: count
    reads in fixed windows, fit a parametric mean-dispersion trend
    (``alpha(mu) = a0 + a1/mu``, the DESeq2 form), test each window against the
    depth-scaled control under NB, BH-correct, and merge adjacent significant
    windows into regions.

    DESeq2 is a differential-abundance method, not a peak caller; this is the
    standard way its model is used for enrichment, and it is the correct
    comparator for the *background model*, not for segmentation.  Because it
    shares FlexPeak's NB numerics, the difference between this row and FlexPeak
    isolates segmentation, local-lambda background and sub-peak detection.

    NOT DESeq2 itself -- see the module docstring.
    """
    t0 = time.perf_counter()
    tw = treat.rebin(window) if treat.bin_size != window else treat
    cw = control.rebin(window) if control is not None and control.bin_size != window else control

    scale = 1.0
    if cw is not None:
        ct, cc = tw.total_signal, cw.total_signal
        scale = (ct / cc) if cc > 0 else 1.0

    all_counts, all_mu, spans = [], [], []
    for chrom in sorted(tw.bins):
        counts = tw.bins[chrom].astype(np.float64)
        if cw is not None and chrom in cw.bins:
            mu = cw.bins[chrom].astype(np.float64)[: counts.size] * scale
        else:
            mu = np.full(counts.size, tw.mean_signal())
        mu = np.maximum(mu, max(tw.mean_signal() * 0.05, 1e-3))
        all_counts.append(counts)
        all_mu.append(mu)
        spans.append((chrom, counts.size))

    counts_all = np.concatenate(all_counts)
    mu_all = np.concatenate(all_mu)
    a0, a1 = estimate_dispersion(counts_all, mu_all)
    alpha_all = np.maximum(a0 + a1 / np.maximum(mu_all, 1e-9), 1e-8)

    nlp = nb_neglog10_sf(counts_all, mu_all, alpha_all)
    q = qvalue_from_neglog10p(nlp)
    sig = q >= -np.log10(qvalue)

    regions: List[Interval] = []
    off = 0
    for chrom, n in spans:
        s = sig[off:off + n]
        off += n
        if not s.any():
            continue
        idx = np.flatnonzero(s)
        starts = [idx[0]]
        ends = [idx[0]]
        for i in idx[1:]:
            if i == ends[-1] + 1:          # adjacent windows only; no gap rule
                ends[-1] = i
            else:
                starts.append(i)
                ends.append(i)
        chrom_len = tw.chrom_sizes.get(chrom, n * window)
        for a, b in zip(starts, ends):
            st = min(a * window, chrom_len)
            en = min((b + 1) * window, chrom_len)
            if en - st >= min_width:
                regions.append((chrom, st, en))

    return {
        "intervals": regions,
        "runtime_s": time.perf_counter() - t0,
        "command": f"deseq2_like(window={window}, q={qvalue}, min_width={min_width})",
        "note": "re-implementation of windowed NB (csaw/DESeq2) testing, not DESeq2",
    }


# -- MACS-style local Poisson, as a controlled ablation ---------------------

def run_poisson_ablation(treat, control, params: Optional[CallParams] = None) -> Dict:
    """FlexPeak with the NB background replaced by Poisson, nothing else changed.

    This is the cleanest available test of the background-model claim:
    identical segmentation, identical thresholds, identical sub-peak detection,
    one distributional assumption swapped.
    """
    params = (params or CallParams()).replace(background_model="poisson")
    t0 = time.perf_counter()
    peaks = call_peaks(treat, control, params)
    return {
        "intervals": [(r.chrom, r.start, r.end) for r in peaks],
        "runtime_s": time.perf_counter() - t0,
        "command": "flexpeak.call_peaks(background_model='poisson')",
        "note": "ablation: FlexPeak with a Poisson background",
    }
