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

        res = tune_fn(treat, control, base=params, n_chroms=2)
        params = res.params
        note = "tuned: " + ", ".join(
            f"{k}={v}" for k, v in sorted(res.params.to_dict().items())
            if k in {"posterior_cutoff", "min_fold", "min_width",
                     "segment_bins_per_scale", "segment_bin_size"})
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


# -- helpers for the external tools ----------------------------------------

def _run(cmd, cwd=None, env=None) -> Tuple[float, "subprocess.CompletedProcess"]:
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=env)
    return time.perf_counter() - t0, proc


def _fail(cmd, runtime, proc, note="FAILED") -> Dict:
    tail = (proc.stderr or proc.stdout or "").strip()[-400:]
    return {"intervals": [], "runtime_s": runtime,
            "command": " ".join(map(str, cmd)) if isinstance(cmd, (list, tuple)) else str(cmd),
            "note": f"{note} rc={proc.returncode}: {tail}"}


def bam_to_bed(bam: str, bed: str) -> str:
    """``bedtools bamtobed``, cached beside the BAM."""
    if not (os.path.exists(bed) and os.path.getsize(bed) > 0):
        with open(bed, "w") as fh:
            subprocess.run(["bedtools", "bamtobed", "-i", bam], stdout=fh, check=True)
    return bed


def bam_to_bedgraph(bam: str, bg: str) -> str:
    """``bedtools genomecov -bg``, cached beside the BAM (SEACR input)."""
    if not (os.path.exists(bg) and os.path.getsize(bg) > 0):
        with open(bg, "w") as fh:
            subprocess.run(["bedtools", "genomecov", "-bg", "-ibam", bam],
                           stdout=fh, check=True)
    return bg


def chrom_sizes_from_bam(bam: str, path: str) -> str:
    import pysam

    with pysam.AlignmentFile(bam) as af, open(path, "w") as fh:
        for n, l in zip(af.references, af.lengths):
            fh.write(f"{n}\t{l}\n")
    return path


# -- SICER2 (real tool) -----------------------------------------------------

def run_sicer2(treat_bam: str, control_bam: Optional[str], genome_size: int,
               outdir: str, window: int = 200, gap: int = 600,
               fragment: int = 50, fdr: float = 0.05, species: str = "mm10",
               species_size: int = 2_730_871_774, evalue: float = 1000) -> Dict:
    """SICER2 as a subprocess (``sicer``).

    SICER needs a named species for its chromosome list; the simulated genome
    uses mm10 chromosome names, so ``-s mm10`` with the effective genome
    fraction set to the simulated genome's share of mm10 gives it the correct
    background rate.  Windows and gap are SICER's own parameters: the defaults
    (200/600) and a broad-mark setting (1000/3000) are both run.
    """
    os.makedirs(outdir, exist_ok=True)
    t_bed = bam_to_bed(treat_bam, treat_bam[:-4] + ".bed")
    cmd = ["sicer", "-t", t_bed, "-s", species, "-w", str(window), "-g", str(gap),
           "-f", str(fragment), "-egf", f"{genome_size / species_size:.6f}",
           "-rt", "1", "-o", outdir, "-cpu", "1"]
    if control_bam:
        cmd += ["-c", bam_to_bed(control_bam, control_bam[:-4] + ".bed"),
                "-fdr", str(fdr)]
    else:
        cmd += ["-e", str(evalue)]
    runtime, proc = _run(cmd)
    if proc.returncode != 0:
        return _fail(cmd, runtime, proc)
    import glob

    base = os.path.basename(t_bed)[:-4]
    pats = ([f"{base}-W{window}-G{gap}-FDR{fdr}-island.bed",
             f"{base}-W{window}-G{gap}-FDR{fdr}-island.bed"] if control_bam
            else [f"{base}-W{window}-G{gap}-E{evalue:g}.scoreisland",
                  f"{base}-W{window}-G{gap}-E{int(evalue)}.scoreisland"])
    out = None
    for pat in pats + [f"{base}-W{window}-G{gap}*island*.bed", f"{base}-W{window}-G{gap}*.scoreisland"]:
        hits = glob.glob(os.path.join(outdir, pat))
        if hits:
            out = hits[0]
            break
    if out is None:
        return {"intervals": [], "runtime_s": runtime, "command": " ".join(cmd),
                "note": f"no island file found in {outdir}: {os.listdir(outdir)[:6]}"}
    return {"intervals": read_bed_like(out), "runtime_s": runtime,
            "command": " ".join(cmd), "note": ""}


# -- SEACR (real tool) ------------------------------------------------------

def run_seacr(seacr_sh: str, treat_bam: str, control_bam: Optional[str], outdir: str,
              mode: str = "stringent", norm: str = "norm",
              threshold: float = 0.01) -> Dict:
    """SEACR 1.3 as a subprocess: ``SEACR_1.3.sh exp.bg [ctrl.bg|thr] norm mode out``.

    SEACR is built for CUT&RUN fragment bedgraphs; here the simulated reads are
    their own fragments (50 bp), so ``bedtools genomecov -bg`` is the exact
    input it expects.  With a control, SEACR thresholds on the control's own
    signal distribution; without, ``threshold`` is the fraction of top regions
    kept -- both of SEACR's documented modes are run.
    """
    os.makedirs(outdir, exist_ok=True)
    t_bg = bam_to_bedgraph(treat_bam, treat_bam[:-4] + ".bedgraph")
    second = (bam_to_bedgraph(control_bam, control_bam[:-4] + ".bedgraph")
              if control_bam else str(threshold))
    prefix = os.path.join(outdir, os.path.basename(treat_bam)[:-4] + f"_{norm}_{mode}")
    cmd = ["bash", seacr_sh, t_bg, second, norm if control_bam else "non", mode, prefix]
    runtime, proc = _run(cmd, cwd=outdir)
    out = f"{prefix}.{mode}.bed"
    if proc.returncode != 0 or not os.path.exists(out):
        return _fail(cmd, runtime, proc)
    return {"intervals": read_bed_like(out), "runtime_s": runtime,
            "command": " ".join(cmd), "note": ""}


# -- differential comparators ------------------------------------------------

def run_flexpeak_diff(reps_a, reps_b, controls=None, bin_size: int = 100,
                      norm: str = "background", **kw) -> Dict:
    """FlexPeak differential on in-memory coverage.  Returns signed intervals."""
    from flexpeak.diff import DiffParams, differential_from_coverage

    t0 = time.perf_counter()
    params = DiffParams(bin_size=bin_size, norm=norm, **kw)
    ra = [r.rebin(bin_size) for r in reps_a]
    rb = [r.rebin(bin_size) for r in reps_b]
    ctrl = [c.rebin(bin_size) for c in (controls or [])] or None
    res = differential_from_coverage(ra, rb, ctrl, ctrl, params)
    return {
        "gain": res.intervals("gain"), "loss": res.intervals("loss"),
        "runtime_s": time.perf_counter() - t0,
        "command": f"flexpeak.diff.differential_from_coverage(params={params.to_dict()})",
        "note": f"rho={res.diagnostics['rho']:.4f} ({res.diagnostics['rho_source']})",
    }


def run_thor(bams_a: List[str], bams_b: List[str], chrom_sizes: str, outdir: str,
             name: str = "thor", inputs_a: Optional[List[str]] = None,
             inputs_b: Optional[List[str]] = None, pvalue: float = 0.05,
             extra: Sequence[str] = ()) -> Dict:
    """THOR (rgt-THOR) as a subprocess.

    THOR's narrowPeak output marks each differential peak's direction in the
    strand column: ``+`` higher in the first condition, ``-`` higher in the
    second (RGT documentation, "THOR output").  Condition A is given first, so
    ``+`` maps to ``gain``.
    """
    # THOR refuses to run if --output-dir already exists; write the config
    # beside it and let THOR create it.
    import shutil as _sh

    if os.path.isdir(outdir):
        _sh.rmtree(outdir)
    os.makedirs(os.path.dirname(outdir) or ".", exist_ok=True)
    cfg = os.path.join(os.path.dirname(outdir) or ".", f"{name}.config")
    with open(cfg, "w") as fh:
        fh.write("#rep1\n" + "\n".join(bams_a) + "\n#rep2\n" + "\n".join(bams_b) + "\n")
        # THOR pairs inputs with replicates one to one; a single input is
        # repeated for every replicate of the condition.
        if inputs_a:
            fh.write("#inputs1\n" + "\n".join((inputs_a * len(bams_a))[: len(bams_a)]) + "\n")
        if inputs_b:
            fh.write("#inputs2\n" + "\n".join((inputs_b * len(bams_b))[: len(bams_b)]) + "\n")
        fh.write(f"#chrom_sizes\n{chrom_sizes}\n")
    cmd = ["rgt-THOR", cfg, "--output-dir", outdir, "--name", name,
           "--pvalue", str(pvalue), "--no-correction"] + list(extra)
    runtime, proc = _run(cmd)
    out = os.path.join(outdir, f"{name}-diffpeaks.narrowPeak")
    if proc.returncode != 0 or not os.path.exists(out):
        r = _fail(cmd, runtime, proc)
        r.update(gain=[], loss=[])
        return r
    gain, loss = [], []
    with open(out) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 6 or line.startswith("#"):
                continue
            iv = (f[0], int(f[1]), int(f[2]))
            (gain if f[5] == "+" else loss).append(iv)
    return {"gain": gain, "loss": loss, "runtime_s": runtime,
            "command": " ".join(cmd), "note": ""}


def run_macs2_bdgdiff(exe: str, bams_a: List[str], bams_b: List[str], control_bam: str,
                      genome_size: int, outdir: str, name: str = "bdgdiff",
                      extsize: int = 50, cutoff: float = 3.0) -> Dict:
    """MACS2's documented differential workflow: callpeak -B per condition, then
    ``bdgdiff`` on the pileup/lambda tracks with the two depths."""
    os.makedirs(outdir, exist_ok=True)
    cmds, depths = [], []
    t0 = time.perf_counter()
    for tag, bams in (("A", bams_a), ("B", bams_b)):
        cmd = [exe, "callpeak", "-t", *bams, "-c", control_bam, "-g", str(genome_size),
               "-n", f"{name}_{tag}", "--outdir", outdir, "--nomodel", "--extsize",
               str(extsize), "-B", "-q", "0.05"]
        cmds.append(" ".join(cmd))
        _, proc = _run(cmd)
        if proc.returncode != 0:
            r = _fail(cmd, time.perf_counter() - t0, proc)
            r.update(gain=[], loss=[])
            return r
        # Depth after filtering, from the callpeak log ("tags after filtering in treatment").
        m = [l for l in proc.stderr.splitlines() if "tags after filtering in treatment" in l]
        depths.append(int(m[0].split(":")[-1].strip()) if m else 1)
    d1, d2 = depths
    cmd = [exe, "bdgdiff", "--t1", os.path.join(outdir, f"{name}_A_treat_pileup.bdg"),
           "--c1", os.path.join(outdir, f"{name}_A_control_lambda.bdg"),
           "--t2", os.path.join(outdir, f"{name}_B_treat_pileup.bdg"),
           "--c2", os.path.join(outdir, f"{name}_B_control_lambda.bdg"),
           "--d1", str(d1), "--d2", str(d2), "-g", "60", "-l", "120", "--cutoff", str(cutoff),
           "--o-prefix", os.path.join(outdir, name)]
    cmds.append(" ".join(cmd))
    _, proc = _run(cmd)
    runtime = time.perf_counter() - t0
    c1 = os.path.join(outdir, f"{name}_c3.0_cond1.bed")
    c2 = os.path.join(outdir, f"{name}_c3.0_cond2.bed")
    if proc.returncode != 0:
        r = _fail(cmd, runtime, proc)
        r.update(gain=[], loss=[])
        return r
    import glob

    c1 = (glob.glob(os.path.join(outdir, f"{name}_c*_cond1.bed")) or [c1])[0]
    c2 = (glob.glob(os.path.join(outdir, f"{name}_c*_cond2.bed")) or [c2])[0]
    return {"gain": read_bed_like(c1), "loss": read_bed_like(c2), "runtime_s": runtime,
            "command": " ; ".join(cmds), "note": f"depths {d1}/{d2}"}


def run_sicer_df(bams_a: List[str], bams_b: List[str], control_bam: Optional[str],
                 genome_size: int, outdir: str, window: int = 200, gap: int = 600,
                 fragment: int = 50, fdr: float = 0.05, species: str = "mm10",
                 species_size: int = 2_730_871_774) -> Dict:
    """SICER2's differential mode (``sicer_df``): islands on each condition,
    then increased/decreased islands between them.  Replicates are pooled
    because sicer_df takes one file per condition."""
    os.makedirs(outdir, exist_ok=True)

    def pooled_bed(bams, tag):
        path = os.path.join(outdir, f"{tag}_pooled.bed")
        if not os.path.exists(path):
            with open(path, "w") as fh:
                for b in bams:
                    subprocess.run(["bedtools", "bamtobed", "-i", b], stdout=fh, check=True)
        return path

    a_bed, b_bed = pooled_bed(bams_a, "A"), pooled_bed(bams_b, "B")
    cmd = ["sicer_df", "-t", a_bed, b_bed, "-s", species, "-w", str(window), "-g", str(gap),
           "-f", str(fragment), "-egf", f"{genome_size / species_size:.6f}", "-rt", "1",
           "-fdr_df", str(fdr), "-o", outdir, "-cpu", "1"]
    if control_bam:
        c_bed = bam_to_bed(control_bam, control_bam[:-4] + ".bed")
        cmd += ["-c", c_bed, c_bed, "-fdr", str(fdr)]
    else:
        cmd += ["-e", "1000"]
    runtime, proc = _run(cmd, cwd=outdir)
    if proc.returncode != 0:
        r = _fail(cmd, runtime, proc)
        r.update(gain=[], loss=[])
        return r
    import glob

    inc = glob.glob(os.path.join(outdir, "*increased*"))
    dec = glob.glob(os.path.join(outdir, "*decreased*"))
    if not inc or not dec:
        return {"gain": [], "loss": [], "runtime_s": runtime, "command": " ".join(cmd),
                "note": f"no increased/decreased files: {os.listdir(outdir)[:8]}"}
    # "increased" is relative to the first file (A): higher in A = gain.
    return {"gain": read_bed_like(inc[0]), "loss": read_bed_like(dec[0]),
            "runtime_s": runtime, "command": " ".join(cmd), "note": ""}
