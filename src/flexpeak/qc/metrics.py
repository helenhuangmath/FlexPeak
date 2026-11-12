"""QC metrics (DESIGN.md section 9).

Deliberately a subset for v0: the library/enrichment/peak statistics that can be
computed from binned coverage plus a call set.  The HTML report and the
replicate/IDR section land in P4.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..peaks import NestedPeakSet
from ..signal.coverage import Coverage

__all__ = ["qc_metrics", "format_qc", "tss_enrichment", "fragment_size_metrics",
           "read_tss"]


#: Bins the distribution summaries (Gini) are computed on.
_QC_MAX_BINS = 8_000_000


def _gini(x) -> float:
    """Coverage inequality. Near 0 = flat (no enrichment), near 1 = highly focal."""
    x = np.asarray(x).ravel()
    if x.size > _QC_MAX_BINS:  # a strided sample: sorting 300M bins is 10 s
        x = x[:: int(np.ceil(x.size / _QC_MAX_BINS))]
    x = np.sort(x.astype(np.float64))
    n = x.size
    if n == 0 or x.sum() <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float(2.0 * (idx * x).sum() / (n * x.sum()) - (n + 1) / n)


def _genome_block(cov: Coverage, n: int) -> np.ndarray:
    """A contiguous run of ``n`` bins from the middle of the concatenated
    genome (the bins a block-limited statistic would have read from the
    concatenation), without building the concatenation."""
    total = cov.n_bins
    if total <= n:
        return cov.concat()
    lo = (total - n) // 2
    hi = lo + n
    parts, pos = [], 0
    for c in sorted(cov.bins):
        a = cov.bins[c]
        s, e = max(lo - pos, 0), min(hi - pos, a.size)
        if e > s:
            parts.append(a[s:e])
        pos += a.size
        if pos >= hi:
            break
    return np.concatenate(parts)


def _autocorrelation_length(x, max_lag: int = 400) -> int:
    """Lag (in bins) at which autocorrelation first drops below 1/e.

    A direct read on whether the signal is punctate or domain-like, and the
    reason a mark's natural scale can be inferred without being told.
    FFT-based on a strided sample: the lag-by-lag loop was 400 passes over
    the whole genome, minutes on a 10 bp mouse genome.
    """
    from ..stats.nbtest import autocorrelation_length_bp

    x = np.asarray(x)
    if x.size < 10:
        return 0
    # A contiguous block, not a strided sample: striding by s measures the
    # lag in units of s bins, and the whole genome strided down to 4M bins
    # (s ~ 75 at 10 bp) reported a length that many times too short.
    cap = 4_000_000
    if x.size > cap:
        mid = (x.size - cap) // 2
        x = x[mid: mid + cap]
    lag_bp = autocorrelation_length_bp(x, 1, max_lag_bp=max_lag)
    return int(min(lag_bp, max_lag))


def _frip(cov: Coverage, peaks: NestedPeakSet) -> float:
    """Fraction of signal in peaks."""
    total = cov.total_signal
    if total <= 0:
        return 0.0
    bs = cov.bin_size
    inside = 0.0
    for r in peaks:
        a = cov.bins.get(r.chrom)
        if a is None:
            continue
        s = min(r.start // bs, a.size)
        e = min(r.end // bs, a.size)
        inside += float(a[s:e].sum())
    return float(inside / total)


def _bimodality(widths: List[int]) -> float:
    """Sarle's bimodality coefficient on log widths.

    Above ~0.555 suggests two width populations -- i.e. a genuinely mixed mark
    with both focal peaks and broad domains (DESIGN.md section 9).
    """
    if len(widths) < 8:
        return 0.0
    x = np.log10(np.asarray(widths, dtype=np.float64))
    s = x.std(ddof=1)
    if s == 0:
        return 0.0
    z = (x - x.mean()) / s
    n = x.size
    g1 = float((z ** 3).mean())
    g2 = float((z ** 4).mean() - 3.0)
    denom = g2 + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    if denom <= 0:
        return 0.0
    return float((g1 * g1 + 1.0) / denom)


def read_tss(path) -> Dict[str, np.ndarray]:
    """Transcription start sites from a GTF/GFF (``gene`` features, falling
    back to ``transcript``) or a BED (start of ``+`` intervals, end of ``-``;
    column 6 is the strand, ``+`` when absent).  Returns sorted, de-duplicated
    0-based positions per chromosome.  ``.gz`` is read transparently."""
    import gzip

    path = str(path)
    opener = gzip.open if path.endswith(".gz") else open
    is_gtf = path.lower().replace(".gz", "").endswith((".gtf", ".gff", ".gff3"))
    genes: Dict[str, list] = {}
    transcripts: Dict[str, list] = {}
    with opener(path, "rt") as fh:
        for ln in fh:
            if not ln or ln[0] == "#" or ln.startswith(("track", "browser")):
                continue
            if is_gtf and "\tgene\t" not in ln and "\ttranscript\t" not in ln:
                continue  # most of a GTF is exons and CDS: skip before splitting
            f = ln.rstrip("\n").split("\t", 8 if is_gtf else -1)
            if is_gtf:
                if len(f) < 8 or f[2] not in ("gene", "transcript"):
                    continue
                pos = int(f[3]) - 1 if f[6] != "-" else int(f[4]) - 1
                (genes if f[2] == "gene" else transcripts).setdefault(f[0], []).append(pos)
            else:
                if len(f) < 3:
                    continue
                strand = f[5] if len(f) > 5 else "+"
                pos = int(f[1]) if strand != "-" else int(f[2]) - 1
                genes.setdefault(f[0], []).append(pos)
    src = genes or transcripts
    return {c: np.unique(np.asarray(v, dtype=np.int64)) for c, v in src.items()}


def tss_enrichment(treatment: Coverage, tss: Dict[str, np.ndarray],
                   flank_bp: int = 2000, edge_bp: int = 100,
                   smooth_bp: int = 50) -> Optional[Dict]:
    """TSS enrichment score from binned fragment coverage (ENCODE ATAC QC).

    Coverage is aggregated over every TSS +/- ``flank_bp``, divided by the
    mean of the outer ``edge_bp`` at both ends (the local background), and
    the score is the maximum of that profile (smoothed over ``smooth_bp``)
    within the central 1 kb.  ENCODE reads it for GRCh38 as < 5 concerning,
    5-7 acceptable, > 7 ideal; the cut-offs depend on the annotation, and
    this is computed on fragment coverage rather than on cut sites, which
    runs a little lower.  Returns ``{"score", "n_tss", "profile"}`` or None
    when no TSS falls on a loaded chromosome.
    """
    bs = treatment.bin_size
    half = max(2, int(round(flank_bp / bs)))
    offsets = np.arange(-half, half + 1)
    total = np.zeros(offsets.size, dtype=np.float64)
    n = 0
    for chrom, arr in treatment.bins.items():
        pos = tss.get(chrom)
        if pos is None or len(pos) == 0:
            continue
        centre = np.asarray(pos, dtype=np.int64) // bs
        centre = centre[(centre - half >= 0) & (centre + half < arr.size)]
        # In blocks: a 2 kb window at 10 bp is 401 bins per TSS.
        for i in range(0, centre.size, 20_000):
            idx = centre[i: i + 20_000, None] + offsets[None, :]
            total += arr[idx].sum(axis=0, dtype=np.float64)
        n += int(centre.size)
    if n == 0:
        return None
    edge = max(1, int(round(edge_bp / bs)))
    base = float(np.concatenate([total[:edge], total[-edge:]]).mean())
    if base <= 0:
        return None
    profile = total / base
    k = max(1, int(round(smooth_bp / bs)))
    sm = np.convolve(profile, np.ones(k) / k, mode="same") if k > 1 else profile
    mid = np.abs(offsets * bs) <= 500
    return {"score": float(sm[mid].max()), "n_tss": n,
            "profile": [round(float(v), 3) for v in profile]}


def fragment_size_metrics(sizes) -> Optional[Dict]:
    """Nucleosome-ladder summary of a paired-end fragment-size histogram.

    ``sizes[i]`` is the number of fragments of length ``i`` bp.  Reports the
    median length, the nucleosome-free share (< 147 bp), the mono-nucleosome
    share (147-294 bp), their ratio (the "nucleosome signal": mono / free;
    above ~4 the library is dominated by nucleosomal fragments), and whether
    the ladder is visible: a mono-nucleosome mode in 160-250 bp that stands
    above the trough before it.  A Tn5 library with no ladder at all is
    over-digested or naked DNA.
    """
    h = np.asarray(sizes if sizes is not None else [], dtype=np.float64)
    tot = float(h.sum())
    if h.size < 300 or tot <= 0:
        return None
    cum = np.cumsum(h)
    free = float(h[:147].sum())
    mono = float(h[147:295].sum())
    k = 11
    sm = np.convolve(h, np.ones(k) / k, mode="same")
    peak = 160 + int(np.argmax(sm[160:251]))
    trough = 100 + int(np.argmin(sm[100:peak + 1])) if peak > 100 else peak
    contrast = float(sm[peak] / max(sm[trough], 1e-9))
    return {
        "fragment_median_bp": int(np.searchsorted(cum, tot / 2.0)),
        "nucleosome_free_fraction": free / tot,
        "mononucleosome_fraction": mono / tot,
        "nucleosome_signal": (mono / free) if free > 0 else float("inf"),
        "mononucleosome_peak_bp": peak,
        "nucleosome_ladder_contrast": contrast,
        "nucleosome_ladder": bool(trough < peak and contrast >= 1.1),
    }


def qc_metrics(
    treatment: Coverage,
    peaks: Optional[NestedPeakSet] = None,
    control: Optional[Coverage] = None,
    background=None,
    segment_bin_size: Optional[int] = None,
    assay: Optional[str] = None,
    tss: Optional[Dict[str, np.ndarray]] = None,
) -> Dict:
    """Compute QC metrics and a coarse quality grade.

    ``assay`` (``"atac"`` or None) selects the metrics and the wording that
    fit the experiment: an ATAC library has no IP to fail and no strand
    cross-correlation, and is judged on TSS enrichment (``tss``: positions
    from :func:`read_tss`) and its fragment-size ladder instead.
    """
    # Whole-genome numbers per chromosome, the distribution summaries on a
    # strided sample and the autocorrelation on one contiguous block: no
    # concatenated copy of the genome (1.2 GB at 10 bp).
    n_bins = int(treatment.n_bins)
    total = float(treatment.total_signal)
    n_zero = int(sum(int(a.size - np.count_nonzero(a)) for a in treatment.bins.values()))
    step = max(1, int(np.ceil(n_bins / _QC_MAX_BINS)))
    sample = (np.concatenate([a[::step] for a in treatment.bins.values()])
              if treatment.bins else np.zeros(0, dtype=np.float32))
    block = _genome_block(treatment, 4_000_000)
    m: Dict = {
        "source": treatment.source,
        "bin_size": treatment.bin_size,
        "n_bins": n_bins,
        "n_chroms": len(treatment.bins),
        "total_signal": total,
        "mean_signal": total / n_bins if n_bins else 0.0,
        "n_fragments": treatment.n_fragments,
        "fragment_length": treatment.fragment_length,
        "is_count_data": treatment.is_count,
        "zero_bin_fraction": n_zero / n_bins if n_bins else 0.0,
        "gini": _gini(sample),
        "autocorrelation_bins": _autocorrelation_length(block),
        # Strand cross-correlation exists for single-end data only; a cache
        # written before that distinction stored 0.0 for paired-end.
        "nsc": None if treatment.meta.get("paired") else treatment.meta.get("nsc"),
        "paired": treatment.meta.get("paired"),
        "assay": assay,
    }
    m["autocorrelation_bp"] = m["autocorrelation_bins"] * treatment.bin_size

    frag = fragment_size_metrics(treatment.meta.get("fragment_sizes"))
    if frag:
        m.update(frag)
    if tss:
        te = tss_enrichment(treatment, tss)
        if te:
            m["tss_enrichment"] = te["score"]
            m["tss_n"] = te["n_tss"]
            m["tss_profile"] = te["profile"]

    if background is not None:
        m["dispersion_a0"] = background.a0
        m["dispersion_a1"] = background.a1
        m["used_control"] = background.used_control
        m["dispersion_at_mean"] = float(
            background.alpha(np.array([max(background.genome_mean, 1e-9)]))[0]
        )
        meta = getattr(background, "meta", None) or {}
        m["background_excluded_fraction"] = meta.get("excluded_fraction")
        m["background_exclude_fold"] = meta.get("exclude_fold")
        m["background_windows"] = meta.get("windows")
        m["fragments_per_kb"] = meta.get("fragments_per_kb")
        seg = (getattr(peaks, "params", None) or {}).get("segment_dispersion") if peaks else None
        if seg:
            m["segment_dispersion_a0"] = seg.get("a0")
            m["segment_dispersion_a1"] = seg.get("a1")

    if peaks is not None:
        widths = peaks.widths()
        genome = sum(treatment.chrom_sizes.values()) or 1
        m.update({
            "n_regions": len(peaks),
            "n_domains": peaks.n_domains,
            "n_subpeaks": peaks.n_subpeaks,
            "total_peak_bp": peaks.total_bp,
            "genome_fraction": peaks.total_bp / float(genome),
            "median_width": float(np.median(widths)) if widths else 0.0,
            "max_width": int(max(widths)) if widths else 0,
            "frip": _frip(treatment, peaks),
            "width_bimodality": _bimodality(widths),
        })
        seg = segment_bin_size or (peaks.params or {}).get("segment_bin_size")
        if seg:
            m["segment_bin_size"] = int(seg)
            # Regions that are exactly one segmentation bin wide are not domains
            # the model joined up -- they are single blocks, and their width is
            # the bin size showing through.  A call set where that is the
            # typical outcome has width by quantisation, and reporting a median
            # width without saying so would be misleading.
            m["single_bin_fraction"] = float(
                np.mean(np.asarray(widths, dtype=np.float64) <= int(seg))
            )

    m["grade"], m["grade_reasons"] = _grade(m)
    return m


def _grade(m: Dict):
    """Coarse A-D grade with explicit reasons.

    QC as an explanation of the caller's behaviour, not a wall of numbers.
    """
    score = 0
    reasons: List[str] = []

    atac = (m.get("assay") or "").lower() == "atac"
    frip = m.get("frip")
    if frip is not None:
        if frip >= 0.20:
            score += 2
        elif frip >= 0.05:
            score += 1
            reasons.append(
                f"FRiP {frip:.3f} is modest (>=0.05 acceptable, >=0.20 good)")
        elif atac:
            reasons.append(
                f"FRiP {frip:.3f} is low: few fragments fall in called accessible "
                f"regions (low signal-to-background library, or too shallow to "
                f"call most of its peaks)")
        else:
            reasons.append(f"FRiP {frip:.3f} is low: weak enrichment or failed IP")

    tsse = m.get("tss_enrichment")
    gini = m.get("gini", 0.0)
    if atac and tsse is not None:
        # For ATAC the enrichment component is TSS enrichment, the assay's
        # own signal-to-background measure (ENCODE: > 7 ideal, 5-7
        # acceptable on GRCh38), not the Gini of the pileup.
        if tsse >= 7:
            score += 2
        elif tsse >= 5:
            score += 1
            reasons.append(f"TSS enrichment {tsse:.1f} is acceptable (5-7; > 7 ideal)")
        else:
            reasons.append(
                f"TSS enrichment {tsse:.1f} is low (< 5): weak signal over "
                f"background -- over-transposed, dead cells or free DNA")
    elif gini >= 0.6:
        score += 2
    elif gini >= 0.35:
        score += 1
        reasons.append(f"coverage Gini {gini:.2f}: enrichment present but diffuse")
    else:
        reasons.append(f"coverage Gini {gini:.2f}: signal is nearly flat")
    if atac and tsse is None:
        reasons.append("TSS enrichment not computed: pass --tss <GTF or BED> "
                       "(the standard ATAC signal-to-background metric)")

    if atac and m.get("nucleosome_ladder") is not None:
        if not m["nucleosome_ladder"]:
            reasons.append(
                f"no mono-nucleosome peak in the fragment sizes (contrast "
                f"{m.get('nucleosome_ladder_contrast', 0):.2f}): the nucleosome "
                f"ladder is missing -- size-selected, over-digested or naked DNA")
        elif m.get("nucleosome_signal", 0) > 4:
            reasons.append(
                f"nucleosome signal {m['nucleosome_signal']:.1f} (mono / "
                f"nucleosome-free fragments) is high: few sub-nucleosomal fragments")

    disp = m.get("dispersion_at_mean")
    if disp is not None and disp <= 1.5e-4 and m.get("is_count_data"):
        reasons.append(
            "per-bin background is at the Poisson limit (no overdispersion "
            "measurable at this depth): region significance rests on the "
            "fragment-overlap correction between neighbouring bins")
    if disp is not None and disp > 0.5:
        reasons.append(
            f"background dispersion alpha={disp:.2f} is high: noisy library, "
            f"the NB model is doing real work here (a Poisson caller would "
            f"report inflated significance)")

    sbf = m.get("single_bin_fraction")
    if sbf is not None and sbf >= 0.5 and m.get("segment_bin_size", 0) > 1000:
        reasons.append(
            f"{sbf:.0%} of regions are one segmentation bin "
            f"({m['segment_bin_size']:,} bp) or narrower: their width is the "
            f"segmentation scale showing through, not domains the model joined "
            f"up. Treat the median width as an upper bound and try a finer "
            f"--segment-bin-size")

    excl = m.get("background_excluded_fraction")
    if excl is not None and excl > 0.5:
        reasons.append(
            f"{excl:.0%} of the genome was held out of the background fit as "
            f"apparent signal: the local-background windows are probably too "
            f"narrow for this mark, and what is left is not a background")

    zero = m.get("zero_bin_fraction")
    if zero is not None and zero > 0.9:
        reasons.append(f"{zero:.1%} of bins are empty: consider a larger bin size")

    bim = m.get("width_bimodality")
    if bim is not None and bim >= 0.555:
        reasons.append(
            f"peak-width bimodality {bim:.2f}: this looks like a MIXED mark "
            f"(both focal peaks and broad domains) -- nested output is "
            f"meaningful here")

    grade = "A" if score >= 5 else "B" if score >= 3 else "C" if score >= 1 else "D"
    return grade, reasons


_ROWS = (
    ("source", "source"),
    ("n_fragments", "fragments"),
    ("fragment_length", "fragment length (bp)"),
    ("paired", "paired-end"),
    ("bin_size", "bin size (bp)"),
    ("n_bins", "bins"),
    ("mean_signal", "mean signal/bin"),
    ("zero_bin_fraction", "empty bins"),
    ("gini", "coverage Gini"),
    ("autocorrelation_bp", "autocorrelation length (bp)"),
    ("nsc", "NSC (single-end)"),
    ("tss_enrichment", "TSS enrichment"),
    ("tss_n", "  TSSs used"),
    ("fragment_median_bp", "fragment median (bp)"),
    ("nucleosome_free_fraction", "  nucleosome-free (<147 bp)"),
    ("mononucleosome_fraction", "  mono-nucleosome (147-294)"),
    ("nucleosome_signal", "  nucleosome signal"),
    ("mononucleosome_peak_bp", "  mono-nucleosome peak (bp)"),
    ("fragments_per_kb", "fragments per kb"),
    ("background_windows", "background windows (bp)"),
    ("dispersion_at_mean", "NB dispersion at mean"),
    ("dispersion_a0", "  trend a0"),
    ("dispersion_a1", "  trend a1"),
    ("segment_dispersion_a0", "  segmentation-bin a0"),
    ("segment_dispersion_a1", "  segmentation-bin a1"),
    ("used_control", "control used"),
    ("background_exclude_fold", "background exclude fold"),
    ("background_excluded_fraction", "  genome held out as signal"),
    ("n_regions", "regions"),
    ("n_domains", "  domains"),
    ("n_subpeaks", "  subpeaks"),
    ("segment_bin_size", "segmentation bin (bp)"),
    ("single_bin_fraction", "  regions <= 1 segment bin"),
    ("median_width", "median width (bp)"),
    ("max_width", "max width (bp)"),
    ("genome_fraction", "genome fraction"),
    ("frip", "FRiP"),
    ("width_bimodality", "width bimodality"),
)


def format_qc(m: Dict) -> str:
    """Plain-text QC report."""
    lines = ["FlexPeak QC", "=" * 60]
    for key, label in _ROWS:
        v = m.get(key)
        if v is None:
            continue
        if isinstance(v, bool):
            txt = "yes" if v else "no"
        elif isinstance(v, float):
            txt = f"{v:,.4f}"
        elif isinstance(v, int):
            txt = f"{v:,}"
        else:
            txt = str(v)
        lines.append("  " + f"{label:<28}" + " " + txt)
    lines.append("")
    lines.append("  QUALITY GRADE: " + m.get("grade", "?"))
    for r in m.get("grade_reasons", []):
        lines.append("    - " + r)
    return "\n".join(lines)
