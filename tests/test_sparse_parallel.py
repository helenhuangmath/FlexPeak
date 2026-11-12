"""Behaviour pinned by the sparse-library / speed work (SNU449 ATAC, 7-8M
fragments at 10 bp): a depth-aware background, per-chromosome workers, the
assembled-chromosome default, sub-peaks searched after the region FDR, the
width cap and the ATAC QC metrics."""

import numpy as np
import pytest

from flexpeak import call_peaks
from flexpeak.background.nb import (
    BackgroundModel, effective_windows, _signal_threshold,
)
from flexpeak.caller import _cap_width, preset_params
from flexpeak.io.bam import _bin_fragments
from flexpeak.io.load import primary_chromosomes
from flexpeak.peaks import NestedPeakSet, Region, SubPeak
from flexpeak.qc.metrics import fragment_size_metrics, qc_metrics, tss_enrichment
from flexpeak.signal.coverage import Coverage
from flexpeak.stats.nbtest import autocorrelation_length_bp

BIN = 10
SIZES = {"chr1": 4_000_000, "chr2": 2_500_000}
RATE = 2.5e-3          # background fragments per bp: a 7-8M fragment human library
PEAKS_PER_CHROM = 40
PEAK_FRAGMENTS = 30


def _sparse_atac(seed=0, peaks=True):
    """Fragment pileup of a shallow ATAC-like library: Poisson background at
    2.5 fragments per kb plus 300 bp peaks of 30 fragments."""
    rng = np.random.default_rng(seed)
    bins, truth, n_frag = {}, [], 0
    for chrom, size in SIZES.items():
        n_bg = rng.poisson(RATE * size)
        starts = rng.integers(0, size - 400, n_bg)
        lengths = rng.integers(60, 280, n_bg)
        if peaks:
            centres = np.linspace(50_000, size - 50_000, PEAKS_PER_CHROM).astype(int)
            for c in centres:
                s = c + rng.integers(-150, 150, PEAK_FRAGMENTS) - 75
                starts = np.concatenate([starts, s])
                lengths = np.concatenate([lengths, rng.integers(60, 200, PEAK_FRAGMENTS)])
                truth.append((chrom, int(c) - 150, int(c) + 150))
        bins[chrom] = _bin_fragments(starts, starts + lengths, size, BIN)
        n_frag += int(starts.size)
    cov = Coverage(bins=bins, bin_size=BIN, chrom_sizes=dict(SIZES), is_count=True,
                   n_fragments=n_frag, meta={"paired": True})
    return cov, truth


@pytest.fixture(scope="module")
def sparse():
    return _sparse_atac(seed=1)


def _hit(truth, peaks):
    regs = [(r.chrom, r.start, r.end) for r in peaks]
    return sum(any(c == rc and s < re and e > rs for rc, rs, re in regs)
               for c, s, e in truth)


# -- depth-aware background -------------------------------------------------

def test_effective_windows_follow_the_fragment_rate():
    # 2.5 fragments per kb: 20 fragments need 8 kb, and 8 kb / 10 kb merge.
    assert effective_windows((1000, 5000, 10000), 2.5e-3, 10) == (8000,)
    # A deep library keeps the ladder it was given.
    assert effective_windows((1000, 5000, 10000), 50e-3, 10) == (1000, 5000, 10000)
    # Unknown depth (a normalised track): untouched.
    assert effective_windows((1000, 5000, 10000), None, 10) == (1000, 5000, 10000)


def test_signal_threshold_needs_more_than_one_fragment_when_sparse():
    level = np.array([0.4], dtype=np.float32)
    # Deep / unknown depth: the plain fold rule.
    assert _signal_threshold(level, 2.0, 10, None)[0] == pytest.approx(0.8)
    # Sparse: one fragment (17 bin-units over a 10-bin window = 1.7) must not
    # count as signal at a background of 0.4 per bin.
    assert _signal_threshold(level, 2.0, 10, 17.0)[0] > 1.7
    # With plenty of fragments in the window the fold rule is the larger one.
    assert _signal_threshold(np.array([40.0], dtype=np.float32), 2.0, 10, 17.0)[0] \
        == pytest.approx(80.0)


def test_sparse_background_is_the_rate_not_the_empty_bins(sparse):
    cov, _ = sparse
    rate = cov.mean_signal()
    aware = BackgroundModel.fit(cov, windows=(1000, 5000, 10000))
    plain = BackgroundModel.fit(cov, windows=(1000, 5000, 10000), depth_aware=False)
    med_aware = np.median(np.concatenate(list(aware.mu.values())))
    med_plain = np.median(np.concatenate(list(plain.mu.values())))
    # Depth-aware: the background sits at the library's rate and almost
    # nothing is set aside as signal.
    assert 0.75 * rate < med_aware < 1.35 * rate
    assert aware.meta["excluded_fraction"] < 0.05
    assert min(aware.meta["windows"]) > 5000      # widened to hold ~20 fragments
    # The plain fold rule holds out every covered stretch and fits the gaps.
    assert med_plain < 0.7 * med_aware


def test_sparse_atac_calls_peaks_not_domains(sparse):
    cov, truth = sparse
    peaks = call_peaks(cov, params=preset_params("atac"))
    assert _hit(truth, peaks) >= 0.9 * len(truth)
    assert max(peaks.widths()) <= 5_000
    # Few calls away from the planted peaks.
    assert len(peaks) <= 1.2 * len(truth)


def test_null_sparse_library_stays_quiet():
    cov, _ = _sparse_atac(seed=5, peaks=False)
    peaks = call_peaks(cov, params=preset_params("atac"))
    assert len(peaks) <= 2


# -- workers ----------------------------------------------------------------

def test_parallel_call_equals_serial(sparse):
    cov, _ = sparse
    p = preset_params("atac")
    serial = call_peaks(cov, params=p)
    forked = call_peaks(cov, params=p, n_jobs=2)
    key = lambda ps: [(r.chrom, r.start, r.end, r.kind, round(r.neg_log10_q, 6),
                       tuple((s.start, s.end, s.summit) for s in r.subpeaks))
                      for r in ps]
    assert key(serial) == key(forked)


def test_segmentation_cache_survives_workers(sparse):
    cov, _ = sparse
    p = preset_params("atac")
    cache = {}
    first = call_peaks(cov, params=p, segmentation_cache=cache, n_jobs=2)
    fits = [k for k in cache if isinstance(k, tuple) and k[0] in cov.bins]
    assert len(fits) == len(cov.bins)
    for k in fits:  # posterior back at reporting resolution
        assert cache[k].posterior.size == cov.bins[k[0]].size
    again = call_peaks(cov, params=p, segmentation_cache=cache)
    assert [(r.chrom, r.start, r.end) for r in first] == \
        [(r.chrom, r.start, r.end) for r in again]


# -- chromosomes --------------------------------------------------------------

def test_primary_chromosomes_drop_contigs_and_chrM():
    sizes = {"chr1": 248_000_000, "chr2": 242_000_000, "chrX": 156_000_000,
             "chrY": 57_000_000, "chrM": 16_569, "chrEBV": 171_823,
             "chrUn_KI270302v1": 2_274, "chr1_KI270706v1_random": 175_055,
             "chr6_GL000250v2_alt": 4_672_374, "GL000220.1": 161_802,
             "HLA-A*01:01:01:01": 3_503}
    assert primary_chromosomes(sizes) == ["chr1", "chr2", "chrX", "chrY"]
    # Other naming schemes are kept whole rather than emptied.
    refseq = {"NC_000001.11": 248_000_000, "NC_000002.12": 242_000_000}
    assert primary_chromosomes(refseq) == list(refseq)
    assert primary_chromosomes({"sim1": 10, "sim2": 5}) == ["sim1", "sim2"]
    assert primary_chromosomes({"2L": 23_000_000, "chrIV": 1_500_000, "MT": 16_000}) \
        == ["2L", "chrIV"]


def test_subset_keeps_signal_per_fragment(sparse):
    cov, _ = sparse
    sub = cov.subset(["chr2"])
    assert sub.total_signal / sub.n_fragments == pytest.approx(
        cov.total_signal / cov.n_fragments, rel=1e-3)


# -- width cap, blacklist ------------------------------------------------------

def test_cap_width_reports_subpeaks_of_an_oversized_region():
    sub = SubPeak("chr1", 5_000, 5_400, 5_200, score=40.0, neg_log10_p=9.0,
                  neg_log10_q=6.0, fold_enrichment=8.0)
    wide = Region("chr1", 1_000, 201_000, kind="domain", subpeaks=[sub])
    empty = Region("chr1", 300_000, 420_000, kind="domain")
    ok = Region("chr1", 500_000, 501_000)
    out, n = _cap_width(NestedPeakSet([wide, empty, ok], {}, {"chr1": 10**6}), 50_000)
    assert n == 2
    assert [(r.start, r.end, r.kind) for r in out] == \
        [(5_000, 5_400, "peak"), (500_000, 501_000, "peak")]
    assert out.regions[0].neg_log10_q == 6.0


def test_exclude_matches_brute_force():
    rng = np.random.default_rng(3)
    starts = np.sort(rng.integers(0, 990_000, 400))
    regions = [Region("chr1", int(s), int(s) + int(w))
               for s, w in zip(starts, rng.integers(50, 3_000, 400))]
    black = [("chr1", int(s), int(s) + 5_000) for s in rng.integers(0, 990_000, 30)]
    black += [("chr1", 100, 200), ("chr1", 150, 400), ("chr2", 0, 10**6)]
    ps = NestedPeakSet(regions, {}, {"chr1": 10**6})
    kept = {(r.start, r.end) for r in ps.exclude(black)}
    brute = {(r.start, r.end) for r in regions
             if not any(c == "chr1" and s < r.end and e > r.start for c, s, e in black)}
    assert kept == brute
    half = {(r.start, r.end) for r in ps.exclude(black, min_overlap_frac=0.5)}
    assert kept <= half


# -- scale estimate -------------------------------------------------------------

def test_autocorrelation_length_is_not_shortened_on_long_arrays():
    rng = np.random.default_rng(0)
    k = np.ones(60) / 60.0                     # features ~60 bins wide
    x = np.convolve(rng.poisson(0.05, 1_200_000).astype(float), k, mode="same")
    short = autocorrelation_length_bp(x[:300_000], 10)
    long_ = autocorrelation_length_bp(x, 10, max_bins=300_000)   # was strided x4
    assert short > 150
    assert abs(long_ - short) <= 0.25 * short


# -- ATAC QC ---------------------------------------------------------------------

def test_tss_enrichment_reads_signal_over_flanks():
    n = 200_000
    arr = np.full(n, 1.0, dtype=np.float32)
    tss = np.arange(10_000, 190_000, 10_000) * 10           # bp positions
    for t in tss // 10:
        arr[t - 10: t + 10] = 8.0
    cov = Coverage(bins={"chr1": arr}, bin_size=10, chrom_sizes={"chr1": n * 10})
    te = tss_enrichment(cov, {"chr1": tss})
    assert te["n_tss"] == tss.size
    assert 6.0 < te["score"] <= 8.0
    assert tss_enrichment(cov, {"chr9": tss}) is None


def test_fragment_size_metrics_sees_the_nucleosome_ladder():
    x = np.arange(1001)
    ladder = 5_000 * np.exp(-x / 60.0) + 900 * np.exp(-0.5 * ((x - 195) / 22.0) ** 2)
    ladder[:30] = 0
    m = fragment_size_metrics(ladder)
    assert m["nucleosome_ladder"] and 180 <= m["mononucleosome_peak_bp"] <= 210
    assert 0 < m["nucleosome_free_fraction"] < 1
    flat = fragment_size_metrics(5_000 * np.exp(-x / 60.0))
    assert not flat["nucleosome_ladder"]
    assert fragment_size_metrics(None) is None


def test_atac_qc_wording_and_paired_end_nsc(sparse):
    cov, _ = sparse
    peaks = call_peaks(cov, params=preset_params("atac"))
    m = qc_metrics(cov, peaks, assay="atac")
    assert m["nsc"] is None                       # not 0.0 for paired-end data
    text = " ".join(m["grade_reasons"])
    assert "failed IP" not in text
    assert "TSS enrichment not computed" in text


# -- speed work: cache, EM on a sample, single QC figure ----------------------

def test_cache_is_compact_exact_and_single(tmp_path, sparse):
    from flexpeak.io import load as L
    from flexpeak.simulate import write_bam

    pytest.importorskip("pysam")
    cov, _ = sparse
    small = cov.subset(["chr2"])
    bam = write_bam(str(tmp_path / "t.bam"), small, seed=1)
    cdir = tmp_path / "cache"
    whole = L.load_signal(bam, bin_size=10, cache_dir=cdir)
    files = list(cdir.glob("*.npz"))
    assert len(files) == 1
    z = np.load(files[0], allow_pickle=True)
    assert z["chr2"].dtype.kind == "u"                    # stored as small integers
    again = L.load_signal(bam, bin_size=10, cache_dir=cdir)
    assert again.bins["chr2"].dtype == np.float32
    assert np.array_equal(again.bins["chr2"], whole.bins["chr2"])
    # A coarser bin size is rebinned in memory, not written as a second cache.
    coarse = L.load_signal(bam, bin_size=50, cache_dir=cdir)
    assert coarse.bin_size == 50 and len(list(cdir.glob("*.npz"))) == 1


def test_subset_is_served_from_the_whole_genome_cache(tmp_path):
    from flexpeak.io.load import _superset_cache, _prune_caches, _stamp, cache_path

    src = tmp_path / "x.bam"
    src.write_bytes(b"not a real bam")
    cdir = tmp_path / "cache"
    cdir.mkdir()
    rng = np.random.default_rng(0)
    bins = {c: rng.poisson(1.0, 5_000).astype(np.float32) for c in ("chr1", "chr2", "chr3")}

    def save(chroms):
        cov = Coverage(bins={c: bins[c] for c in chroms}, bin_size=10,
                       chrom_sizes={c: 50_000 for c in chroms}, n_fragments=3_000,
                       meta={"cache_stamp": _stamp(src),
                             "chrom_signal": {c: float(bins[c].sum()) for c in chroms}})
        p = cache_path(src, 10, chroms, cdir)
        cov.save(p)
        return p

    part = save(["chr2"])
    full = save(["chr1", "chr2", "chr3"])
    sub = _superset_cache(src, 10, ["chr2", "chr3"], cdir)
    assert sub is not None and list(sub.bins) == ["chr2", "chr3"]
    assert np.array_equal(sub.bins["chr3"], bins["chr3"])
    assert 0 < sub.n_fragments < 3_000                    # scaled to the signal kept
    _prune_caches(src, 10, full, ["chr1", "chr2", "chr3"], cdir)
    assert not part.exists() and full.exists()            # the subset cache is redundant


def test_em_on_a_block_sample_gives_the_same_segmentation():
    from flexpeak.segment.hmm import NBHMM

    rng = np.random.default_rng(2)
    n = 600_000
    mu = np.full(n, 4.0)
    rate = mu.copy()
    for s in rng.integers(0, n - 60, 3_000):
        rate[s: s + 40] *= 5.0
    counts = rng.poisson(rate).astype(np.float64)
    full = NBHMM(decode_path=False).fit_predict(counts, mu, np.full(n, 0.05))
    part = NBHMM(decode_path=False, fit_fraction=0.25).fit_predict(counts, mu, np.full(n, 0.05))
    assert part.posterior.size == n and part.state.size == 0
    a, b = full.posterior >= 0.5, part.posterior >= 0.5
    assert (a & b).sum() / max((a | b).sum(), 1) > 0.97
    assert np.allclose(part.multipliers, full.multipliers, rtol=0.1)


def test_qc_figure_is_one_file(tmp_path, sparse):
    pytest.importorskip("matplotlib")
    from flexpeak.plots import qc_figure

    cov, _ = sparse
    peaks = call_peaks(cov, params=preset_params("atac"))
    m = qc_metrics(cov, peaks, assay="atac")
    x = np.arange(1001)
    sizes = 5_000 * np.exp(-x / 60.0) + 900 * np.exp(-0.5 * ((x - 195) / 22.0) ** 2)
    out = qc_figure(peaks, str(tmp_path / "s_qc.pdf"),
                    qc=dict(m, fragment_sizes=sizes, tss_profile=[1.0] * 200 + [6.0] + [1.0] * 200))
    assert list(tmp_path.iterdir()) == [tmp_path / "s_qc.pdf"] and out.endswith("s_qc.pdf")


# -- caches are removed after callpeak --------------------------------------------

def _small_bam(tmp_path, name="s"):
    from flexpeak.simulate import simulate, write_bam

    pytest.importorskip("pysam")
    sim = simulate(chrom_sizes={"chr1": 300_000}, n_narrow=8, n_broad=0, n_mixed=0, seed=7)
    return write_bam(str(tmp_path / f"{name}.bam"), sim.treatment, seed=1)


@pytest.mark.parametrize("keep", [False, True])
def test_callpeak_removes_its_cache_unless_kept(tmp_path, keep):
    from flexpeak import cli

    bam = _small_bam(tmp_path)
    cdir = tmp_path / "cache"
    other = cdir / "s.25bp.0000000000000000.npz"          # same stem, other source
    cdir.mkdir()
    Coverage(bins={"chr1": np.ones(10, np.float32)}, bin_size=25,
             chrom_sizes={"chr1": 250}, source=str(tmp_path / "elsewhere" / "s.bam")).save(other)
    argv = ["callpeak", "-t", bam, "-n", "s", "--outdir", str(tmp_path / "out"),
            "--no-tune", "--cache-dir", str(cdir), "--quiet"] + (["--keep-cache"] if keep else [])
    assert cli.main(argv) == 0
    assert (tmp_path / "out" / "s_peaks.narrowPeak").exists()
    mine = [f for f in cdir.glob("*.npz") if f != other]
    assert bool(mine) == keep
    assert other.exists()                                   # never touched


def test_default_cache_directory_is_removed_when_empty(tmp_path):
    from flexpeak import cli

    bam = _small_bam(tmp_path)
    assert cli.main(["callpeak", "-t", bam, "-n", "s", "--outdir", str(tmp_path / "out"),
                     "--no-tune", "--quiet"]) == 0
    assert not (tmp_path / ".flexpeak_cache").exists()
