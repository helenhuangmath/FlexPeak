"""Behaviour pinned by the accuracy/speed work: the two-sided tuning objective,
the read split, genome-scoped FDR, sub-peaks judged against their domain, and
the vectorised BAM loader."""

import os
import tempfile

import numpy as np
import pytest

from flexpeak import CallParams, call_peaks, tune
from flexpeak.simulate import simulate
from flexpeak.stats.nbtest import qvalue_from_neglog10p
from flexpeak.tune.objective import HeldOut, TuningObjective, heldout_gain

SIZES = {"chr1": 1_500_000, "chr2": 750_000}


def _bp_f1(peaks, truth):
    from flexpeak.peaks import interval_overlap_bp, interval_total_bp

    calls = [(r.chrom, r.start, r.end) for r in peaks]
    inter = interval_overlap_bp(calls, truth)
    p = inter / max(interval_total_bp(calls), 1)
    r = inter / max(interval_total_bp(truth), 1)
    return 2 * p * r / max(p + r, 1e-9)


@pytest.fixture(scope="module")
def mixed():
    return simulate(chrom_sizes=SIZES, n_narrow=15, n_broad=2, n_mixed=2,
                    dispersion=0.15, seed=103)


@pytest.fixture(scope="module")
def lowqual():
    return simulate(chrom_sizes=SIZES, n_narrow=20, n_broad=2, n_mixed=0,
                    narrow_fold=3.0, broad_fold=1.6, background_level=6.0,
                    dispersion=0.40, seed=104)


# -- Coverage.split ---------------------------------------------------------

def test_split_halves_are_disjoint_and_sum_to_whole(mixed):
    a, b = mixed.treatment.split(seed=3)
    for c in mixed.treatment.bins:
        whole = np.rint(mixed.treatment.bins[c])
        assert np.array_equal(a.bins[c] + b.bins[c], whole)
        assert (a.bins[c] >= 0).all() and (b.bins[c] >= 0).all()
    # Independent halves of the same signal: strongly correlated where there
    # is signal, but not identical.
    x, y = a.bins["chr1"], b.bins["chr1"]
    assert 0.3 < np.corrcoef(x, y)[0, 1] < 0.99


# -- the held-out objective is two-sided -----------------------------------

def test_heldout_gain_penalises_undercalling_and_noise(mixed):
    a, b = mixed.treatment.split(seed=0)
    good = CallParams(bin_size=25)
    strict = CallParams(bin_size=25, min_fold=3.0, posterior_cutoff=0.7)
    ra = call_peaks(a, mixed.control, good, return_details=True)
    rb = call_peaks(b, mixed.control, good, return_details=True)
    ho_a, ho_b = HeldOut(a, ra.background.mu), HeldOut(b, rb.background.mu)

    g_good = heldout_gain(ra.peaks, ho_a, ho_b)
    strict_calls = call_peaks(a, mixed.control, strict, background=ra.background)
    g_strict = heldout_gain(strict_calls, ho_a, ho_b)
    assert g_good > 0
    # Dropping the domains forfeits their evidence: the strict setting that
    # the old objective preferred must score lower here.
    assert g_strict < 0.6 * g_good

    # Calls on pure background predict nothing on the other half.
    null = simulate(chrom_sizes=SIZES, n_narrow=0, n_broad=0, n_mixed=0,
                    dispersion=0.25, seed=105)
    na, nb = null.treatment.split(seed=0)
    rn = call_peaks(na, null.control, CallParams(bin_size=25, qvalue=1.0,
                                                 min_fold=1.0),
                    return_details=True)
    rnb = call_peaks(nb, null.control, CallParams(bin_size=25), return_details=True)
    g_null = heldout_gain(rn.peaks, HeldOut(na, rn.background.mu),
                          HeldOut(nb, rnb.background.mu))
    assert g_null < 0.05 * g_good


def test_tune_does_not_collapse_on_mixed_data(mixed):
    truth = mixed.truth_intervals()
    res = tune(mixed.treatment, mixed.control, base=CallParams(bin_size=25),
               n_chroms=2)
    f1_default = _bp_f1(call_peaks(mixed.treatment, mixed.control,
                                   CallParams(bin_size=25)), truth)
    f1_tuned = _bp_f1(call_peaks(mixed.treatment, mixed.control, res.params), truth)
    assert f1_tuned > 0.9
    assert f1_tuned > f1_default - 0.05
    assert res.score.heldout_gain is not None and res.score.heldout > 0.9
    # The report carries the new term.
    assert "held-out fit" in res.report()


def test_tuning_objective_requires_territory_penalty():
    with pytest.raises(ValueError):
        TuningObjective(territory_weight=0.0)


# -- weak broad marks and the FDR scope ------------------------------------

def test_weak_domains_are_called_at_default_min_fold(lowqual):
    """1.6x domains at dispersion 0.4.  The old 1.5 fold floor erased them
    outright; the defaults now recover most of the territory, and the tuner
    -- whose scale axis exists for exactly this sample -- recovers the rest."""
    truth = lowqual.truth_intervals()
    peaks = call_peaks(lowqual.treatment, lowqual.control, CallParams(bin_size=25))
    assert _bp_f1(peaks, truth) > 0.7
    old_floor = call_peaks(lowqual.treatment, lowqual.control,
                           CallParams(bin_size=25, min_fold=1.5, fdr_scope="candidates"))
    assert _bp_f1(peaks, truth) > _bp_f1(old_floor, truth)

    # The tuner must not make it worse (it used to collapse to F1 ~0.1 here).
    # It does not yet make it better either: at this depth the objective
    # cannot tell a 100 bp from a 500 bp segmentation, which is the open
    # limitation recorded in Benchmark/README.md.
    res = tune(lowqual.treatment, lowqual.control, base=CallParams(bin_size=25),
               n_chroms=2)
    tuned = call_peaks(lowqual.treatment, lowqual.control, res.params)
    assert _bp_f1(tuned, truth) >= _bp_f1(peaks, truth) - 0.02
    assert _bp_f1(tuned, truth) > 0.7


def test_genome_fdr_scope_is_more_conservative():
    p = np.array([8.0, 3.0, 1.5, 1.2, 0.4])
    q_cand = qvalue_from_neglog10p(p)
    q_gen = qvalue_from_neglog10p(p, n_tests=10_000)
    assert (q_gen <= q_cand + 1e-12).all()
    assert q_gen[0] < q_cand[0]
    # Ranks are unchanged by the padding, so ordering is preserved.
    assert np.array_equal(np.argsort(-q_gen), np.argsort(-q_cand))


def test_null_data_stays_quiet():
    null = simulate(chrom_sizes=SIZES, n_narrow=0, n_broad=0, n_mixed=0,
                    dispersion=0.25, seed=105)
    peaks = call_peaks(null.treatment, null.control, CallParams(bin_size=25))
    genome = sum(SIZES.values())
    assert peaks.total_bp / genome < 0.002


# -- sub-peaks inside domains ----------------------------------------------

def _overlaps(iv, others):
    return any(c == iv[0] and min(e, iv[2]) > max(s, iv[1]) for c, s, e in others)


def test_focal_subpeaks_inside_domains_are_found(mixed):
    peaks = call_peaks(mixed.treatment, mixed.control, CallParams(bin_size=25))
    focal = mixed.subpeak_intervals()
    assert focal
    dom_subs = [(r.chrom, s.start, s.end) for r in peaks
                if r.kind == "domain" for s in r.subpeaks]
    found = sum(_overlaps(f, dom_subs) for f in focal)
    assert found >= 0.8 * len(focal)
    # Narrow regions still carry their own summit.
    narrow = [(t.chrom, t.start, t.end) for t in mixed.truth if t.kind == "narrow"]
    pk_subs = [(r.chrom, s.start, s.end) for r in peaks
               if r.kind == "peak" for s in r.subpeaks]
    assert sum(_overlaps(n, pk_subs) for n in narrow) >= 0.8 * len(narrow)


def test_flat_noisy_domains_carry_no_subpeaks(lowqual):
    # Segmentation pinned so the domains come out whole and only the sub-peak
    # logic is under test.
    p = CallParams(bin_size=25, segment_bin_size=500)
    peaks = call_peaks(lowqual.treatment, lowqual.control, p)
    assert sum(r.kind == "domain" for r in peaks) >= 4
    n_dom_subs = sum(len(r.subpeaks) for r in peaks if r.kind == "domain")
    assert n_dom_subs <= 2
    # Ablation: without the domain-level reference the bumps come back.
    abl = call_peaks(lowqual.treatment, lowqual.control,
                     p.replace(subpeak_vs_domain=False))
    assert sum(len(r.subpeaks) for r in abl if r.kind == "domain") > 20


# -- the caller's reusable pieces ------------------------------------------

def test_background_and_segmentation_reuse_give_identical_calls(mixed):
    from flexpeak.caller import fit_background

    p = CallParams(bin_size=25)
    plain = call_peaks(mixed.treatment, mixed.control, p)
    bg = fit_background(mixed.treatment, mixed.control, p)
    cache = {}
    first = call_peaks(mixed.treatment, mixed.control, p, background=bg,
                       segmentation_cache=cache)
    second = call_peaks(mixed.treatment, mixed.control,
                        p.replace(posterior_cutoff=0.5, min_fold=1.2),
                        background=bg, segmentation_cache=cache)
    key = lambda pk: [(r.chrom, r.start, r.end, len(r.subpeaks)) for r in pk]
    assert key(plain) == key(first) == key(second)
    assert any(isinstance(k, tuple) for k in cache)


# -- BAM loader -------------------------------------------------------------

pysam = pytest.importorskip("pysam")


def _write_bam(path, paired):
    hdr = {"HD": {"VN": "1.6", "SO": "coordinate"},
           "SQ": [{"SN": "chrA", "LN": 100_000}, {"SN": "chrB", "LN": 40_000}]}
    rng = np.random.default_rng(5)
    recs = []
    for tid, L in ((0, 100_000), (1, 40_000)):
        for i in range(3000):
            s = int(rng.integers(0, L - 1500))
            mq = int(rng.choice([10, 60]))
            dup = rng.random() < 0.05
            if paired:
                tl = int(rng.integers(80, 1300))
                for mate, (pos, flag) in enumerate(((s, 99), (s + tl - 50, 147))):
                    a = pysam.AlignedSegment()
                    a.query_name = f"q{tid}_{i}"
                    a.query_sequence = "A" * 50
                    a.query_qualities = pysam.qualitystring_to_array("I" * 50)
                    a.flag = flag | (0x400 if dup else 0)
                    a.reference_id, a.reference_start = tid, pos
                    a.mapping_quality, a.cigartuples = mq, [(0, 50)]
                    a.next_reference_id = tid
                    a.next_reference_start = s if mate else s + tl - 50
                    a.template_length = tl if mate == 0 else -tl
                    recs.append(a)
            else:
                a = pysam.AlignedSegment()
                a.query_name = f"q{tid}_{i}"
                a.query_sequence = "A" * 50
                a.query_qualities = pysam.qualitystring_to_array("I" * 50)
                a.flag = (16 if rng.random() < 0.5 else 0) | (0x400 if dup else 0)
                a.reference_id, a.reference_start = tid, s
                a.mapping_quality, a.cigartuples = mq, [(0, 50)]
                recs.append(a)
    recs.sort(key=lambda a: (a.reference_id, a.reference_start))
    tmp = path + ".unsorted.bam"
    with pysam.AlignmentFile(tmp, "wb", header=hdr) as out:
        for a in recs:
            out.write(a)
    pysam.sort("-o", path, tmp)
    pysam.index(path)
    os.remove(tmp)


def _reference_coverage(path, bin_size, mapq, paired, frag_len, keep_dup):
    """Slow per-read reference: what the loader must reproduce."""
    bad = 0x4 | 0x100 | 0x200 | 0x800 | (0 if keep_dup else 0x400)
    out = {}
    with pysam.AlignmentFile(path) as af:
        for chrom, L in zip(af.references, af.lengths):
            nb = L // bin_size + 2
            cov = np.zeros(nb, dtype=np.int64)
            for r in af.fetch(chrom):
                if r.flag & bad or r.mapping_quality < mapq:
                    continue
                if paired:
                    if not r.is_proper_pair or r.template_length <= 0 or r.template_length > 1000:
                        continue
                    s, e = r.reference_start, r.reference_start + r.template_length
                elif r.is_reverse:
                    e = r.reference_end
                    s = e - frag_len
                else:
                    s = r.reference_start
                    e = s + frag_len
                s, e = max(0, s), min(L, e)
                cov[s // bin_size:(e - 1) // bin_size + 1] += 1
            out[chrom] = cov.astype(np.float32)
    return out


@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("kw", [dict(), dict(keep_duplicates=True, mapq=0),
                                dict(n_jobs=2)])
def test_load_bam_matches_per_read_reference(paired, kw):
    from flexpeak.io.bam import load_bam

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.bam")
        _write_bam(path, paired)
        cov = load_bam(path, bin_size=50, fragment_length=None if paired else 120, **kw)
        ref = _reference_coverage(path, 50, kw.get("mapq", 30), paired, 120,
                                  kw.get("keep_duplicates", False))
        assert cov.meta["paired"] is paired
        for c in ref:
            assert np.array_equal(cov.bins[c], ref[c]), c
        assert cov.n_fragments > 0


def test_fragment_length_estimate_runs_on_single_end():
    from flexpeak.io.bam import estimate_fragment_length, load_bam

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "se.bam")
        _write_bam(path, paired=False)
        frag, nsc = estimate_fragment_length(path)
        assert 50 <= frag <= 500
        cov = load_bam(path, bin_size=50)
        assert cov.fragment_length == frag
        assert cov.meta["nsc"] == pytest.approx(nsc)
