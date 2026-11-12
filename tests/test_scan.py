"""The quick scan: background by sigma clipping, signal excess, and the
narrow / broad / null classification it feeds into parameter choice."""

import numpy as np
import pytest

from flexpeak import CallParams, call_peaks
from flexpeak.scan import profile_bins, scan_coverage, sigma_clip
from flexpeak.simulate import simulate

SIZES = {"chr1": 2_000_000, "chr2": 500_000}
BINS = (25, 50, 100)  # the simulator bins at 25 bp


def test_sigma_clip_recovers_background_under_contamination():
    rng = np.random.default_rng(0)
    bg = rng.negative_binomial(5, 5 / (5 + 8.0), size=200_000).astype(float)
    x = bg.copy()
    x[:4000] += 80.0                       # 2% of bins carry strong signal
    mean, sd, keep = sigma_clip(x, 5.0)
    assert abs(mean - bg.mean()) < 0.15 * bg.mean()
    assert keep[:4000].sum() == 0          # signal bins are all clipped
    assert keep[4000:].mean() > 0.995      # background bins are nearly all kept


def test_profile_null_has_no_excess():
    rng = np.random.default_rng(1)
    x = rng.negative_binomial(4, 4 / (4 + 6.0), size=100_000).astype(float)
    p = profile_bins(x, 25, n_sd=5.0)
    assert p.signal_excess < 2e-3
    assert all(v < 2e-3 for v in p.smoothed_excess.values())


@pytest.mark.parametrize("kind", ["narrow", "broad", "null"])
def test_scan_classifies_simulated_marks(kind):
    # 15% baseline drift: the default 35% at a 40 kb scale is an adversarial
    # setting for the caller's local background, but for a scan that judges
    # domains against a 200 kb baseline it would hide an 80 kb 2.5x domain
    # inside the drift (3.9 drift-SDs), which no scale-based rule can resolve.
    kw = dict(chrom_sizes=SIZES, seed=7, dispersion=0.15, drift_sd=0.15)
    if kind == "narrow":
        sim = simulate(n_narrow=60, n_broad=0, n_mixed=0, narrow_fold=10.0, **kw)
    elif kind == "broad":
        sim = simulate(n_narrow=0, n_broad=6, n_mixed=0, broad_fold=2.5,
                       broad_width=80_000, **kw)
    else:
        sim = simulate(n_narrow=0, n_broad=0, n_mixed=0, **kw)
    prof = scan_coverage(sim.treatment, BINS, chrom="chr1")
    rec = prof.recommend()
    assert prof.bin_sizes == list(BINS)
    assert "class" in prof.report(rec)
    if kind == "null":
        assert rec.signal_class == "null"
    elif kind == "narrow":
        assert rec.signal_class in ("narrow", "mixed")
        assert rec.bin_size <= 50
        assert rec.windows == (1_000, 5_000, 10_000)
    else:
        assert rec.signal_class == "broad"
        assert rec.bin_size >= 50
        assert min(rec.windows) >= 50_000
        assert rec.segment_bin_size and rec.segment_bin_size >= rec.bin_size
    # The sweep is small (axes are swept one at a time), and every override
    # and every grid axis is a real CallParams field.
    assert sum(len(v) for v in rec.grid.values()) <= 12
    CallParams().replace(**rec.overrides())
    CallParams().replace(**{k: v[0] for k, v in rec.grid.items()})


def test_scan_json_roundtrip_and_recommendation_calls(tmp_path):
    sim = simulate(chrom_sizes=SIZES, n_narrow=20, n_broad=2, n_mixed=1, seed=9)
    prof = scan_coverage(sim.treatment, BINS, chrom="chr1")
    rec = prof.recommend()
    path = tmp_path / "scan.json"
    prof.to_json(path, rec)
    import json

    d = json.loads(path.read_text())
    assert d["recommendation"]["signal_class"] == rec.signal_class
    assert set(d["profiles"]) == {str(b) for b in BINS}
    params = CallParams().replace(**rec.overrides())
    peaks = call_peaks(sim.treatment, sim.control, params)
    truth = sim.truth_intervals()
    from flexpeak.peaks import interval_overlap_bp, interval_total_bp

    inter = interval_overlap_bp(peaks.intervals(), truth)
    recall = inter / interval_total_bp(truth)
    assert recall > 0.8


def test_similarity_gated_merge_only_joins_enriched_gaps():
    from flexpeak.caller import _merge_similar

    mu = np.full(100, 10.0)
    counts = mu.copy()
    counts[10:20] = 40; counts[20:25] = 25; counts[25:35] = 40     # enriched gap
    counts[50:60] = 40; counts[60:65] = 10; counts[65:75] = 40     # background gap
    ivs = [(10, 20), (25, 35), (50, 60), (65, 75)]
    out = _merge_similar(ivs, 8, counts, mu, 0.5, 1.2)
    assert out == [(10, 35), (50, 60), (65, 75)]
    assert _merge_similar(ivs, 0, counts, mu, 0.5, 1.2) == ivs
    assert _merge_similar(ivs, 8, counts, mu, 0.0, 1.2) == [(10, 35), (50, 75)]


def test_scan_window_and_finer_cache_reuse(tmp_path):
    from flexpeak.scan import scan_window
    from flexpeak.io.load import load_signal
    from flexpeak.simulate import simulate, write_bigwig

    pytest.importorskip("pyBigWig")
    assert scan_window(1_000_000, 60_000_000) == (0, 1_000_000)
    s, e = scan_window(195_000_000, 60_000_000)
    assert e - s == 60_000_000 and s > 0
    sim = simulate(chrom_sizes={"chr1": 400_000}, n_narrow=10, n_broad=0, n_mixed=0, seed=3)
    bw = write_bigwig(tmp_path / "t.bw", sim.treatment)
    cdir = tmp_path / "cache"
    fine = load_signal(bw, bin_size=25, cache_dir=cdir)
    coarse = load_signal(bw, bin_size=100, cache_dir=cdir)      # served by rebinning the 25 bp cache
    assert coarse.bin_size == 100
    assert np.allclose(coarse.bins["chr1"], fine.rebin(100).bins["chr1"])
    # Rebinned in memory: the derived coverage is not written as a second cache.
    assert len(list(cdir.glob("*.100bp.*.npz"))) == 0
    assert len(list(cdir.glob("*.25bp.*.npz"))) == 1
