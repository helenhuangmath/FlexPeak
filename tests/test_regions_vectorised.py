"""The vectorised region builder must reproduce the per-region builder."""

import numpy as np

from flexpeak import CallParams, call_peaks
from flexpeak.caller import _build_region, _build_regions
from flexpeak.simulate import simulate


def test_build_regions_matches_per_region():
    rng = np.random.default_rng(0)
    counts = rng.poisson(8.0, 5000).astype(float)
    counts[100:140] += 40; counts[1000:1200] += 15; counts[3000:3010] += 60
    mu = np.full(5000, 8.0); alpha = np.full(5000, 0.1)
    post = np.clip(counts / 60.0, 0, 1)
    ivs = [(100, 140), (1000, 1200), (3000, 3010), (4000, 4004)]
    p = CallParams(bin_size=25, min_fold=1.2)
    vec = _build_regions("chr1", ivs, counts, mu, alpha, post, 25, 125_000, p, 1.3)
    single = [r for r in (_build_region("chr1", a, b, counts, mu, alpha, post, 25, 125_000, p, 1.3)
                          for a, b in ivs) if r is not None]
    assert len(vec) == len(single)
    for v, s in zip(vec, single):
        assert (v.start, v.end, v.kind) == (s.start, s.end, s.kind)
        assert np.isclose(v.fold_enrichment, s.fold_enrichment)
        assert np.isclose(v.neg_log10_p, s.neg_log10_p, rtol=1e-6)
        assert np.isclose(v.mean_posterior, s.mean_posterior)
        assert [(x.start, x.end) for x in v.subpeaks] == [(x.start, x.end) for x in s.subpeaks]


def test_tune_returns_subpeaks_on_final_call():
    from flexpeak import tune

    sim = simulate(chrom_sizes={"chr1": 1_000_000, "chr2": 500_000}, n_narrow=10,
                   n_broad=1, n_mixed=1, seed=21)
    res = tune(sim.treatment, sim.control, base=CallParams(bin_size=25), n_chroms=2)
    assert res.params.call_subpeaks is True
    peaks = call_peaks(sim.treatment, sim.control, res.params)
    assert peaks.n_subpeaks > 0
