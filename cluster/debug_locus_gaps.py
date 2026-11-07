"""Why are small H3K9me3 peaks at Pdcd1 missed, and why do H3K27me3 domains bridge gaps?

Re-runs chr1+chr11 with the exact call parameters stored in the track-figure
outputs and prints, at 1 kb resolution across each locus: signal, wide
background, focal background, HMM posterior and the focal test's -log10 p.
"""
import json
import os

import numpy as np
from scipy.ndimage import uniform_filter1d

from flexpeak import load_signal, CallParams, call_peaks
from flexpeak.background.nb import BackgroundModel, estimate_dispersion
from flexpeak.caller import _coarsen, _dwell_floor
from flexpeak.stats.nbtest import nb_neglog10_sf, qvalue_from_neglog10p

R = os.environ["REPO"]
CACHE = f"{R}/testing/cache"
DATA = f"{R}/testing/data"
RUNS = {
    "H3K9me3": f"{R}/testing/results/real/k9_quick/H3K9me3_Cl13_peaks.nested.tsv",
    "H3K27me3": f"{R}/testing/results/tracks/calls_H3K27me3/H3K27me3_Cl13_peaks.nested.tsv",
}
LOCI = {
    "H3K9me3": [("Pdcd1", "chr1", 93_938_000, 94_153_000), ("Havcr2", "chr11", 46_429_000, 46_502_000)],
    "H3K27me3": [("Tcf7", "chr11", 52_227_000, 52_308_000), ("Pdcd1", "chr1", 94_013_000, 94_078_000),
                 ("Havcr2", "chr11", 46_429_000, 46_502_000)],
}

for mark, path in RUNS.items():
    hdr = json.loads(open(path).readline()[2:])
    call = hdr["params"]["call"]
    call["windows"] = tuple(call["windows"]); call["bandwidths"] = tuple(call["bandwidths"])
    call["focal_windows"] = tuple(call["focal_windows"])
    params = CallParams(**call)
    cov = load_signal(f"{DATA}/GSE285245_pool_{mark}_Cl13.bw", bin_size=params.bin_size,
                      chroms=["chr1", "chr11"], cache_dir=CACHE)
    bs = cov.bin_size
    res = call_peaks(cov, None, params, return_details=True)
    d = res.diagnostics
    print(f"\n################ {mark}  bin {bs}  seg {d['segment_bin_size']}  scale {d['signal_scale_bp']}  "
          f"min_fold {params.min_fold} post {params.posterior_cutoff} max_gap {params.max_gap}", flush=True)
    print("dwell floor stay", _dwell_floor(d["signal_scale_bp"], d["segment_bin_size"], params),
          "-> mean enriched run", int(min(max(params.min_width, d["signal_scale_bp"]), params.max_dwell)), "bp")
    for c, cd in d["chroms"].items():
        print(f"  {c}: multipliers {np.round(cd['multipliers'], 2)} fitted dwell {cd['fitted_dwell_bp']}")
    print("  focal diag", {k: v for k, v in d.get("focal", {}).items()})
    print(f"  regions {len(res.peaks)}  subpeaks {sum(len(r.subpeaks) for r in res.peaks)}  "
          f"focal standalone {sum(r.source == 'focal' for r in res.peaks)}")

    # Re-create the focal test so its per-position p-values can be inspected.
    bg = res.background
    fd = d["focal"]
    seg = fd["segment_bin_size"]; factor = seg // bs; win_bp = fd["test_window_bp"]
    bg_loc = BackgroundModel.fit(cov, None, windows=tuple(fd["windows"]), exclude_fold=0.0)
    sums = {}
    for c in sorted(cov.bins):
        x = cov.bins[c].astype(np.float64)
        muf = np.maximum(bg_loc.mu[c], bg.mu[c])
        cs, ms = _coarsen(x, factor), _coarsen(muf, factor)
        k = max(1, int(round(win_bp / seg)))
        cw = uniform_filter1d(cs, k, mode="nearest") * k
        mw = uniform_filter1d(ms, k, mode="nearest") * k
        sums[c] = (cw, np.maximum(mw, 1e-9))
    a0, a1 = estimate_dispersion(np.concatenate([v[0] for v in sums.values()]),
                                 np.concatenate([v[1] for v in sums.values()]))
    nlp = {c: nb_neglog10_sf(cw, mw, np.maximum(a0 + a1 / mw, 1e-10)) for c, (cw, mw) in sums.items()}
    allp = np.concatenate(list(nlp.values()))
    q = qvalue_from_neglog10p(allp)
    passing = allp[q >= -np.log10(params.qvalue)]
    print(f"  window dispersion a0 {a0:.3f} a1 {a1:.3f}; BH needs -log10p >= "
          f"{passing.min() if passing.size else float('nan'):.2f}; "
          f"alpha at window mean {a0 + a1 / np.median(allp * 0 + np.median(np.concatenate([v[1] for v in sums.values()]))):.3f}")
    # what fold over the focal mean does a single window need to pass?
    mwm = float(np.median(np.concatenate([v[1] for v in sums.values()])))
    for fold in (1.5, 2, 3, 5, 8):
        p = nb_neglog10_sf(np.array([fold * mwm]), np.array([mwm]), np.array([a0 + a1 / mwm]))[0]
        print(f"    window at {fold}x over median focal mean ({mwm:.1f}): -log10p {p:.2f}")

    for name, c, s, e in LOCI[mark]:
        print(f"\n  --- {mark} {name} {c}:{s}-{e}  (1 kb rows; sig=sum/1kb, wide/focal = bg sum/1kb)")
        x = cov.bins[c].astype(np.float64)
        post = res.diagnostics  # noqa
        # posterior: rerun cached? take from regions instead
        regs = [(r.start, r.end) for r in res.peaks if r.chrom == c and r.end > s and r.start < e]
        print("   called:", regs, [(sp.start, sp.end) for r in res.peaks if r.chrom == c and r.end > s and r.start < e for sp in r.subpeaks])
        muf = np.maximum(bg_loc.mu[c], bg.mu[c])
        for p0 in range(s, e, 1000):
            b0, b1 = p0 // bs, (p0 + 1000) // bs
            sg = x[b0:b1].sum(); wd = bg.mu[c][b0:b1].sum(); fc = muf[b0:b1].sum()
            j0, j1 = p0 // seg, max(p0 // seg + 1, (p0 + 1000) // seg)
            mp = float(nlp[c][j0:j1].max())
            called = any(a <= p0 < b for a, b in regs)
            bar = "#" * int(min(60, sg / max(wd, 1e-9) * 3))
            print(f"   {p0:>11,} sig {sg:8.1f} wide {wd:6.1f} fold {sg / max(wd, 1e-9):5.1f} focal {fc:7.1f} "
                  f"f/foc {sg / max(fc, 1e-9):4.1f} nlp {mp:6.2f} {'C' if called else ' '} {bar}")
