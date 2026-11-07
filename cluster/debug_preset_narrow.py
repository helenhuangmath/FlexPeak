"""Why did preset h3k27me3 on the *narrow* simulated set go from 85% to 29% bp F1?

Toggles the two November-6 changes independently on the benchmark's own
narrow dataset: the finite-length AR(1) factor and the segmentation-scale
dispersion fit.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Benchmark"))
import numpy as np
import run_benchmark as RB
import metrics as MT
import flexpeak.caller as C
import flexpeak.stats.nbtest as NB
import flexpeak.background.nb as BGM
from flexpeak import preset_params, call_peaks

ds = RB.build_dataset("narrow", RB.dataset_specs(False)["narrow"])
treat, ctrl, truth, G = ds["treat"], ds["control"], ds["truth"], ds["genome_size"]
params = preset_params("h3k27me3").replace(bin_size=RB.BIN_SIZE)
bg = C.fit_background(treat, ctrl, params, None)
print(f"bg a0={bg.a0:.4f} a1={bg.a1:.4f} inflation={bg.autocorr_inflation:.3f}")

orig_ar1 = NB.ar1_factor
def limit_ar1(n, rho):
    return np.full(np.shape(n) if np.ndim(n) else (), (1 + rho) / (1 - rho)) if np.ndim(n) else (1 + rho) / (1 - rho)
orig_est = BGM.estimate_dispersion
def bin_trend(counts, mu, **kw):
    return bg.a0, bg.a1

def run(label):
    cache = {}
    res = call_peaks(treat, ctrl, params, background=bg, return_details=True, segmentation_cache=cache)
    peaks, diag = res.peaks, res.diagnostics
    iv = [(r.chrom, r.start, r.end) for r in peaks]
    m = MT.evaluate(iv, truth, G)
    sd = diag.get("segment_dispersion", cache.get(("seg_dispersion", diag.get("segment_bin_size", 0)//RB.BIN_SIZE, params.background_model)))
    print(f"{label:<28} seg_bin={diag.get('segment_bin_size')} seg_disp={sd} n={m['n_calls']} called_bp={m['called_bp']} "
          f"prec={m['bp_precision']:.3f} rec={m['bp_recall']:.3f} f1={m['bp_f1']:.3f}")
    w = sorted(e - s for _, s, e in iv)
    print(f"    widths: min {w[0]} median {w[len(w)//2]} max {w[-1]}")

run("current")
C.ar1_factor = limit_ar1; NB.ar1_factor = limit_ar1
run("AR1 limit (old)")
C.ar1_factor = orig_ar1; NB.ar1_factor = orig_ar1
BGM.estimate_dispersion = bin_trend
run("seg disp = bin trend")
C.ar1_factor = limit_ar1; NB.ar1_factor = limit_ar1
run("both old")
# exact old emission: _aggregate_alpha with the limit factor -> alpha = inflation*a1/mu_seg (a0 = 0 here)
def old_agg(counts, mu, **kw):
    return bg.a0 * bg.autocorr_inflation, bg.a1 * bg.autocorr_inflation
BGM.estimate_dispersion = old_agg; C.ar1_factor = limit_ar1; NB.ar1_factor = limit_ar1
run("old _aggregate_alpha, AR1 limit")
C.ar1_factor = orig_ar1; NB.ar1_factor = orig_ar1
run("old _aggregate_alpha, AR1 finite")
BGM.estimate_dispersion = orig_est
for cut in (0.45, 0.6):
    for mf in (1.2, 1.5):
        p2 = params.replace(posterior_cutoff=cut, min_fold=mf)
        res = call_peaks(treat, ctrl, p2, background=bg)
        iv = [(r.chrom, r.start, r.end) for r in res]
        m = MT.evaluate(iv, truth, G)
        w = sorted(e - s for _, s, e in iv)
        print(f"current cut={cut} min_fold={mf}: n={m['n_calls']} called_bp={m['called_bp']} f1={m['bp_f1']:.3f} maxw={w[-1]}")
for fp in (False,):
    p2 = params.replace(focal_pass=fp)
    res = call_peaks(treat, ctrl, p2, background=bg)
    iv = [(r.chrom, r.start, r.end) for r in res]
    m = MT.evaluate(iv, truth, G)
    print(f"current, focal_pass={fp}: n={m['n_calls']} called_bp={m['called_bp']} f1={m['bp_f1']:.3f}")
