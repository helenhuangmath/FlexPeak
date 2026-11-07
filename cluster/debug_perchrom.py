"""Per-replicate CTCF genome-wide: why is chr11 sane and every other chromosome a blob?"""
import glob, os, sys
import numpy as np
from flexpeak import load_signal, CallParams, call_peaks
from flexpeak.scan import scan_file, sigma_clip
from flexpeak.background.nb import BackgroundModel
from scipy.ndimage import uniform_filter1d

R = os.environ["REPO"]; CACHE = f"{R}/testing/cache"
BW = "/vast/projects/wherry/foundation-models-immuno/hhua/data_lab_internal/Huang_SR2025_CnR/geo_processed/per_sample_bw"
f = glob.glob(f"{BW}/*_Cl13_*_CTCF_Rep1_*.scale.bw")[0]
rec = scan_file(f, cache_dir=CACHE).recommend()
print("scan", rec.signal_class, rec.overrides(), flush=True)
chroms = sys.argv[1].split(",") if len(sys.argv) > 1 and sys.argv[1] != "all" else None
bin_override = int(sys.argv[2]) if len(sys.argv) > 2 else None
ov = dict(rec.overrides())
if bin_override:
    ov["bin_size"] = bin_override
cov = load_signal(f, bin_size=ov["bin_size"], chroms=chroms, cache_dir=CACHE)
params = CallParams().replace(**ov)
bs = cov.bin_size
print(f"loaded {len(cov.bins)} chroms @ {bs} bp; params windows {params.windows} exclude_fold {params.background_exclude_fold} smooth {params.background_exclude_smooth_bp}", flush=True)
bg = BackgroundModel.fit(cov, None, windows=params.windows, exclude_fold=params.background_exclude_fold,
                         exclude_iters=params.background_exclude_iters, exclude_smooth_bp=params.background_exclude_smooth_bp)
print(f"bg a0 {bg.a0:.3f} a1 {bg.a1:.3f} gmean {bg.genome_mean:.3f} infl {bg.autocorr_inflation:.2f}", flush=True)
print(f"bg excluded_fraction {bg.meta.get('excluded_fraction')}", flush=True)
if os.environ.get("TUNE") == "1":
    from flexpeak.tune import tune
    tr = tune(cov, None, base=params, n_chroms=1, grid=rec.grid)
    params = tr.params
    print(f"tuned on {tr.chroms}: post {params.posterior_cutoff} min_fold {params.min_fold} stable {tr.stable} notes {tr.notes}", flush=True)
if os.environ.get("A1"):
    bg = BackgroundModel(mu=bg.mu, a0=bg.a0, a1=float(os.environ["A1"]), genome_mean=bg.genome_mean, windows=bg.windows,
                         used_control=bg.used_control, covariate_coef=bg.covariate_coef,
                         autocorr_inflation=bg.autocorr_inflation, meta=bg.meta)
    print(f"a1 forced to {bg.a1}", flush=True)
res = call_peaks(cov, None, params, return_details=True, background=bg)
nd = sum(1 for r in res.peaks.regions if getattr(r, "source", "hmm") == "hmm")
print(f"regions {len(res.peaks.regions)} hmm {nd} focal {len(res.peaks.regions)-nd} dissolved {res.diagnostics.get('n_dissolved')}", flush=True)
terr = {}
for r in res.peaks.regions:
    terr[r.chrom] = terr.get(r.chrom, 0) + r.end - r.start
print(f"segdisp {res.diagnostics.get('segment_dispersion')} segbin {res.diagnostics['segment_bin_size']} scale {res.diagnostics['signal_scale_bp']}")
print(f"{'chrom':<6} {'nbins':>9} {'mean':>6} {'frac0':>6} {'q99':>6} {'smoothed mode':>14} {'mu q01':>7} {'mu q50':>7} {'mu min':>7} {'terr Mb':>8} {'post>=cut':>9}")
for c in sorted(cov.bins):
    x = cov.bins[c].astype(np.float64); mu = bg.mu[c]
    k = max(1, int(round(200 / bs)))
    sm = uniform_filter1d(x.astype(np.float32), size=k, mode="nearest")
    mode, csd, cmask = sigma_clip(sm, 5.0, averaged_over=k)
    mode = float(np.ravel(mode)[0]); kept_frac = float(np.mean(cmask)) if hasattr(cmask, "__len__") else float("nan")
    d = res.diagnostics["chroms"].get(c, {})
    print(f"{c:<6} {x.size:>9} {x.mean():>6.2f} {(x==0).mean():>6.3f} {np.percentile(x,99):>6.1f} {mode:>14.4f} keep={kept_frac:.3f} med={np.median(sm):.3f} "
          f"{np.percentile(mu,1):>7.3f} {np.median(mu):>7.3f} {mu.min():>7.4f} {terr.get(c,0)/1e6:>8.2f} {str(d.get('fitted_dwell_bp','?')):>9}", flush=True)
