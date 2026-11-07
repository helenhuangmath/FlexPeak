"""H3K27me3 Cl13 chr11 at Havcr2: where is the local lambda, the floor and the
posterior across 46.40-46.56 Mb, under the floor2 params; and what does the
once-read floor do vs the old iterated floor."""
import json, os, sys, dataclasses
import numpy as np
from flexpeak import load_signal, CallParams, call_peaks
from flexpeak.background.nb import BackgroundModel
import flexpeak.background.nb as NB
from flexpeak.scan import sigma_clip
from scipy.ndimage import uniform_filter1d

R = os.environ["REPO"]; CACHE = f"{R}/testing/cache"
bw = f"{R}/testing/data/GSE285245_pool_H3K27me3_Cl13.bw"
hdr = open(f"{R}/testing/results/tracks/floor2_H3K27me3/H3K27me3_Cl13_peaks.broadPeak").readline().lstrip("# ").strip()
call = json.loads(hdr)["params"]["call"]
names = {f.name for f in dataclasses.fields(CallParams)}
params = CallParams(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in call.items() if k in names})
print("params: bin", params.bin_size, "windows", params.windows, "cut", params.posterior_cutoff, "min_fold", params.min_fold, "segbin", params.segment_bin_size, "exclude", params.background_exclude_fold, params.background_exclude_smooth_bp)
cov = load_signal(bw, bin_size=params.bin_size, chroms=["chr11"], cache_dir=CACHE)
x = cov.bins["chr11"].astype(np.float64); bs = cov.bin_size
lo, hi = 46_400_000 // bs, 46_560_000 // bs
print(f"chr11 mean {x.mean():.2f} median {np.median(x):.2f} q25 {np.percentile(x,25):.2f}")
bg = BackgroundModel.fit(cov, None, windows=params.windows, exclude_fold=params.background_exclude_fold,
                         exclude_iters=params.background_exclude_iters, exclude_smooth_bp=params.background_exclude_smooth_bp)
mu = bg.mu["chr11"]
print(f"lambda chr11: q01 {np.percentile(mu,1):.2f} q50 {np.median(mu):.2f} q99 {np.percentile(mu,99):.2f}; a0 {bg.a0:.3f} a1 {bg.a1:.3f} infl {bg.autocorr_inflation:.2f}")
res = call_peaks(cov, None, params, return_details=True, background=bg)
regs = [(r.start, r.end, round(r.fold_enrichment, 2)) for r in res.peaks.regions if r.end > 46_300_000 and r.start < 46_900_000]
print("regions near Havcr2:", regs)
k = 10  # 1 kb summary
print(f"{'pos kb':>8} {'signal':>7} {'lambda':>7} {'fold':>6}")
for b in range(lo, hi, k):
    print(f"{b*bs/1000:>8.0f} {x[b:b+k].mean():>7.1f} {mu[b:b+k].mean():>7.2f} {x[b:b+k].mean()/max(mu[b:b+k].mean(),1e-9):>6.2f}")
# what the floor is
from flexpeak.background.nb import exclusion_smoothing_bp
sm_bp = params.background_exclude_smooth_bp or exclusion_smoothing_bp(bs, params.windows)
ks = max(1, int(round(sm_bp / bs)))
judged = uniform_filter1d(x.astype(np.float32), size=ks, mode="nearest")
mode, sd, mask = sigma_clip(judged, 5.0, averaged_over=ks)
print(f"exclusion smoothing {sm_bp} bp; once-read mode {float(np.ravel(mode)[0]):.3f} sd {float(np.ravel(sd)[0]):.3f}; genome_mean {bg.genome_mean:.2f}")
