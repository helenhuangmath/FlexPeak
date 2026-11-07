import numpy as np, os, glob
from flexpeak import load_signal, CallParams, call_peaks
from flexpeak.scan import scan_file
from flexpeak.background.nb import local_lambda
R = os.environ["REPO"]; CACHE = f"{R}/testing/cache"
BW = "/vast/projects/wherry/foundation-models-immuno/hhua/data_lab_internal/Huang_SR2025_CnR/geo_processed/per_sample_bw"
for mark in ("CTCF", "H3K9me3"):
    f = glob.glob(f"{BW}/*_Cl13_*_{mark}_Rep1_*.scale.bw")[0]
    rec = scan_file(f, cache_dir=CACHE).recommend()
    print(f"\n===== {mark} Rep1: scan {rec.signal_class} {rec.overrides()}")
    cov = load_signal(f, bin_size=rec.bin_size, chroms=["chr11"], cache_dir=CACHE)
    x = cov.bins["chr11"].astype(float); bs = cov.bin_size
    print(f"  bins {x.size} @ {bs} bp: mean {x.mean():.2f} frac0 {(x==0).mean():.3f} q50 {np.median(x):.2f} q90 {np.percentile(x,90):.2f} q99 {np.percentile(x,99):.2f} max {x.max():.0f}")
    base = CallParams().replace(**rec.overrides())
    def run(tag, p):
        seg = {}
        res = call_peaks(cov, None, p, return_details=True, segmentation_cache=seg)
        d = res.diagnostics["chroms"]["chr11"]; bg = res.background; mu = bg.mu["chr11"]
        post = [v.posterior for k, v in seg.items() if isinstance(k, tuple) and k[0] == "chr11"]
        pf = (post[0] >= p.posterior_cutoff).mean() if post else float("nan")
        print(f"  -- {tag}: segdisp {res.diagnostics.get('segment_dispersion')} regions {len(res.peaks)} territory {res.peaks.total_bp/1e6:.1f} Mb subpeaks {res.peaks.n_subpeaks} | lambda q10/50/90 {np.percentile(mu,10):.2f}/{np.median(mu):.2f}/{np.percentile(mu,90):.2f} gmean {bg.genome_mean:.2f} a0 {bg.a0:.2f} infl {bg.autocorr_inflation:.1f} | mult {np.round(d['multipliers'],2)} dwell {d['fitted_dwell_bp']} segbin {res.diagnostics['segment_bin_size']} scale {res.diagnostics['signal_scale_bp']} | posterior>=cut {pf:.3f}")
    run("scan params", base)
    run("exclusion off", base.replace(background_exclude_fold=0.0))
    run("posterior 0.7", base.replace(posterior_cutoff=0.7))
    run("min_fold 3", base.replace(min_fold=3.0))
    run("segment 100 forced", base.replace(segment_bin_size=100))
    run("no dwell floor", base.replace(dwell_from_autocorrelation=False))
