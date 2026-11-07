import numpy as np, os
from flexpeak import load_signal, CallParams, call_peaks
from flexpeak.scan import scan_file
R = os.environ["REPO"]; CACHE = f"{R}/testing/cache"
bw = f"{R}/testing/data/GSE285245_pool_H3K27me3_Cl13.bw"
rec = scan_file(bw, cache_dir=CACHE).recommend()
print("scan:", rec.signal_class, rec.overrides())
cov = load_signal(bw, bin_size=100, chroms=["chr11"], cache_dir=CACHE)
base = CallParams().replace(**rec.overrides()).replace(bin_size=100)
loci = {"Havcr2_left": (46_429_000, 46_438_000), "Havcr2": (46_454_000, 46_477_000), "Tcf7": (52_252_000, 52_283_000), "Havcr2_right": (46_488_000, 46_502_000)}
def run(tag, p):
    seg = {}
    res = call_peaks(cov, None, p, return_details=True, segmentation_cache=seg)
    d = res.diagnostics["chroms"]["chr11"]; bg = res.background
    post = [v.posterior for k, v in seg.items() if isinstance(k, tuple)][0]
    arr = cov.bins["chr11"]; mu = bg.mu["chr11"]
    bs = 100
    print(f"--- {tag}: regions {len(res.peaks)} territory {res.peaks.total_bp/1e6:.1f} Mb | mult {np.round(d['multipliers'],2)} dwell {d['fitted_dwell_bp']} segbin {res.diagnostics['segment_bin_size']} | bg a0 {bg.a0:.3f} gmean {bg.genome_mean:.2f} floor-meta {bg.meta.get('excluded_fraction'):.3f}")
    for name, (s, e) in loci.items():
        b0, b1 = s // bs, e // bs
        c = arr[b0:b1]; m = mu[b0:b1]; pp = post[b0:b1]
        n_in = sum(1 for r in res.peaks if r.chrom == "chr11" and r.end > s and r.start < e)
        print(f"   {name:12s} counts mean {c.mean():6.2f} p90 {np.percentile(c,90):6.2f} | lambda mean {m.mean():6.2f} | fold {c.mean()/m.mean():4.2f} | posterior mean {pp.mean():.2f} max {pp.max():.2f} | regions overlapping {n_in}")
run("scan params (new floor, 1M/5M windows)", base)
run("posterior 0.3", base.replace(posterior_cutoff=0.3))
run("segment 600", base.replace(segment_bin_size=600))
run("no dwell floor", base.replace(dwell_from_autocorrelation=False))
