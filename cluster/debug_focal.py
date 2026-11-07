import numpy as np, os
from scipy.ndimage import uniform_filter1d
from flexpeak import load_signal, CallParams, call_peaks
from flexpeak.scan import scan_file
from flexpeak.background.nb import BackgroundModel
from flexpeak.caller import _aggregate_alpha, _coarsen
from flexpeak.stats.nbtest import nb_neglog10_sf, qvalue_from_neglog10p
R = os.environ["REPO"]; CACHE = f"{R}/testing/cache"
bw = f"{R}/testing/data/GSE285245_pool_H3K4me3_Cl13.bw"
rec = scan_file(bw, cache_dir=CACHE).recommend()
cov = load_signal(bw, bin_size=25, chroms=["chr11"], cache_dir=CACHE)
params = CallParams().replace(**rec.overrides()).replace(bin_size=25)
res = call_peaks(cov, None, params, return_details=True)
print("focal diag:", res.diagnostics.get("focal"))
s, e = 46_429_000, 46_480_000
for r in [r for r in res.peaks if r.chrom == "chr11" and r.end > s and r.start < e]:
    print(f"region {r.start}-{r.end} kind {r.kind} fold {r.fold_enrichment:.2f} subpeaks {len(r.subpeaks)}")
    for sp in r.subpeaks[:12]: print(f"    sub {sp.start}-{sp.end} w {sp.end-sp.start} fold {sp.fold_enrichment:.2f} p {sp.neg_log10_p:.1f} q {sp.neg_log10_q:.1f}")
bgw = res.background
bgl = BackgroundModel.fit(cov, None, windows=(5000, 10000), exclude_fold=0.0)
counts = cov.bins["chr11"].astype(float); mu = np.maximum(bgl.mu["chr11"], bgw.mu["chr11"]).astype(float)
alpha = bgw.alpha(mu)
factor = 4; k_w = 5   # 100 bp seg, 500 bp window
c_seg, mu_seg = _coarsen(counts, factor), _coarsen(mu, factor)
a_seg = _aggregate_alpha(mu, alpha, factor, bgw.autocorr_inflation)
c_w = uniform_filter1d(c_seg, k_w, mode="nearest") * k_w; mu_w = uniform_filter1d(mu_seg, k_w, mode="nearest") * k_w
var_w = uniform_filter1d(mu_seg + a_seg * mu_seg**2, k_w, mode="nearest") * k_w
a_w = np.maximum((var_w - mu_w) / np.maximum(mu_w**2, 1e-12), 1e-10)
nlp = nb_neglog10_sf(c_w, np.maximum(mu_w, 1e-9), a_w)
q = qvalue_from_neglog10p(nlp)
print(f"bg: a0 {bgw.a0:.3f} a1 {bgw.a1:.3f} inflation {bgw.autocorr_inflation:.2f}; chr11 bins {nlp.size}; frac q>=1.3: {(q >= 1.3).mean():.4f}; frac nlp>=3: {(nlp>=3).mean():.4f}")
b0, b1 = s // 100, e // 100
print("alpha per-bin at Havcr2: mean", alpha[s//25:e//25].mean().round(3), "alpha_seg mean", a_seg[b0:b1].mean().round(3), "alpha_w mean", a_w[b0:b1].mean().round(4))
for pos in (46437000, 46448500, 46454900, 46462100, 46474300):
    i = pos // 100
    print(f"  {pos}: c_w {c_w[i]:.0f} mu_w {mu_w[i]:.0f} fold {c_w[i]/mu_w[i]:.2f} alpha_w {a_w[i]:.4f} nlp {nlp[i]:.2f} q {q[i]:.2f}")
