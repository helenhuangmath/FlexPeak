import time, sys, numpy as np, pyBigWig
f = sys.argv[1]
bw = pyBigWig.open(f)
ch = bw.chroms(); print("n chroms", len(ch), "first", list(ch.items())[:5])
print("header", bw.header())
t = time.perf_counter(); iv = bw.intervals("chr1", 0, 20_000_000); print(f"intervals chr1[0:20Mb]: {len(iv):,} in {time.perf_counter()-t:.1f}s; first {iv[:3]}")
t = time.perf_counter(); v = bw.values("chr1", 0, 20_000_000, numpy=True); print(f"values chr1[0:20Mb]: {time.perf_counter()-t:.1f}s, nan frac {np.isnan(v).mean():.3f}, mean {np.nanmean(v):.4f}, max {np.nanmax(v):.2f}")
t = time.perf_counter(); v = bw.values("chr1", 20_000_000, 60_000_000, numpy=True); print(f"values chr1[20:60Mb]: {time.perf_counter()-t:.1f}s")
from flexpeak.io.load import load_signal
t = time.perf_counter(); cov = load_signal(f, bin_size=5, chroms=["chr1"], cache=False); print(f"load chr1 @5bp: {time.perf_counter()-t:.1f}s {cov}")
from flexpeak.scan import scan_coverage
t = time.perf_counter(); prof = scan_coverage(cov, (10, 25, 50), chrom="chr1"); print(f"scan: {time.perf_counter()-t:.1f}s class {prof.recommend().signal_class}")
