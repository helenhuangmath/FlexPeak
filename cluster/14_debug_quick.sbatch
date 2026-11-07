#!/bin/bash
#SBATCH --job-name=flexpeak_quickdbg
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=8
#SBATCH --mem=44G
#SBATCH --time=01:30:00
#SBATCH --output=/vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/logs/%x_%j.out
#SBATCH --error=/vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/logs/%x_%j.err
source /vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/_env.sh
BW=/vast/projects/wherry/foundation-models-immuno/hhua/data_lab_internal/Huang_SR2025_CnR/geo_processed/per_sample_bw
f=$(ls $BW/*_Cl13_*_CTCF_Rep1_*.scale.bw)
CHR=$(python -c "print(','.join([f'chr{i}' for i in range(1,20)]+['chrX']))")
OUT=$REPO/tmp/quickdbg; mkdir -p $OUT
python - "$f" "$CHR" "$OUT" <<'PY' 2>&1
import sys, time
t0 = time.perf_counter()
def T(msg): print(f"[{time.perf_counter()-t0:7.1f}s] {msg}", flush=True)
f, chr_s, out = sys.argv[1:4]
chroms = chr_s.split(",")
from flexpeak.scan import scan_file
from flexpeak import CallParams, call_peaks, load_signal, tune
prof = scan_file(f, cache_dir=out); rec = prof.recommend(); T(f"scan done: {rec.signal_class} bin {rec.bin_size} grid {rec.grid}")
params = CallParams().replace(**rec.overrides())
treat = load_signal(f, bin_size=params.bin_size, chroms=chroms, cache_dir=out); T(f"loaded {treat}")
res = tune(treat, None, base=params, n_chroms=1, grid=rec.grid, verbose=True); T(f"tuned: {res.params.to_dict()['posterior_cutoff']} {res.params.min_fold}")
r = call_peaks(treat, None, res.params, return_details=True); T(f"called {len(r.peaks)} regions, seg bin {r.diagnostics['segment_bin_size']}, scale {r.diagnostics['signal_scale_bp']}")
import cProfile, pstats
cProfile.run("call_peaks(treat.subset(['chr19']), None, res.params)", f"{out}/prof.out")
pstats.Stats(f"{out}/prof.out").sort_stats("cumulative").print_stats(28)
PY
echo "=== done $(date)"
