#!/bin/bash
#SBATCH --job-name=flexpeak_inhouse
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=8
#SBATCH --mem=44G
#SBATCH --time=10:00:00
#SBATCH --output=/vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/logs/%x_%j.out
#SBATCH --error=/vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/logs/%x_%j.err
set -uo pipefail
source /vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/_env.sh
BW=/vast/projects/wherry/foundation-models-immuno/hhua/data_lab_internal/Huang_SR2025_CnR/geo_processed/per_sample_bw
export OUT=$REPO/testing/results/inhouse
CACHE=$REPO/testing/cache
mkdir -p "$OUT" "$CACHE"
CHR=$(python -c "print(','.join([f'chr{i}' for i in range(1,20)]+['chrX']))")
MARKS="${1:-CTCF H3K4me3 H3K27ac H3K27me3 H3K9me3}"
SECTIONS="${2:-A B}"
run_section() { case " $SECTIONS " in *" $1 "*) return 0;; *) return 1;; esac; }

run_section A && {
echo "### A. per-replicate quick calls (Cl13) and replicate agreement"
for m in ${MARKS_A:-CTCF H3K9me3}; do
  echo "--- $m"
  for f in $BW/*_Cl13_*_${m}_Rep[1-4]_*.scale.bw; do
    n=$(basename $f .scale.bw | sed 's/^GSM[0-9]*_//')
    /usr/bin/time -f "   $n: wall %e s  rss %M KB" flexpeak callpeak -t "$f" --chroms "$CHR" --quick \
        --cache-dir "$CACHE" --outdir "$OUT/calls_$m" -n "$n" --quiet 2>&1 | grep -E "wall|Error|Traceback" || true
  done
done
python - <<'PY'
import os, glob, itertools
from flexpeak.peaks import read_nested_tsv, interval_jaccard
out = os.environ.get("OUT")
for d in sorted(glob.glob(f"{out}/calls_*")):
    sets = {}
    for p in sorted(glob.glob(f"{d}/*_peaks.nested.tsv")):
        s = read_nested_tsv(p); sets[os.path.basename(p).split("_peaks")[0]] = s
        sc = s.params.get("scan", {})
        print(f"{os.path.basename(d):<16} {os.path.basename(p).split('_peaks')[0]:<40} regions {len(s):>7} domains {s.n_domains:>6} "
              f"subpeaks {s.n_subpeaks:>6} territory {s.total_bp/1e6:7.2f} Mb  scan={sc.get('class')}/{sc.get('scale_bp')}bp")
    ks = list(sets)
    j = [interval_jaccard(sets[a].intervals(), sets[b].intervals()) for a, b in itertools.combinations(ks, 2)]
    if j:
        print(f"{os.path.basename(d):<16} pairwise Jaccard between replicates: mean {sum(j)/len(j):.3f} min {min(j):.3f} max {max(j):.3f}")
PY
}

run_section B && {
echo "### B. differential Arm vs Cl13 with 4 replicates per condition"
for m in $MARKS; do
  echo "--- $m"
  A=$(ls $BW/*_Arm_*_${m}_Rep[1-4]_*.scale.bw); B=$(ls $BW/*_Cl13_*_${m}_Rep[1-4]_*.scale.bw)
  /usr/bin/time -f "   wall %e s  rss %M KB" flexpeak diff -a $A -b $B --chroms "$CHR" \
      --cache-dir "$CACHE" --outdir "$OUT/diff_$m" -n ${m}_Arm_vs_Cl13 2>&1 | grep -vE "^\[flexpeak diff\] loading" | tail -22
done
}
echo "=== done $(date)"
