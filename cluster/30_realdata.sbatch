#!/bin/bash
#SBATCH --job-name=flexpeak_real
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=8
#SBATCH --mem=44G
#SBATCH --time=08:00:00
#SBATCH --output=/vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/logs/%x_%j.out
#SBATCH --error=/vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/logs/%x_%j.err
set -uo pipefail
source /vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/_env.sh
DATA=$REPO/testing/data
OUT=$REPO/testing/results/real
CACHE=$REPO/testing/cache
mkdir -p "$DATA" "$OUT" "$CACHE"
GEO=https://ftp.ncbi.nlm.nih.gov/geo/series/GSE285nnn/GSE285245/suppl
SECTIONS="${1:-1 2 3}"   # which numbered sections to run
CHR=$(python -c "print(','.join([f'chr{i}' for i in range(1,20)]+['chrX']))")
run_section() { case " $SECTIONS " in *" $1 "*) return 0;; *) return 1;; esac; }
fetch() {  # fetch <mark> <cond>
  local f="GSE285245_pool_${1}_${2}.bw"
  if [ ! -s "$DATA/$f" ]; then
    echo "downloading $f"; curl -sSL --retry 3 -o "$DATA/$f" "$GEO/GSE285245%5Fpool%5F${1}%5F${2}%2Ebw" || echo "!! download failed: $f"
  fi
}
for m in CTCF H3K4me3 H3K27ac H3K27me3 H3K9me3; do fetch $m Cl13; done
for m in H3K9me3 H3K27me3 H3K4me3; do fetch $m Arm; done

run_section 1 && {
echo "### 1. quick scan of chr1 at 10/25/50 bp for every mark (Cl13)"
for m in CTCF H3K4me3 H3K27ac H3K27me3 H3K9me3; do
  f=$DATA/GSE285245_pool_${m}_Cl13.bw; [ -s "$f" ] || continue
  echo "--- $m"
  /usr/bin/time -f "   wall %e s  rss %M KB" flexpeak scan -t "$f" --cache-dir "$CACHE" -o "$OUT/scan_${m}_Cl13.json" 2>&1
done
}

run_section 2 && {
echo "### 2. H3K9me3 Cl13, chr1-19+X: full tuning vs --quick (timing + agreement)"
f=$DATA/GSE285245_pool_H3K9me3_Cl13.bw
rm -rf "$CACHE"/GSE285245_pool_H3K9me3_Cl13.*; echo "   (cache cleared before each timed run)"
/usr/bin/time -f "   full tune: wall %e s  rss %M KB" flexpeak callpeak -t "$f" --chroms "$CHR" --preset h3k9me3 \
    --cache-dir "$CACHE" --outdir "$OUT/k9_full" -n H3K9me3_Cl13 --threads 8 2> "$OUT/k9_full.log"; tail -1 "$OUT/k9_full.log"
rm -rf "$CACHE"/GSE285245_pool_H3K9me3_Cl13.*
/usr/bin/time -f "   quick:     wall %e s  rss %M KB" flexpeak callpeak -t "$f" --chroms "$CHR" --quick \
    --cache-dir "$CACHE" --outdir "$OUT/k9_quick" -n H3K9me3_Cl13 --threads 8 2> "$OUT/k9_quick.log"; tail -1 "$OUT/k9_quick.log"
rm -rf "$CACHE"/GSE285245_pool_H3K9me3_Cl13.*
/usr/bin/time -f "   no-tune:   wall %e s  rss %M KB" flexpeak callpeak -t "$f" --chroms "$CHR" --preset h3k9me3 --no-tune \
    --cache-dir "$CACHE" --outdir "$OUT/k9_notune" -n H3K9me3_Cl13 --threads 8 2> "$OUT/k9_notune.log"; tail -1 "$OUT/k9_notune.log"
python - <<'PY'
import os, glob
from flexpeak.peaks import read_nested_tsv, interval_jaccard
out = os.environ.get("OUT", "testing/results/real")
sets = {}
for k in ("k9_full", "k9_quick", "k9_notune"):
    p = glob.glob(f"{out}/{k}/*_peaks.nested.tsv")
    if p:
        sets[k] = read_nested_tsv(p[0])
        s = sets[k]
        print(f"{k:<10} regions {len(s):>6}  domains {s.n_domains:>6}  subpeaks {s.n_subpeaks:>6}  "
              f"territory {s.total_bp/1e6:7.2f} Mb")
keys = list(sets)
for i in range(len(keys)):
    for j in range(i+1, len(keys)):
        print(f"jaccard {keys[i]} vs {keys[j]}: "
              f"{interval_jaccard(sets[keys[i]].intervals(), sets[keys[j]].intervals()):.3f}")
PY
}

run_section 3 && {
echo "### 3. differential Arm vs Cl13 (H3K9me3, H3K27me3, H3K4me3), chr1-19+X"
for m in H3K9me3 H3K27me3 H3K4me3; do
  a=$DATA/GSE285245_pool_${m}_Arm.bw; b=$DATA/GSE285245_pool_${m}_Cl13.bw
  [ -s "$a" ] && [ -s "$b" ] || continue
  echo "--- $m"
  /usr/bin/time -f "   wall %e s  rss %M KB" flexpeak diff -a "$a" -b "$b" --chroms "$CHR" \
      --cache-dir "$CACHE" --outdir "$OUT/diff_$m" -n ${m}_Arm_vs_Cl13 2>&1 | tail -25
done
}
echo "=== done $(date)"
