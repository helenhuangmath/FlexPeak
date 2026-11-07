#!/bin/bash
#SBATCH --job-name=flexpeak_tr11
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=8
#SBATCH --mem=44G
#SBATCH --time=01:30:00
#SBATCH --output=/vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/logs/%x_%j.out
#SBATCH --error=/vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/logs/%x_%j.err
set -uo pipefail
source /vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/_env.sh
DATA=$REPO/testing/data; CACHE=$REPO/testing/cache; OUT=$REPO/testing/results/tracks; TAG="${1:-focal}"
mkdir -p "$OUT"
LOCI="--locus Tcf7 chr11:52252000-52283000 25000 --locus Havcr2 chr11:46454000-46477000 25000"
for m in CTCF H3K4me3 H3K27ac H3K27me3; do
  f=$DATA/GSE285245_pool_${m}_Cl13.bw
  /usr/bin/time -f "   $m: wall %e s" flexpeak callpeak -t "$f" --chroms chr11 --quick --cache-dir "$CACHE" \
      --outdir "$OUT/${TAG}_$m" -n ${m}_Cl13 2> "$OUT/${TAG}_$m.log"; grep -E "wall" "$OUT/${TAG}_$m.log"; grep -E "regions \(|focal" "$OUT/${TAG}_$m.log" | tail -2
done
T=""; for m in CTCF H3K4me3 H3K27ac H3K27me3; do T="$T --track $m $DATA/GSE285245_pool_${m}_Cl13.bw $OUT/${TAG}_$m/${m}_Cl13_peaks.broadPeak $OUT/${TAG}_$m/${m}_Cl13_peaks.narrowPeak"; done
python examples/locus_tracks.py --out "$OUT/${TAG}_marks" --prefix $TAG $LOCI $T
echo "=== done $(date)"
