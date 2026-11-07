#!/bin/bash
#SBATCH --job-name=flexpeak_tmm
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=22G
#SBATCH --time=00:40:00
#SBATCH --output=/vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/logs/%x_%j.out
#SBATCH --error=/vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/logs/%x_%j.err
source /vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/_env.sh
BW=/vast/projects/wherry/foundation-models-immuno/hhua/data_lab_internal/Huang_SR2025_CnR/geo_processed/per_sample_bw
OUT=$REPO/testing/results/inhouse; CACHE=$REPO/testing/cache
CHR=$(python -c "print(','.join([f'chr{i}' for i in range(1,20)]+['chrX']))")
for m in H3K4me3 CTCF; do
  A=$(ls $BW/*_Arm_*_${m}_Rep[1-4]_*.scale.bw); B=$(ls $BW/*_Cl13_*_${m}_Rep[1-4]_*.scale.bw)
  echo "--- $m tmm"
  flexpeak diff -a $A -b $B --chroms "$CHR" --norm tmm --cache-dir "$CACHE" --outdir "$OUT/diff_${m}_tmm" -n ${m}_Arm_vs_Cl13_tmm 2>&1 | grep -E "normalisation|regions  |median width|state proportions"
done
echo "=== done"
