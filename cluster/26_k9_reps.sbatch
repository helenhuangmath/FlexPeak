#!/bin/bash
#SBATCH --job-name=flexpeak_k9_reps
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=22G
#SBATCH --time=01:00:00
#SBATCH --exclude=epyc-2-10
#SBATCH --output=/vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/logs/%x_%j.out
#SBATCH --error=/vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/logs/%x_%j.err
set -uo pipefail
source /vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/_env.sh
python cluster/check_k9_replicates.py
exit 0
T=""; for i in 1 2 3 4; do f=$(ls $BW/*_Cl13_*_H3K9me3_Rep${i}_*.scale.bw); T="$T --track Cl13_rep$i $f - -"; done
python examples/locus_tracks.py --out $REPO/testing/results/tracks_multiscale/k9_reps --prefix k9reps \
  --locus Pdcd1 chr1:94038305-94052553 100000 \
  --track pooled $REPO/testing/data/GSE285245_pool_H3K9me3_Cl13.bw $REPO/testing/results/tracks_multiscale/calls_H3K9me3/H3K9me3_Cl13_peaks.broadPeak $REPO/testing/results/tracks_multiscale/calls_H3K9me3/H3K9me3_Cl13_peaks.narrowPeak $T
echo "=== done $(date)"
