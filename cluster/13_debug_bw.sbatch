#!/bin/bash
#SBATCH --job-name=flexpeak_bwdbg
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=22G
#SBATCH --time=00:40:00
#SBATCH --output=/vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/logs/%x_%j.out
#SBATCH --error=/vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/logs/%x_%j.err
source /vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/_env.sh
BW=/vast/projects/wherry/foundation-models-immuno/hhua/data_lab_internal/Huang_SR2025_CnR/geo_processed/per_sample_bw
f=$(ls $BW/*_Cl13_*_CTCF_Rep1_*.scale.bw); echo "$f"; ls -la "$f"
timeout 1500 python cluster/debug_bw.py "$f" 2>&1
echo "=== done $(date)"
