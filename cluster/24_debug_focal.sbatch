#!/bin/bash
#SBATCH --job-name=flexpeak_dbg_focal
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=8
#SBATCH --mem=44G
#SBATCH --time=01:00:00
#SBATCH --exclude=epyc-2-10
#SBATCH --output=/vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/logs/%x_%j.out
#SBATCH --error=/vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/logs/%x_%j.err
set -uo pipefail
source /vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/_env.sh
python cluster/debug_focal_scales.py H3K9me3 chr1,chr11
python cluster/debug_focal_scales.py H3K4me3 chr1,chr11
python cluster/debug_focal_scales.py H3K27me3 chr1,chr11
echo "=== done $(date)"
