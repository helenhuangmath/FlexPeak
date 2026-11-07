#!/bin/bash
#SBATCH --job-name=flexpeak_dbgc
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=8
#SBATCH --mem=44G
#SBATCH --time=01:30:00
#SBATCH --exclude=epyc-2-10
#SBATCH --output=/vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/logs/%x_%j.out
#SBATCH --error=/vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/logs/%x_%j.err
set -uo pipefail
source /vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/_env.sh
python cluster/debug_perchrom.py "${1:-chr1,chr9,chr11,chr19}" ${2:-} 2>&1 | grep -v "^\[W::"
echo "=== done $(date)"
