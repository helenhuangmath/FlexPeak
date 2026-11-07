#!/bin/bash
#SBATCH --job-name=flexpeak_dbge
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=22G
#SBATCH --time=00:30:00
#SBATCH --exclude=epyc-2-10
#SBATCH --output=/vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/logs/%x_%j.out
#SBATCH --error=/vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/logs/%x_%j.err
set -uo pipefail
source /vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/_env.sh
python cluster/debug_k27_edge.py 2>&1 | grep -v "^\[W::"
echo "=== done $(date)"
