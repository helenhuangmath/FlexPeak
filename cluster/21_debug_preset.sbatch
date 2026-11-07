#!/bin/bash
#SBATCH --job-name=flexpeak_dbgp
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=22G
#SBATCH --time=00:40:00
#SBATCH --output=/vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/logs/%x_%j.out
#SBATCH --error=/vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/logs/%x_%j.err
set -uo pipefail
source /vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/_env.sh
python cluster/debug_preset_narrow.py 2>&1 | grep -v "^\[W::"
python -m pytest tests -q -x -p no:cacheprovider 2>&1 | tail -5
echo "=== done $(date)"
