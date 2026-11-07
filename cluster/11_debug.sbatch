#!/bin/bash
#SBATCH --job-name=flexpeak_dbg
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=22G
#SBATCH --time=01:00:00
#SBATCH --output=/vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/logs/%x_%j.out
#SBATCH --error=/vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/logs/%x_%j.err
set -uo pipefail
source /vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/_env.sh
python cluster/debug_scan.py 2>&1 | grep -E "^(NULL|BROAD|NARROW|MIXED|REAL)"
python -m pytest tests -q --durations=10 -p no:cacheprovider 2>&1 | tail -40
echo "=== done $(date)"
