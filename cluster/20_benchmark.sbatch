#!/bin/bash
#SBATCH --job-name=flexpeak_bench
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=8
#SBATCH --mem=44G
#SBATCH --time=08:00:00
#SBATCH --output=/vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/logs/%x_%j.out
#SBATCH --error=/vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/logs/%x_%j.err
set -uo pipefail
source /vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/_env.sh
MODE="${1:-all}"     # all | peaks | diff
EXTRA="${2:-}"       # e.g. --quick
if [ "$MODE" = all ] || [ "$MODE" = peaks ]; then
  echo "### peak-calling benchmark"
  python Benchmark/run_benchmark.py $EXTRA 2>&1
  python Benchmark/make_figures.py 2>&1 | tail -3
fi
if [ "$MODE" = all ] || [ "$MODE" = diff ]; then
  echo "### differential benchmark"
  python Benchmark/run_diff_benchmark.py $EXTRA 2>&1
  python Benchmark/make_diff_figures.py 2>&1 | tail -3
fi
echo "=== done $(date)"
