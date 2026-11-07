#!/bin/bash
#SBATCH --job-name=flexpeak_bench_ms
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=8
#SBATCH --mem=44G
#SBATCH --time=04:00:00
#SBATCH --exclude=epyc-2-10
#SBATCH --output=/vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/logs/%x_%j.out
#SBATCH --error=/vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/logs/%x_%j.err
# Peak benchmark with the multi-scale focal pass, WITHOUT replacing the published
# metrics/figures: back them up, run, keep the new table aside, restore.
set -uo pipefail
source /vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/_env.sh
BK=$REPO/testing/results/bench_multiscale; mkdir -p $BK/published $BK/new
cp Benchmark/results/metrics.tsv Benchmark/results/commands.txt Benchmark/results/versions.txt $BK/published/
cp Benchmark/figures/B*.png $BK/published/
python Benchmark/run_benchmark.py 2>&1 | grep -v "^\s*$"
cp Benchmark/results/metrics.tsv $BK/new/metrics.tsv
cp $BK/published/metrics.tsv $BK/published/commands.txt $BK/published/versions.txt Benchmark/results/
cp $BK/published/B*.png Benchmark/figures/
echo "### FlexPeak rows: published vs new"
for f in $BK/published/metrics.tsv $BK/new/metrics.tsv; do echo "== $f"; head -1 $f; grep -i flexpeak $f; done
echo "=== done $(date)"
