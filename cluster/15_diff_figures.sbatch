#!/bin/bash
#SBATCH --job-name=flexpeak_figs
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=2
#SBATCH --mem=11G
#SBATCH --time=00:20:00
#SBATCH --output=/vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/logs/%x_%j.out
#SBATCH --error=/vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/logs/%x_%j.err
source /vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/_env.sh
python Benchmark/make_diff_figures.py 2>&1; ls -la Benchmark/figures/D*.png
