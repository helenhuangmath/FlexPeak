# Shared environment for the FlexPeak cluster jobs.  Source from sbatch scripts.
# No conda activation: binaries are called from the env prefix directly.
export TOOLS=/vast/projects/wherry/foundation-models-immuno/hhua/tools
export REPO=/vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak
export ENV=$TOOLS/flexpeak_env
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export SEACR_SH=$TOOLS/SEACR/SEACR_1.3.sh
export RGTDATA=$TOOLS/rgtdata
export TMPDIR=$REPO/tmp
export MPLCONFIGDIR=$REPO/tmp/mpl
export NUMBA_CACHE_DIR=$REPO/tmp/numba
export PYTHONPYCACHEPREFIX=$REPO/tmp/pycache
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"
cd "$REPO"
echo "=== $(date)  host $(hostname)  python $(python --version 2>&1)  flexpeak $(python -c 'import flexpeak;print(flexpeak.__version__)' 2>&1)"
