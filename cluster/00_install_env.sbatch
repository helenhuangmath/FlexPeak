#!/bin/bash
#SBATCH --job-name=flexpeak_env
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=8
#SBATCH --mem=44G
#SBATCH --time=04:00:00
#SBATCH --output=/vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/logs/%x_%j.out
#SBATCH --error=/vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/logs/%x_%j.err
set -euo pipefail

TOOLS=/vast/projects/wherry/foundation-models-immuno/hhua/tools
ENV=$TOOLS/flexpeak_env
REPO=/vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak
export TMPDIR=$REPO/tmp; mkdir -p $TMPDIR
export CONDA_PKGS_DIRS=$TOOLS/conda_pkgs
export PIP_CACHE_DIR=$TOOLS/conda_cache/pip

module purge 2>/dev/null || true
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
echo "=== conda $(conda --version) at $(date)"

if [ ! -x "$ENV/bin/python" ]; then
  conda create -y -p "$ENV" -c conda-forge -c bioconda \
    python=3.10 numpy scipy pysam pybigwig numba matplotlib pytest pandas \
    macs3 macs2 sicer2 bedtools samtools ucsc-bigwigtobedgraph ucsc-bedgraphtobigwig \
    r-base=4.3 htseq pip
fi
conda activate "$ENV"
echo "=== python: $(python --version); macs2: $(macs2 --version 2>&1 | head -1); macs3: $(macs3 --version 2>&1 | head -1)"
echo "=== sicer2: $(sicer --version 2>&1 | head -1 || true); bedtools: $(bedtools --version); R: $(R --version | head -1)"

# THOR (RGT) from pip; its genome data goes in the project tree, not $HOME.
export RGTDATA=$TOOLS/rgtdata
mkdir -p "$RGTDATA"
pip install --no-input RGT 2>&1 | tail -3 || echo "!! RGT pip install failed"
python -c "import rgt, sys; print('rgt', rgt.__version__)" || true
which rgt-THOR || echo "!! rgt-THOR not on PATH"
# rgt-data: the mm10 chrom sizes are shipped with the package; populate the data dir.
python - <<'PY' || true
import os, shutil, rgt
src = os.path.join(os.path.dirname(rgt.__file__), "..", "data")
print("rgt package data:", os.path.abspath(src), os.path.exists(src))
PY
ls "$RGTDATA" || true

# SEACR: bash + R script, no install step.
if [ ! -d "$TOOLS/SEACR" ]; then
  git clone --depth 1 https://github.com/FredHutch/SEACR "$TOOLS/SEACR"
fi
ls "$TOOLS/SEACR"

# FlexPeak itself, editable.
pip install --no-input --no-deps -e "$REPO" 2>&1 | tail -2
python -c "import flexpeak; print('flexpeak', flexpeak.__version__, flexpeak.__file__)"
pip list 2>/dev/null | grep -iE '^(numpy|scipy|pysam|pyBigWig|numba|matplotlib|pytest|RGT|MACS2|MACS3|SICER2|HTSeq|hmmlearn) '
echo "=== done $(date)"

# --- compatibility patches for the comparators (idempotent) ---
# THOR 1.0.2 stringifies a tuple that numpy >= 2 prints with np.float64(x) wrappers.
# SICER2 1.0.3: sicer_df never sets args.input_type, and uses the removed scipy.array alias.
python - <<'PY'
import os, site
E = site.getsitepackages()[0]
p = os.path.join(E, "rgt/THOR/dpc_help.py"); s = open(p).read()
old = "        tmp = el.data.split(',')\n"
new = ("        _data = __import__('re').sub(r'np\\.(?:float64|float32|int64|int32)\\(([^)]*)\\)', r'\\1', el.data)\n"
       "        tmp = _data.split(',')\n")
if old in s and "_data" not in s: open(p, "w").write(s.replace(old, new)); print("patched THOR dpc_help.py")
p = os.path.join(E, "sicer/main/run_SICER_df.py"); s = open(p).read()
old = "    args_1.df = False\n    args_2.df = False\n"
new = old + '    for _a in (args_1, args_2):\n        if not hasattr(_a, "input_type"):\n            _a.input_type = "SE"\n'
if old in s and "input_type" not in s: open(p, "w").write(s.replace(old, new)); print("patched sicer_df input_type")
p = os.path.join(E, "sicer/src/compare_two_libraries_on_islands.py"); s = open(p).read()
if "scipy.array(" in s: open(p, "w").write(s.replace("scipy.array(", "__import__('numpy').array(")); print("patched sicer scipy.array")
PY
