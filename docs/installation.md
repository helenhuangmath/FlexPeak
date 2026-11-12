# Installation

FlexPeak requires Python 3.9 or later. The core depends on NumPy and SciPy
only; file-format support and acceleration are optional extras.

## From source

```bash
git clone https://github.com/helenhuangmath/FlexPeak.git
cd FlexPeak
pip install -e ".[bam,bigwig,fast,figures]"
```

## Optional extras

| Extra | Installs | Required for |
|---|---|---|
| `bam` | pysam | BAM and CRAM input |
| `bigwig` | pyBigWig | bigWig input |
| `fast` | numba | compiled HMM kernels (a NumPy implementation with matching results is used otherwise) |
| `figures` | matplotlib | `--figures` and `flexpeak stats` |
| `all` | all of the above, plus pandas and jinja2 | everything |
| `dev` | pytest, ruff | the test suite |

## Conda environment

FlexPeak itself is installed with pip, but the compiled dependencies can come
from conda-forge and bioconda:

```bash
conda create -n flexpeak -c conda-forge -c bioconda python=3.11 numpy scipy pysam pybigwig numba matplotlib
conda activate flexpeak
pip install -e /path/to/FlexPeak
```

## Verify the installation

```bash
flexpeak --version
python -m pytest tests -q
```

If `flexpeak` is not on your `PATH`, use `python -m flexpeak.cli` instead.
