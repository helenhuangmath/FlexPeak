#!/bin/bash
#SBATCH --job-name=flexpeak_agree
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=2
#SBATCH --mem=11G
#SBATCH --time=00:20:00
#SBATCH --output=/vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/logs/%x_%j.out
#SBATCH --error=/vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/logs/%x_%j.err
source /vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/_env.sh
export OUT=$REPO/testing/results/inhouse
python - <<'PY'
import os, glob, itertools
from flexpeak.peaks import read_nested_tsv, interval_jaccard
out = os.environ["OUT"]
for d in sorted(glob.glob(f"{out}/calls_*")):
    sets = {}
    for p in sorted(glob.glob(f"{d}/*_peaks.nested.tsv")):
        s = read_nested_tsv(p); k = os.path.basename(p).split("_peaks")[0]; sets[k] = s
        sc = s.params.get("scan", {})
        print(f"{os.path.basename(d):<14} {k:<38} regions {len(s):>6} domains {s.n_domains:>6} subpeaks {s.n_subpeaks:>6} territory {s.total_bp/1e6:7.1f} Mb scan={sc.get('class')}")
    j = [interval_jaccard(sets[a].intervals(), sets[b].intervals()) for a, b in itertools.combinations(sets, 2)]
    if j: print(f"{os.path.basename(d):<14} pairwise Jaccard between replicates: mean {sum(j)/len(j):.3f} min {min(j):.3f} max {max(j):.3f}")
PY
