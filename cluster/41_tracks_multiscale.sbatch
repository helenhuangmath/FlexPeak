#!/bin/bash
#SBATCH --job-name=flexpeak_multiscale
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=8
#SBATCH --mem=44G
#SBATCH --time=03:00:00
#SBATCH --exclude=epyc-2-10
#SBATCH --output=/vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/logs/%x_%j.out
#SBATCH --error=/vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/logs/%x_%j.err
# Multi-scale focal pass (scan-derived sliding windows, left-out flank background):
# tests, re-call the five GEO marks on the locus chromosomes, old-vs-new track figures.
set -uo pipefail
source /vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/_env.sh
DATA=$REPO/testing/data; OLD=$REPO/testing/results/tracks; OLDK9=$REPO/testing/results/real/k9_quick
CACHE=$REPO/testing/cache; OUT=$REPO/testing/results/tracks_multiscale
mkdir -p "$OUT"
CHR=chr1,chr4,chr6,chr10,chr11,chr15
LOCI="--locus Pdcd1 chr1:94038305-94052553 FLANK --locus Tox chr4:6686000-6999000 FLANK --locus Tcf7 chr11:52252000-52283000 FLANK --locus Havcr2 chr11:46454000-46477000 FLANK"

echo "### 0. tests"
python -m pytest tests -q -x -p no:cacheprovider 2>&1 | tail -15

echo "### 1. --quick calls with the multi-scale focal pass"
for m in ${MARKS:-H3K9me3 H3K27me3 H3K4me3 H3K27ac CTCF}; do
  /usr/bin/time -f "   $m: wall %e s, max RSS %M kB" flexpeak callpeak -t $DATA/GSE285245_pool_${m}_Cl13.bw \
     --chroms $CHR --quick --cache-dir "$CACHE" --outdir "$OUT/calls_$m" -n ${m}_Cl13 2> "$OUT/calls_${m}.log"
  grep -E "scan:|focal windows|regions \(|wall|Error|Traceback" "$OUT/calls_${m}.log"
  cat "$OUT/calls_$m/${m}_Cl13_summary.txt"
done

echo "### 2. old (same chromosomes) vs new: counts"
for m in H3K9me3 H3K27me3 H3K4me3 H3K27ac CTCF; do
  o=$OLD/calls_$m/${m}_Cl13_peaks.nested.tsv; [ "$m" = H3K9me3 ] && o=$OLDK9/H3K9me3_Cl13_peaks.nested.tsv
  n=$OUT/calls_$m/${m}_Cl13_peaks.nested.tsv
  for f in "$o" "$n"; do
    [ -s "$f" ] || continue
    awk -v F="$f" -F'\t' 'BEGIN{split("chr1 chr4 chr6 chr10 chr11 chr15",a," ");for(i in a)k[a[i]]=1}
      /^#/||$1=="chrom"{next} !($1 in k){next}
      {if($6=="."){nr++; if($5=="domain")nd++; bp+=$3-$2} else ns++}
      END{printf "   %-9s %-4s regions %7d domains %6d subpeaks %7d territory %7.1f Mb  %s\n","'$m'",(F~/multiscale/?"new":"old"),nr,nd,ns,bp/1e6,F}' "$f"
  done
done

echo "### 3. track figures: old vs new"
python examples/locus_tracks.py --out "$OUT/k9" --prefix k9 ${LOCI//FLANK/100000} \
  --track "old" $DATA/GSE285245_pool_H3K9me3_Cl13.bw $OLDK9/H3K9me3_Cl13_peaks.broadPeak $OLDK9/H3K9me3_Cl13_peaks.narrowPeak \
  --track "multi-scale" $DATA/GSE285245_pool_H3K9me3_Cl13.bw $OUT/calls_H3K9me3/H3K9me3_Cl13_peaks.broadPeak $OUT/calls_H3K9me3/H3K9me3_Cl13_peaks.narrowPeak
python examples/locus_tracks.py --out "$OUT/k27me3" --prefix k27me3 ${LOCI//FLANK/40000} \
  --track "old" $DATA/GSE285245_pool_H3K27me3_Cl13.bw $OLD/calls_H3K27me3/H3K27me3_Cl13_peaks.broadPeak $OLD/calls_H3K27me3/H3K27me3_Cl13_peaks.narrowPeak \
  --track "multi-scale" $DATA/GSE285245_pool_H3K27me3_Cl13.bw $OUT/calls_H3K27me3/H3K27me3_Cl13_peaks.broadPeak $OUT/calls_H3K27me3/H3K27me3_Cl13_peaks.narrowPeak
python examples/locus_tracks.py --out "$OUT/marks" --prefix marks ${LOCI//FLANK/25000} \
  --track "CTCF" $DATA/GSE285245_pool_CTCF_Cl13.bw $OUT/calls_CTCF/CTCF_Cl13_peaks.broadPeak $OUT/calls_CTCF/CTCF_Cl13_peaks.narrowPeak \
  --track "H3K4me3" $DATA/GSE285245_pool_H3K4me3_Cl13.bw $OUT/calls_H3K4me3/H3K4me3_Cl13_peaks.broadPeak $OUT/calls_H3K4me3/H3K4me3_Cl13_peaks.narrowPeak \
  --track "H3K27ac" $DATA/GSE285245_pool_H3K27ac_Cl13.bw $OUT/calls_H3K27ac/H3K27ac_Cl13_peaks.broadPeak $OUT/calls_H3K27ac/H3K27ac_Cl13_peaks.narrowPeak \
  --track "H3K27me3" $DATA/GSE285245_pool_H3K27me3_Cl13.bw $OUT/calls_H3K27me3/H3K27me3_Cl13_peaks.broadPeak $OUT/calls_H3K27me3/H3K27me3_Cl13_peaks.narrowPeak \
  --track "H3K9me3" $DATA/GSE285245_pool_H3K9me3_Cl13.bw $OUT/calls_H3K9me3/H3K9me3_Cl13_peaks.broadPeak $OUT/calls_H3K9me3/H3K9me3_Cl13_peaks.narrowPeak
ls "$OUT"/*/
echo "=== done $(date)"
