#!/bin/bash
#SBATCH --job-name=flexpeak_generalized
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=8
#SBATCH --mem=44G
#SBATCH --time=04:00:00
#SBATCH --exclude=epyc-2-10
#SBATCH --output=/vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/logs/%x_%j.out
#SBATCH --error=/vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/logs/%x_%j.err
# Stratified multi-chromosome scan, robust background start, weak-domain pass,
# replicate-group tuning halves + coordinate search: tests, scan speed/agreement,
# re-calls with replicates, replicate-support evaluation (tuning vs held-out
# chromosomes), track figures.
set -uo pipefail
source /vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/_env.sh
DATA=$REPO/testing/data; CACHE=$REPO/testing/cache
OLD=$REPO/testing/results/tracks_multiscale; OUT=$REPO/testing/results/tracks_generalized
BW=/vast/projects/wherry/foundation-models-immuno/hhua/data_lab_internal/Huang_SR2025_CnR/geo_processed/per_sample_bw
mkdir -p "$OUT"
CHR=chr1,chr4,chr6,chr10,chr11,chr15
MARKS=${MARKS:-H3K9me3 H3K27me3 H3K4me3 H3K27ac CTCF}

echo "### 0. tests"
python -m pytest tests -q -x -p no:cacheprovider 2>&1 | tail -15

echo "### 1. scan: chr1 only vs 6 stratified windows (same 60 Mb)"
for m in $MARKS; do
  for mode in "--scan-chrom chr1" "--scan-windows 6"; do
    /usr/bin/time -f "      wall %e s" flexpeak scan -t $DATA/GSE285245_pool_${m}_Cl13.bw --cache-dir "$CACHE" $mode \
      2>&1 | grep -E "chromosome|elapsed|class|recommended|focal windows|tuning grid|wall" | sed "s/^/   $m ${mode%% *} /"
  done
done

echo "### 2. --quick calls with replicate-group tuning"
for m in $MARKS; do
  R=$(ls $BW/*_Cl13_*_${m}_Rep*_*.scale.bw | tr '\n' ' ')
  /usr/bin/time -f "   $m: wall %e s, max RSS %M kB" flexpeak callpeak -t $DATA/GSE285245_pool_${m}_Cl13.bw \
     --replicates $R --chroms $CHR --quick --cache-dir "$CACHE" --outdir "$OUT/calls_$m" -n ${m}_Cl13 \
     2> "$OUT/calls_${m}.log"
  sed -n '/self-tuning report/,/calling peaks/p' "$OUT/calls_${m}.log"
  grep -E "scan:|regions \(|wall|Error|Traceback" "$OUT/calls_${m}.log"
done

echo "### 3. replicate support: previous (multi-scale) vs generalized"
TUNE=$(grep -m1 "tuning chromosomes" "$OUT/calls_H3K9me3.log" | sed 's/.*: //; s/ //g' | tr ',' '\n' | cut -d: -f1 | sort -u | paste -sd,)
grep -m1 "tuning chromosomes" "$OUT/calls_H3K9me3.log"
echo "   tuning chromosomes: $TUNE"
for m in $MARKS; do
  python cluster/eval_replicate_support.py $m "$TUNE" previous=$OLD/calls_$m generalized=$OUT/calls_$m
done

echo "### 4. track figures"
LOCI="--locus Pdcd1 chr1:94038305-94052553 FLANK --locus Tox chr4:6686000-6999000 FLANK --locus Tcf7 chr11:52252000-52283000 FLANK --locus Havcr2 chr11:46454000-46477000 FLANK"
for pair in "H3K9me3 k9 100000" "H3K27me3 k27me3 40000"; do
  set -- $pair; m=$1; p=$2; fl=$3
  T=""; for i in 1 2 3 4; do f=$(ls $BW/*_Cl13_*_${m}_Rep${i}_*.scale.bw); T="$T --track Cl13_rep$i $f - -"; done
  python examples/locus_tracks.py --out "$OUT/$p" --prefix $p ${LOCI//FLANK/$fl} \
    --track "previous" $DATA/GSE285245_pool_${m}_Cl13.bw $OLD/calls_$m/${m}_Cl13_peaks.broadPeak $OLD/calls_$m/${m}_Cl13_peaks.narrowPeak \
    --track "generalized" $DATA/GSE285245_pool_${m}_Cl13.bw $OUT/calls_$m/${m}_Cl13_peaks.broadPeak $OUT/calls_$m/${m}_Cl13_peaks.narrowPeak \
    $T
done
python examples/locus_tracks.py --out "$OUT/marks" --prefix marks ${LOCI//FLANK/25000} \
  $(for m in CTCF H3K4me3 H3K27ac H3K27me3 H3K9me3; do echo --track $m $DATA/GSE285245_pool_${m}_Cl13.bw $OUT/calls_$m/${m}_Cl13_peaks.broadPeak $OUT/calls_$m/${m}_Cl13_peaks.narrowPeak; done)
ls "$OUT"/*/
echo "=== done $(date)"

echo "### 5. same-node timing, H3K9me3: old-style (chr1 scan, 1 tuning chrom, no replicates) vs new default"
for mode in "--scan-chrom chr1 --tune-chroms 1" "--replicates $(ls $BW/*_Cl13_*_H3K9me3_Rep*_*.scale.bw | tr '\n' ' ')"; do
  /usr/bin/time -f "   ${mode:0:40}: wall %e s" flexpeak callpeak -t $DATA/GSE285245_pool_H3K9me3_Cl13.bw --chroms $CHR --quick \
    --cache-dir "$CACHE" --outdir "$OUT/timing" -n t --quiet $mode 2>&1 | grep wall
  /usr/bin/time -f "   ${mode:0:40} --no-tune: wall %e s" flexpeak callpeak -t $DATA/GSE285245_pool_H3K9me3_Cl13.bw --chroms $CHR --quick \
    --cache-dir "$CACHE" --outdir "$OUT/timing" -n t --quiet --no-tune $mode 2>&1 | grep wall
done
echo "=== timing done $(date)"
