#!/bin/bash
#SBATCH --job-name=flexpeak_tracks
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=8
#SBATCH --mem=44G
#SBATCH --time=03:00:00
#SBATCH --output=/vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/logs/%x_%j.out
#SBATCH --error=/vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/logs/%x_%j.err
set -uo pipefail
source /vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/_env.sh
DATA=$REPO/testing/data; REAL=$REPO/testing/results/real; IH=$REPO/testing/results/inhouse
CACHE=$REPO/testing/cache; OUT=$REPO/testing/results/tracks; FIG=$REPO/docs/figures/tracks
mkdir -p "$OUT" "$FIG"
LOCI="--locus Pdcd1 chr1:94038305-94052553 FLANK --locus Sell chr1:164062000-164083000 FLANK --locus Tox chr4:6686000-6999000 FLANK --locus Cd8a chr6:71373000-71380000 FLANK --locus Ifng chr10:118440000-118446000 FLANK --locus Tcf7 chr11:52252000-52283000 FLANK --locus Havcr2 chr11:46454000-46477000 FLANK --locus Il7r chr15:9506000-9531000 FLANK"
CHR=chr1,chr4,chr6,chr10,chr11,chr15

echo "### 1. quick calls on the pooled Cl13 bigWigs for the locus chromosomes"
for m in CTCF H3K4me3 H3K27ac H3K27me3; do
  f=$DATA/GSE285245_pool_${m}_Cl13.bw
  [ -s "$OUT/calls_$m/${m}_Cl13_peaks.broadPeak" ] && continue
  /usr/bin/time -f "   $m: wall %e s" flexpeak callpeak -t "$f" --chroms $CHR --quick --cache-dir "$CACHE" \
      --outdir "$OUT/calls_$m" -n ${m}_Cl13 --quiet 2>&1 | grep -E "wall|Error|Traceback"
done

echo "### 2. five marks per locus"
python examples/locus_tracks.py --out "$OUT/marks" --prefix marks ${LOCI//FLANK/25000} \
  --track "CTCF" $DATA/GSE285245_pool_CTCF_Cl13.bw $OUT/calls_CTCF/CTCF_Cl13_peaks.broadPeak $OUT/calls_CTCF/CTCF_Cl13_peaks.narrowPeak \
  --track "H3K4me3" $DATA/GSE285245_pool_H3K4me3_Cl13.bw $OUT/calls_H3K4me3/H3K4me3_Cl13_peaks.broadPeak $OUT/calls_H3K4me3/H3K4me3_Cl13_peaks.narrowPeak \
  --track "H3K27ac" $DATA/GSE285245_pool_H3K27ac_Cl13.bw $OUT/calls_H3K27ac/H3K27ac_Cl13_peaks.broadPeak $OUT/calls_H3K27ac/H3K27ac_Cl13_peaks.narrowPeak \
  --track "H3K27me3" $DATA/GSE285245_pool_H3K27me3_Cl13.bw $OUT/calls_H3K27me3/H3K27me3_Cl13_peaks.broadPeak $OUT/calls_H3K27me3/H3K27me3_Cl13_peaks.narrowPeak \
  --track "H3K9me3" $DATA/GSE285245_pool_H3K9me3_Cl13.bw $REAL/k9_quick/H3K9me3_Cl13_peaks.broadPeak $REAL/k9_quick/H3K9me3_Cl13_peaks.narrowPeak

echo "### 3. H3K9me3: full sweep vs --quick vs no-tune"
python examples/locus_tracks.py --out "$OUT/k9_compare" --prefix k9 ${LOCI//FLANK/100000} \
  --track "full sweep" $DATA/GSE285245_pool_H3K9me3_Cl13.bw $REAL/k9_full/H3K9me3_Cl13_peaks.broadPeak $REAL/k9_full/H3K9me3_Cl13_peaks.narrowPeak \
  --track "quick scan" $DATA/GSE285245_pool_H3K9me3_Cl13.bw $REAL/k9_quick/H3K9me3_Cl13_peaks.broadPeak $REAL/k9_quick/H3K9me3_Cl13_peaks.narrowPeak \
  --track "no tuning" $DATA/GSE285245_pool_H3K9me3_Cl13.bw $REAL/k9_notune/H3K9me3_Cl13_peaks.broadPeak $REAL/k9_notune/H3K9me3_Cl13_peaks.narrowPeak

echo "### 4. differential Arm vs Cl13 (in-house 4 vs 4), pooled signal shown"
for m in H3K4me3 H3K27me3 H3K9me3; do
  python examples/locus_tracks.py --out "$OUT/diff_$m" --prefix diff_$m ${LOCI//FLANK/60000} \
    --track "$m Arm" $DATA/GSE285245_pool_${m}_Arm.bw - - \
    --track "$m Cl13" $DATA/GSE285245_pool_${m}_Cl13.bw - - \
    --track "4 vs 4 diff" $DATA/GSE285245_pool_${m}_Cl13.bw $IH/diff_$m/${m}_Arm_vs_Cl13_diffpeaks.narrowPeak -
done

echo "### 5. IGV session"
python - <<'PY'
import os
repo = os.environ["REPO"]; data = f"{repo}/testing/data"; out = f"{repo}/testing/results/tracks"
real = f"{repo}/testing/results/real"; ih = f"{repo}/testing/results/inhouse"
res = []
def add(path, name):
    if os.path.exists(path): res.append((os.path.realpath(path), name))
for m in ("CTCF", "H3K4me3", "H3K27ac", "H3K27me3", "H3K9me3"):
    add(f"{data}/GSE285245_pool_{m}_Cl13.bw", f"{m} Cl13 signal")
    add(f"{data}/GSE285245_pool_{m}_Arm.bw", f"{m} Arm signal")
    d = f"{real}/k9_quick" if m == "H3K9me3" else f"{out}/calls_{m}"
    add(f"{d}/{m}_Cl13_peaks.broadPeak", f"{m} Cl13 regions (FlexPeak --quick)")
    add(f"{d}/{m}_Cl13_peaks.narrowPeak", f"{m} Cl13 sub-peaks")
    add(f"{d}/{m}_Cl13_nested.bed", f"{m} Cl13 nested (BED12)")
    add(f"{ih}/diff_{m}/{m}_Arm_vs_Cl13_gain.bed", f"{m} higher in Arm (4 vs 4)")
    add(f"{ih}/diff_{m}/{m}_Arm_vs_Cl13_loss.bed", f"{m} higher in Cl13 (4 vs 4)")
for k in ("k9_full", "k9_notune"):
    add(f"{real}/{k}/H3K9me3_Cl13_peaks.broadPeak", f"H3K9me3 Cl13 regions ({k})")
lines = ['<?xml version="1.0" encoding="UTF-8"?>',
         '<Session genome="mm10" locus="chr1:94038305-94052553" version="8">', "  <Resources>"]
lines += [f'    <Resource path="{p}" name="{n}"/>' for p, n in res]
lines += ["  </Resources>", "</Session>"]
open(f"{out}/igv_session_mm10.xml", "w").write("\n".join(lines) + "\n")
open(f"{out}/TRACKS.txt", "w").write("# load in IGV (File > Open Session) or drag single files in\n"
                                    + "\n".join(f"{n}\t{p}" for p, n in res) + "\n")
print(f"wrote {out}/igv_session_mm10.xml with {len(res)} tracks")
PY

echo "### 6. representative subset into the repo docs"
for f in "$OUT"/marks/marks_Pdcd1.png "$OUT"/marks/marks_Tcf7.png "$OUT"/marks/marks_Havcr2.png \
         "$OUT"/k9_compare/k9_Tox.png "$OUT"/k9_compare/k9_Pdcd1.png \
         "$OUT"/diff_H3K4me3/diff_H3K4me3_Pdcd1.png "$OUT"/diff_H3K27me3/diff_H3K27me3_Tcf7.png "$OUT"/diff_H3K9me3/diff_H3K9me3_Tox.png; do
  [ -f "$f" ] && cp "$f" "$FIG/"
done
ls "$FIG"
echo "=== done $(date)"
