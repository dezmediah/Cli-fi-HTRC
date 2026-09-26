#!/usr/bin/env bash
# One short command per step, for the Remote Desktop terminal in SECURE mode.
#   bash ~/Desktop/Clifi-htrc/capsule/run_secure.sh <step>
# Steps, in order:  look | bertopic | keyness | status | booknlp | finalize | pmi | request-b | done | kwic | w2v
# Notebooks run headless as COPIES on the secure volume (her files are never modified); each copy
# keeps its outputs, including the releaseresults output of the notebooks that submit themselves.
set -u
R=$HOME/Desktop/Clifi-htrc; SV=/media/secure_volume; TXT=$SV/fa50b375-3216-4edd-a685-98488562b723
PY=$HOME/anaconda3/bin/python; JUP=$HOME/anaconda3/bin/jupyter; META=$HOME/metadata_august2026.csv
LOG=$SV/logs; mkdir -p "$LOG" 2>/dev/null

nb() {  # nb <notebook name> [sed expression applied to the COPY]
  local name=$1 expr=${2:-} run=$SV/run_$1.ipynb
  cp "$R/notebooks/$name.ipynb" "$run" || exit 1
  [ -n "$expr" ] && sed -i "$expr" "$run"
  nohup "$JUP" nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 "$run" > "$LOG/$name.log" 2>&1 &
  echo "started $name, pid $!  log: $LOG/$name.log  executed copy: $run"
}

case ${1:-} in
  look)
    echo "== secure volume"; ls -la "$SV"; echo; echo "volumes: $(ls "$TXT" 2>/dev/null | wc -l) (expect 270)"; df -h "$SV" | tail -1
    for d in "$SV"/out_* "$SV"/bnlp "$SV"/whole_corpus_word2vec_models; do [ -e "$d" ] && du -sh "$d"; done; true ;;
  bertopic)  nb htrc_bertopic_pipeline 's/TORCH_THREADS = None/TORCH_THREADS = 32/' ;;   # ~7 h; out_bertopic/ (release) + work_bertopic/ (never)
  keyness)   nb SF_novel_keyness_v2 ;;                                                    # ~4 h; submits itself at the end (request A)
  booknlp)
    cd "$R/scripts" && nohup "$HOME/venv-booknlp/bin/python" booknlp_batch.py --input "$TXT" --out "$SV/bnlp" --procs 6 --threads 8 > "$LOG/booknlp.log" 2>&1 &
    echo "started booknlp, pid $!  log: $LOG/booknlp.log  (all 270 volumes, ~5 h; re-run this step after any crash, it resumes)" ;;
  finalize)  "$HOME/venv-booknlp/bin/python" "$R/scripts/booknlp_batch.py" --finalize --out "$SV/bnlp" --metadata "$META" ;;
  pmi)
    nohup "$PY" "$R/capsule/extras_analysis.py" pmi --input "$TXT" --metadata "$META" --out "$SV/out_extras" --procs 40 > "$LOG/pmi.log" 2>&1 &
    echo "started pmi, pid $!  log: $LOG/pmi.log  (minutes)" ;;
  request-b)
    git -C "$R" rev-parse HEAD > "$SV/COMMIT.txt"
    echo "== sizes (total must be under 1 GB)"; du -shc "$SV/out_bertopic" "$SV/bnlp/release" "$SV/out_extras/pmi" 2>&1 | tail -4
    releaseresults add "$SV/out_bertopic" "$SV/bnlp/release" "$SV/out_extras/pmi" "$META" "$SV/COMMIT.txt"
    echo; echo "Read the list above. If it is right:  bash $R/capsule/run_secure.sh done" ;;
  done)      releaseresults done ;;
  kwic)      nb SF_keyword_context_v4 ;;                                                  # last; submits itself (request C)
  w2v)       nb SF_word2vec_whole_corpus_v2 ;;                                            # optional; 3 seeds, submits itself (request D)
  status)
    echo "== running"; pgrep -af 'nbconvert|booknlp_batch|extras_analysis' | grep -v pgrep || echo "(nothing)"
    echo "== load"; uptime
    for f in "$LOG"/*.log; do [ -e "$f" ] && { echo "== $f"; tail -4 "$f"; }; done
    echo "== outputs"; du -sh "$SV"/out_* "$SV"/bnlp/release "$SV"/whole_corpus_word2vec_models 2>/dev/null; true ;;
  *) sed -n 2,5p "$0"; exit 1 ;;
esac
