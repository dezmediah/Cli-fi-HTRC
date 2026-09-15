#!/usr/bin/env bash
# Stage everything the secure-mode runs need. Run in MAINTENANCE mode only (needs network).
# Idempotent: re-running skips what is already present. Items staged in the home directory
# survive the switch into secure mode. Do not apt upgrade or reboot; a power cycle is a mode switch.
#
# Usage:   bash stage_capsule.sh [--skip-bge]      --skip-bge omits the 2.3 GB long-context model

set -uo pipefail

SKIP_BGE=0
[[ "${1:-}" == "--skip-bge" ]] && SKIP_BGE=1

UV_BIN="$HOME/.local/bin/uv"
VENV="$HOME/venv-booknlp"
BNLP_MODELS="$HOME/booknlp_models"
CLIFI="$HOME/clifi"
REQ="$CLIFI/booknlp-requirements.txt"
SPACY_WHL="https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
BAMMAN="http://people.ischool.berkeley.edu/~dbamman/booknlp_models"

ok()   { printf '  \033[32mok\033[0m   %s\n' "$1"; }
skip() { printf '  --   %s (already present)\n' "$1"; }
die()  { printf '  \033[31mFAIL\033[0m %s\n' "$1"; exit 1; }
step() { printf '\n\033[1m%s\033[0m\n' "$1"; }

step "0. Preflight"
[ -e /media/secure_volume ] && die "secure_volume is mounted -- you are in SECURE mode. Staging needs maintenance mode."
curl -sI --max-time 20 https://pypi.org/simple/ >/dev/null 2>&1 || die "no network. Are you in secure mode?"
ok "maintenance mode, network reachable"
AVAIL=$(df -Pk "$HOME" | awk 'NR==2{print int($4/1048576)}')
echo "     $AVAIL GB free"
[ "$AVAIL" -lt 6 ] && die "need ~6 GB free (3.4 GB staged + build headroom)"

step "1. uv (static binary, no dependencies)"
if [ -x "$UV_BIN" ]; then skip "uv"; else
  mkdir -p "$HOME/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" sh >/dev/null 2>&1 \
    || die "uv install failed"
  ok "uv installed"
fi
"$UV_BIN" --version || die "uv not runnable"

step "2. Isolated BookNLP venv (its own Python -- touches nothing in ~/.local or anaconda)"
[ -f "$REQ" ] || die "missing $REQ -- copy capsule/staging/booknlp-requirements.txt to $CLIFI/ first"
if [ -x "$VENV/bin/python" ]; then skip "venv"; else
  "$UV_BIN" venv --python 3.10 "$VENV" || die "venv creation failed"
  ok "venv created"
fi
if [ "${REBUILD:-0}" = 1 ]; then
  rm -rf "$VENV"; "$UV_BIN" venv --python 3.10 "$VENV" || die "venv rebuild failed"
  ok "venv rebuilt from scratch"
fi
# --no-deps is required: the file is a complete freeze; resolving deps drags in TensorFlow and CUDA wheels.
"$UV_BIN" pip install --python "$VENV/bin/python" --no-deps -r "$REQ" || die "pin install failed"
ok "pinned packages installed (--no-deps: no TensorFlow, no CUDA)"
# The pinned torch is a CUDA build that fails to import without nvidia libs; swap in the CPU wheel.
"$UV_BIN" pip install --python "$VENV/bin/python" \
  --index-url https://download.pytorch.org/whl/cpu "torch==2.13.0+cpu" \
  || die "CPU torch install failed"
ok "torch swapped to the CPU build (no CUDA libs needed)"
"$VENV/bin/python" -c "import torch, transformers, spacy, numpy" 2>/dev/null \
  || die "core imports failed -- the freeze is missing something"
ok "core imports verified"
"$VENV/bin/python" -c "import en_core_web_sm" 2>/dev/null \
  && skip "en_core_web_sm" \
  || { "$UV_BIN" pip install --python "$VENV/bin/python" "$SPACY_WHL" || die "spacy model failed"; ok "en_core_web_sm installed"; }

step "3. BookNLP's own models (small config, ~155 MB, plain HTTP from Berkeley)"
mkdir -p "$BNLP_MODELS"
for m in entities_google_bert_uncased_L-4_H-256_A-4-v1.0.model \
         coref_google_bert_uncased_L-2_H-256_A-4-v1.0.model \
         speaker_google_bert_uncased_L-8_H-256_A-4-v1.0.1.model; do
  if [ -s "$BNLP_MODELS/$m" ]; then skip "$m"; else
    curl -fL --retry 3 -o "$BNLP_MODELS/$m.part" "$BAMMAN/$m" || die "download failed: $m"
    mv "$BNLP_MODELS/$m.part" "$BNLP_MODELS/$m"; ok "$m"
  fi
done

step "4. The three Google BERT models BookNLP loads from HuggingFace at runtime"
# BookNLP calls from_pretrained() on these at runtime; the cache must be warm before secure mode.
"$VENV/bin/python" - <<'PY' || die "HF prefetch failed"
from transformers import BertTokenizer, BertModel
for n in ["google/bert_uncased_L-4_H-256_A-4",
          "google/bert_uncased_L-2_H-256_A-4",
          "google/bert_uncased_L-8_H-256_A-4"]:
    BertTokenizer.from_pretrained(n, do_lower_case=False, do_basic_tokenize=False)
    BertModel.from_pretrained(n)
    print("  cached", n)
PY
ok "HF cache warm"

step "5. bge-m3 (arm B long-context encoder, ~2.3 GB)"
if [ "$SKIP_BGE" = 1 ]; then
  echo "  skipped (--skip-bge). Arm B cannot run without it, and there is no network in secure mode."
elif [ -d "$HOME/models/bge-m3" ] && [ -n "$(ls -A "$HOME/models/bge-m3" 2>/dev/null)" ]; then
  skip "bge-m3"
else
  mkdir -p "$HOME/models"
  "$VENV/bin/python" - <<'PY' || die "bge-m3 download failed"
from huggingface_hub import snapshot_download
import os
p = snapshot_download("BAAI/bge-m3", local_dir=os.path.expanduser("~/models/bge-m3"),
                      allow_patterns=["*.json","*.txt","*.model","*.safetensors","1_Pooling/*"])
print("  ->", p)
PY
  ok "bge-m3 staged"
fi

step "6. OFFLINE VERIFICATION -- the whole point of this script"
# Runs BookNLP with the network forced dead. If this passes, secure mode will work.
printf 'The reactor flooded at dawn. Dr. Reyes watched the water rise.\n"We have to go," she said.\n' > /tmp/_stagecheck.txt
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  http_proxy=http://127.0.0.1:9 https_proxy=http://127.0.0.1:9 all_proxy=http://127.0.0.1:9 \
  "$VENV/bin/python" - <<'PY'
import sys, warnings, tempfile, os
warnings.filterwarnings("ignore")
from booknlp.booknlp import BookNLP
b = BookNLP("en", {"pipeline":"entity,quote,supersense,event,coref", "model":"small"})
out = tempfile.mkdtemp()
b.process("/tmp/_stagecheck.txt", out, "check")
need = ["check.entities","check.quotes","check.supersense","check.tokens","check.book"]
missing = [f for f in need if not os.path.exists(os.path.join(out, f))]
sys.exit(1 if missing else 0)
PY
[ $? -eq 0 ] && ok "BookNLP runs with NO network -- staging is complete" || die "BookNLP still needs the network. DO NOT switch to secure mode."
rm -f /tmp/_stagecheck.txt

step "Summary"
du -sh "$VENV" "$BNLP_MODELS" "$HOME/.cache/huggingface" "$HOME/models" 2>/dev/null
cat <<'TXT'

Staged. Next:
  1. Copy the notebooks/scripts into ~/clifi/ if not already there.
  2. Switch to SECURE mode from the HTRC Analytics page.
  3. First command in secure mode:  ls /media/secure_volume
     If that errors you are not in secure mode and nothing you do will persist.
  4. Run word2vec, then w2v_export.py IMMEDIATELY -- do not leave models unexported.

  Reminder: releaseresults `add` is NOT cumulative. One `add` with every path, then `done`.
TXT
