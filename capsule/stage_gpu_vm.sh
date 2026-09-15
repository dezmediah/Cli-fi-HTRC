#!/usr/bin/env bash
# Stage BookNLP + BERTopic on the HTRC GPU VM (A100 40 GB, Ubuntu 24.04; address from HTRC).
# Reached by ssh hop from the capsule in maintenance mode. Run while the VM is still "open to the
# world" (network on); HTRC wires it into secure mode afterwards, and there is no network then.
# Idempotent: re-running skips what is already present.
#
# Sibling of stage_capsule.sh. Differences: two venvs (BookNLP's freeze and the BERTopic pins
# disagree on transformers/numpy), the CUDA torch is KEPT, both BookNLP model configs (small +
# big) are staged because the big one is what a GPU is for, and bge-m3 is on by default.
#
# Usage:   bash stage_gpu_vm.sh [--skip-bge]

set -uo pipefail

SKIP_BGE=0
[[ "${1:-}" == "--skip-bge" ]] && SKIP_BGE=1

UV_BIN="$HOME/.local/bin/uv"
PYVER="3.10"
VENV_B="$HOME/venv-booknlp"
VENV_T="$HOME/venv-bertopic"
BNLP_MODELS="$HOME/booknlp_models"
CLIFI="$HOME/clifi"
REQ="$CLIFI/booknlp-requirements.txt"
SPACY_WHL="https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
BAMMAN="http://people.ischool.berkeley.edu/~dbamman/booknlp_models"

ok()   { printf '  \033[32mok\033[0m   %s\n' "$1"; }
skip() { printf '  --   %s (already present)\n' "$1"; }
die()  { printf '  \033[31mFAIL\033[0m %s\n' "$1"; exit 1; }
step() { printf '\n\033[1m%s\033[0m\n' "$1"; }
OFFLINE="env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 http_proxy=http://127.0.0.1:9 https_proxy=http://127.0.0.1:9 all_proxy=http://127.0.0.1:9"

step "0. Preflight"
nvidia-smi -L >/dev/null 2>&1 || die "no nvidia-smi -- this is not the GPU VM"
nvidia-smi -L | sed 's/^/     /'
curl -sI --max-time 20 https://pypi.org/simple/ >/dev/null 2>&1 || die "no network"
ok "network reachable"
AVAIL=$(df -Pk "$HOME" | awk 'NR==2{print int($4/1048576)}')
echo "     $AVAIL GB free"
[ "$AVAIL" -lt 15 ] && die "need ~15 GB free (two CUDA torches + models)"
[ -f "$REQ" ] || die "missing $REQ -- untar the staging bundle into $CLIFI first"

step "1. uv"
if [ -x "$UV_BIN" ]; then skip "uv"; else
  mkdir -p "$HOME/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" sh >/dev/null 2>&1 \
    || die "uv install failed"
  ok "uv installed"
fi
"$UV_BIN" --version || die "uv not runnable"
# System python is 3.12; BookNLP's freeze (numpy 1.23.5) and the BERTopic pins were measured on 3.10.
"$UV_BIN" python install "$PYVER" >/dev/null 2>&1 || die "uv could not install python $PYVER"
ok "python $PYVER available to uv"

step "2. BookNLP venv ($VENV_B) -- the capsule freeze, CUDA torch kept"
if [ -x "$VENV_B/bin/python" ]; then skip "venv"; else
  "$UV_BIN" venv --python "$PYVER" "$VENV_B" || die "venv creation failed"; ok "venv created"
fi
# --no-deps: the freeze is complete; resolving deps drags in TensorFlow.
"$UV_BIN" pip install --python "$VENV_B/bin/python" --no-deps -r "$REQ" || die "pin install failed"
ok "pinned packages installed"
# The pinned torch is a CUDA build and needs the nvidia libs the freeze left out. Installing it
# WITH deps pulls exactly those (and nothing else the freeze didn't already pin).
"$UV_BIN" pip install --python "$VENV_B/bin/python" "torch==2.13.0" || die "torch (CUDA) install failed"
"$VENV_B/bin/python" -c "import torch; assert torch.cuda.is_available(), 'cuda not available'; print('     torch', torch.__version__, torch.cuda.get_device_name(0))" \
  || die "torch cannot see the GPU"
ok "torch sees the A100"
"$VENV_B/bin/python" -c "import transformers, spacy, numpy, booknlp" 2>/dev/null || die "core imports failed"
ok "core imports verified"
"$VENV_B/bin/python" -c "import en_core_web_sm" 2>/dev/null \
  && skip "en_core_web_sm" \
  || { "$UV_BIN" pip install --python "$VENV_B/bin/python" "$SPACY_WHL" || die "spacy model failed"; ok "en_core_web_sm installed"; }

step "3. BookNLP's own models: small (the capsule config) AND big (the GPU config)"
mkdir -p "$BNLP_MODELS"
for m in entities_google_bert_uncased_L-4_H-256_A-4-v1.0.model \
         coref_google_bert_uncased_L-2_H-256_A-4-v1.0.model \
         speaker_google_bert_uncased_L-8_H-256_A-4-v1.0.1.model \
         entities_google_bert_uncased_L-6_H-768_A-12-v1.0.model \
         coref_google_bert_uncased_L-12_H-768_A-12-v1.0.model \
         speaker_google_bert_uncased_L-12_H-768_A-12-v1.0.1.model; do
  if [ -s "$BNLP_MODELS/$m" ]; then skip "$m"; else
    curl -fL --retry 3 -o "$BNLP_MODELS/$m.part" "$BAMMAN/$m" || die "download failed: $m"
    mv "$BNLP_MODELS/$m.part" "$BNLP_MODELS/$m"; ok "$m"
  fi
done

step "4. The Google BERT bases BookNLP loads from HuggingFace at runtime (both configs)"
"$VENV_B/bin/python" - <<'PY' || die "HF prefetch failed"
from transformers import BertTokenizer, BertModel
for n in ["google/bert_uncased_L-4_H-256_A-4", "google/bert_uncased_L-2_H-256_A-4",
          "google/bert_uncased_L-8_H-256_A-4", "google/bert_uncased_L-6_H-768_A-12",
          "google/bert_uncased_L-12_H-768_A-12"]:
    BertTokenizer.from_pretrained(n, do_lower_case=False, do_basic_tokenize=False)
    BertModel.from_pretrained(n)
    print("  cached", n)
PY
ok "HF cache warm"

step "5. BERTopic venv ($VENV_T) -- capsule pins, s-t bumped (bertopic 0.17.4 imports StaticEmbedding, 3.x+), CUDA torch"
if [ -x "$VENV_T/bin/python" ]; then skip "venv"; else
  "$UV_BIN" venv --python "$PYVER" "$VENV_T" || die "venv creation failed"; ok "venv created"
fi
"$UV_BIN" pip install --python "$VENV_T/bin/python" \
  "bertopic==0.17.4" "sentence-transformers>=3,<6" "umap-learn==0.5.12" "scikit-learn==1.7.2" \
  "numpy==1.24.3" "hdbscan>=0.8.33" "gensim>=4.3" "transformers<5" "torch" \
  "pandas>=2.0" "tqdm" "nltk>=3.8" "jupyter" "nbconvert" "ipykernel" "matplotlib" \
  || die "BERTopic stack install failed"
"$VENV_T/bin/python" -c "import torch; assert torch.cuda.is_available(); print('     torch', torch.__version__, torch.cuda.get_device_name(0))" \
  || die "torch cannot see the GPU (bertopic venv)"
"$VENV_T/bin/python" -c "import bertopic, sentence_transformers, umap, hdbscan, gensim, sklearn; print('     bertopic', bertopic.__version__, 's-t', sentence_transformers.__version__)" \
  || die "BERTopic imports failed"
"$VENV_T/bin/python" -m ipykernel install --user --name bertopic-gpu --display-name "BERTopic (GPU)" >/dev/null 2>&1 && ok "jupyter kernel 'bertopic-gpu' registered"
"$VENV_T/bin/python" -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('stopwords', quiet=True)" \
  && ok "nltk punkt + stopwords"

step "6. Embedding models"
mkdir -p "$HOME/models"
if [ -f "$HOME/models/minilm/config.json" ]; then skip "minilm"; else
  "$VENV_T/bin/python" -c "from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/all-MiniLM-L6-v2').save('$HOME/models/minilm')" \
    || die "minilm download failed"
  ok "minilm staged (arm A)"
fi
if [ "$SKIP_BGE" = 1 ]; then
  echo "  skipped bge-m3 (--skip-bge)"
elif [ -d "$HOME/models/bge-m3" ] && [ -n "$(ls -A "$HOME/models/bge-m3" 2>/dev/null)" ]; then
  skip "bge-m3"
else
  "$VENV_T/bin/python" - <<'PY' || die "bge-m3 download failed"
from huggingface_hub import snapshot_download
import os
p = snapshot_download("BAAI/bge-m3", local_dir=os.path.expanduser("~/models/bge-m3"),
                      allow_patterns=["*.json","*.txt","*.model","pytorch_model.bin","1_Pooling/*"])  # bge-m3 ships .bin, no safetensors; onnx/ excluded (2.1 GB dup)
print("  ->", p)
PY
  ok "bge-m3 staged (arm B, viable on the A100)"
fi

step "7. OFFLINE VERIFICATION on the GPU -- the whole point"
printf 'The reactor flooded at dawn. Dr. Reyes watched the water rise.\n"We have to go," she said.\n' > /tmp/_stagecheck.txt
$OFFLINE "$VENV_B/bin/python" - <<'PY'
import sys, warnings, tempfile, os, torch
warnings.filterwarnings("ignore")
assert torch.cuda.is_available()
from booknlp.booknlp import BookNLP
for cfg in ("small", "big"):
    b = BookNLP("en", {"pipeline":"entity,quote,supersense,event,coref", "model":cfg})
    out = tempfile.mkdtemp()
    b.process("/tmp/_stagecheck.txt", out, "check")
    need = ["check.entities","check.quotes","check.supersense","check.tokens","check.book"]
    missing = [f for f in need if not os.path.exists(os.path.join(out, f))]
    if missing: sys.exit(1)
    print("  booknlp", cfg, "ok, no network")
PY
[ $? -eq 0 ] && ok "BookNLP small + big run with NO network" || die "BookNLP still needs the network. Tell HTRC NOTHING yet."
$OFFLINE "$VENV_T/bin/python" - <<'PY'
import os, torch
from sentence_transformers import SentenceTransformer
assert torch.cuda.is_available()
m = SentenceTransformer(os.path.expanduser("~/models/minilm"), device="cuda")
e = m.encode(["the reactor flooded at dawn"]*64, batch_size=64)
assert e.shape == (64, 384), e.shape
print("  minilm on cuda ok", e.shape)
from bertopic import BERTopic
docs = [f"the {w} rose over the drowned city" for w in ("water","river","sea","flood","tide","rain")]*20
t = BERTopic(embedding_model=m, min_topic_size=5, verbose=False).fit(docs)
print("  bertopic fit ok, topics:", len(t.get_topic_info()))
PY
[ $? -eq 0 ] && ok "MiniLM + BERTopic run on the GPU with NO network" || die "BERTopic arm failed offline."
rm -f /tmp/_stagecheck.txt

step "Summary"
du -sh "$VENV_B" "$VENV_T" "$BNLP_MODELS" "$HOME/.cache/huggingface" "$HOME/models" 2>/dev/null
df -h "$HOME" | tail -1
cat <<'TXT'

Staged. Next:
  1. Tell HTRC (Ryan) the environment is ready so they connect this VM to the capsule's secure mode.
  2. Ask him where the text will appear here, whether releaseresults runs here, and whether this
     disk survives the mode change. Nothing restricted goes on this box until he says it is wired.
  3. BookNLP:  ~/venv-booknlp/bin/python ~/clifi/booknlp_batch.py --ids ~/clifi/clifi_htids_capsule.txt ...
     BERTopic: ~/venv-bertopic/bin/jupyter nbconvert --execute ~/clifi/htrc_bertopic_pipeline.ipynb ...
TXT
