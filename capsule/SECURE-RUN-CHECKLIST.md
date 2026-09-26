# Secure-mode run checklist (written 2026-09-26, for the last capsule window before Sep 30)

Box: dc6, 50 vCPU, 58 GB RAM, 22 GB free. Everything below runs in SECURE mode via the HTRC
portal's Remote Desktop, inside a terminal there. Outputs go under `/media/secure_volume/` only;
anything written elsewhere is lost at the next flip. Start long jobs with `nohup ... &` so closing
the terminal window doesn't kill them, and log to the secure volume.

## 0. Before the flip (maintenance mode) — DONE 2026-09-26 18:50 EDT

Clone `~/Desktop/Clifi-htrc` pulled to `1333adb` (PR #3 + PR #4): `scripts/booknlp_batch.py`,
`capsule/extras_analysis.py` (reads 125 lexicon words from the v4 notebook on the box) and the
fixed BERTopic notebook are all present and smoke-tested. Nothing else needs the network.

## 1. First look after the flip

```
ls /media/secure_volume/
ls /media/secure_volume/fa50b375-3216-4edd-a685-98488562b723 | wc -l      # expect 270
df -h /media/secure_volume
```
Write down what is already there (out_keyword_context_v2/v3/v4, keyness, w2v models, bertopic, bnlp).

## Dry-run result (2026-09-26 20:30, local venv mirroring the box: py3.10, pandas 1.5.3, gensim 4.3.0)

Keyness, KWIC v4 and whole-corpus w2v all ran end to end on a synthetic 8-volume corpus and wrote
their tables, parts, manifests and models. The only failure was the final `releaseresults` cell,
which exists only on the capsule. So the code is sound; what needs care is the release cells:

- **Her three notebooks SUBMIT THEMSELVES.** Each ends with `releaseresults add …` and, in the
  next cell, `releaseresults done`. "Restart & Run All" therefore files a release request the
  moment the notebook finishes. That is fine, as long as no manual `add` is pending at that moment.
- **BERTopic, BookNLP and PMI do NOT submit.** They only print the `add` line; those are manual.
- `add` wipes whatever was queued before. Never start a notebook's release cells while a manual
  `add` is waiting for its `done`.

## 2. Night 1: BERTopic + keyness together (both fit in 50 cores)

BERTopic, in Jupyter (`jupyter notebook` from the anaconda base, open `notebooks/htrc_bertopic_pipeline.ipynb`):
set `TORCH_THREADS = 32` in the config cell, then Kernel > Restart & Run All. ~7 h.
Writes `/media/secure_volume/out_bertopic/` (release) and `work_bertopic/` (never release). No auto-submit.

Keyness, same Jupyter: open `notebooks/SF_novel_keyness_v2.ipynb`, Kernel > Restart & Run All. ~4 h.
Writes `/media/secure_volume/out_novel_keyness_v2/tables/` and then **submits itself (request A)**.
The cell-7 permission error from 09-23 was maintenance mode; in secure mode the path exists.

## 3. Day 2: BookNLP, then PMI, then the manual request, then KWIC v4

```
cd ~/Desktop/Clifi-htrc/scripts
nohup ~/venv-booknlp/bin/python booknlp_batch.py --input /media/secure_volume/fa50b375-3216-4edd-a685-98488562b723 --out /media/secure_volume/bnlp --procs 6 --threads 8 > /media/secure_volume/bnlp.log 2>&1 &
tail -f /media/secure_volume/bnlp.log        # all 270 volumes, ~5 h; resumable, re-run the same line after any crash
~/venv-booknlp/bin/python booknlp_batch.py --finalize --out /media/secure_volume/bnlp --metadata ~/metadata_august2026.csv
```
Release only `/media/secure_volume/bnlp/release/`.

```
python3 ~/Desktop/Clifi-htrc/capsule/extras_analysis.py pmi --input /media/secure_volume/fa50b375-3216-4edd-a685-98488562b723 --metadata ~/metadata_august2026.csv --out /media/secure_volume/out_extras --procs 40
```
Minutes. Uses the 101 + 24 lexicon from the v4 notebook. Release `/media/secure_volume/out_extras/pmi/`.

**Request B (manual, once BERTopic, BookNLP and PMI are all done and request A's `done` has run):**
```
git -C ~/Desktop/Clifi-htrc rev-parse HEAD > /media/secure_volume/COMMIT.txt
du -sh /media/secure_volume/out_bertopic /media/secure_volume/bnlp/release /media/secure_volume/out_extras/pmi   # total must stay under 1 GB
releaseresults add /media/secure_volume/out_bertopic /media/secure_volume/bnlp/release /media/secure_volume/out_extras/pmi ~/metadata_august2026.csv /media/secure_volume/COMMIT.txt
releaseresults done
```

**KWIC v4 (request C, last, only after request B's `done`):** Jupyter, `notebooks/SF_keyword_context_v4.ipynb`,
Restart & Run All. It splits CSVs at 900 MB per part and submits itself; if the manifest shows more
than one part per lexicon the total exceeds 1 GB, so stop before the release cells and add one part.

**Whole-corpus w2v (optional, request D):** `notebooks/SF_word2vec_whole_corpus_v2.ipynb` trains
`SEEDS = [1, 2, 3]` and adds every model file; three whole-corpus models will exceed 1 GB. Set
`SEEDS = [1]` before running, and check `du -sh` of the model dir before the release cells.
Era models are already out (Alex 09-26), so skip this if time is short.

## 4. Order of requests

A keyness (auto, end of night 1) → B BERTopic + BookNLP + PMI (manual, day 2) → C KWIC v4 (auto)
→ D whole-corpus w2v (auto, optional). One `done` before the next `add`, always.

Never add: `work_bertopic/`, anything `.npy` except a w2v sidecar, per-volume BookNLP `vol/` folders, raw text.
If something drops, keep in this order: BERTopic tables, BookNLP release, w2v, PMI, KWIC v4, keyness.
