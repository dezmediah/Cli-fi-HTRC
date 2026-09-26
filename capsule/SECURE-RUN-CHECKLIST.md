# Secure-mode run checklist (written 2026-09-26, for the last capsule window before Sep 30)

Box: dc6, 50 vCPU, 58 GB RAM, 22 GB free. Everything below runs in SECURE mode via the HTRC
portal's Remote Desktop, inside a terminal there. Outputs go under `/media/secure_volume/` only;
anything written elsewhere is lost at the next flip. Start long jobs with `nohup ... &` so closing
the terminal window doesn't kill them, and log to the secure volume.

## 0. Before the flip (maintenance mode, Alex's terminal on the Mac)

```
ssh -i ~/.ssh/id_rsa_htrc -p 16040 dcuser@dc6.htrc.indiana.edu 'git -C ~/Desktop/Clifi-htrc pull --ff-only origin main && git -C ~/Desktop/Clifi-htrc log --oneline -1 && ls ~/Desktop/Clifi-htrc/capsule ~/Desktop/Clifi-htrc/scripts'
scp -i ~/.ssh/id_rsa_htrc -P 16040 ~/Code/SF-Nexus/sf-hathitrust-mining/data/clifi/clifi_htids_capsule.txt ~/Code/SF-Nexus/sf-hathitrust-mining/capsule/extras_analysis.py dcuser@dc6.htrc.indiana.edu:clifi/
```
Expected: clone at `21d4be9`, `scripts/booknlp_batch.py` present, `~/clifi/` has the id list (234 lines) and `extras_analysis.py`.

## 1. First look after the flip

```
ls /media/secure_volume/
ls /media/secure_volume/fa50b375-3216-4edd-a685-98488562b723 | wc -l      # expect 270
df -h /media/secure_volume
```
Write down what is already there (out_keyword_context_v2/v3/v4, keyness, w2v models, bertopic, bnlp).

## 2. Night 1: BERTopic + keyness together (both fit in 50 cores)

BERTopic, in Jupyter (`jupyter notebook` from the anaconda base, open `notebooks/htrc_bertopic_pipeline.ipynb`):
set `TORCH_THREADS = 32` in the config cell, then Kernel > Restart & Run All. ~7 h.
Writes `/media/secure_volume/out_bertopic/` (release) and `work_bertopic/` (never release).

Keyness, same Jupyter: open `notebooks/SF_novel_keyness_v2.ipynb`, Kernel > Restart & Run All. ~4 h.
Writes `/media/secure_volume/out_novel_keyness_v2/tables/`. The cell-7 permission error from 09-23
was maintenance mode; in secure mode the path exists.

## 3. Day 2: BookNLP, then PMI, then KWIC v4

```
cd ~/Desktop/Clifi-htrc/scripts
nohup ~/venv-booknlp/bin/python booknlp_batch.py --input /media/secure_volume/fa50b375-3216-4edd-a685-98488562b723 --out /media/secure_volume/bnlp --procs 6 --threads 8 > /media/secure_volume/bnlp.log 2>&1 &
tail -f /media/secure_volume/bnlp.log        # all 270 volumes, ~5 h (Alex 09-26: whole corpus, not the 234 list); resumable, re-run the same line after any crash
~/venv-booknlp/bin/python booknlp_batch.py --finalize --out /media/secure_volume/bnlp --metadata ~/metadata_august2026.csv
```
Release only `/media/secure_volume/bnlp/release/`.

```
python3 ~/clifi/extras_analysis.py pmi --input /media/secure_volume/fa50b375-3216-4edd-a685-98488562b723 --metadata ~/metadata_august2026.csv --out /media/secure_volume/out_extras --procs 40
```
Minutes. Uses the 101 + 24 lexicon from the v4 notebook. Release `/media/secure_volume/out_extras/pmi/`.

KWIC v4: Jupyter, `notebooks/SF_keyword_context_v4.ipynb`, Restart & Run All. Check the size of
`/media/secure_volume/out_keyword_context_v4/` before adding it; it must stay well under 1 GB.

Optional, only if everything above is in: whole-corpus w2v, `notebooks/SF_word2vec_whole_corpus_v2.ipynb` (era models are already out, Alex 09-26).

## 4. Release requests (one person, never two `add`s before a `done`)

Request 1 (as soon as night 1 finishes): keyness tables + `out_bertopic/` + their `MANIFEST`/`run_manifest.json` files + `~/metadata_august2026.csv` (the 270 htids + years) + a text file with the commit hash (`git -C ~/Desktop/Clifi-htrc rev-parse HEAD > /media/secure_volume/COMMIT.txt`).
```
releaseresults add /media/secure_volume/out_novel_keyness_v2/tables /media/secure_volume/out_bertopic ~/metadata_august2026.csv /media/secure_volume/COMMIT.txt
releaseresults done
```
Request 2: `/media/secure_volume/bnlp/release` + `/media/secure_volume/out_extras/pmi`.
Request 3 (last, alone): `/media/secure_volume/out_keyword_context_v4`.

Never add: `work_bertopic/`, anything `.npy`, per-volume BookNLP `vol/` folders, raw text.
If something drops, keep in this order: BERTopic tables, BookNLP release, w2v, PMI, KWIC v4, keyness.
