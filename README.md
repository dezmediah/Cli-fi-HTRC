# Cli-fi HTRC

Environmental language in American science fiction, 1945 to 1980, read from the full text of
in-copyright novels inside the HathiTrust Research Center Data Capsule. The question the
notebooks are built around is how environmental contamination is framed across the period, and
whether that framing shifts around the emergence of environmental discourse in the 1960s and
1970s. The capsule is non-consumptive: the text never leaves, only aggregate results do, after
HTRC review.

## Corpus

`notebooks/metadata_august2026.csv`, one row per work: `htid, title, author, year, source`.

| | |
|---|---|
| rows | 2,892 (2,881 unique volumes; 11 multi-work volumes are listed once per work) |
| `source=laure` | 2,779 novels from the Temple SF holdings crosswalked to ISFDB, one HathiTrust copy per work, earliest edition |
| `source=temple` | 113 volumes Temple digitized and deposited itself (`ppt.ssfcbz…` ids) |
| years | 1945 to 1980, original printing |

Inside the capsule the text sits at `/media/secure_volume/fa50b375-3216-4edd-a685-98488562b723/<htid>/<page>.txt` (not `/data/sf_corpus`, which loads nothing), one folder per volume and
one file per page. 2,868 of the volumes had text on the last run.

Eras are cut at two events rather than into equal buckets:

| era | years | anchor |
|---|---|---|
| `era_a` | 1945 to 1961 | before *Silent Spring* |
| `era_b` | 1962 to 1971 | *Silent Spring* to the Clean Water Act |
| `era_c` | 1972 to 1980 | Clean Water Act onward |

## Files

| file | what it does |
|---|---|
| `notebooks/SF_word2vec_eras_v2.ipynb` | the current word2vec notebook: trains per era, three seeds each, and compares an expanded, WordNet-checked environmental lexicon across eras: neighbours, word-pair contrasts, seed stability, Procrustes alignment. Models save to `MODEL_DIR` (`v_2_word2vec_models`) on the secure volume. |
| `scripts/w2v_export_v2.py` | reads the v2 models and writes release-safe tables (pair contrasts, neighbours, stability, frequency per million and shift, semantic shift, a small vector matrix per era), six directories all under 1 MB. Lexicon matches the v2 notebook. |
| `notebooks/SF_word2vec_eras.ipynb`, `scripts/w2v_export.py` | the first lexicon and its export script, kept for the August run's models in `w2v_models` |
| `notebooks/htrc_bertopic_pipeline.ipynb` | BERTopic over 165-word chunks with an era axis. A preflight cell checks the capsule before anything expensive; embeddings and assignments checkpoint to the secure volume so a killed session resumes. |
| `notebooks/old_files/` | 2023 exploratory notebooks, kept for reference |

Both notebooks share one identical OCR-cleaning cell, run when volumes are loaded. It drops running
heads and page-number lines, rejoins words hyphenated across line breaks, and drops pages whose
share of real words is below a threshold, where "real" means a word that appears in at least ten
volumes of the corpus. No extra package. The settings are in each notebook's configuration cell:

    RUNNING_HEAD_MIN_PAGES = 3
    RUNNING_HEAD_FRAC      = 0.05
    RUNNING_HEAD_MAX_CHARS = 60
    DICT_MIN_VOLS          = 10
    PAGE_MIN_DICT_RATE     = 0.55   # None keeps every page

Each era prints what was removed, so the effect is visible before training starts.

Everything here has run end to end on a synthetic corpus shaped like the capsule's, including one
with planted running heads, hyphenation and garbage pages, all of which were caught. It has not yet
run on the real text.

## Three rules for the capsule

1. Save everything under `/media/secure_volume/`. A mode switch or power cycle restores a snapshot
   and destroys anything outside that volume. That is how the first nine word2vec models were
   lost. Both notebooks refuse or warn on any other path.
2. `releaseresults add` is not cumulative. Each call wipes the spool, so only the last `add` before
   `done` is released. One `add` naming every path, then one `done`.
3. No `apt upgrade`, no reboot. A power cycle is a mode switch.

## Before the secure session (maintenance mode, network on)

Already on the capsule: `~/models/minilm`, NLTK punkt and stopwords, gensim and the BERTopic stack,
`~/metadata_august2026.csv`. This repo is cloned at `~/Desktop/Clifi-htrc`; `git pull` there while
the network is on. Secure mode has no network, so anything missing costs a full mode switch.

## The secure session

First two commands, because the secure volume's size is unknown:

    ls /media/secure_volume
    df -h /media/secure_volume

1. Word2vec, then export immediately. Run `SF_word2vec_eras_v2.ipynb`, then:

        python scripts/w2v_export_v2.py --models /media/secure_volume/v_2_word2vec_models --out /media/secure_volume/out_w2v

2. BERTopic, about 7 hours on the capsule's CPUs. Run it detached so it survives the browser
   session:

        tmux new -s bertopic
        source activate BERT
        jupyter nbconvert --to notebook --execute htrc_bertopic_pipeline.ipynb \
            --output run_$(date +%F).ipynb --ExecutePreprocessor.timeout=-1

   `Ctrl-b d` detaches, `tmux attach -t bertopic` returns. Checkpoints are on the secure volume, so
   a rerun resumes. `ASSIGN_MODE` in the configuration cell decides whether the run is hours or
   days. `N_TOPICS_REDUCE = 60` keeps the per-volume table small.

3. Release: one `releaseresults add` with every file under `out_w2v` and the BERTopic `OUT_DIR`,
   then `releaseresults done`. Nothing in a `work` directory leaves.

The package is a few MB. The hard refusal is 1 GB; human review starts asking questions above
1 MB, so name the files with htrc-help in advance.

## Next

A BookNLP runner over the same corpus (characters, quotes, events, entities, reduced to tables
per volume) is built and tested and follows once these two arms have run.
