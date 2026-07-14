# Cli-fi HTRC — Plan

## Current state (Jul 2025)

- Repo initialized at github.com/dezmediah/Cli-fi-HTRC
- 5 notebooks + 4 volume-id txt files tracked on main
- Data capsule: dc6.htrc.indiana.edu:16040, user dcuser
- Disk: 41G/62G used (19G free) after initial cleanup

## Pending: uv migration (replace conda)

Four overlapping conda envs (7.2G total) can be consolidated into one uv venv:

| Env | Size | Packages |
|-----|------|----------|
| BERT | 2.9G | bertopic, sentence-transformers, transformers, pytorch CPU, hdbscan, umap, gensim, nltk, pyLDAvis |
| BERTTopic | 2.0G | near-duplicate of BERT |
| word2vec | 1.8G | gensim, nltk, matplotlib |
| gensim_LDA | 541M | gensim, nltk, pyLDAvis, scikit-learn |

**Plan:**
1. Install uv on the data capsule
2. Create .venv in ~/Desktop/Clifi-htrc/
3. Install: bertopic, sentence-transformers, transformers, pytorch (CPU-only), hdbscan, umap-learn, gensim, pyLDAvis, nltk, scikit-learn, matplotlib, pandas, numpy, tqdm, jupyter/ipykernel
4. Pin to versions matching current conda envs: bertopic 0.15, sentence-transformers 2.2.2, transformers 4.29-4.30, gensim 4.3
5. Register as Jupyter kernel
6. Test imports against all notebooks
7. Remove conda envs: BERT, BERTTopic, word2vec, gensim_LDA

## Pending: further disk cleanup

- ~/.local/lib/python3.10/site-packages/nvidia/ — 2.7G CUDA libs (no GPU on capsule; reinstall torch CPU-only during uv migration)
- ~/nltk_data.zip — 48M (redundant with unzipped ~/nltk_data/)
- ~/anaconda3/pkgs/ — 9.5G remaining (mostly extracted packages tied to envs; safe to remove after conda envs are deleted)
- Consolidate or remove ~/.local/lib/python3.10/ pip installs (4.8G) after migration

## Pending: repo hygiene

- Decide whether volume ids doc*.txt files should stay tracked
- Set git user.name and user.email (currently commits as dcuser)
- Consider gitignoring *.txt if volume lists are data, not code

## Notes

- HTRC/data/sample_volumes/ (2.8G) is gitignored and stays local
- models/minilm/ (87M) is the sentence-transformer model for BERTopic
- Metadata_BERT_processing.csv is gitignored (active working data)
