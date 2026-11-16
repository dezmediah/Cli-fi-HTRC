#!/usr/bin/env python3
"""
Build the interactive "Word neighbors, 1945-1980" page from released results.

Runs OUTSIDE the capsule, on files that have already been released:
  --models    folder holding era_a/, era_b/, era_c/, each with sf_model_<era>_seed1..3
              (the word2vec models from SF_word2vec_eras_v2.ipynb)
  --freq      out_w2v_v2/tables/freq.csv        (from scripts/w2v_export_v2.py)
  --keyness   novel_keyness_top_by_word.csv     (from SF_novel_keyness_v2.ipynb)
  --out       where to write the finished page, e.g. docs/index.html

For each chosen word the page shows, per era: the 20 nearest neighbors (cosine similarity
averaged over the three seeds), similarity to four word groups (with the seed range),
frequency per million tokens, and the 10 volumes with the highest keyness score.

The page is one self-contained HTML file (plain JavaScript and SVG, no libraries), so it
can be hosted as-is on GitHub Pages. No passages of text are used anywhere.

Example:
  python scripts/build_word_neighbors.py \
      --models v_2_word2vec_models \
      --freq v_2_word2vec_models/out_w2v_v2/tables/freq.csv \
      --keyness novel_keyness_top_by_word.csv \
      --out docs/index.html
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

ERAS = ["era_a", "era_b", "era_c"]
SEEDS = [1, 2, 3]
DEFAULT_WORDS = ["pollution", "toxic", "climate", "nature", "resource"]
PICKER_LABEL = "Environment"

# Word groups the meaning-profile chart compares each word against. These lists are an
# analytic choice, not a result: changing them changes the lines on the chart (the page's
# "Where these numbers come from" section prints the same lists).
GROUPS = {
    "bodily": ["bacterial", "bacteria", "disease", "decay", "decomposition", "infection", "putrid"],
    "crowding": ["overpopulation", "overcrowding", "population_explosion", "overpopulated"],
    "industrial": ["chemical", "industrial", "smog", "sewage", "factories", "fumes"],
    "nuclear": ["radioactivity", "fallout", "radiation", "radioactive", "nuclear", "atomic"],
}

# Titles known to be nonfiction. Everything else is flagged "edited volume" only when the
# author field is empty or the title says "edited by" / starts like a "best of" anthology.
# These flags are a reading of the metadata and are not verified.
NONFICTION_TITLES = ("Exploring New Ethics for Survival", "The universe makers")

N_NEIGHBORS = 20
N_BOOKS = 10


def centroid(model, words, exclude):
    vecs = [model.wv.get_vector(w, norm=True) for w in words if w in model.wv.key_to_index and w != exclude]
    c = np.mean(vecs, axis=0)
    return c / np.linalg.norm(c)


def load_models(models_dir):
    return {
        era: [Word2Vec.load(str(Path(models_dir) / era / f"sf_model_{era}_seed{s}"), mmap="r") for s in SEEDS]
        for era in ERAS
    }


def book_flag(title, author):
    if any(t in title for t in NONFICTION_TITLES):
        return "nonfiction"
    t = title.lower()
    if author == "" or "edited by" in t or t.startswith("best sf") or t.startswith("the best science fiction"):
        return "edited volume"
    return ""


def compute_word(word, models, freq, keyness):
    d = {"nbrs": {}, "uses": {}, "groups": {g: {} for g in GROUPS}, "freq": {}, "novels": []}
    for era in ERAS:
        ms = models[era]
        d["uses"][era] = int(ms[0].wv.get_vecattr(word, "count"))

        top = [{w for w, _ in m.wv.most_similar(word, topn=N_NEIGHBORS)} for m in ms]
        candidates = set()
        for m in ms:
            candidates |= {w for w, _ in m.wv.most_similar(word, topn=120)}
        rows = []
        for w in candidates:
            if w == word or not all(w in m.wv.key_to_index for m in ms):
                continue
            sims = [float(m.wv.similarity(word, w)) for m in ms]
            rows.append([w.replace("_", " "), float(np.mean(sims)),
                         sum(w in t for t in top), int(ms[0].wv.get_vecattr(w, "count"))])
        # unrounded score first, then the word: ties must not depend on set/hash order,
        # or two builds from the same models can differ at the rank-20 boundary
        rows.sort(key=lambda r: (-r[1], r[0]))
        d["nbrs"][era] = [[w, round(s, 3), k, c] for w, s, k, c in rows[:N_NEIGHBORS]]

        for g, ws in GROUPS.items():
            vals = [float(m.wv.get_vector(word, norm=True) @ centroid(m, ws, word)) for m in ms]
            d["groups"][g][era] = [round(float(np.mean(vals)), 3), round(min(vals), 3), round(max(vals), 3)]

        hit = freq[(freq.word == word) & (freq.era == era)]
        if hit.empty:
            raise SystemExit(f"'{word}' is not in the frequency table for {era}")
        d["freq"][era] = [int(hit.iloc[0]["count"]), round(float(hit.iloc[0]["per_million"]), 2)]

    top_books = keyness[keyness.word == word].sort_values("g2", ascending=False).head(N_BOOKS)
    for _, r in top_books.iterrows():
        title = str(r["title"])
        author = "" if pd.isna(r["author"]) else str(r["author"])
        d["novels"].append([title, author, int(r["year"]), r["era"], int(r["count_in_novel"]),
                            round(float(r["rate_per_10k_in_novel"]), 2), round(float(r["g2"]), 1),
                            book_flag(title, author)])
    return d


def render(template_path, data):
    """Wrap the template (written for the Artifact viewer) as a standalone HTML document."""
    t = Path(template_path).read_text(encoding="utf-8")
    title = re.search(r"<title>(.*?)</title>", t, re.S).group(1)
    body = re.sub(r"<title>.*?</title>\s*", "", t, count=1, flags=re.S)
    body = re.sub(r'<meta name="claude:viewport-fit-sides"[^>]*>\s*', "", body)
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    assert "/*DATA*/" in body, "template has no /*DATA*/ placeholder"
    body = body.replace("/*DATA*/", payload)
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{title}</title>\n<style>html{{color-scheme:light}}body{{margin:0}}</style>\n"
        f"</head>\n<body>\n{body}\n</body>\n</html>\n"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True)
    ap.add_argument("--freq", required=True)
    ap.add_argument("--keyness", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--template", default=str(Path(__file__).with_name("word_neighbors_template.html")))
    ap.add_argument("--words", nargs="+", default=DEFAULT_WORDS)
    ap.add_argument("--json-out", help="also write the computed data as JSON (for checking)")
    args = ap.parse_args()

    freq = pd.read_csv(args.freq)
    keyness = pd.read_csv(args.keyness)
    models = load_models(args.models)

    data = {"words": args.words, "picker": [{"label": PICKER_LABEL, "words": args.words}], "w": {}}
    for w in args.words:
        print(f"computing {w} ...")
        data["w"][w] = compute_word(w, models, freq, keyness)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(args.template, data), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
