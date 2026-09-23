"""
Novel-level keyness: for each seed word in ENV_WORD_GROUPS (SF_word2vec_eras_v2.ipynb),
find which novels use that word unusually often relative to the rest of the corpus.

This answers "which novels predominate for particular seed words" -- a document-vs-corpus
question, which is a different (and more directly interpretable) tool than word2vec's
neighbor lists. Uses the same Dunning log-likelihood (G2) test already used for
tables/freq_shift_g2.csv, just applied novel-vs-corpus instead of era-vs-era.

Must run INSIDE THE CAPSULE, in secure mode -- raw per-novel text only exists at
TEXT_DIR on the secure volume and never leaves it.

Reuses the OCR-cleaning pipeline verbatim from SF_word2vec_eras_v2.ipynb (discover_volumes,
_hyphen_break, build_dictionary, clean_volume) so results are directly comparable to the
word2vec analysis. Unlike the notebook's load_docs(), this keeps each document's htid so
results can be attributed back to a specific novel via METADATA_CSV.

Usage (inside capsule):
    python novel_keyness_v2.py \
        --text-dir /media/secure_volume/fa50b375-3216-4edd-a685-98488562b723 \
        --metadata /home/dcuser/metadata_august2026.csv \
        --out /media/secure_volume/out_novel_keyness_v2
"""

import argparse
import json
import math
import os
import re
from collections import Counter
from multiprocessing import get_context
from pathlib import Path

import pandas as pd

# ---- OCR cleaning: copied verbatim from SF_word2vec_eras_v2.ipynb (cell 46a9ca82) ----
# so novel-level word counts here are directly comparable to the word2vec pipeline's.

_norm_ws = re.compile(r"\s+")
_only_number = re.compile(r"^\s*[\divxlcIVXLC]+\s*$")
_hyphen_break = re.compile(r"(\w)-\s*\n\s*(\w)")
_word = re.compile(r"[A-Za-z']+")


def discover_volumes(base_dir, ids=None):
    vols = {}
    for d in sorted(p for p in Path(base_dir).iterdir() if p.is_dir() and not p.name.startswith(".")):
        if ids is not None and d.name not in ids:
            continue
        pages = sorted(d.glob("*.txt"))
        if pages:
            vols[d.name] = pages
    return vols


def _norm_line(line):
    return _norm_ws.sub(" ", re.sub(r"\d+", "", line)).strip().lower()


def _volume_vocab(page_paths):
    words = set()
    for p in page_paths:
        text, _ = _hyphen_break.subn(r"\1\2", p.read_text(encoding="utf-8", errors="replace"))
        words.update(w.lower() for w in _word.findall(text) if len(w) > 1)
    return words


def build_dictionary(volumes, min_vols, procs=None):
    df = Counter()
    with get_context("fork").Pool(procs or min(16, os.cpu_count() or 1)) as pool:
        for vocab in pool.imap_unordered(_volume_vocab, list(volumes.values()), chunksize=8):
            df.update(vocab)
    return {w for w, c in df.items() if c >= min_vols}


def page_dict_rate(text, dictionary):
    words = [w.lower() for w in _word.findall(text) if len(w) > 1]
    return sum(1 for w in words if w in dictionary) / len(words) if words else 0.0


def clean_volume(page_paths, dictionary, running_head_min_pages, running_head_frac,
                  running_head_max_chars, page_min_dict_rate):
    pages = [p.read_text(encoding="utf-8", errors="replace") for p in page_paths]
    line_pages = Counter()
    per_page_lines = []
    for text in pages:
        lines = text.split("\n")
        per_page_lines.append(lines)
        line_pages.update({_norm_line(l) for l in lines if 0 < len(l.strip()) <= running_head_max_chars})
    thresh = max(running_head_min_pages, int(running_head_frac * len(pages)))
    heads = {l for l, c in line_pages.items() if c >= thresh and l}
    kept = []
    for lines in per_page_lines:
        out = []
        for l in lines:
            if (len(l.strip()) <= running_head_max_chars and _norm_line(l) in heads) or _only_number.match(l):
                continue
            out.append(l)
        page_text, _ = _hyphen_break.subn(r"\1\2", "\n".join(out))
        if dictionary is not None and page_min_dict_rate and page_dict_rate(page_text, dictionary) < page_min_dict_rate:
            continue
        kept.append(page_text)
    text, _ = _hyphen_break.subn(r"\1\2", "\n".join(kept))
    return text


def load_docs_keep_id(base_dir, dictionary, ids=None, **clean_kwargs):
    """Like the notebook's load_docs(), but keeps {htid: cleaned_text} instead of discarding htid."""
    docs = {}
    for htid, pages in discover_volumes(base_dir, ids).items():
        docs[htid] = clean_volume(pages, dictionary, **clean_kwargs)
    return docs


# ---- era assignment: copied from SF_word2vec_eras_v2.ipynb (cell 7030de76) ----

def assign_era(year, cutoffs=(1962, 1972), labels=("era_a", "era_b", "era_c")):
    for cutoff, label in zip(cutoffs, labels):
        if year < cutoff:
            return label
    return labels[-1]


# ---- keyword list: copied from SF_word2vec_eras_v2.ipynb (cell 520bb515) ----
# Keep in sync with the notebook if the lexicon changes there.

ENV_WORD_GROUPS = {
    "landscape_baseline": ["river", "creek", "stream", "water", "forest", "nature", "wilderness", "jungle",
                            "ocean", "landscape", "levee", "dam", "reservoir", "estuary", "wetland",
                            "marsh", "watershed"],
    "ecology_concept": ["ecology", "ecosystem", "environment", "biosphere", "habitat", "balance", "cycle"],
    "contamination": ["contamination", "waste", "smog", "fumes", "chemical", "pesticide", "insecticide",
                       "pollutant", "exhaust", "toxic", "polluted", "pollution"],
    "waste_infrastructure": ["sewer", "sewage", "drainage", "effluent", "runoff", "wastewater",
                              "cesspool", "sludge", "septic", "cistern", "culvert", "plumbing"],
    "population_scarcity": ["overpopulation", "population", "famine", "scarcity", "starvation", "resource", "drought"],
    "energy": ["oil", "fuel", "energy", "coal", "power"],
    "nuclear_atomic": ["radiation", "radioactive", "fallout", "nuclear", "atomic", "bomb", "meltdown"],
    "cosmic_natural_causation": ["solar", "cosmic", "celestial", "geological", "planetary"],
    "human_agency": ["mankind", "humanity", "civilization", "industrial"],
    "disaster_collapse": ["wasteland", "extinction", "collapse", "barren", "dying", "decay", "catastrophe",
                           "apocalypse", "plague"],
    "climate_weather": ["climate", "weather", "warming", "greenhouse", "atmosphere", "temperature",
                         "flood", "flooding", "storm", "hurricane", "ice", "glacier", "carbon", "ozone"],
    "space_earth_framing": ["earth", "homeworld", "colony", "frontier", "terraform", "alien"],
}
ENV_WORDS = sorted({w for group in ENV_WORD_GROUPS.values() for w in group})


# ---- G2 log-likelihood keyness (same test as tables/freq_shift_g2.csv, novel-vs-corpus) ----

def log_likelihood_g2(a, b, c, d):
    """a = count of word in doc, b = count in rest of corpus,
    c = total tokens in doc, d = total tokens in rest of corpus."""
    total_word = a + b
    total_tok = c + d
    if total_word == 0 or total_tok == 0:
        return 0.0
    e1 = c * total_word / total_tok
    e2 = d * total_word / total_tok
    g2 = 0.0
    if a > 0 and e1 > 0:
        g2 += a * math.log(a / e1)
    if b > 0 and e2 > 0:
        g2 += b * math.log(b / e2)
    return 2 * g2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text-dir", required=True, help="folder of htid subfolders, each containing page .txt files")
    ap.add_argument("--metadata", required=True, help="csv with htid, title, author, year columns")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--min-count", type=int, default=3, help="minimum occurrences of a word in a novel to be scored (default 3)")
    ap.add_argument("--topn", type=int, default=15, help="top N novels per word to keep (default 15)")
    ap.add_argument("--dict-min-vols", type=int, default=10)
    ap.add_argument("--page-min-dict-rate", type=float, default=0.55)
    ap.add_argument("--running-head-min-pages", type=int, default=3)
    ap.add_argument("--running-head-frac", type=float, default=0.05)
    ap.add_argument("--running-head-max-chars", type=int, default=60)
    args = ap.parse_args()

    out_dir = Path(args.out)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(args.metadata)
    meta["htid"] = meta["htid"].astype(str)
    meta_by_id = meta.set_index("htid").to_dict("index")

    print(f"discovering volumes under {args.text_dir} ...")
    all_volumes = discover_volumes(args.text_dir)
    print(f"found {len(all_volumes):,} volumes")

    print(f"building corpus dictionary (word must appear in >= {args.dict_min_vols} volumes) ...")
    dictionary = build_dictionary(all_volumes, args.dict_min_vols)
    print(f"dictionary: {len(dictionary):,} words")

    print("cleaning + loading all volumes (this is the slow step) ...")
    docs = load_docs_keep_id(
        args.text_dir, dictionary,
        running_head_min_pages=args.running_head_min_pages,
        running_head_frac=args.running_head_frac,
        running_head_max_chars=args.running_head_max_chars,
        page_min_dict_rate=args.page_min_dict_rate,
    )
    print(f"loaded {len(docs):,} cleaned documents")

    # per-doc word counts (all words, so we have accurate totals + counts for every seed word)
    doc_counts = {}
    doc_totals = {}
    for htid, text in docs.items():
        toks = [w.lower() for w in _word.findall(text) if len(w) > 1]
        doc_counts[htid] = Counter(toks)
        doc_totals[htid] = len(toks)

    corpus_total = sum(doc_totals.values())
    corpus_counts = Counter()
    for c in doc_counts.values():
        corpus_counts.update(c)

    print(f"corpus: {corpus_total:,} tokens across {len(docs):,} novels")
    print(f"scoring {len(ENV_WORDS)} seed words against every novel ...")

    rows = []
    for word in ENV_WORDS:
        word_corpus_count = corpus_counts.get(word, 0)
        if word_corpus_count == 0:
            continue
        for htid, counts in doc_counts.items():
            a = counts.get(word, 0)
            if a < args.min_count:
                continue
            c = doc_totals[htid]
            b = word_corpus_count - a
            d = corpus_total - c
            rate_doc = a / c if c else 0.0
            rate_rest = b / d if d else 0.0
            if rate_doc <= rate_rest:
                continue  # only keep novels where the word is OVER-represented, not under
            g2 = log_likelihood_g2(a, b, c, d)
            info = meta_by_id.get(htid, {})
            rows.append({
                "word": word,
                "htid": htid,
                "title": info.get("title"),
                "author": info.get("author"),
                "year": info.get("year"),
                "era": assign_era(info["year"]) if pd.notna(info.get("year")) else None,
                "count_in_novel": a,
                "novel_total_tokens": c,
                "rate_per_10k_in_novel": round(rate_doc * 10000, 2),
                "rate_per_10k_rest_of_corpus": round(rate_rest * 10000, 2),
                "g2": round(g2, 2),
            })

    result = pd.DataFrame(rows)
    result = result.sort_values(["word", "g2"], ascending=[True, False])
    top = result.groupby("word", group_keys=False).head(args.topn)
    top.to_csv(out_dir / "tables" / "novel_keyness_top_by_word.csv", index=False)
    print(f"tables/novel_keyness_top_by_word.csv: {len(top):,} rows "
          f"(top {args.topn} novels per word, {top['word'].nunique()} words with any hits)")

    manifest = {
        "n_novels": len(docs),
        "corpus_total_tokens": corpus_total,
        "n_seed_words": len(ENV_WORDS),
        "min_count": args.min_count,
        "topn": args.topn,
        "words_with_no_hits": sorted(set(ENV_WORDS) - set(result["word"].unique())),
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"MANIFEST.json written. {len(manifest['words_with_no_hits'])} words had no qualifying novel "
          f"(too rare / never clears --min-count anywhere): {manifest['words_with_no_hits']}")


if __name__ == "__main__":
    main()
