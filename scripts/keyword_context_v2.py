"""
KWIC / concordance export: for every occurrence of a keyword (environmental lexicon,
or the new technology lexicon), pull a ~5-sentence window of surrounding context
(2 sentences before + the keyword sentence + 2 after), tagged with book/author/year/era.

Two output CSVs (environment_context*.csv, technology_context*.csv), long format:
one row per PASSAGE, not per occurrence. If two keyword hits are close enough that
their windows overlap or touch, they're merged into a single passage with a
`keywords_found` column listing everything that matched -- this avoids exporting
many near-duplicate, heavily-overlapping rows when a word clusters densely in one
spot, and keeps each row's footprint small.

No cap on occurrences: every keyword hit in the whole corpus is captured. That means
a word that saturates a given novel (e.g. "water" throughout a nautical story) can
still produce merged passages that add up to a large share of that book's own text.
Because the goal here is grounding quantitative findings in real passages, not
re-exporting whole books, this script tracks that directly:

  tables/book_summary.csv -- one row per novel, per lexicon: total words in the book
  vs. total words captured across all its merged passages, as a percentage. Any book
  at or above --flag-threshold-pct (default 30) is marked `flagged` in that file AND
  printed as a warning at the end of the run, so it can't be missed and silently
  released. This is a report-and-flag design, not a silent cap -- review flagged
  books yourself before deciding what (if anything) to trim.

Each output CSV auto-splits into part files (part1, part2, ...) before hitting
--max-csv-bytes (default ~900MB, a safety margin under the capsule's 1GB export
limit), so you're never stuck manually splitting a file after the fact.

Must run INSIDE THE CAPSULE, in secure mode -- raw per-novel text only exists at
TEXT_DIR on the secure volume and never leaves it.

Usage (inside capsule):
    python keyword_context_v2.py \
        --text-dir /media/secure_volume/fa50b375-3216-4edd-a685-98488562b723 \
        --metadata /home/dcuser/metadata_august2026.csv \
        --out /media/secure_volume/out_keyword_context_v2
"""

import argparse
import csv
import json
import os
import re
from collections import Counter
from multiprocessing import get_context
from pathlib import Path

import nltk
import pandas as pd
from nltk.tokenize import TreebankWordTokenizer, sent_tokenize

# ---- OCR cleaning: copied verbatim from SF_word2vec_eras_v2.ipynb (cell 46a9ca82),
# same as scripts/novel_keyness_v2.py -- keeps counts/text comparable across scripts. ----

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


# ---- tokenization for keyword matching: same TOKEN_RE as SF_word2vec_eras_v2.ipynb (cell be5f5689) ----

_tokenizer = TreebankWordTokenizer()
TOKEN_RE = re.compile(r"^[a-z]+(?:'[a-z]+)?$")


def clean_tokens(tokens):
    return [t for t in tokens if TOKEN_RE.match(t) and len(t) > 1]


def sentences_with_tokens(text):
    """Raw (readable, original-case) sentences paired with their lowercased match-tokens."""
    raw_sents = sent_tokenize(text)
    out = []
    for s in raw_sents:
        toks = set(clean_tokens(_tokenizer.tokenize(s.lower())))
        out.append((s, toks))
    return out


# ---- era assignment: copied from SF_word2vec_eras_v2.ipynb (cell 7030de76) ----

def assign_era(year, cutoffs=(1962, 1972), labels=("era_a", "era_b", "era_c")):
    for cutoff, label in zip(cutoffs, labels):
        if year < cutoff:
            return label
    return labels[-1]


# ---- lexicons ----
# Environmental lexicon: copied from SF_word2vec_eras_v2.ipynb (cell 520bb515). Keep in
# sync with the notebook if it changes there.

ENV_WORD_GROUPS = {
    "landscape_baseline": ["river", "creek", "stream", "water", "forest", "nature", "wilderness", "jungle",
                            "ocean", "landscape", "levee", "dam", "reservoir", "estuary", "wetland",
                            "marsh", "watershed"],
    "ecology_concept": ["ecology", "ecosystem", "environment", "biosphere", "habitat", "balance"],
    "contamination": ["contamination", "waste", "smog", "fumes", "chemical", "pesticide", "insecticide",
                       "pollutant", "exhaust", "toxic", "polluted", "pollution"],
    "waste_infrastructure": ["sewer", "sewage", "drainage", "effluent", "runoff", "wastewater",
                              "cesspool", "sludge", "septic", "plumbing"],
    "population_scarcity": ["overpopulation", "population", "famine", "scarcity", "starvation", "resource", "drought"],
    "energy": ["oil", "fuel", "energy", "coal"],
    "nuclear_atomic": ["radiation", "radioactive", "fallout", "nuclear", "atomic", "bomb", "meltdown"],
    "cosmic_natural_causation": ["solar", "cosmic", "celestial", "geological", "planetary"],
    "human_agency": ["mankind", "humanity", "civilization", "industrial", "war"],
    "disaster_collapse": ["wasteland", "extinction", "collapse", "barren", "dying", "decay", "catastrophe",
                           "apocalypse", "plague"],
    "climate_weather": ["climate", "weather", "warming", "greenhouse", "atmosphere", "temperature",
                         "flood", "flooding", "storm", "hurricane", "glacier", "carbon", "ozone"],
    "space_earth_framing": ["earth", "homeworld", "colony", "frontier", "terraform", "alien"],
}

# Technology lexicon: drafted 2026-09-23, WordNet-audited (each word confirmed a real,
# cleanly-defined lexical item), reviewed and approved by Dez. Zero overlap with the
# environmental lexicon above. "machine" dropped 2026-09-23 (too generic -- machine gun,
# "political machine" -- machinery/mechanical/automaton/automation/automated still cover
# the theme).
TECH_WORD_GROUPS = {
    "automation_machinery": ["machinery", "mechanical", "automaton", "automation", "automated"],
    "artificial_beings": ["robot", "android", "cyborg"],
    "computing_electronics": ["computer", "cybernetic", "electronic", "circuitry"],
    "engineering_industry": ["engineering", "engineer", "technology", "technological", "factory"],
    "synthetic_material": ["synthetic", "artificial"],
    "space_energy_tech": ["rocket", "spacecraft", "satellite", "laser", "reactor"],
}


def word_to_group_map(word_groups):
    return {w: g for g, ws in word_groups.items() for w in ws}


# ---- window-merging KWIC extraction ----

def extract_passages(sent_list, lexicon_words, word_to_group, sentences_before=2, sentences_after=2):
    """sent_list: [(raw_sentence, token_set), ...] for one novel.
    Returns list of dicts: one per merged passage (keywords_found, groups_found, context,
    n_sentences, sent_start, sent_end)."""
    n = len(sent_list)
    hits = []  # (sentence_idx, {matched words in this sentence})
    for i, (_, toks) in enumerate(sent_list):
        matched = toks & lexicon_words
        if matched:
            hits.append((i, matched))
    if not hits:
        return []

    # windows, sorted by construction since hits are in sentence order
    windows = []
    for i, matched in hits:
        start = max(0, i - sentences_before)
        end = min(n - 1, i + sentences_after)
        windows.append([start, end, set(matched)])

    merged = [windows[0]]
    for start, end, matched in windows[1:]:
        last = merged[-1]
        if start <= last[1] + 1:  # overlapping or adjacent -> merge
            last[1] = max(last[1], end)
            last[2] |= matched
        else:
            merged.append([start, end, set(matched)])

    passages = []
    for start, end, matched in merged:
        context = " ".join(sent_list[j][0] for j in range(start, end + 1))
        groups = sorted({word_to_group[w] for w in matched})
        passages.append({
            "keywords_found": ", ".join(sorted(matched)),
            "groups_found": ", ".join(groups),
            "context": context,
            "n_sentences": end - start + 1,
            "sent_start": start,
            "sent_end": end,
            "context_words": len(context.split()),
        })
    return passages


# ---- rolling CSV writer: auto-splits into part files before hitting max_bytes ----

class RollingCSVWriter:
    def __init__(self, out_dir, base_name, fieldnames, max_bytes):
        self.out_dir = Path(out_dir)
        self.base_name = base_name
        self.fieldnames = fieldnames
        self.max_bytes = max_bytes
        self.part = 1
        self.f = None
        self.writer = None
        self.rows_written = 0
        self.parts_written = []
        self._open_new_part()

    def _open_new_part(self):
        if self.f is not None:
            self.f.close()
        path = self.out_dir / f"{self.base_name}_part{self.part}.csv"
        self.f = open(path, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.f, fieldnames=self.fieldnames)
        self.writer.writeheader()
        self.parts_written.append(path)

    def write(self, row):
        self.writer.writerow(row)
        self.rows_written += 1
        if self.f.tell() >= self.max_bytes:
            self.part += 1
            self._open_new_part()

    def close(self):
        if self.f is not None:
            self.f.close()


CONTEXT_FIELDNAMES = ["keywords_found", "groups_found", "context", "n_sentences", "context_words",
                       "sent_start", "sent_end", "htid", "title", "author", "year", "era"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text-dir", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sentences-before", type=int, default=2)
    ap.add_argument("--sentences-after", type=int, default=2)
    ap.add_argument("--flag-threshold-pct", type=float, default=30.0,
                     help="flag a book if a lexicon's merged passages cover >= this %% of its total words")
    ap.add_argument("--max-csv-bytes", type=int, default=900_000_000,
                     help="split into a new part file before a CSV reaches this size (default ~900MB, under the 1GB export limit)")
    ap.add_argument("--dict-min-vols", type=int, default=10)
    ap.add_argument("--page-min-dict-rate", type=float, default=0.55)
    ap.add_argument("--running-head-min-pages", type=int, default=3)
    ap.add_argument("--running-head-frac", type=float, default=0.05)
    ap.add_argument("--running-head-max-chars", type=int, default=60)
    args = ap.parse_args()

    out_dir = Path(args.out)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    for pkg in ["punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg)

    meta = pd.read_csv(args.metadata)
    meta["htid"] = meta["htid"].astype(str)
    meta_by_id = meta.set_index("htid").to_dict("index")

    print(f"discovering volumes under {args.text_dir} ...")
    all_volumes = discover_volumes(args.text_dir)
    print(f"found {len(all_volumes):,} volumes")

    print(f"building corpus dictionary (word must appear in >= {args.dict_min_vols} volumes) ...")
    dictionary = build_dictionary(all_volumes, args.dict_min_vols)
    print(f"dictionary: {len(dictionary):,} words")

    env_words = set(word_to_group_map(ENV_WORD_GROUPS))
    tech_words = set(word_to_group_map(TECH_WORD_GROUPS))
    env_word_to_group = word_to_group_map(ENV_WORD_GROUPS)
    tech_word_to_group = word_to_group_map(TECH_WORD_GROUPS)

    env_writer = RollingCSVWriter(out_dir, "environment_context", CONTEXT_FIELDNAMES, args.max_csv_bytes)
    tech_writer = RollingCSVWriter(out_dir, "technology_context", CONTEXT_FIELDNAMES, args.max_csv_bytes)

    book_summary_rows = []
    flagged_books = []

    print(f"processing {len(all_volumes):,} volumes (cleaning, sentence-splitting, keyword matching) ...")
    for count, htid in enumerate(all_volumes, 1):
        pages = all_volumes[htid]
        text = clean_volume(
            pages, dictionary,
            running_head_min_pages=args.running_head_min_pages,
            running_head_frac=args.running_head_frac,
            running_head_max_chars=args.running_head_max_chars,
            page_min_dict_rate=args.page_min_dict_rate,
        )
        book_total_words = len(text.split())
        if book_total_words == 0:
            continue

        sent_list = sentences_with_tokens(text)
        info = meta_by_id.get(htid, {})
        year = info.get("year")
        row_meta = {
            "htid": htid,
            "title": info.get("title"),
            "author": info.get("author"),
            "year": year,
            "era": assign_era(year) if pd.notna(year) else None,
        }

        book_row = {**row_meta, "book_total_words": book_total_words}

        for label, words, word_to_group, writer in [
            ("env", env_words, env_word_to_group, env_writer),
            ("tech", tech_words, tech_word_to_group, tech_writer),
        ]:
            passages = extract_passages(sent_list, words, word_to_group,
                                         args.sentences_before, args.sentences_after)
            exported_words = sum(p["context_words"] for p in passages)
            pct = round(100 * exported_words / book_total_words, 2)
            flagged = pct >= args.flag_threshold_pct
            book_row[f"{label}_passages"] = len(passages)
            book_row[f"{label}_exported_words"] = exported_words
            book_row[f"{label}_pct_exported"] = pct
            book_row[f"{label}_flagged"] = flagged
            if flagged:
                flagged_books.append((label, row_meta["title"], row_meta["author"], row_meta["year"], pct))
            for p in passages:
                writer.write({**p, **row_meta})

        book_summary_rows.append(book_row)

        if count % 100 == 0:
            print(f"  ... {count:,}/{len(all_volumes):,} volumes processed")

    env_writer.close()
    tech_writer.close()

    summary_df = pd.DataFrame(book_summary_rows)
    summary_df.to_csv(out_dir / "tables" / "book_summary.csv", index=False)

    print()
    print(f"environment_context: {env_writer.rows_written:,} passages across {len(env_writer.parts_written)} file(s)")
    for p in env_writer.parts_written:
        print(f"  {p} ({p.stat().st_size / 1e6:.1f} MB)")
    print(f"technology_context: {tech_writer.rows_written:,} passages across {len(tech_writer.parts_written)} file(s)")
    for p in tech_writer.parts_written:
        print(f"  {p} ({p.stat().st_size / 1e6:.1f} MB)")
    print(f"tables/book_summary.csv: {len(summary_df):,} rows")

    print()
    if flagged_books:
        print(f"*** {len(flagged_books)} lexicon/book combinations flagged at >= {args.flag_threshold_pct}% "
              f"of the book's own words exported -- REVIEW BEFORE RELEASING: ***")
        for label, title, author, year, pct in sorted(flagged_books, key=lambda x: -x[4]):
            print(f"  [{label}] {pct:.1f}%  {title} ({author}, {year})")
    else:
        print(f"no book exceeded the {args.flag_threshold_pct}% flag threshold for either lexicon.")

    manifest = {
        "n_volumes": len(all_volumes),
        "sentences_before": args.sentences_before,
        "sentences_after": args.sentences_after,
        "flag_threshold_pct": args.flag_threshold_pct,
        "env_lexicon_size": len(env_words),
        "tech_lexicon_size": len(tech_words),
        "env_passages": env_writer.rows_written,
        "tech_passages": tech_writer.rows_written,
        "n_flagged": len(flagged_books),
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print("MANIFEST.json written.")


if __name__ == "__main__":
    main()
