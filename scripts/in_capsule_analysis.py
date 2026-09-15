#!/usr/bin/env python3
"""
Lexicon counts, collocates and KWIC over the capsule corpus (secure mode).

    python in_capsule_analysis.py --input /media/secure_volume/fa50b375-3216-4edd-a685-98488562b723 --out /media/secure_volume/out_lex --no-ner

Corpus layout: /media/secure_volume/fa50b375-3216-4edd-a685-98488562b723/<htid>/<page>.txt (a flat folder of .txt files also works).
Release-safe outputs: climate_term_counts.csv, climate_collocates.csv, place_entity_counts.csv.
NOT releasable: kwic_REVIEW_SEPARATELY/climate_kwic.csv holds ordered OCR windows. Read it
inside the capsule; clear it with htrc-help@hathitrust.org before any of it leaves.
"""
import argparse, csv, json, re
from pathlib import Path
from collections import Counter

# Climate lexicon
CLIMATE_TERMS = [
    "flood", "drought", "warming", "extinction", "refuge", "refugee", "carbon",
    "emission", "wildfire", "hurricane", "famine", "sea level", "greenhouse",
    "pollution", "toxic", "contamination", "collapse", "wasteland", "melt",
]
KWIC_WINDOW = 12  # tokens either side
MAX_KWIC_PER_TERM_PER_VOL = 25  # caps exported windows only, never counts
TOP_COLLOCATES_PER_TERM = 100
EXPORT_SIZE_WARN_BYTES = 1_048_576  # HTRC review threshold

STOPWORDS = frozenset("""
the a an and or but if then than that this these those there here of in on at to
from by for with without into onto upon over under again further is was were be
been being am are it its it's as so not no nor only own same too very can will
just should now he she they them his her their we you i me my our us who whom
which what when where why how all any both each few more most other some such
had has have having do does did doing would could may might must shall about
up down out off above below between through during before after once while
""".split())


def iter_volumes(input_dir: Path):
    """Yield (htid, full_text) per VOLUME.

    Capsule layout: one folder per HTID under input_dir, one .txt
    per PAGE inside it. Pages are concatenated in filename order into a single text.
    A bare .txt directly under input_dir is still treated as one whole volume, so the
    local Gutenberg dry run keeps working.

    The old rglob loader yielded one record per .txt, which on the capsule layout meant
    one record per page, all under the same htid -- counts came out per page, not per
    volume, with no error. Same bug the v1 BERTopic notebook had.
    """
    for p in sorted(input_dir.iterdir()):
        if p.name.startswith("."):
            continue
        if p.is_dir():
            pages = sorted(p.glob("*.txt"))
            if not pages:
                continue
            try:
                yield p.name, "\n".join(pg.read_text(errors="ignore") for pg in pages)
            except Exception as e:
                print(f"  skip {p}: {e}")
        elif p.suffix == ".txt":
            try:
                yield p.stem, p.read_text(errors="ignore")
            except Exception as e:
                print(f"  skip {p}: {e}")


def tokenize(text: str):
    return re.findall(r"[A-Za-z']+", text.lower())


def kwic(tokens, term, window):
    """Count every hit of `term`; return (total_hits, capped context windows, collocate Counter).

    The cap applies to exported windows only, never to the count.
    """
    parts = term.split()
    n = len(parts)
    total = 0
    out = []
    colloc = Counter()
    for i in range(len(tokens) - n + 1):
        if tokens[i:i + n] == parts:
            total += 1
            lo, hi = max(0, i - window), min(len(tokens), i + n + window)
            if len(out) < MAX_KWIC_PER_TERM_PER_VOL:
                out.append(tokens[lo:hi])
            for t in tokens[lo:i] + tokens[i + n:hi]:
                if t not in STOPWORDS and len(t) > 2:
                    colloc[t] += 1
    return total, out, colloc


# Per-volume results are appended here so an interrupted run resumes; kept outside the export dir.
PARTIAL = "partial.jsonl"


def _load_partial(work_dir: Path):
    """Return per-volume records already computed, keyed by htid."""
    fp = work_dir / PARTIAL
    if not fp.exists():
        return {}
    done = {}
    with open(fp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # a run killed mid-write leaves one torn line; drop it and rerun that volume
                print(f"  ! dropping torn partial line, that volume will re-run")
                continue
            done[rec["htid"]] = rec
    return done


def analyze(input_dir: Path, out_dir: Path, work_dir: Path, do_ner: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    done = _load_partial(work_dir)
    if done:
        print(f"resuming: {len(done)} volumes already done in {work_dir / PARTIAL}")

    nlp = None
    if do_ner:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
        nlp.max_length = 4_000_000

    partial_fp = open(work_dir / PARTIAL, "a")
    for htid, text in iter_volumes(input_dir):
        if htid in done:
            print(f"  skip {htid} (already done)")
            continue
        toks = tokenize(text)
        rec = {"htid": htid, "tokens": len(toks), "terms": {}, "kwic": [],
               "colloc": {}, "ents": {}}
        for term in CLIMATE_TERMS:
            total, windows, colloc = kwic(toks, term, KWIC_WINDOW)
            if total:
                rec["terms"][term] = total
                rec["kwic"] += [{"term": term, "window": " ".join(w)} for w in windows]
                rec["colloc"][term] = dict(colloc)
        if nlp is not None:
            # NER for place/setting geography — export only the aggregate counts.
            ec = Counter()
            for chunk_start in range(0, len(text), 900_000):
                doc = nlp(text[chunk_start:chunk_start + 900_000])
                for ent in doc.ents:
                    if ent.label_ in ("GPE", "LOC", "FAC"):
                        # OCR text is hard-wrapped, so entities routinely span a line
                        # break ("the united\nstates"). Without this, one place is
                        # counted as several distinct entities.
                        name = re.sub(r"\s+", " ", ent.text).strip().lower()
                        if name:
                            ec[f"{ent.label_}\t{name}"] += 1
            rec["ents"] = dict(ec)
        partial_fp.write(json.dumps(rec) + "\n")
        partial_fp.flush()
        done[htid] = rec
        print(f"  done {htid}: {sum(rec['terms'].values())} term-hits "
              f"across {len(rec['terms'])} terms")
    partial_fp.close()

    freq_rows, kwic_rows = [], []
    ent_counter = Counter()
    colloc_by_term = {}
    for htid, rec in sorted(done.items()):
        for term, count in rec["terms"].items():
            freq_rows.append({"htid": htid, "term": term, "count": count,
                              "volume_tokens": rec.get("tokens", "")})
        for k in rec["kwic"]:
            kwic_rows.append({"htid": htid, "term": k["term"], "window": k["window"]})
        for term, counts in rec.get("colloc", {}).items():
            c = colloc_by_term.setdefault(term, Counter())
            for w, n in counts.items():
                c[w] += n
        for key, c in rec.get("ents", {}).items():
            ent_counter[key] += c

    _write_csv(out_dir / "climate_term_counts.csv", freq_rows,
               ["htid", "term", "count", "volume_tokens"])
    colloc_rows = [{"term": t, "collocate": w, "count": n}
                   for t, c in sorted(colloc_by_term.items())
                   for w, n in c.most_common(TOP_COLLOCATES_PER_TERM)]
    _write_csv(out_dir / "climate_collocates.csv", colloc_rows,
               ["term", "collocate", "count"])
    if do_ner:
        rows = [{"label": k.split("\t")[0], "entity": k.split("\t")[1], "count": c}
                for k, c in ent_counter.most_common()]
        _write_csv(out_dir / "place_entity_counts.csv", rows, ["label", "entity", "count"])

    # KWIC holds running OCR text: written last, into its own subdir, never released with the rest.
    kwic_dir = out_dir / "kwic_REVIEW_SEPARATELY"
    kwic_dir.mkdir(exist_ok=True)
    _write_csv(kwic_dir / "climate_kwic.csv", kwic_rows, ["htid", "term", "window"])

    safe = ["climate_term_counts.csv", "climate_collocates.csv"] + \
           (["place_entity_counts.csv"] if do_ner else [])
    (out_dir / "MANIFEST.json").write_text(json.dumps({
        "non_consumptive_outputs": safe,
        "note": "Aggregate derived data only: exhaustive term counts, unordered collocate "
                "counts, aggregate place-entity counts. Word order is discarded, so nothing "
                "here can be reassembled into passages.",
        "kwic_window": KWIC_WINDOW,
        "max_kwic_per_term_per_volume": MAX_KWIC_PER_TERM_PER_VOL,
        "top_collocates_per_term": TOP_COLLOCATES_PER_TERM,
        "counts_are_complete": "term and collocate counts are exhaustive; only KWIC is capped",
        "volumes": len(done),
        "terms": CLIMATE_TERMS,
        "kwic_caveat": "kwic_REVIEW_SEPARATELY/ holds ordered token windows, i.e. running OCR "
                       "text. HTRC guidelines prohibit exporting OCR text. Do NOT include it "
                       "in a releaseresults batch without clearing it with htrc-help first.",
    }, indent=2))

    print(f"\nWrote outputs to {out_dir}")
    print(f"Working state (resume data) is in {work_dir} — not part of the export.")
    _export_check(out_dir, safe, kwic_dir)


def _export_check(out_dir: Path, safe, kwic_dir: Path):
    """Report the release-safe payload size against HTRC's 1 MB review threshold."""
    total = sum((out_dir / f).stat().st_size for f in safe) + \
        (out_dir / "MANIFEST.json").stat().st_size
    kwic_bytes = sum(p.stat().st_size for p in kwic_dir.glob("*.csv"))
    print(f"\n  release-safe payload : {total:>10,} bytes"
          f"  ({'OK' if total < EXPORT_SIZE_WARN_BYTES else 'OVER 1 MB — aggregate further'})")
    print(f"  kwic (NOT releasable): {kwic_bytes:>10,} bytes  in {kwic_dir.name}/")
    # `releaseresults add` is not cumulative: every path goes in ONE add.
    print("\n  Release with ONE add — repeated `add` calls OVERWRITE each other:")
    print("    releaseresults add " + " ".join(str(out_dir / f) for f in safe))
    print("    releaseresults done")


def _write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {path.name}: {len(rows)} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="dir of pulled volume text")
    ap.add_argument("--out", required=True, help="export dir (on the secure volume)")
    ap.add_argument("--work", default=None,
                    help="resume/working dir; default <out>_work. Kept out of the export dir.")
    ap.add_argument("--no-ner", action="store_true", help="skip spaCy NER (faster dry-run)")
    a = ap.parse_args()
    out = Path(a.out)
    work = Path(a.work) if a.work else out.with_name(out.name + "_work")
    analyze(Path(a.input), out, work, do_ner=not a.no_ner)
