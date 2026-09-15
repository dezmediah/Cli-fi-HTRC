#!/usr/bin/env python
"""
Run BookNLP over the capsule corpus and keep only derived tables; raw output is deleted per volume.

Use the isolated venv and cap threads (torch on all 50 cores is 6x slower); about 6 workers:

    ~/venv-booknlp/bin/python booknlp_batch.py --input /media/secure_volume/fa50b375-3216-4edd-a685-98488562b723 --out /media/secure_volume/bnlp \
        --workset clifi_htids_capsule.txt --procs 6 --threads 8
    ~/venv-booknlp/bin/python booknlp_batch.py --finalize --out /media/secure_volume/bnlp \
        --metadata ~/metadata_august2026.csv

Resumable: .done / .failed markers per volume, --retry-failed to redo failures. Per volume it keeps
summary.json, characters.csv, entity_counts.csv, entity_names.csv, supersense_counts.csv,
event_lemmas.csv, pos_counts.csv, quotes_by_char.csv. --finalize stacks them under release/ and
prints the single `releaseresults add`. Nothing kept is running text; .tokens, .quotes and
.book.html are always deleted.
"""

import argparse
import csv
import json
import os
import shutil
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

# ----------------------------------------------------------------------------- config

GENDER_KEYS = {"he/him/his": "p_he", "she/her": "p_she", "they/them/their": "p_they"}
NAME_CATS = ("LOC", "GPE", "FAC", "ORG", "VEH")     # PER names are covered by characters.csv
DEFAULT_CUTOFFS = (1962, 1972)                       # matches SF_word2vec_eras_v2.ipynb

# ----------------------------------------------------------------------------- corpus

def read_workset(path):
    ids = []
    for line in Path(path).read_text().splitlines():
        tok = line.strip().split()[0] if line.strip() else ""
        if tok and tok.lower() != "htid":
            ids.append(tok.split(",")[0])
    return ids


def list_volumes(input_dir, workset=None, limit=None):
    input_dir = Path(input_dir)
    if workset:
        ids = [h for h in read_workset(workset) if (input_dir / h).is_dir()]
        missing = len(read_workset(workset)) - len(ids)
        if missing:
            print(f"[warn] {missing} workset ids not present under {input_dir}", file=sys.stderr)
    else:
        ids = sorted(p.name for p in input_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
    return ids[:limit] if limit else ids


def volume_text(input_dir, htid):
    pages = sorted((Path(input_dir) / htid).glob("*.txt"))
    return "\n\n".join(p.read_text(errors="ignore") for p in pages), len(pages)


# ----------------------------------------------------------------------------- worker

_NLP = None
_ARGS = None


def _init_worker(args):
    global _NLP, _ARGS
    _ARGS = args
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # secure mode has no network; make every HF/torch code path fail fast instead of hanging
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    torch.set_num_threads(args.threads)
    from booknlp.booknlp import BookNLP
    params = {"pipeline": "entity,quote,supersense,event,coref", "model": args.model}
    if args.model_path:
        params["model_path"] = args.model_path
    _NLP = BookNLP("en", params)


def _top(counter, n):
    return "|".join(f"{w}:{c}" for w, c in counter.most_common(n))


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def aggregate_volume(raw_dir, htid, vol_dir, top_chars, top_lemmas, top_names):
    """Reduce BookNLP's raw output for one volume to the release-safe tables. Returns summary dict."""
    raw_dir = Path(raw_dir)
    vol_dir = Path(vol_dir)
    vol_dir.mkdir(parents=True, exist_ok=True)

    # ---- tokens: counts only, never the words
    n_tok = 0
    sents = set()
    paras = set()
    pos = Counter()
    events = Counter()
    with open(raw_dir / f"{htid}.tokens", newline="") as f:
        r = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in r:
            n_tok += 1
            sents.add((row["paragraph_ID"], row["sentence_ID"]))
            paras.add(row["paragraph_ID"])
            pos[row["POS_tag"]] += 1
            if row["event"] == "EVENT":
                events[row["lemma"].lower()] += 1
    _write_csv(vol_dir / "pos_counts.csv", ["pos", "count"], sorted(pos.items(), key=lambda x: -x[1]))
    _write_csv(vol_dir / "event_lemmas.csv", ["lemma", "count"], events.most_common(top_lemmas))

    # ---- entities
    ent = Counter()
    names = defaultdict(Counter)
    n_ent = 0
    with open(raw_dir / f"{htid}.entities", newline="") as f:
        r = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in r:
            n_ent += 1
            ent[(row["cat"], row["prop"])] += 1
            if row["prop"] == "PROP" and row["cat"] in NAME_CATS:
                names[row["cat"]][row["text"]] += 1
    _write_csv(vol_dir / "entity_counts.csv", ["cat", "prop", "count"],
               [(c, p, n) for (c, p), n in sorted(ent.items())])
    name_rows = []
    for cat in NAME_CATS:
        name_rows += [(cat, t, n) for t, n in names[cat].most_common(top_names)]
    _write_csv(vol_dir / "entity_names.csv", ["cat", "name", "count"], name_rows)

    # ---- supersenses
    ss = Counter()
    ss_types = defaultdict(set)
    with open(raw_dir / f"{htid}.supersense", newline="") as f:
        r = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in r:
            ss[row["supersense_category"]] += 1
            ss_types[row["supersense_category"]].add(row["text"].lower())
    _write_csv(vol_dir / "supersense_counts.csv", ["supersense", "count", "n_types"],
               [(k, n, len(ss_types[k])) for k, n in sorted(ss.items(), key=lambda x: -x[1])])

    # ---- quotes: count and length only
    q_by_char = Counter()
    q_tok_by_char = Counter()
    n_q = 0
    q_tok = 0
    with open(raw_dir / f"{htid}.quotes", newline="") as f:
        r = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in r:
            n_q += 1
            ln = int(row["quote_end"]) - int(row["quote_start"]) + 1
            q_tok += ln
            cid = row["char_id"]
            q_by_char[cid] += 1
            q_tok_by_char[cid] += ln
    _write_csv(vol_dir / "quotes_by_char.csv", ["char_id", "n_quotes", "n_quote_tokens"],
               [(c, n, q_tok_by_char[c]) for c, n in q_by_char.most_common(top_chars)])

    # ---- characters (.book JSON)
    book = json.loads((raw_dir / f"{htid}.book").read_text())
    chars = sorted(book["characters"], key=lambda c: -c["count"])
    rows = []
    for rank, c in enumerate(chars[:top_chars], 1):
        proper = c["mentions"].get("proper", [])
        best = proper[0]["n"] if proper else ""
        g = c.get("g") or {}
        inf = g.get("inference", {})
        agent = Counter(d["w"].lower() for d in c["agent"])
        patient = Counter(d["w"].lower() for d in c["patient"])
        mod = Counter(d["w"].lower() for d in c["mod"])
        poss = Counter(d["w"].lower() for d in c["poss"])
        rows.append([
            rank, c["id"], best, len(proper), c["count"],
            sum(m["c"] for m in proper),
            sum(m["c"] for m in c["mentions"].get("common", [])),
            sum(m["c"] for m in c["mentions"].get("pronoun", [])),
            g.get("argmax", ""), g.get("max", ""),
            inf.get("he/him/his", ""), inf.get("she/her", ""), inf.get("they/them/their", ""),
            len(c["agent"]), len(c["patient"]), len(c["mod"]), len(c["poss"]),
            _top(agent, 15), _top(patient, 15), _top(mod, 15), _top(poss, 15),
        ])
    _write_csv(vol_dir / "characters.csv",
               ["rank", "char_id", "name", "n_proper_names", "mentions", "proper", "common", "pronoun",
                "gender", "gender_p", "p_he", "p_she", "p_they",
                "n_agent", "n_patient", "n_mod", "n_poss",
                "top_agent", "top_patient", "top_mod", "top_poss"], rows)

    summary = {
        "htid": htid, "n_tokens": n_tok, "n_sentences": len(sents), "n_paragraphs": len(paras),
        "n_entities": n_ent, "n_characters": len(chars), "n_quotes": n_q,
        "quote_token_share": round(q_tok / n_tok, 4) if n_tok else 0.0,
        "n_events": sum(events.values()), "n_event_types": len(events),
    }
    return summary


def _process_one(htid):
    args = _ARGS
    out = Path(args.out)
    vol_dir = out / "vol" / htid
    if (vol_dir / ".done").exists():
        return htid, "skip", 0, 0
    if (vol_dir / ".failed").exists():
        (vol_dir / ".failed").unlink()
    work = out / "work" / htid
    t0 = time.time()
    try:
        text, n_pages = volume_text(args.input, htid)
        n_words = len(text.split())
        if n_words < args.min_words:
            vol_dir.mkdir(parents=True, exist_ok=True)
            (vol_dir / ".failed").write_text(f"too short: {n_words} words in {n_pages} pages\n")
            return htid, "short", n_words, time.time() - t0
        work.mkdir(parents=True, exist_ok=True)
        txt = work / f"{htid}.txt"
        txt.write_text(text)
        _NLP.process(str(txt), str(work), htid)
        summary = aggregate_volume(work, htid, vol_dir, args.top_chars, args.top_lemmas, args.top_names)
        dt = time.time() - t0
        summary.update({"n_pages": n_pages, "n_words": n_words, "seconds": round(dt, 1),
                        "words_per_sec": round(n_words / dt, 1) if dt else None})
        (vol_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        if args.keep_raw:
            for junk in (f"{htid}.tokens", f"{htid}.book.html", f"{htid}.txt"):
                p = work / junk
                if p.exists():
                    p.unlink()
        else:
            shutil.rmtree(work, ignore_errors=True)
        (vol_dir / ".done").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
        return htid, "ok", n_words, dt
    except Exception:
        vol_dir.mkdir(parents=True, exist_ok=True)
        (vol_dir / ".failed").write_text(traceback.format_exc())
        shutil.rmtree(work, ignore_errors=True)
        return htid, "fail", 0, time.time() - t0


def run(args):
    out = Path(args.out)
    if not str(out).startswith("/media/secure_volume") and not args.allow_unsafe_out:
        print(f"[warn] --out {out} is not under /media/secure_volume — anything written elsewhere is "
              f"destroyed by a mode switch. Pass --allow-unsafe-out if this is a local test.", file=sys.stderr)
        sys.exit(2)
    (out / "vol").mkdir(parents=True, exist_ok=True)
    (out / "work").mkdir(parents=True, exist_ok=True)
    ids = list_volumes(args.input, args.workset, args.limit)
    todo = [h for h in ids if not (out / "vol" / h / ".done").exists()]
    if not args.retry_failed:
        todo = [h for h in todo if not (out / "vol" / h / ".failed").exists()]
    print(f"{len(ids)} volumes, {len(ids) - len(todo)} done or failed, {len(todo)} to run "
          f"with {args.procs} procs × {args.threads} threads", flush=True)
    if not todo:
        return
    import multiprocessing as mp
    log = open(out / "log.jsonl", "a")
    t0 = time.time()
    words = 0
    n_done = 0
    ctx = mp.get_context("spawn")   # never fork a process that has torch loaded
    with ctx.Pool(args.procs, initializer=_init_worker, initargs=(args,)) as pool:
        for htid, status, n_words, dt in pool.imap_unordered(_process_one, todo):
            n_done += 1
            words += n_words
            el = time.time() - t0
            rec = {"htid": htid, "status": status, "words": n_words, "seconds": round(dt, 1),
                   "t": time.strftime("%Y-%m-%d %H:%M:%S")}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            rate = words / el if el else 0
            left = len(todo) - n_done
            eta = (left * (words / max(n_done, 1))) / rate / 3600 if rate else float("nan")
            print(f"[{n_done}/{len(todo)}] {htid} {status} {n_words}w {dt:.0f}s | "
                  f"aggregate {rate:.0f} w/s | eta {eta:.1f} h", flush=True)
    log.close()
    fails = [p.parent.name for p in (out / "vol").glob("*/.failed")]
    print(f"done: {n_done} processed, {len(fails)} failed" + (f": {fails[:10]}" if fails else ""))


# ----------------------------------------------------------------------------- finalize

def load_metadata(path, cutoffs):
    meta = {}
    if not path:
        return meta
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                year = int(str(row.get("year", "")).strip()[:4])
            except ValueError:
                year = None
            era = ""
            if year is not None:
                idx = sum(year >= c for c in cutoffs)
                era = "era_" + "abcdefgh"[idx]
            meta[row["htid"]] = {"year": year, "era": era, "source": row.get("source", "")}
    return meta


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def finalize(args):
    out = Path(args.out)
    rel = out / "release"
    rel.mkdir(exist_ok=True)
    cutoffs = tuple(int(c) for c in args.cutoffs.split(","))
    meta = load_metadata(args.metadata, cutoffs)
    vols = sorted(p.parent.name for p in (out / "vol").glob("*/.done"))
    if not vols:
        print("nothing finished yet", file=sys.stderr)
        sys.exit(1)

    def tag(h):
        m = meta.get(h, {})
        return [h, m.get("year", ""), m.get("era", "")]

    # volume summary + gender/agency aggregates
    vs_rows = []
    char_rows = []
    ent_wide = {}
    ss_wide = {}
    pos_wide = {}
    ev_by_era = defaultdict(Counter)
    ev_by_vol = []
    names_by_era = defaultdict(Counter)
    names_by_vol = []
    q_rows = []
    ent_keys, ss_keys, pos_keys = set(), set(), set()
    for h in vols:
        d = out / "vol" / h
        s = json.loads((d / "summary.json").read_text())
        chars = _read_csv(d / "characters.csv")
        agg = Counter()
        for c in chars:
            g = c["gender"]
            key = {"he/him/his": "he", "she/her": "she"}.get(g, "other")
            agg[f"mentions_{key}"] += int(c["mentions"])
            agg[f"agent_{key}"] += int(c["n_agent"])
            agg[f"patient_{key}"] += int(c["n_patient"])
            char_rows.append(tag(h) + [c[k] for k in c])
        vs_rows.append(tag(h) + [s.get(k, "") for k in (
            "n_pages", "n_words", "n_tokens", "n_sentences", "n_paragraphs", "n_entities",
            "n_characters", "n_quotes", "quote_token_share", "n_events", "n_event_types",
            "seconds", "words_per_sec")] + [agg[k] for k in (
            "mentions_he", "mentions_she", "mentions_other",
            "agent_he", "agent_she", "patient_he", "patient_she")])
        e = {(r["cat"], r["prop"]): int(r["count"]) for r in _read_csv(d / "entity_counts.csv")}
        ent_wide[h] = e
        ent_keys |= e.keys()
        ss = {r["supersense"]: int(r["count"]) for r in _read_csv(d / "supersense_counts.csv")}
        ss_wide[h] = ss
        ss_keys |= ss.keys()
        ps = {r["pos"]: int(r["count"]) for r in _read_csv(d / "pos_counts.csv")}
        pos_wide[h] = ps
        pos_keys |= ps.keys()
        era = meta.get(h, {}).get("era", "")
        for r in _read_csv(d / "event_lemmas.csv"):
            ev_by_era[era][r["lemma"]] += int(r["count"])
        ev_by_vol += [tag(h) + [r["lemma"], r["count"]] for r in _read_csv(d / "event_lemmas.csv")[:args.top_lemmas_release]]
        for r in _read_csv(d / "entity_names.csv"):
            names_by_era[(era, r["cat"])][r["name"]] += int(r["count"])
        names_by_vol += [tag(h) + [r["cat"], r["name"], r["count"]] for r in _read_csv(d / "entity_names.csv")[:args.top_names_release]]
        q_rows += [tag(h) + [r["char_id"], r["n_quotes"], r["n_quote_tokens"]] for r in _read_csv(d / "quotes_by_char.csv")]

    T = ["htid", "year", "era"]
    _write_csv(rel / "volume_summary.csv", T + [
        "n_pages", "n_words", "n_tokens", "n_sentences", "n_paragraphs", "n_entities", "n_characters",
        "n_quotes", "quote_token_share", "n_events", "n_event_types", "seconds", "words_per_sec",
        "mentions_he", "mentions_she", "mentions_other", "agent_he", "agent_she", "patient_he", "patient_she"], vs_rows)
    char_header = list(_read_csv(out / "vol" / vols[0] / "characters.csv")[0].keys()) if _read_csv(out / "vol" / vols[0] / "characters.csv") else []
    _write_csv(rel / "characters.csv", T + char_header, char_rows)
    ek = sorted(ent_keys)
    _write_csv(rel / "entity_counts.csv", T + [f"{c}_{p}" for c, p in ek],
               [tag(h) + [ent_wide[h].get(k, 0) for k in ek] for h in vols])
    sk = sorted(ss_keys)
    _write_csv(rel / "supersense_counts.csv", T + sk, [tag(h) + [ss_wide[h].get(k, 0) for k in sk] for h in vols])
    pk = sorted(pos_keys)
    _write_csv(rel / "pos_counts.csv", T + pk, [tag(h) + [pos_wide[h].get(k, 0) for k in pk] for h in vols])
    _write_csv(rel / "event_lemmas_by_era.csv", ["era", "lemma", "count"],
               [(era, l, n) for era in sorted(ev_by_era) for l, n in ev_by_era[era].most_common(args.top_lemmas_era)])
    _write_csv(rel / "event_lemmas_by_volume.csv", T + ["lemma", "count"], ev_by_vol)
    _write_csv(rel / "entity_names_by_era.csv", ["era", "cat", "name", "count"],
               [(era, cat, nme, n) for (era, cat) in sorted(names_by_era) for nme, n in names_by_era[(era, cat)].most_common(args.top_names_era)])
    _write_csv(rel / "entity_names_by_volume.csv", T + ["cat", "name", "count"], names_by_vol)
    _write_csv(rel / "quotes_by_char.csv", T + ["char_id", "n_quotes", "n_quote_tokens"], q_rows)

    files = sorted(p for p in rel.iterdir() if p.is_file() and p.name != "MANIFEST.json")
    manifest = {"volumes": len(vols), "cutoffs": cutoffs, "metadata": args.metadata,
                "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                "files": {p.name: p.stat().st_size for p in files}}
    (rel / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    total = sum(manifest["files"].values())
    print(f"release/: {len(files)} files, {total/1e6:.2f} MB total, {len(vols)} volumes")
    for p in files:
        sz = p.stat().st_size
        print(f"  {p.name:32s} {sz/1e3:9.1f} KB" + ("   (over the 1 MB review threshold — mention it)" if sz > 1_000_000 else ""))
    print("\nrelease with ONE add (a second add wipes the first package):")
    print(f"    releaseresults add {rel}")
    print("    releaseresults done")


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="/media/secure_volume/fa50b375-3216-4edd-a685-98488562b723")
    ap.add_argument("--out", default="/media/secure_volume/bnlp")
    ap.add_argument("--workset", help="file of htids, one per line (default: every folder under --input)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--threads", type=int, default=8, help="torch threads PER process; 8 measured best")
    ap.add_argument("--model", default="small", choices=["small", "big"])
    ap.add_argument("--model-path", help="BookNLP model dir (default ~/booknlp_models)")
    ap.add_argument("--min-words", type=int, default=2000, help="skip volumes shorter than this")
    ap.add_argument("--top-chars", type=int, default=30)
    ap.add_argument("--top-lemmas", type=int, default=500)
    ap.add_argument("--top-names", type=int, default=50, help="per entity category per volume")
    ap.add_argument("--keep-raw", action="store_true", help="keep .entities/.quotes/.supersense/.book (NOT .tokens)")
    ap.add_argument("--retry-failed", action="store_true", help="re-run volumes that have a .failed marker")
    ap.add_argument("--allow-unsafe-out", action="store_true")
    ap.add_argument("--finalize", action="store_true", help="stack per-volume tables into release/")
    ap.add_argument("--metadata", help="csv with htid,year (finalize only)")
    ap.add_argument("--cutoffs", default=",".join(map(str, DEFAULT_CUTOFFS)))
    ap.add_argument("--top-lemmas-release", type=int, default=100, help="event lemmas per volume in the release table")
    ap.add_argument("--top-lemmas-era", type=int, default=3000)
    ap.add_argument("--top-names-release", type=int, default=100, help="entity names per volume in the release table")
    ap.add_argument("--top-names-era", type=int, default=500)
    args = ap.parse_args()
    if args.finalize:
        finalize(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
