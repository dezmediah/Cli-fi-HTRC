#!/usr/bin/env python
"""
Three install-free extractions over the capsule corpus, under the default interpreter. Counts only.

    python extras_analysis.py ocr --input /media/secure_volume/fa50b375-3216-4edd-a685-98488562b723 --metadata ~/metadata_august2026.csv \
        --out /media/secure_volume/out_extras --procs 40
    python extras_analysis.py pmi ...
    python extras_analysis.py lda ... --topics 50 --chunk-words 1000

ocr: per-page dictionary-hit rate, summarised per volume. Run it first; it takes minutes.
pmi: per-era PMI/NPMI collocates (window 5) for the environmental lexicon, plus frequency per million.
lda: gensim LdaMulticore baseline on 1,000-word chunks; topic terms and per-volume / per-era means.
Release dirs are <out>/{ocr,pmi,lda}; <out>/work is never released. One `releaseresults add`.
"""

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
DEFAULT_CUTOFFS = (1962, 1972)


# ----------------------------------------------------------------------------- shared

DEFAULT_LEXICON = Path.home() / "Desktop" / "Clifi-htrc" / "notebooks" / "SF_keyword_context_v4.ipynb"


def lexicon(path=None):
    """ENV_WORD_GROUPS (+ TECH_WORD_GROUPS if present) read out of a notebook or .py file --
    by default Dez's KWIC v4 notebook, the 101 + 24 word lexicon. Falls back to w2v_export_v2."""
    path = Path(path or DEFAULT_LEXICON)
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".ipynb":
            text = "\n".join("".join(c["source"]) for c in json.loads(text)["cells"] if c["cell_type"] == "code")
        ns, out = {}, {}
        for name in ("ENV_WORD_GROUPS", "TECH_WORD_GROUPS"):
            m = re.search(r"^" + name + r"\s*=\s*\{.*?^\}", text, re.S | re.M)
            if m:
                exec(m.group(0), ns)
                out.update({w: g for g, ws in ns[name].items() for w in ws})
        if out:
            return out
    try:
        try:
            from w2v_export_v2 import ENV_WORD_GROUPS      # the current lexicon (WordNet-checked)
        except ImportError:
            from w2v_export import ENV_WORD_GROUPS
        return {w: g for g, ws in ENV_WORD_GROUPS.items() for w in ws}
    except Exception:
        from in_capsule_analysis import CLIMATE_TERMS
        return {w: "climate" for w in CLIMATE_TERMS}


def stopwords():
    try:
        from nltk.corpus import stopwords as sw
        return set(sw.words("english"))
    except Exception:
        return set("""a an the and or but if of to in on at by for with from as is are was were be been
        being it its this that these those he she they them his her their we you i me my our your not no
        so than then there here which who whom what when where why how all any some such very can will
        would could should may might must do does did have has had into out up down over under again
        off only own same too s t just now about above after before below between both each few more
        most other because until while during through""".split())


def load_metadata(path, cutoffs):
    meta = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                year = int(str(row.get("year", "")).strip()[:4])
            except ValueError:
                continue
            idx = sum(year >= c for c in cutoffs)
            meta[row["htid"]] = {"year": year, "era": "era_" + "abcdefgh"[idx]}
    return meta


def list_volumes(input_dir, limit=None, workset=None):
    ids = sorted(p.name for p in Path(input_dir).iterdir() if p.is_dir() and not p.name.startswith("."))
    if workset:
        keep = {l.split()[0] for l in Path(workset).read_text().splitlines() if l.strip()}
        ids = [h for h in ids if h in keep]
    return ids[:limit] if limit else ids


def pages_of(input_dir, htid):
    for p in sorted((Path(input_dir) / htid).glob("*.txt")):
        yield p.name, p.read_text(errors="ignore")


def tokens_of(text):
    return TOKEN_RE.findall(text.lower())


def write_csv(path, header, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def report(out_sub):
    files = sorted(p for p in Path(out_sub).rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    print(f"{out_sub}: {len(files)} files, {total/1e6:.2f} MB")
    for p in files:
        print(f"  {p.relative_to(out_sub)!s:36s} {p.stat().st_size/1e3:9.1f} KB")


_G = {}


def _pool(procs, init=None, initargs=()):
    import multiprocessing as mp
    return mp.get_context("fork").Pool(procs, initializer=init, initargs=initargs)


# ----------------------------------------------------------------------------- pmi

def _pmi_init(input_dir, targets, window):
    _G.update(input_dir=input_dir, targets=set(targets), window=window)


def _pmi_volume(htid):
    uni = Counter()
    co = defaultdict(Counter)
    W = _G["window"]
    T = _G["targets"]
    for _, text in pages_of(_G["input_dir"], htid):
        toks = tokens_of(text)
        uni.update(toks)
        for i, t in enumerate(toks):
            if t in T:
                lo, hi = max(0, i - W), min(len(toks), i + W + 1)
                c = co[t]
                for j in range(lo, hi):
                    if j != i:
                        c[toks[j]] += 1
    return htid, uni, co


def cmd_pmi(args):
    meta = load_metadata(args.metadata, args.cutoffs)
    lex = lexicon(args.lexicon)
    sw = stopwords()
    ids = [h for h in list_volumes(args.input, args.limit, args.workset) if h in meta]
    print(f"pmi: {len(ids)} volumes, {len(lex)} targets, window ±{args.window}, {args.procs} procs", flush=True)
    era_uni = defaultdict(Counter)
    era_co = defaultdict(lambda: defaultdict(Counter))
    era_vols = Counter()
    t0 = time.time()
    with _pool(args.procs, _pmi_init, (args.input, list(lex), args.window)) as pool:
        for n, (htid, uni, co) in enumerate(pool.imap_unordered(_pmi_volume, ids, chunksize=4), 1):
            era = meta[htid]["era"]
            era_uni[era].update(uni)
            era_vols[era] += 1
            for t, c in co.items():
                era_co[era][t].update(c)
            if n % 50 == 0 or n == len(ids):
                print(f"  [{n}/{len(ids)}] {time.time()-t0:.0f}s", flush=True)
    out = Path(args.out) / "pmi"
    rows = []
    freq_rows = []
    totals = []
    for era in sorted(era_uni):
        uni = era_uni[era]
        N = sum(uni.values())
        totals.append((era, era_vols[era], N, len(uni)))
        for t in sorted(lex):
            ct = uni.get(t, 0)
            freq_rows.append((era, lex[t], t, ct, round(ct / N * 1e6, 3) if N else 0))
            if ct < args.min_target:
                continue
            c = era_co[era][t]
            pairs = ct * 2 * args.window
            scored = []
            for w, cw in c.items():
                if cw < args.min_cooc or w in sw or w == t or len(w) < 3 or uni[w] < args.min_word:
                    continue
                p_tw = cw / pairs
                p_w = uni[w] / N
                pmi = math.log2(p_tw / p_w)
                npmi = pmi / -math.log2(cw / N)
                scored.append((w, cw, pmi, npmi))
            scored.sort(key=lambda x: -x[2])
            rows += [(era, lex[t], t, ct, w, cw, round(pmi, 3), round(npmi, 3), r)
                     for r, (w, cw, pmi, npmi) in enumerate(scored[:args.topn], 1)]
    write_csv(out / "collocates.csv",
              ["era", "group", "target", "target_count", "word", "cooc", "pmi", "npmi", "rank"], rows)
    write_csv(out / "lexicon_freq.csv", ["era", "group", "word", "count", "per_million"], freq_rows)
    write_csv(out / "era_totals.csv", ["era", "n_volumes", "n_tokens", "n_types"], totals)
    (out / "MANIFEST.json").write_text(json.dumps({
        "window": args.window, "min_cooc": args.min_cooc, "min_word": args.min_word,
        "topn": args.topn, "cutoffs": args.cutoffs, "volumes": len(ids), "lexicon": len(lex),
        "lexicon_source": str(args.lexicon) if Path(args.lexicon).exists() else "w2v_export_v2 fallback"}, indent=1))
    report(out)


# ----------------------------------------------------------------------------- lda

def _lda_init(input_dir, chunk_words, sw):
    _G.update(input_dir=input_dir, chunk_words=chunk_words, sw=sw)


def _lda_chunks(htid):
    """Return (htid, [chunk token lists]) — filtered tokens, fixed-size chunks."""
    sw = _G["sw"]
    toks = []
    for _, text in pages_of(_G["input_dir"], htid):
        toks += [t for t in tokens_of(text) if len(t) >= 3 and t not in sw]
    n = _G["chunk_words"]
    chunks = [toks[i:i + n] for i in range(0, len(toks), n)]
    if chunks and len(chunks[-1]) < n // 2 and len(chunks) > 1:
        tail = chunks.pop()
        chunks[-1] += tail
    return htid, chunks


def cmd_lda(args):
    from gensim.corpora import Dictionary, MmCorpus
    from gensim.models import LdaMulticore
    meta = load_metadata(args.metadata, args.cutoffs)
    sw = stopwords()
    ids = [h for h in list_volumes(args.input, args.limit, args.workset) if h in meta]
    work = Path(args.out) / "work" / "lda"
    work.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) / "lda"
    print(f"lda: {len(ids)} volumes, {args.topics} topics, {args.chunk_words}-word chunks", flush=True)

    # pass 1: dictionary
    t0 = time.time()
    dictionary = Dictionary()
    n_chunks = 0
    with _pool(args.procs, _lda_init, (args.input, args.chunk_words, sw)) as pool:
        for n, (htid, chunks) in enumerate(pool.imap(_lda_chunks, ids, chunksize=2), 1):
            dictionary.add_documents(chunks)
            n_chunks += len(chunks)
            if n % 100 == 0:
                print(f"  dict [{n}/{len(ids)}] {n_chunks} chunks {time.time()-t0:.0f}s", flush=True)
    dictionary.filter_extremes(no_below=args.no_below, no_above=args.no_above, keep_n=args.keep_n)
    dictionary.compactify()
    dictionary.save(str(work / "dictionary.gensim"))
    print(f"  dictionary: {len(dictionary)} types, {n_chunks} chunks, {time.time()-t0:.0f}s", flush=True)

    # pass 2: serialise the bag-of-words corpus to disk (streaming, so RAM stays flat)
    chunk_htid = []

    def bow_stream():
        with _pool(args.procs, _lda_init, (args.input, args.chunk_words, sw)) as pool:
            for htid, chunks in pool.imap(_lda_chunks, ids, chunksize=2):
                for c in chunks:
                    chunk_htid.append(htid)
                    yield dictionary.doc2bow(c)

    MmCorpus.serialize(str(work / "corpus.mm"), bow_stream())
    corpus = MmCorpus(str(work / "corpus.mm"))
    (work / "chunk_htid.txt").write_text("\n".join(chunk_htid))
    print(f"  corpus serialised: {len(corpus)} docs, {time.time()-t0:.0f}s", flush=True)

    # train
    lda = LdaMulticore(corpus, id2word=dictionary, num_topics=args.topics, workers=max(1, args.procs - 1),
                       passes=args.passes, chunksize=2000, random_state=args.seed, eval_every=None)
    lda.save(str(work / "lda.gensim"))
    print(f"  trained: {time.time()-t0:.0f}s", flush=True)

    # topic terms
    rows = []
    for k in range(args.topics):
        for r, (w, p) in enumerate(lda.show_topic(k, topn=args.topn), 1):
            rows.append((k, r, w, round(float(p), 6)))
    write_csv(out / "topic_terms.csv", ["topic", "rank", "word", "prob"], rows)

    # doc-topic → per volume / per era means
    K = args.topics
    vol_sum = defaultdict(lambda: [0.0] * K)
    vol_n = Counter()
    for i, bow in enumerate(corpus):
        h = chunk_htid[i]
        for k, p in lda.get_document_topics(bow, minimum_probability=0.0):
            vol_sum[h][k] += float(p)
        vol_n[h] += 1
        if (i + 1) % 20000 == 0:
            print(f"  infer [{i+1}/{len(corpus)}] {time.time()-t0:.0f}s", flush=True)
    vrows = []
    era_sum = defaultdict(lambda: [0.0] * K)
    era_n = Counter()
    for h in ids:
        if vol_n[h] == 0:
            continue
        means = [s / vol_n[h] for s in vol_sum[h]]
        m = meta[h]
        vrows.append([h, m["year"], m["era"], vol_n[h]] + [round(x, 5) for x in means])
        for k in range(K):
            era_sum[m["era"]][k] += means[k]
        era_n[m["era"]] += 1
    write_csv(out / "volume_topics.csv", ["htid", "year", "era", "n_chunks"] + [f"t{k}" for k in range(K)], vrows)
    write_csv(out / "era_topics.csv", ["era", "n_volumes"] + [f"t{k}" for k in range(K)],
              [[e, era_n[e]] + [round(s / era_n[e], 5) for s in era_sum[e]] for e in sorted(era_n)])
    (out / "MANIFEST.json").write_text(json.dumps({
        "topics": K, "chunk_words": args.chunk_words, "passes": args.passes, "seed": args.seed,
        "vocab": len(dictionary), "chunks": len(corpus), "volumes": len(vrows),
        "no_below": args.no_below, "no_above": args.no_above, "keep_n": args.keep_n,
        "cutoffs": args.cutoffs, "seconds": round(time.time() - t0)}, indent=1))
    report(out)


# ----------------------------------------------------------------------------- ocr

def _ocr_init(input_dir):
    _G.update(input_dir=input_dir)


def _ocr_volume(htid):
    """Pass 1: per-page token stats + the volume's vocabulary (for the corpus-internal dictionary)."""
    pages = []
    vocab = set()
    for name, text in pages_of(_G["input_dir"], htid):
        raw = text.split()
        toks = tokens_of(text)
        vocab.update(toks)
        pages.append((name, len(raw), len(toks), Counter(toks)))
    return htid, pages, vocab


def cmd_ocr(args):
    meta = load_metadata(args.metadata, args.cutoffs)
    ids = [h for h in list_volumes(args.input, args.limit, args.workset) if h in meta]
    out = Path(args.out) / "ocr"
    work = Path(args.out) / "work" / "ocr"
    work.mkdir(parents=True, exist_ok=True)
    print(f"ocr: {len(ids)} volumes, {args.procs} procs", flush=True)

    dictionary = set(stopwords())
    try:
        from nltk.corpus import wordnet as wn
        dictionary |= {l for l in wn.all_lemma_names() if "_" not in l and l.isalpha()}
        print(f"  wordnet: {len(dictionary)} dictionary words", flush=True)
    except Exception as e:
        print(f"  [warn] wordnet unavailable ({e}); using stopwords + corpus vocabulary only", flush=True)

    t0 = time.time()
    per_vol = {}
    df = Counter()
    with _pool(args.procs, _ocr_init, (args.input,)) as pool:
        for n, (htid, pages, vocab) in enumerate(pool.imap_unordered(_ocr_volume, ids, chunksize=4), 1):
            per_vol[htid] = pages
            df.update(vocab)
            if n % 200 == 0 or n == len(ids):
                print(f"  [{n}/{len(ids)}] {time.time()-t0:.0f}s", flush=True)
    corpus_dict = {w for w, c in df.items() if c >= args.min_df}
    dictionary |= corpus_dict
    print(f"  dictionary incl. corpus words in >= {args.min_df} vols: {len(dictionary)}", flush=True)

    page_rows = []
    vol_rows = []
    for h in ids:
        pages = per_vol.get(h, [])
        shares = []
        n_empty = 0
        for name, n_raw, n_tok, cnt in pages:
            if n_tok == 0:
                n_empty += 1
                page_rows.append((h, name, n_raw, 0, "", "", ""))
                continue
            in_dict = sum(c for w, c in cnt.items() if w in dictionary)
            share = in_dict / n_tok
            alpha_share = n_tok / n_raw if n_raw else 0
            mean_len = sum(len(w) * c for w, c in cnt.items()) / n_tok
            shares.append(share)
            page_rows.append((h, name, n_raw, n_tok, round(share, 4), round(alpha_share, 4), round(mean_len, 2)))
        m = meta[h]
        if shares:
            s = sorted(shares)
            p10 = s[int(0.1 * (len(s) - 1))]
            vol_rows.append((h, m["year"], m["era"], len(pages), n_empty,
                             sum(p[2] for p in pages), round(sum(shares) / len(shares), 4),
                             round(p10, 4), round(min(shares), 4),
                             round(sum(1 for x in shares if x < args.bad_page) / len(shares), 4)))
        else:
            vol_rows.append((h, m["year"], m["era"], len(pages), n_empty, 0, "", "", "", ""))
    write_csv(out / "volume_quality.csv",
              ["htid", "year", "era", "n_pages", "n_empty_pages", "n_tokens", "dict_share_mean",
               "dict_share_p10", "dict_share_min", "share_pages_below_threshold"], vol_rows)
    write_csv(work / "page_quality.csv",
              ["htid", "page", "n_raw", "n_tokens", "dict_share", "alpha_share", "mean_token_len"], page_rows)
    (out / "MANIFEST.json").write_text(json.dumps({
        "dictionary_size": len(dictionary), "min_df": args.min_df, "bad_page_threshold": args.bad_page,
        "volumes": len(vol_rows), "pages": len(page_rows), "cutoffs": args.cutoffs,
        "page_table": str(work / "page_quality.csv") + "  (numbers only; in work/ because of size, release if wanted)"}, indent=1))
    report(out)
    print(f"  page table: {work / 'page_quality.csv'} ({(work / 'page_quality.csv').stat().st_size/1e6:.1f} MB)")


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["pmi", "lda", "ocr"])
    ap.add_argument("--input", default="/media/secure_volume/fa50b375-3216-4edd-a685-98488562b723")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out", default="/media/secure_volume/out_extras")
    ap.add_argument("--workset")
    ap.add_argument("--lexicon", default=str(DEFAULT_LEXICON), help="notebook or .py holding ENV_WORD_GROUPS (+ TECH_WORD_GROUPS)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--procs", type=int, default=40)
    ap.add_argument("--cutoffs", default=",".join(map(str, DEFAULT_CUTOFFS)))
    # pmi
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--min-cooc", type=int, default=5)
    ap.add_argument("--min-word", type=int, default=20)
    ap.add_argument("--min-target", type=int, default=10)
    ap.add_argument("--topn", type=int, default=50)
    # lda
    ap.add_argument("--topics", type=int, default=50)
    ap.add_argument("--chunk-words", type=int, default=1000)
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--no-below", type=int, default=20)
    ap.add_argument("--no-above", type=float, default=0.5)
    ap.add_argument("--keep-n", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=1)
    # ocr
    ap.add_argument("--min-df", type=int, default=10)
    ap.add_argument("--bad-page", type=float, default=0.7)
    args = ap.parse_args()
    args.cutoffs = tuple(int(c) for c in args.cutoffs.split(","))
    if not str(args.out).startswith("/media/secure_volume"):
        print(f"[warn] --out {args.out} is not on the secure volume (fine for a local test)", file=sys.stderr)
    {"pmi": cmd_pmi, "lda": cmd_lda, "ocr": cmd_ocr}[args.cmd](args)


if __name__ == "__main__":
    main()
