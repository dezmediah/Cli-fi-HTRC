#!/usr/bin/env python3
"""
Export release-safe tables from the era word2vec models trained by SF_word2vec_eras_v2.ipynb.
Run in secure mode right after training:

    python w2v_export_v2.py --models /media/secure_volume/v_2_word2vec_models --out /media/secure_volume/out_w2v_v2

Both paths must be under /media/secure_volume/. Outputs: neighbours/, pairwise/, tables/,
matrix/<era>/ and MANIFEST.json. Everything is aggregate (types, cosines, counts, a PCA-reduced
type matrix); no running text. Release all of it in ONE `releaseresults add`.
"""
import argparse, csv, json, itertools, math
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec
from scipy.linalg import orthogonal_procrustes

# Copied from SF_word2vec_eras_v2.ipynb; keep the two in sync.
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
WORD_PAIR_CONTRASTS = [
    ("earth", "home", "earth", "resource"),
    ("nature", "machine", "nature", "technology"),
    ("earth", "garden", "earth", "wasteland"),
    ("disaster", "nuclear", "disaster", "industrial"),
    ("disaster", "mankind", "disaster", "cosmic"),
    ("catastrophe", "humanity", "catastrophe", "solar"),
    ("wasteland", "radioactive", "wasteland", "polluted"),
]
ERA_LABELS = ["era_a", "era_b", "era_c"]
SEEDS = [1, 2, 3]
GATE_BYTES = 1_000_000


def load_models(models_dir: Path):
    out = {}
    for era in ERA_LABELS:
        for seed in SEEDS:
            p = models_dir / f"sf_model_{era}_seed{seed}"
            if p.exists():
                out[(era, seed)] = Word2Vec.load(str(p))
    return out


def write_csv(path: Path, rows, fields):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  {path.relative_to(path.parents[1])}: {len(rows):,} rows")


def neighbours(primary, lexicon, topn):
    rows = []
    for era, m in primary.items():
        for group, word in lexicon:
            if word not in m.wv:
                continue
            for rank, (nb, cos) in enumerate(m.wv.most_similar(word, topn=topn), 1):
                rows.append(dict(era=era, word=word, group=group, rank=rank, neighbour=nb, cosine=round(float(cos), 4)))
    return rows


def pairwise(primary, words):
    rows = []
    for era, m in primary.items():
        present = [w for w in words if w in m.wv]
        for a, b in itertools.combinations(present, 2):
            rows.append(dict(era=era, word_a=a, word_b=b, cosine=round(float(m.wv.similarity(a, b)), 4)))
    return rows


def contrasts(primary):
    rows = []
    for a, b1, _, b2 in WORD_PAIR_CONTRASTS:
        for era, m in primary.items():
            s1 = float(m.wv.similarity(a, b1)) if a in m.wv and b1 in m.wv else None
            s2 = float(m.wv.similarity(a, b2)) if a in m.wv and b2 in m.wv else None
            rows.append(dict(era=era, word_a=a, b1=b1, b2=b2,
                             sim_a_b1=None if s1 is None else round(s1, 4),
                             sim_a_b2=None if s2 is None else round(s2, 4),
                             diff=None if None in (s1, s2) else round(s1 - s2, 4)))
    return rows


def stability(models, words, topn=10):
    """Her neighbor_stability(): mean pairwise Jaccard of top-N neighbour sets across seeds."""
    rows = []
    for era in ERA_LABELS:
        for word in words:
            sets = []
            for seed in SEEDS:
                m = models.get((era, seed))
                if m is not None and word in m.wv:
                    sets.append({w for w, _ in m.wv.most_similar(word, topn=topn)})
            if len(sets) < 2:
                continue
            jac = [len(a & b) / len(a | b) for a, b in itertools.combinations(sets, 2)]
            rows.append(dict(era=era, word=word, jaccard_stability=round(sum(jac) / len(jac), 4), n_seeds=len(sets)))
    return rows


def freq(primary, words):
    """Counts come from the model's own vocabulary (min_count-filtered), totals from
    corpus_total_words, which gensim records at build time. Not identical to the
    notebook's sentence-level freq_df, but computed from the same training input."""
    rows, counts = [], {}
    for era, m in primary.items():
        total = m.corpus_total_words or 1
        for word in words:
            c = int(m.wv.get_vecattr(word, "count")) if word in m.wv else 0
            counts[(era, word)] = (c, total)
            rows.append(dict(era=era, word=word, count=c, per_million=round(c / total * 1e6, 2)))
    return rows, counts


def g2(counts, words):
    """Dunning log-likelihood for a word between two eras, so a neighbour shift can be
    checked against whether the word's raw frequency also moved."""
    rows = []
    eras = [e for e in ERA_LABELS if any((e, w) in counts for w in words)]
    for x, y in itertools.combinations(eras, 2):
        for word in words:
            if (x, word) not in counts or (y, word) not in counts:
                continue
            a, na = counts[(x, word)]
            b, nb = counts[(y, word)]
            if a + b == 0:
                continue
            e1 = na * (a + b) / (na + nb)
            e2 = nb * (a + b) / (na + nb)
            ll = 2 * sum(o * math.log(o / e) for o, e in ((a, e1), (b, e2)) if o > 0)
            rows.append(dict(word=word, era_x=x, era_y=y, count_x=a, count_y=b, g2=round(ll, 2),
                             direction="up" if b / nb > a / na else "down"))
    return rows


def semantic_shift(primary, words, base_era):
    """Her align_vectors() + semantic_shift(): orthogonal Procrustes over shared vocabulary,
    then cosine distance moved per lexicon word."""
    rows = []
    base = primary[base_era]
    for era, other in primary.items():
        if era == base_era:
            continue
        shared = [w for w in base.wv.index_to_key if w in other.wv]
        B = np.array([base.wv[w] for w in shared])
        O = np.array([other.wv[w] for w in shared])
        R, _ = orthogonal_procrustes(O, B)
        for word in words:
            if word not in base.wv or word not in other.wv:
                continue
            v1, v2 = base.wv[word], other.wv[word] @ R
            cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            rows.append(dict(word=word, base_era=base_era, other_era=era, cosine_distance=round(1 - cos, 4),
                             shared_vocab=len(shared)))
    return rows


def matrix(primary, out_dir: Path, top_k: int, dims: int):
    """Top-K most frequent types, PCA-reduced from 300d to `dims`, float16. gensim keeps
    index_to_key sorted by count, so the first K rows are the K most frequent."""
    sizes = {}
    for era, m in primary.items():
        vocab = m.wv.index_to_key[:top_k]
        X = np.array([m.wv[w] for w in vocab], dtype=np.float32)
        X = X - X.mean(axis=0)
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        Z = (X @ Vt[:dims].T).astype(np.float16)
        d = out_dir / era
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "matrix.npy", Z)
        (d / "vocab.txt").write_text("\n".join(vocab) + "\n")
        sizes[era] = dict(rows=len(vocab), dims=dims)
        print(f"  matrix/{era}: {len(vocab):,} x {dims} float16")
    return sizes


def dir_bytes(p: Path):
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="dir holding sf_model_{era}_seed{seed}")
    ap.add_argument("--out", required=True, help="export dir (tables/ and matrix/ created inside)")
    ap.add_argument("--topn", type=int, default=50, help="neighbours per lexicon word")
    ap.add_argument("--top-k", type=int, default=5000, help="types kept in the matrix")
    ap.add_argument("--dims", type=int, default=50, help="PCA dims for the matrix")
    ap.add_argument("--base-era", default="era_a", help="alignment target for semantic_shift")
    a = ap.parse_args()

    models_dir, out = Path(a.models), Path(a.out)
    for p in (models_dir, out):
        if not str(p.resolve()).startswith("/media/secure_volume"):
            print(f"WARNING: {p} is not under /media/secure_volume/ -- it will not survive a mode switch")

    models = load_models(models_dir)
    if not models:
        raise SystemExit(f"no sf_model_* found under {models_dir}")
    primary = {era: models[(era, SEEDS[0])] for era in ERA_LABELS if (era, SEEDS[0]) in models}
    print(f"loaded {len(models)} models; primary eras: {list(primary)}")

    lexicon = [(g, w) for g, ws in ENV_WORD_GROUPS.items() for w in ws]
    words = [w for _, w in lexicon]

    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    print("tables/")
    write_csv(tables / "lexicon.csv", [dict(group=g, word=w) for g, w in lexicon], ["group", "word"])
    nb_dir, pw_dir = out / "neighbours", out / "pairwise"
    nb_dir.mkdir(exist_ok=True); pw_dir.mkdir(exist_ok=True)
    write_csv(nb_dir / "neighbours.csv", neighbours(primary, lexicon, a.topn),
              ["era", "word", "group", "rank", "neighbour", "cosine"])
    write_csv(pw_dir / "pairwise_cosine.csv", pairwise(primary, words), ["era", "word_a", "word_b", "cosine"])
    write_csv(tables / "pair_contrasts.csv", contrasts(primary),
              ["era", "word_a", "b1", "b2", "sim_a_b1", "sim_a_b2", "diff"])
    write_csv(tables / "stability.csv", stability(models, words), ["era", "word", "jaccard_stability", "n_seeds"])
    frows, counts = freq(primary, words)
    write_csv(tables / "freq.csv", frows, ["era", "word", "count", "per_million"])
    write_csv(tables / "freq_shift_g2.csv", g2(counts, words),
              ["word", "era_x", "era_y", "count_x", "count_y", "g2", "direction"])
    if a.base_era in primary:
        write_csv(tables / "semantic_shift.csv", semantic_shift(primary, words, a.base_era),
                  ["word", "base_era", "other_era", "cosine_distance", "shared_vocab"])

    msizes = matrix(primary, out / "matrix", a.top_k, a.dims)

    manifest = dict(
        params=dict(topn=a.topn, top_k=a.top_k, dims=a.dims, base_era=a.base_era, seeds=SEEDS,
                    models_loaded=[f"{e}_seed{s}" for e, s in sorted(models)],
                    vocab_size={e: len(m.wv) for e, m in primary.items()},
                    corpus_total_words={e: int(m.corpus_total_words) for e, m in primary.items()}),
        matrix=msizes,
        bytes={"tables": dir_bytes(tables), "neighbours": dir_bytes(nb_dir), "pairwise": dir_bytes(pw_dir),
               **{f"matrix/{e}": dir_bytes(out / "matrix" / e) for e in msizes}},
        gate_bytes=GATE_BYTES,
    )
    manifest["over_gate"] = [k for k, v in manifest["bytes"].items() if v > GATE_BYTES]
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    print("\nrelease directories (the ~1 MB threshold is a REVIEW conversation, not a hard cap;")
    print("the tool itself only refuses at 1 GB):")
    for k, v in manifest["bytes"].items():
        print(f"  {k:<14} {v/1e6:5.2f} MB  {'over ~1 MB - flag to htrc-help' if v > GATE_BYTES else 'ok'}")
    # `releaseresults add` is not cumulative: every path goes in ONE add.
    print("\nrelease with ONE add (repeated `add` calls OVERWRITE, they do not accumulate):")
    print(f"  releaseresults add {out}/tables {out}/neighbours {out}/pairwise {out}/matrix")
    print("  releaseresults done")
    print("\nnothing here contains running text: types, cosines, counts, and a PCA'd type matrix only.")


if __name__ == "__main__":
    main()
