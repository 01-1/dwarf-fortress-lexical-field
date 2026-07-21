#!/usr/bin/env python3
"""Build the browser-ready semantic map from df_name_words.txt.

Usage:
    python generate_embedding.py /path/to/glove-wiki-gigaword-50.gz

The input vector file is the word2vec-formatted GloVe 6B 50d release from
gensim-data. The generated data file is intentionally dependency-free in the
browser, so index.html can also be opened directly from disk.
"""

from __future__ import annotations

import gzip
import base64
import json
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Ridge
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "df_name_words.txt"
OUTPUT = ROOT / "embedding-data.js"
SIMILARITY_OUTPUT = ROOT / "all-pair-similarity.js"
SEED = 42


def read_words() -> tuple[list[tuple[str, str]], Counter]:
    parsed: list[tuple[str, str]] = []
    pattern = re.compile(r"^(.+?) \((n|v|adj|\?)\)$")
    for line_number, raw in enumerate(SOURCE.read_text().splitlines(), 1):
        if not raw.strip():
            continue
        match = pattern.fullmatch(raw.strip())
        if not match:
            raise ValueError(f"Malformed line {line_number}: {raw!r}")
        parsed.append((match.group(1).lower(), match.group(2)))

    counts = Counter(parsed)
    # dict preserves first appearance, which makes generated output stable.
    return list(dict.fromkeys(parsed)), counts


def lookup_candidates(word: str) -> list[str]:
    candidates = {word, word.replace("-", ""), word.replace("-", "_")}
    candidates.update(part for part in re.split(r"[-_]", word) if part)
    suffixes = (
        "lessness", "fulness", "ation", "ition", "iveness", "ousness",
        "ability", "ibility", "ically", "ingly", "edly", "ness", "ment",
        "ship", "hood", "ance", "ence", "ality", "ility", "icity", "ious",
        "less", "ful", "ous", "ive", "able", "ible", "ing", "ers", "er",
        "est", "ied", "ies", "ed", "ly", "ity", "s",
    )
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            stem = word[: -len(suffix)]
            candidates.add(stem)
            candidates.add(stem + "e")
            if suffix in {"ied", "ies"}:
                candidates.add(stem + "y")
    return sorted(candidates)


def load_needed_vectors(path: Path, words: list[str]) -> dict[str, np.ndarray]:
    needed = set().union(*(lookup_candidates(word) for word in words))
    found: dict[str, np.ndarray] = {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
        first = handle.readline().split()
        if len(first) != 2 or not all(piece.isdigit() for piece in first):
            handle.seek(0)
        for line in handle:
            pieces = line.rstrip().split()
            if len(pieces) < 3 or pieces[0] not in needed:
                continue
            found[pieces[0]] = np.asarray(pieces[1:], dtype=np.float32)
    return found


def resolve_vector(word: str, vectors: dict[str, np.ndarray]) -> np.ndarray | None:
    if word in vectors:
        return vectors[word]
    parts = [vectors[p] for p in re.split(r"[-_]", word) if p in vectors]
    if parts:
        return np.mean(parts, axis=0)
    candidates = [c for c in lookup_candidates(word) if c in vectors]
    return vectors[candidates[0]] if candidates else None


def make_embeddings(entries: list[tuple[str, str]], raw: dict[str, np.ndarray]):
    words = [word for word, _ in entries]
    resolved = [resolve_vector(word, raw) for word in words]
    known_mask = np.array([vector is not None for vector in resolved])

    # Estimate rare coined/derived forms from spelling. This keeps every source
    # word visible while the "estimated" flag remains available to the UI.
    char_features = HashingVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), n_features=2048,
        alternate_sign=False, norm="l2",
    ).transform(words)
    known_vectors = np.vstack([v for v in resolved if v is not None])
    regressor = Ridge(alpha=7.5).fit(char_features[known_mask], known_vectors)
    estimates = regressor.predict(char_features[~known_mask])
    estimate_index = 0
    complete = []
    for vector in resolved:
        if vector is None:
            complete.append(estimates[estimate_index])
            estimate_index += 1
        else:
            complete.append(vector)

    matrix = np.asarray(complete, dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True).clip(min=1e-8)
    return matrix, ~known_mask


def project_and_cluster(matrix: np.ndarray):
    projection = TSNE(
        n_components=2,
        perplexity=36,
        learning_rate="auto",
        max_iter=1500,
        init="pca",
        metric="cosine",
        random_state=SEED,
    ).fit_transform(matrix)

    # Normalize around the robust 99th-percentile radius so extreme points do
    # not make the main cloud unreadably small.
    projection -= np.median(projection, axis=0)
    radius = np.percentile(np.linalg.norm(projection, axis=1), 99)
    projection /= max(radius, 1e-6)

    model = KMeans(n_clusters=16, n_init=20, random_state=SEED).fit(matrix)
    return projection, model.labels_, model.cluster_centers_


def spread_homographs(entries: list[tuple[str, str]], projection: np.ndarray) -> None:
    """Give POS variants of the same spelling a tiny, deterministic separation."""
    groups: dict[str, list[int]] = {}
    for index, (word, _pos) in enumerate(entries):
        groups.setdefault(word, []).append(index)
    for members in groups.values():
        if len(members) < 2:
            continue
        for offset, index in enumerate(members):
            angle = -math.pi / 2 + (2 * math.pi * offset / len(members))
            projection[index] += np.array([math.cos(angle), math.sin(angle)]) * 0.018


def write_similarity_payload(similarities: np.ndarray) -> None:
    """Store the upper triangle as 8-bit cosine values for live all-pair forces."""
    rows, columns = np.triu_indices(len(similarities), 1)
    encoded = np.rint((np.clip(similarities[rows, columns], -1, 1) + 1) * 127.5).astype(np.uint8)
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    SIMILARITY_OUTPUT.write_text(
        "window.ALL_PAIR_SIMILARITY="
        + json.dumps({"count": len(similarities), "encoding": "cosine-u8-upper", "data": payload}, separators=(",", ":"))
        + ";\n"
    )


def similarity_order_curves(similarities: np.ndarray) -> dict:
    """Build the operative square-largest curve and display-only ascending CDF."""
    rows, columns = np.triu_indices(len(similarities), 1)
    ordered = np.sort(similarities[rows, columns])
    maximum_k = math.isqrt(len(ordered))
    square_descending_indices = len(ordered) - np.arange(1, maximum_k + 1, dtype=np.int64) ** 2
    ascending_sample_count = min(4096, len(ordered))
    ascending_indices = np.rint(np.linspace(0, len(ordered) - 1, ascending_sample_count)).astype(np.int64)
    return {
        "pairCount": len(ordered),
        "maximumK": maximum_k,
        "squareLargest": [round(float(value), 5) for value in ordered[square_descending_indices]],
        "ascending": [round(float(value), 5) for value in ordered[ascending_indices]],
    }


def choose_cluster_labels(
    entries: list[tuple[str, str]], matrix: np.ndarray,
    labels: np.ndarray, centers: np.ndarray,
) -> list[dict]:
    result = []
    for cluster_id, center in enumerate(centers):
        members = np.flatnonzero(labels == cluster_id)
        scores = matrix[members] @ (center / max(np.linalg.norm(center), 1e-8))
        ranked = members[np.argsort(-scores)]
        names = []
        for idx in ranked:
            word = entries[int(idx)][0]
            if word not in names:
                names.append(word)
            if len(names) == 3:
                break
        result.append({"id": cluster_id, "name": " · ".join(names), "size": len(members)})
    return result


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python generate_embedding.py VECTOR_FILE[.gz]")
    vector_path = Path(sys.argv[1]).expanduser()
    if not vector_path.exists():
        raise SystemExit(f"Vector file does not exist: {vector_path}")

    entries, counts = read_words()
    raw_vectors = load_needed_vectors(vector_path, [word for word, _ in entries])
    matrix, estimated = make_embeddings(entries, raw_vectors)
    projection, labels, centers = project_and_cluster(matrix)
    spread_homographs(entries, projection)
    similarities = cosine_similarity(matrix)
    write_similarity_payload(similarities)
    similarity_order = similarity_order_curves(similarities)
    np.fill_diagonal(similarities, -np.inf)

    points = []
    approximate_candidate_count = min(
        len(entries) - 1,
        4 * math.ceil(math.sqrt(len(entries))),
    )
    for idx, ((word, pos), xy) in enumerate(zip(entries, projection)):
        approximate_nearest = np.argpartition(
            -similarities[idx], approximate_candidate_count - 1,
        )[:approximate_candidate_count]
        approximate_nearest = approximate_nearest[
            np.argsort(-similarities[idx, approximate_nearest])
        ]
        nearest = approximate_nearest[:8]
        points.append({
            "w": word,
            "p": pos,
            "x": round(float(xy[0]), 5),
            "y": round(float(xy[1]), 5),
            "fx": round(float(xy[0]), 5),
            "fy": round(float(xy[1]), 5),
            "c": int(labels[idx]),
            "nn": [int(value) for value in nearest],
            "ns": [round(float(similarities[idx, value]), 5) for value in nearest],
            "an": [int(value) for value in approximate_nearest],
            "count": counts[(word, pos)],
            "estimated": bool(estimated[idx]),
        })

    cluster_labels = choose_cluster_labels(entries, matrix, labels, centers)
    payload = {
        "meta": {
            "source": SOURCE.name,
            "sourceRows": sum(counts.values()),
            "uniqueEntries": len(entries),
            "estimatedEntries": int(estimated.sum()),
            "model": "GloVe 6B · 50 dimensions",
            "projection": "t-SNE · cosine distance",
            "graphLayout": "pending shared JavaScript force simulation",
            "graphDiagnostics": {},
        },
        "points": points,
        "clusters": cluster_labels,
        "similarityOrder": similarity_order,
    }
    OUTPUT.write_text(
        "window.EMBEDDING_DATA=" + json.dumps(payload, separators=(",", ":")) + ";\n"
    )
    subprocess.run(["node", str(ROOT / "settle-force.js")], cwd=ROOT, check=True)
    print(
        f"Wrote {OUTPUT.name}: {len(entries)} points, "
        f"{estimated.sum()} morphology-estimated, {OUTPUT.stat().st_size / 1024:.0f} KiB"
    )


if __name__ == "__main__":
    main()
