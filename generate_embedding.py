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
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
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


def inverse_square_rank_distances(similarities: np.ndarray) -> np.ndarray:
    """Map cosine to normalized inverse of the square-rank-largest curve."""
    count = len(similarities)
    rows, columns = np.triu_indices(count, 1)
    values = similarities[rows, columns]
    descending_ranks = rankdata(-values, method="average").astype(np.float32)
    targets = np.sqrt(descending_ranks / len(values), dtype=np.float32)
    distances = np.zeros((count, count), dtype=np.float32)
    distances[rows, columns] = targets
    distances[columns, rows] = targets
    return distances


def precompute_force_layout(similarities: np.ndarray, initial: np.ndarray) -> tuple[np.ndarray, dict]:
    """Run the browser force equations over every word until motion settles."""
    targets = inverse_square_rank_distances(similarities)
    count = len(targets)
    ideal_distance = max(12.0, 82.0 * math.sqrt(42 / max(42, count)))
    semantic_weights = 1 / (targets + 1e-3)
    np.fill_diagonal(semantic_weights, 0)

    # Match stepPhysics(): centered acceleration, exact repulsion, dynamic
    # all-pairs semantic stress, velocity damping, and cooling. The t-SNE seed
    # starts near the target-distance scale but does not constrain the result.
    positions = np.asarray(initial, dtype=np.float64) * (ideal_distance * 0.5)
    positions -= np.mean(positions, axis=0)
    jitter_angles = np.arange(count) * 2.399963229728653
    positions += np.column_stack((np.cos(jitter_angles), np.sin(jitter_angles))) * (ideal_distance * 1e-4)
    velocities = np.zeros_like(positions)
    heat = 1.0
    strength = 0.04 / count
    batch_size = 256
    settled_steps = 0
    required_settled_steps = 40
    velocity_tolerance = ideal_distance * 2e-3
    upper_velocity_tolerance = ideal_distance * 8e-3
    iterations = 0

    while settled_steps < required_settled_steps:
        forces = -positions * 0.0014
        heat_factor = 0.45 + heat * 0.55

        for left in range(0, count, batch_size):
            right = min(left + batch_size, count)
            dx = positions[np.newaxis, :, 0] - positions[left:right, np.newaxis, 0]
            dy = positions[np.newaxis, :, 1] - positions[left:right, np.newaxis, 1]
            distances_squared = dx * dx + dy * dy
            local_rows = np.arange(right - left)
            global_rows = np.arange(left, right)
            distances_squared[local_rows, global_rows] = np.inf
            safe_distances_squared = np.maximum(distances_squared, 1e-4)
            distances = np.sqrt(safe_distances_squared)

            repulsion = np.minimum(
                0.04,
                (ideal_distance * ideal_distance / safe_distances_squared) * 0.018,
            ) * heat
            forces[left:right, 0] -= np.sum(dx * repulsion, axis=1)
            forces[left:right, 1] -= np.sum(dy * repulsion, axis=1)

            normalized_targets = targets[left:right]
            errors = distances - ideal_distance * normalized_targets
            errors[local_rows, global_rows] = 0
            raw_proximity = ideal_distance / (distances + ideal_distance * 0.05)
            proximity = np.clip(raw_proximity, 0.35, 4.0)
            proximity_derivative = np.where(
                raw_proximity == proximity,
                -ideal_distance / np.square(distances + ideal_distance * 0.05),
                0,
            )
            combined_weights = semantic_weights[left:right] + proximity
            weighted_errors = combined_weights * errors + 0.5 * proximity_derivative * errors * errors
            semantic_force = weighted_errors * strength * heat_factor
            forces[left:right, 0] += np.sum(dx / distances * semantic_force, axis=1)
            forces[left:right, 1] += np.sum(dy / distances * semantic_force, axis=1)

        velocities = (velocities + forces) * 0.86
        positions += velocities
        heat = max(0.09, heat * 0.992)
        iterations += 1

        speeds = np.linalg.norm(velocities, axis=1)
        rms_velocity = float(np.sqrt(np.mean(speeds * speeds)))
        upper_velocity = float(np.percentile(speeds, 99))
        maximum_velocity = float(np.max(speeds))
        if rms_velocity < velocity_tolerance and upper_velocity < upper_velocity_tolerance:
            settled_steps += 1
        else:
            settled_steps = 0
        if not np.all(np.isfinite(positions)):
            raise RuntimeError("Full force layout diverged before settling")
        if iterations % 100 == 0:
            print(
                f"Force layout iteration {iterations}: "
                f"RMS velocity {rms_velocity:.6f}, p99 {upper_velocity:.6f}, "
                f"max {maximum_velocity:.6f}",
                flush=True,
            )

    diagnostics = {
        "iterations": iterations,
        "settledSteps": settled_steps,
        "requiredSettledSteps": required_settled_steps,
        "rmsVelocity": round(rms_velocity, 8),
        "p99Velocity": round(upper_velocity, 8),
        "maximumVelocity": round(maximum_velocity, 8),
        "velocityTolerance": velocity_tolerance,
        "p99VelocityTolerance": upper_velocity_tolerance,
        "finalHeat": heat,
        "repulsion": "exact O(n^2)",
        "semanticForce": "exact O(n^2)",
        "allNodesMovable": True,
        "pairWeight": "1 / (target + 0.001) + clamp(ideal / (current_distance + 0.05 * ideal), 0.35, 4)",
    }
    layout = positions
    layout -= np.median(layout, axis=0)
    radius = np.percentile(np.linalg.norm(layout, axis=1), 99)
    diagnostics["normalizationRadius"] = round(float(radius), 8)
    return layout / max(radius, 1e-6), diagnostics


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
    force_layout, force_diagnostics = precompute_force_layout(similarities, projection)
    write_similarity_payload(similarities)
    similarity_order = similarity_order_curves(similarities)
    np.fill_diagonal(similarities, -np.inf)

    points = []
    for idx, ((word, pos), xy, force_xy) in enumerate(zip(entries, projection, force_layout)):
        nearest = np.argpartition(-similarities[idx], 8)[:8]
        nearest = nearest[np.argsort(-similarities[idx, nearest])]
        points.append({
            "w": word,
            "p": pos,
            "x": round(float(xy[0]), 5),
            "y": round(float(xy[1]), 5),
            "fx": round(float(force_xy[0]), 5),
            "fy": round(float(force_xy[1]), 5),
            "c": int(labels[idx]),
            "nn": [int(value) for value in nearest],
            "ns": [round(float(similarities[idx, value]), 5) for value in nearest],
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
            "graphLayout": "settled exact all-pairs force simulation",
            "graphDiagnostics": force_diagnostics,
        },
        "points": points,
        "clusters": cluster_labels,
        "similarityOrder": similarity_order,
    }
    OUTPUT.write_text(
        "window.EMBEDDING_DATA=" + json.dumps(payload, separators=(",", ":")) + ";\n"
    )
    print(
        f"Wrote {OUTPUT.name}: {len(entries)} points, "
        f"{estimated.sum()} morphology-estimated, {OUTPUT.stat().st_size / 1024:.0f} KiB"
    )


if __name__ == "__main__":
    main()
