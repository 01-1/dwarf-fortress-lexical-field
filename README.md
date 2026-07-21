# Dwarf Fortress Lexical Field

An interactive semantic map of 2,187 words used by Dwarf Fortress's procedural
name generator. It has four linked views:

- **Embedding map** — the default original t-SNE projection with filters,
  cluster labels, zooming, and nearest-meaning inspection.
- **Physics demo** — an all-pairs target-distance stress model integrated live
  in JavaScript over the selected semantic field. Every pair gets distance
  `sqrt(descending_rank / pair_count)`, the normalized inverse of the global
  square-rank-largest cosine curve, weighted by `1 / (distance + 0.001)` so
  semantically close-pair errors matter more. A second bounded inverse-distance
  weight updates every frame and is added to the Sammon weight, prioritizing
  pairs currently closest in the evolving 2D layout. Every word is draggable,
  and the field-size slider expands from 10 words to all 2,187.
  Additional repulsion can be selected as exact O(n²) or Barnes–Hut
  O(n log n), with automatic selection as the default; semantic stress itself
  remains O(n²). The canvas supports cursor-centered wheel zoom and shift-drag
  panning.
- **Settled force** — a static snapshot made by running the live force equations
  over all 2,187 movable words, using exact repulsion and exact semantic forces
  across all 2,390,391 unique non-self pairs. It stops after satisfying a
  sustained low-velocity settling condition, not an iteration limit. Only
  nearest-neighbor links are drawn for orientation; they do not define the
  layout.
- **Similarity curve** — the operative descending order-statistic curve
  `f(k) = cosine of the k²-th largest pair`, plus an ordinary ascending
  `g(r) = cosine of the r-th smallest pair` shown for reference only. Both
  exclude self-pairs.

Search follows the active tab, so choosing a word selects it in whichever view
is open.

## Run locally

The generated embedding payload is committed and the site has no build step or
server requirement. Open `index.html` directly in a browser.

## Regenerate the embedding

Install the Python dependencies and provide a GloVe-format vector file. The
50-dimensional `glove.6B.50d.txt` model matches the checked-in payload. Node.js
is also required to run the same `force-model.js` implementation used by the
browser when baking the settled-force coordinates.

```sh
python -m pip install -r requirements.txt
python generate_embedding.py /path/to/glove.6B.50d.txt
```

The generator keeps every source term visible. Words missing from the supplied
vector file receive a morphology-based estimated position, which remains marked
in the generated data and interface.
