# Dwarf Fortress Lexical Field

An interactive semantic map of 2,187 words used by Dwarf Fortress's procedural
name generator. Search, filter, pan, zoom, and inspect the nearest meanings in a
two-dimensional projection of GloVe word embeddings.

## Run locally

The generated embedding payload is committed, so the site has no build step:

```sh
python -m http.server 8000
```

Then open <http://localhost:8000>.

## Regenerate the embedding

Install the Python dependencies and provide a GloVe-format vector file. The
50-dimensional `glove.6B.50d.txt` model matches the checked-in payload.

```sh
python -m pip install -r requirements.txt
python generate_embedding.py /path/to/glove.6B.50d.txt
```

The generator keeps every source term visible. Words missing from the supplied
vector file receive a morphology-based estimated position, which remains marked
in the generated data and interface.
