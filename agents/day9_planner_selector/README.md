# Day 9 planner selector

This directory contains the exact package used by the internal arena agent named
`Day 9 planner selector`.

## Provenance

- Arena agent ID: `235bd66241b98bcbf5e3ed454a7cef5baacde82dca3a5480a531d0955352034a`
- Arena version: `historical-cbef5780feef`
- Source archive SHA-256: `cbef5780feeffb7e010be4c04fee9a5eacf4d7e9da152f02b9a39b1243342381`
- Arena image: `sha256:e6b2ccdc9eeef5af1263b0b0babde0439f4fb0c9c76f9a464d74d012c327069c`
- Package created: `2026-09-14T04:35:56.750072+00:00`
- Arena validation completed: `2026-09-14T04:40:47.776097+00:00`

The arena marked the build as verified and the agent as active. `MANIFEST.json`
contains the SHA-256 digest for each package file. The files here retain the
exact bytes from the arena archive.

## Entry points

- `main.py` exposes the Kaggriculture `agent` function.
- `_arena_bridge.py` is the internal arena bridge. The arena runs it with
  `python _arena_bridge.py`.

The package includes a native `agent.so`, so it needs a compatible Linux
runtime. The arena result records local evaluation data, not a Kaggle score.
