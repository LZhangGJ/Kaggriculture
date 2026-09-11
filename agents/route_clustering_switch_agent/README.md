# Route-clustering switch agent

This agent groups historical farming strategies, evaluates route switches by simulation and learns small decision trees that decide when to switch. Its online policy uses a frozen route library through a reactive executor.

- [Complete English method: ideas, components, training, execution and reproduction](docs/METHOD.md)
- [Comparison with the P16 planning approach](../../docs/agent_approaches/README.md)
- [Source provenance](SOURCE_PROVENANCE.md) and [omitted artifacts](OMITTED_ARTIFACTS.md)
- [Maintenance instructions and clustering quality gate](agent.md)

## Use the published agent

[main.py](main.py) is the generated single-file submission. It embeds the deployment assets and exposes `agent(observation, configuration=None)`. NumPy is needed for inference. [runtime/main.py](runtime/main.py) is the expanded version, with inspectable route metadata, action tapes, trees and base executor beside it. Initialize separate policy instances/processes for concurrent games.

The published policy starts with route family `G001`, checks for a switch at steps 144, 168 and 216, and makes at most one actual switch. If an early tree keeps the current route, a later checkpoint may still switch. The current runtime contains 175 route carriers and three trees of depths 2, 4 and 2. There is no PPO, Torch dependency or neural checkpoint in this deployed policy.

```bash
python -I main.py
python -I runtime/main.py
PYTHONPATH=src python -m pytest -q tests
```

These are initialization and focused tests, not a complete match or timing certification.

## Source map

| Path | Purpose |
|---|---|
| `src/meta_agent/` | Editable features, tree controller and route adapter |
| `scripts/` | Replay processing, clustering, search, training, evaluation and export |
| `fast_kaggriculture/` | C++20/pybind11 simulator and differential tests for default rules version 1.32.7 |
| `runtime/` | Frozen expanded deployment assets |
| `main.py` | Frozen generated single-file deployment |
| `configs/` | Replay manifest and pipeline configuration examples |
| `docs/` | Detailed method and archived explanatory charts |

Do not hand-edit the compressed submission. Development source and frozen runtime snapshots are distinct; export and verify a new candidate when changing behavior.

## Reconstruct the offline experiment

Use Linux/WSL with Python, C++20 and OpenMP. Install `requirements.txt`, copy `configs/pipeline.env.example` outside the source tree, fill in replay inputs and an external `OUTPUT_ROOT`, and inspect `scripts/run_pipeline.sh`.

Complete the clustering and representative-quality review before expensive round-robin/search stages. The wrapper does not automatically pause for that review. The [method guide](docs/METHOD.md) explains each stage, exact archived seed ranges, the manual gate and the required rule-fidelity checks.

```bash
python -m pip install -r requirements.txt
bash scripts/run_pipeline.sh /path/to/pipeline.env
```

This is a large reconstruction job, requiring replay inputs not shipped in Git. The source corpus, matrices, caches and historical full experiment outputs are omitted. A new corpus will not necessarily produce the same families or trees.

To repack only the frozen assets, without training or simulation:

```bash
python scripts/export_single_file_submission.py --source-dir runtime \
  --forced-opening G001 --output /tmp/route-switch-repacked.py
```

The explicit opening matters: the controller also supports sampling the stored Nash mixture, but that is not the published fixed-opening policy. This branch fixes the full pipeline's opening propagation and the multi-file exporter's source lookup; the frozen deployment is unchanged.

## Archived evidence and limits

The original experiment reduced 3,563 replay seats to 275 intent families, retained 175 carriers, searched 11 complementary switch targets and selected three checkpoints. These are corpus-specific outcomes.

The final 256-seed, 175-opponent, both-seat holdout records a score of 89.614955%, versus 81.176339% for its fixed-route baseline. Score is `(wins + 0.5 * draws) / games`. [Runtime manifest](runtime/MANIFEST.json).

These are archived author results, not matches rerun during this publication. The opponents are route carriers through the common executor. This result cannot be compared directly with the P16 live-agent pool's win rates, and it does not establish robustness to arbitrary new strategies or universal route feasibility.
