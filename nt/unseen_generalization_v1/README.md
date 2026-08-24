# Unseen Generalization V1

This experiment turns the current FC24B research line into a reproducible pipeline for
unknown-opponent generalization.  The first milestone, **UG0**, does not train a
selector yet.  It freezes the live local evidence corpus, reconstructs Kaggle replay
observations, extracts coarse behavior families, and creates leakage-resistant train /
validation / whole-family holdout splits.

## Current status

Implemented and locally smoke-tested:

- secure synchronization wrapper for the latest own Kaggle submission and replay set;
- deterministic inventory of public notebooks, agent sources, submissions, receipts,
  reports, and replays under `D:\Kaggriculture`;
- second-seat replay delta reconstruction;
- action, market, public-farm, and daily behavior descriptors;
- content hashes, behavior signatures, duplicate detection, and coarse route families;
- group-level split generation that never separates an exact behavior signature;
- machine-readable receipts and a Chinese baseline report.

Not yet claimed:

- live Kaggle download success in this branch;
- official-Python/JAX replay parity for newly downloaded episodes;
- checkpoint restoration, handoff bridge, continuation-route search, or selector
  improvement.

## Security

The token is read from `KAGGLE_API_TOKEN` or the normal Kaggle credential files.  It is
never printed or written into receipts.  `.env`, `access_token`, `kaggle.json`, and
`auth.json` are explicitly excluded from the asset inventory.

## Local quick start

From PowerShell:

```powershell
cd D:\Kaggriculture
python -m pip install -U kaggle pytest

# Set the token only in the current shell; do not put it in Git.
$env:KAGGLE_API_TOKEN = "<set locally>"

powershell -ExecutionPolicy Bypass -File `
  nt\unseen_generalization_v1\tools\run_ug0_windows.ps1 `
  -Workspace D:\Kaggriculture `
  -MaxReplays 128
```

Equivalent explicit commands:

```powershell
$Exp = "D:\Kaggriculture\nt\unseen_generalization_v1"
python "$Exp\tools\sync_live_assets.py" --workspace D:\Kaggriculture --max-replays 128
$env:PYTHONPATH = "$Exp\src"
python "$Exp\tools\build_ug0_manifest.py" --workspace D:\Kaggriculture
python -m pytest "$Exp\tests" -q
```

The synchronization step uses the official simulation competition commands:

- `kaggle competitions submissions kaggriculture -v`
- `kaggle competitions submission-download <submission_id>`
- `kaggle competitions episodes <submission_id> -v`
- `kaggle competitions replay <episode_id>`

If `submission-download` is missing, upgrade the official `kaggle` package before
continuing.

## Outputs

By default, live files and generated manifests are written outside the experiment
source tree:

```text
D:\Kaggriculture\live_assets\
├─ own_latest\submission_<id>\
│  ├─ replays\
│  ├─ logs\                 optional
│  └─ sync_receipt.json
└─ ug0_corpus_v1\
   ├─ source_manifest_v1.json
   ├─ replay_summaries_v1.jsonl
   ├─ replay_players_v1.csv
   ├─ split_manifest_v1.json
   ├─ parse_failures_v1.json
   └─ UG0_CORPUS_BASELINE_V1_ZH.md
```

These files are evidence and training inputs; they should remain local unless an
explicit compact/frozen artifact is selected for Git.

## Project structure

```text
unseen_generalization_v1/
├─ src/unseen_generalization_v1/
│  ├─ schema.py
│  ├─ replay_features.py
│  ├─ inventory.py
│  └─ splits.py
├─ tools/
│  ├─ sync_live_assets.py
│  ├─ build_ug0_manifest.py
│  └─ run_ug0_windows.ps1
├─ tests/
└─ docs/
```

## Next gate

UG1 starts only after UG0 is run on the real local corpus.  It must implement exact
checkpoint restoration and prove that continuing the original agent from every saved
checkpoint reproduces the original actions, states, and terminal cash.
