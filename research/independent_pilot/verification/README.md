# Pilot evidence

Read [REPORT.md](REPORT.md) for the result and its limits. [RESULTS.json](RESULTS.json) contains the frozen analysis. [EVIDENCE_AUDIT.json](EVIDENCE_AUDIT.json) adds coverage, hash and replay checks, training diagnostics and explicitly supplementary comparisons.

The evidence archive contains all twelve trial journals and checkpoints, selection and test rows, complete reference-checked replays, input manifests, correctness receipts and the tested native binary. It also preserves the earlier development receipts and two preparation directories that stopped before training. It does not contain the reserved holdout manifest or its outcomes. The unchanged reference/opponent runtime remains in the base evaluation bundle in this repository.

`evidence-manifest.json` lists every archived file's SHA256 and size, plus the archive's hash. In a fresh checkout, extract from the repository root:

```sh
tar -xzf research/independent_pilot/verification/evidence.tar.gz
python research/independent_pilot_audit.py --root artifacts/independent-pilot/run-v3 --out artifacts/independent-pilot/rechecked-evidence.json
```

The supplementary audit uses the Python standard library. It checks file bytes, score arithmetic, full case coverage, configuration selection, action-stream hashes, replay contents and training totals. It does not rerun the reference interpreter. It was written during selection, after the trial inputs had been frozen, and cannot change the frozen decision rule.

To rerun the original analysis, use the recorded Linux environment and `python -m research.independent_pilot.analyze --root artifacts/independent-pilot/run-v3`. The original runner also checks Python, package, platform and libstdc++ versions. Rebuilding the native extension with a different compiler will produce a different binary hash; keep such a rebuild as a separate run.

Both pilot splits have been consumed. Further tuning needs new development seeds, and final confirmation needs a fresh custodian release. Do not reuse these test outcomes as an independent confirmation.
