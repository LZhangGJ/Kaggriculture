# C07 round2 strategy

**Status: author-validation target missed.** The frozen r4 mean was 187,473.125 over the declared 16 validation seeds in both seats. The paired parent mean was 187,673.875. The complete development mean was 200,709.21875. Source was not changed after validation.

## Run the strategy

1. Extract the complete ZIP into a directory. Keep `main.py`, `build.py`, `COMPILER_FLAGS.json` and `policy/` together.
2. From that directory, rebuild the native runtime offline:

   ```sh
   python3 -B build.py --cxx g++
   ```

3. Use `main.agent(observation, configuration)`. For an independent game context, call `main.create_agent()` and call the returned object's `close()` after the game. A new step0 resets an existing context; the module entrypoint isolates both seats.
4. For a local full-game reproduction on an already-used seed, choose a new result label:

   ```sh
   python3 -B research/tools/run_panel.py --source . --label local_smoke --seeds 1681080428 --seats 0,1 --stage local_reproduction
   ```

The runner refuses to overwrite an existing label. It uses the included frozen interpreter, writes every attempt, and saves full replays. This local host is not Kaggle's sandbox or timeout implementation.

## Inspect the evidence

`REPORT.md` contains results, changes, failures and runtime limits. `EVALUATION_SUMMARY.json` contains exact values and frozen source hashes. `research/audit/all_games.csv` indexes every new game. `research/audit/experiment_inventory.json` locates all thirteen panels and full replays. `research/notes/METHOD_AND_LIMITS.md` explains forecasts and policy-input boundaries. Historical source, original archives and previous results remain under `research/feedback/` and `research/provenance/`.

To verify the shipped files before rebuilding them, run:

```sh
python3 -B verify_package.py
```

A rebuild may replace build receipts. The included Linux x86-64 binary is byte-identical to the independently rebuilt final source in this environment. Other machines should rebuild from source; normal Linux system libraries and a C++20 compiler are required. Runtime Python dependencies are standard-library only.
