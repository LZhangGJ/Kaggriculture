# C06 round 1: delivery report

## Selected checkpoint

The runnable files at the archive root are the unmodified, published A06 r6 revision1 parent from the verified AUTHOR_INPUTS archive. The recorded parent baseline was approximately $209,142 mean terminal cash over 16 completed games. Read the original per-game logs in research_snapshot for exact values, seed/seat assignments, timing, and terminal assertions. This is an economic PASS-opponent baseline, not competitive qualification.

## Research focus and experiments

The completed work investigated shared dawn-deposit capacity, discarded inventory, service/labor scheduling, fertilizer valuation, and cash reserves. Earlier diagnostic reports identified losses at the 100-unit shared dawn deposit. Storage safeguards were neutral on the first four games; cheaper labor valuation helped slightly across 16 games; removing fertilizer or cash reserves hurt. Earlier-return/service-worker experiments are included in the research snapshot, but are not promoted at the root without verified final selection.

## Source and verification boundary

The input ZIP SHA-256 matches 75356640e384979f504fa7393796ab3c4e8dd22b051d5284599d7e4266b2e76d. The earlier verified run reported that all 71 input manifest entries matched. audit/manifest_check.json contains the packaging-time checks supported by the manifest format. All research code and logs have been preserved, including failed or rejected experiments. This report does not convert forecast, partial-game, or missing outputs into terminal results.

## Tool-output limitation

The last continuation could not read tool outputs: commands returned only skipped-message markers. The final root selection is therefore conservative. A final post-packaging compile and game check is not claimed. The source had been built and benchmarked earlier; rebuild it locally with the supplied offline build command.

## Resource measurements

The earlier measured environment had a four-core CPU quota, five-core affinity, and 4 GiB memory. Detailed earlier measurements and benchmark timings are in the preserved research files. Packaging-time measurements are in audit/packaging_resources.json.

## Offline build and execution

From the extracted archive root, run `python3 -B build.py --cxx g++`. Use Python 3 with the standard library and the compiler expected by build.py. Submit/call `main.py:agent(observation, configuration)` after building. The published interface also provides `create_agent()` and `close()`; create a fresh agent for each game and close it afterward. The offline referee and frozen engine are retained under published_inputs.

## Audit files

`audit/SOURCE_HASHES.sha256` covers every delivered file except itself. `audit/experiment_inventory.json` indexes JSONL result files and their actual recorded seeds. `research_snapshot` is a complete copy of available author work except caches. The original archive contents, including historical competitive data, are kept in `published_inputs`; historical rows are not new C06 results.
