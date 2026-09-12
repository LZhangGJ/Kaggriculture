"""Write the source-backed delivery notes; never read holdout seed values."""
from collections import Counter
import json
from pathlib import Path
from audit_seeds import dump,sha
from seed_sets import ROOT


def write(root=ROOT):
    def get(name):return json.loads((root/name).read_text())
    def link(name,label=None):return f'[{label or name}]({(root/name).as_posix()})'
    def save(name,text):(root/name).write_text(text.strip()+'\n',encoding='utf-8')
    provenance=get('provenance.json');audit=get('audit/summary.json');refresh=get('audit_refresh_final.json');held=get('holdout_receipt.json')
    pool=get('selection_result.json');opponents=get('opponents.json')['opponents'];candidates=get('candidates.json')['candidates']
    profiles={}
    for name in ('representative','stress'):
        rows=[json.loads(l) for l in (root/'features'/f'{name}.jsonl').open()]
        profiles[name]=dict(first_reference_shop_counts=dict(Counter(r['reference_first_shop'] for r in rows)),
            distinct_reference_sequences=len({tuple(r['reference_pass_shops']) for r in rows}),
            feature_ranges={k:dict(min=min(r['features'][k] for r in rows),max=max(r['features'][k] for r in rows)) for k in rows[0]['features']})
    dump(root/'economic_coverage.json',profiles)
    hash_table='\n'.join(f'| {n} | `{sha(root/p)}` |' for n,p in [('Representative','manifests/representative.json'),('Stress','manifests/stress.json'),('Sealed holdout','sealed/holdout.json')])
    opponent_table='\n'.join(f"| `{o['id']}` | `{o['runtime_tree_sha256']}` |" for o in opponents)
    candidate_table='\n'.join(f"| `{c['id']}` | `{c.get('package_manifest_sha256',c['runtime_files'].get('nt/latest_20260911_afs_workflow_repair_r1_r2/r2/policy/agent.so'))}` |" for c in candidates)
    ranges=[]
    for k,label in [('ref_shop_units_MILK','PASS/PASS shop milk units'),('ref_shop_units_WOOL','PASS/PASS shop wool units'),
                    ('ref_max_duplicate','PASS/PASS maximum shop copies'),('latent_weed_hits','Latent weed hits in 5,800 draws')]:
        a,b=[profiles[n]['feature_ranges'][k] for n in ('representative','stress')]
        ranges.append(f"| {label} | {a['min']}–{a['max']} | {b['min']}–{b['max']} |")
    save('REPORT.md',f'''
# Kaggriculture evaluation seed sets

Created 256 representative seeds, 128 stress seeds and 256 sealed holdout seeds. All are valid, unique and mutually disjoint. No candidate panel ran. All work used the local CPU; the source checkout, campaign processes, champion and Git state were left untouched.

| Set | Seeds | Games per candidate: 16 opponents × 2 seats | Manifest |
|---|---:|---:|---|
| Representative | 256 | 8,192 | {link('manifests/representative.json')} |
| Stress | 128 | 4,096 | {link('manifests/stress.json')} |
| Sealed holdout | 256 | 8,192 | Separate file named in {link('holdout_receipt.json')} |

The representative set is a uniform pseudorandom sample without replacement from the eligible default 31-bit seed domain. It was drawn before any characterization. The holdout uses an independent random key and rejects the entire 4,096-seed characterization pool as well as the representative set and prior exclusions. Its economic characteristics remain uninspected. Disjointness necessarily conditions these independently generated streams; the sets are not statistically independent without that condition.

The stress set takes both coordinate extremes on 26 frozen dimensions, a witness for each of eight reference first-shop types, all 12 planned contrast cells, and then fills to 128 by farthest-first distance on feature ranks. All extrema refer to the 4,096-seed pool, not the full seed space. No candidate win, draw, cash, reward or action trace influenced selection. {link('selection_protocol.json')} records the draw rule and original code hashes; {link('selection_result.json')} records every selection reason.

## What the economic labels mean

Shop paths are action-dependent in this engine. Each day, weed draws consume the RNG before the shop draw; changing the number of empty farm tiles can change the resulting shop. Seed-only potential draws are fixed, but realized shops and prices are not fixed by the seed alone. The stress set therefore combines latent random-draw features with shop demand under a declared PASS/PASS controller. These conditional labels do not promise the same conditions in candidate-versus-opponent games.

| Feature | Representative range | Stress range |
|---|---:|---:|
{chr(10).join(ranges)}

Both sets include all eight reference first-shop types. The stress set has {profiles['stress']['distinct_reference_sequences']} distinct reference shop sequences. All 12 low/high combinations for reference milk/wool, early/late milk-minus-wool, and potential milk/wool demand have witnesses. No shop consumes melon; its town-center demand is fixed. No seed-only glut, oversupply, price-collapse or sale-timing label is claimed. See {link('FEATURE_DEFINITIONS.md')} for formulas and limits.

## Audit coverage and gaps

The full audit ran from **{audit['inventory_started_utc']}** to **{audit['completed_utc']}**. It inventoried {audit['inventory_files']:,} files ({audit['bytes_at_inventory']:,} bytes), scanned {audit['scanned_files']:,} text or compressed files, and processed {audit['parsed_expanded_bytes']:,} expanded bytes. Coverage includes all supplied `research/robust90` artifacts plus the source tree's `nt`, tests, benchmarks, scripts and other text. It covers saved training, tuning, diagnostic and evaluation seed fields, including future-continuation seeds, filenames, literal Python ranges and declared CLI blocks. The seed array in the NPZ training artifact was also decoded; its seeds were already excluded.

The frozen exclusion union contains **{audit['unique_exclusions']:,}** values, of which **{audit['exclusions_in_sampling_domain']:,}** fall in the sampling domain. This is a conservative superset, not a count of proven played episode seeds: seed counts and constant source ranges can add false positives. Each file has a saved hash, byte limit, extraction method and recovered seed list in {link('audit/files.jsonl')}; {link('audit/summary.json')} and {link('audit/supplement.json')} give the audit details.

The active campaign changed one row file and added two files during that full scan. A final supplement at **{refresh['utc']}** checked {refresh['changed_or_new_files']} changed/new files and brought the union to **{refresh['total_exclusions']:,}**. It found **zero overlap** with any delivered set or the characterization pool. A newly copied opponent binary matched bytes already inventoried; its adjacent metadata was scanned. The supplement trusts unchanged files by size and modification time; its evidence is in {link('audit_refresh_final.json')}. The earlier supplement is retained as {link('audit_refresh.json')}.

Coverage does not include Git history, deleted or missing logs, sibling checkouts, remote training stores, or unrecorded dynamic seeds. Compiled binaries were hashed, not decoded for embedded seeds. Literal seed arrays over 65,000 bytes can escape the text extractor unless their individual seeds also appear in rows. No text read or parse errors occurred. Later campaign activity can add exclusions. Re-audit after candidate freeze and before use; complete historical non-overlap beyond the audited records is not claimed.

## Validation and versions

{link('validation.json')} records passing counts, bounds, uniqueness, exclusion and disjointness checks. All 4,352 characterization rows and all four sampled/selected manifests reproduced exactly; holdout reproduction checked numbers only. The official-interpreter checks used 16 fixed-controller seasons, 11,504 transitions and 59,264 raw RNG/choice checks. PASS/PASS demand and weeds matched in all eight reference cases. Buying NE once changed the shop path in all eight paired cases. Four synthetic tests cover seed recovery, rejection sampling, frozen-file drift and holdout contamination. Six unexecuted job plans contain 36,864 development jobs across the three candidates; no holdout job file was made.

A first reproduction check failed only because JSON reordered an unordered list of constant feature names. The verifier now compares that list without order. The selected seeds and feature values did not change. Original selection code is retained under `source_snapshot/selection_tools`; validation checks that the sampling, feature, selection and build code remains unchanged.

Source: `{provenance['source_root']}` at HEAD `{provenance['source_commit']}`. HEAD alone does not describe uncommitted campaign work; {link('provenance.json')} pins the actual source bytes. The frozen CPU host identifies its interpreter as Kaggriculture 1.32.7. All 16 opponent runtime file sets and all three candidate versions matched their saved hashes. See {link('OPPONENTS.md')} and {link('candidates.json')}.

| Manifest | SHA-256 |
|---|---|
{hash_table}

The holdout is procedurally separated, not encrypted or inaccessible to local agents. Its release rule is in {link('EVALUATION_CONTRACT.md')}. This task ends with these artifacts; it makes no strength, promotion or submission claim.
''')
    save('FEATURE_DEFINITIONS.md','''
# Feature definitions

The frozen official Python source is the authority. All runs use defaults: 720 recorded states, 719 transitions, 24 turns/day, a 10×10 board, 25 unlocked NW tiles per player, weed chance 0.005, shops every three days with replacement, and eight shop instances at most. Town-center consumption occurs every 24 turns; shops consume every four turns. Single-product shops consume two units per tick.

**Randomness.** At end of zero-based day `d`, the engine creates `random.Random((seed * 1_000_003) ^ d)`. It visits both farms in order and calls `random()` only for empty tiles. It then calls `choice(sorted(SHOPS))` when a shop unlocks. Only days 0–28 have end-of-day processing in a 719-transition episode. Shop unlock days are 3, 6, 9, 12, 15, 18, 21 and 24. Farm actions affect RNG consumption, even though the underlying daily random stream is seed-controlled. Initial prices, cash, land and town state do not vary by seed.

**Latent weed fields.** `latent_weed_hits` counts values below 0.005 in the first 200 uniform draws on each of 29 days: 5,800 possible draws. `latent_weed_early_minus_late` is 20 times the day-0–8 hit count minus nine times the day-9–28 count, comparing equal daily rates with integer arithmetic. `latent_weed_stream_half_difference` counts hits in offsets 0–99 minus offsets 100–199 across all days. These are potential draw opportunities, not actual weeds, tile locations or seat advantages. Actions choose how many opportunities are consumed and which tile receives a draw.

**Potential shop table.** For each unlock day, record the shop obtained after `k` prior calls to `random()`, for each `k=0..200`. This table is a deterministic property of the seed and the official RNG. The range covers syntactically possible empty-tile counts across two fully unlocked boards. It does not assert that every count is reachable by that day under the default budget. The table measures sensitivity to occupancy; it is not a new simulator or a forced-shop mode.

`potential_units_sum_PRODUCT` sums lifetime shop demand over all 201 offsets at each unlock, then over unlocks. Divide by 201 to obtain a uniform-offset average. Uniform offset weighting is an explicit analytical convention, not a prior for actual farm occupancy. `potential_milk_minus_wool` subtracts the two sums. `potential_crop_minus_animal` subtracts EGG+MILK+WOOL from CARROT+TOMATO+STRAWBERRY+MELON; WHEAT is omitted because it also feeds animals. `potential_adjacent_choice_agreement` counts equal neighboring choices across eight sets of 200 adjacent offsets; higher values mean less local sensitivity to changing the count by one. No potential-table average is presented as realized exogenous demand.

**PASS/PASS reference controller.** Both players issue PASS every turn, with no market orders, hands, planting or land purchases. Starting farms are empty NW quadrants. Each day's weeds permanently remove empty cells; their order and shop draws match the official engine. This is a legal, deliberately simple characterization controller, not an opponent in the main panel. A second legal controller, buying NE once on the first turn and then passing, is used only to validate action/RNG coupling.

`ref_shop_units_PRODUCT` is total additional shop consumption over transitions 0–718. A shop opening on day `u` has `floor(718/4) - 6*u + 1` consumption ticks. Multiply by two for PET_CAFE/CARROT and YARN_STORE/WOOL; other demanded products use one. The fixed 30 town-center units per non-fertilizer product are excluded. These are consumption units, not money, production or yield. `ref_early_units_PRODUCT` uses end step 215, just before day 9. `ref_late_added_units_PRODUCT` counts lifetime demand from shops added on day 15 or later; it is not total late-game demand.

`ref_milk_minus_wool`, `ref_crop_minus_animal`, and the early/late milk-minus-wool fields apply those differences to the corresponding reference demand fields. `ref_shop_variety` counts unique shop types; `ref_max_duplicate` is the largest number of copies of one shop. `ref_weeds_total` and `ref_weeds_seat_difference` describe weeds under this reference only and are stored for validation; they are not selection coordinates. The stored shop sequence, empty-cell counts and first shop make the reference trace inspectable.

The 26 selection coordinates comprise 13 potential/latent fields and 13 PASS/PASS conditional fields. All are listed in `selection_result.json`. Coordinate extrema receive witnesses. Contrast cells use inclusive 25th/75th pool order-statistic cutoffs, recorded in that same file. Remaining choices maximize their minimum squared Euclidean distance from chosen seeds on doubled integer midranks. Numeric seed order breaks ties. All 128 reference sequences are distinct in this release, but this is not a claim of 128 distinct opponent-induced economies.

No shop buys MELON, so all four shop-only melon fields are zero and are excluded from selection distances. Product market curves and initial inventory are fixed, not seed parameters. Sales, product purchases, inventory, price floors, gluts and cash are action-dependent. No price scenario was selected or imposed. Applying the reference labels to policy outcomes is a predeclared diagnostic grouping; actual economic-condition reports must also use each game's realized public shop and market path, clearly labeled as action-dependent.
''')
    save('EVALUATION_CONTRACT.md',f'''
# Evaluation contract v1

Run Day 3 `connected-day3-v1`, Day 9 `expected-selector-v1`, and `original-afs-r2` on matched sets. Use the exact file hashes in {link('candidates.json')} and the 16 opponent versions in {link('opponents.json')}. Each seed crosses all 16 opponents and both candidate seats: 8,192 representative, 4,096 stress and 8,192 holdout games per candidate. That is 20,480 games per candidate or 61,440 for all three if all stages are later authorized. `stress:delayed_seller` is diagnostic-only and is excluded from these counts.

Use the official default rules and 719 transitions per complete game. Keep native direct evaluation separate from common-randomness or forced-future training engines. Do not reseed or replace the official shop/weed process. Same seed/opponent/seat gives matched starting randomness, but different candidate actions can produce different realized shops. Reset both agents for every game and keep the episode seed out of their observations and configuration. Pin and record runner, simulator, candidate, opponent, configuration and analysis hashes, runtime versions and timeout rules before launch. Verify native/official parity for the actual runner before relying on native results.

Keep representative and stress results separate. Report scheduled, completed, invalid and missing games; wins, draws, losses and strict win rate; each opponent; each candidate seat; and the same breakdown by predeclared economic condition. Draws contribute zero wins. If also reporting `(wins + 0.5*draws)/games`, label it separately. Cash and margin can be secondary diagnostics. Do not drop failed games from the denominator or treat a partial panel as final. Any missing or invalid cell blocks a complete comparison; record repairs and reruns without choosing the favorable outcome.

Use frozen reference-condition labels from the feature manifests, with their controller qualification. Separately record each game's actual shop unlock sequence and times, demand units, market inventory and price-floor exposure. These realized groups are descriptive and action-dependent; comparing candidates within them is not a controlled causal estimate. Do not rename a PASS/PASS label as an exogenous price regime. Define any additional bins before inspecting candidate results. Holdout features may be computed only after freeze/release as part of the frozen analysis.

For uncertainty, cluster by seed, keeping all 32 opponent/seat games together. Use 4,000 bootstrap resamples and fixed bootstrap key `kaggriculture-seed-contract-v1-bootstrap`. Report 95% percentile intervals for win rate and paired candidate differences. Use the same resampled seed indices across candidates. Per-opponent intervals retain both seats within each seed; per-seat and condition intervals retain all eligible games from each sampled seed. Report distinct seed support and mark empty/small groups. Stress bootstrap intervals describe sensitivity within this selected panel, not prevalence or performance in the uniform seed population. Treat many opponent/condition comparisons as exploratory unless a testing rule was frozen in advance.

Before holdout use, freeze all three finalists, their complete runtime files, evaluator, settings, timeout/failure rules and analysis plan. Then rerun the full seed audit and its NPZ/CLI supplement. Resolve new overlap or audit gaps before release. Keep holdout values and generator key in `sealed/`; publish only count and checksum before release. `release_holdout.py` checks the freeze record, pinned versions and fresh audit before copying the manifest to a new release directory. The separation is procedural: local agents can read these files, so it is not cryptographic secrecy.

Use the holdout for one frozen matched comparison. Once any holdout feedback guides code, tuning, opponent choice, metric choice or another development decision, retire the whole holdout and sample a new independent one excluding all prior sets, characterized pools and recovered history. Do not keep the same holdout for another claim of fresh confirmation.

Repeated tuning on representative and stress panels still risks overfitting. Record every tested candidate and selection decision. Sixteen identities do not imply sixteen independent strategy families; the source pool itself notes shared production behavior. A fixed panel cannot prove robustness to every opponent or economic path. This contract defines evidence to collect, not automatic promotion authority.
''')
    save('OPPONENTS.md',f'''
# Pinned main-panel opponents

All 16 runtime file sets matched `research/robust90/PANEL_SOURCES.json` during this task. The membership comes from `research/robust90/POOL.json`. Each hash below is SHA-256 of canonical UTF-8 JSON for that opponent's repository-relative path-to-file-hash mapping (sorted keys; comma/colon separators). It is a composite identity, not a binary hash. {link('opponents.json')} contains every exact runtime file path and SHA-256. Recheck those files before evaluation.

| Opponent | Runtime file-set SHA-256 |
|---|---|
{opponent_table}

`stress:delayed_seller` and `nagatakengo_v70` stay outside the main panel. Shared lineage and observed production patterns limit opponent diversity; do not count names as independent economic strategies.

| Candidate | Package MANIFEST SHA-256; R2 row uses agent.so SHA-256 |
|---|---|
{candidate_table}

Day 3's selector hash is `ea387f80fbcaf725a61d1752daf36316d95f4e9319b6e8aa0267c1dce2f1bdab`; it restores the base on day 8 and uses Day 9 as its suffix. Day 9's selector hash is `6b9147d29a4b8278e660e34efbee39751007f88c8ee66fb9a8c43ab85c05c5ca`. Both use `forecast_features.so` bytes `894212097145c33daea8c6f4961aa1bc00b39f42efb8445f51d4c1cda311160a`. Original AFS R2 uses `179b204db64a32e08af3afb34ae1e6687737c71e4d506f37c4dd3ebe0ec591f2`. Full package file hashes, including each selector and config, are in {link('candidates.json')}. These pins identify the requested comparison; no candidate was promoted.
''')
    save('README.md',fr'''
# Use these seed sets

Bundle: `{root}`

Read {link('REPORT.md')}, then {link('EVALUATION_CONTRACT.md')}. The individual holdout seeds are intentionally absent from the report. {link('holdout_receipt.json')} publishes its count, file location and checksum. Do not load it for diagnostics or pass it to a policy during development.

## Verify the delivery

Run from the bundle directory with Python 3.11 or later. The tools use only the standard library. This release was checked with Python 3.14.3 on Windows. A different Python version should pass `validate.py` and `seed_sets.py reproduce` before use; runtime RNG behavior is checked, not assumed. These commands use one CPU process and never import an agent or a GPU library.

```powershell
python -B tools\check_bundle.py
python -B tools\validate.py
python -B tools\seed_sets.py reproduce
python -B tools\test_tools.py -v
```

`validate.py` runs 16 simulator-only fixed-controller seasons on eight already-characterized development seeds. It makes no holdout simulator call. `reproduce` regenerates the draws and all 4,352 feature rows without changing seed manifests; holdout verification is numeric only. Verify the bundle first: validation reports contain elapsed times and can change when rerun. Original selection code and its hashes are retained in `source_snapshot/selection_tools`. The shipped verifier has one documented correction for unordered metadata; selection code is checked unchanged.

## Integrate into the main campaign

1. Keep this bundle outside the active source checkout. Verify all runtime pins and record the evaluation runner's environment, hashes and settings. Re-audit exclusions before a later run; fail on overlap. Reserve these manifests and the whole characterized pool against future training use.
2. Read the exact seed lists. Do not turn them into a consecutive `seed-start` block. The existing `evaluate.py` CLI only accepts consecutive seeds. The six prepared `job_plans/<candidate>/<set>/jobs.jsonl` files cover representative and stress for all three candidates; they contain 36,864 planned jobs and have executed none. Every row records `candidate_seat` and `opponent_seat = 1 - candidate_seat`.
3. Use a separately reviewed manifest-aware runner in the main campaign. Its job tuple for the existing `evaluate.worker` is `(opponent, seed, opponent_seat, binary, config, trace)`. This recipe requires Linux/WSL because that worker uses POSIX alarms and native `.so` files. Translate source paths explicitly; do not import the worker during seed preparation. Use one frozen engine choice for every candidate, and complete parity checks before the large panels.
4. For Day 3 and Day 9, the runner can set `__engine='direct'` and `__package` to the pinned package folder, with the matching package `agent.so` and `config.json`. For original R2, use its pinned `policy/agent.so` and config, with `__engine='direct'` and no selector, suffix, common-randomness or package override. Review this adapter against the frozen `evaluate.py` before launch. Snapshot both agents fresh for every game and write results in a new experiment directory.
5. Require exactly the planned unique `(seed, opponent, candidate_seat)` keys and 719 transitions per terminal game. Keep the two development sets separate and apply the contract's seed-cluster analysis. Collect actual shop/market paths without feeding the hidden seed or future labels to the policy.

To write another development job plan without running it:

```powershell
python -B tools\plan_jobs.py --set representative --candidate connected-day3-v1 --out C:\path\to\new-job-plan
```

## Release the holdout later

1. Finish development and freeze all three finalists. Create a freeze JSON with `finalists_frozen: true`, ISO UTC `frozen_utc`, `candidates` mapping the three IDs to **absolute runtime-file path → SHA-256** mappings, `evaluator_files` and `analysis_files` mappings, and `opponents_sha256` equal to this bundle's opponent-manifest checksum. Include the complete runner, simulator, analysis implementation, configuration and environment record. Candidate hashes must include all files in `candidates.json`. `freeze_record_template.json` gives the required shape and already fills the pinned candidate file maps.
2. Run a fresh full audit, then the supplement, outside the source tree. The audit must start after the freeze and finish without unresolved file changes or text errors. This stage reads records; it does not stop or change the campaign.

```powershell
python -B tools\audit_seeds.py --repo "{provenance['source_root']}" --out C:\path\to\fresh-audit
python -B tools\supplement_audit.py --repo "{provenance['source_root']}" --out C:\path\to\fresh-audit
python -B tools\release_holdout.py --freeze C:\path\to\freeze.json --fresh-audit C:\path\to\fresh-audit --out C:\path\to\holdout-release
```

3. Pass the release receipt to `plan_jobs.py --set holdout --candidate <one of the three IDs> --release-receipt <release_receipt.json> --out <new plan directory>`. This creates jobs only. The runner must verify frozen runtime bytes again at execution. Run the predeclared matched comparison once. Retire the holdout if its feedback informs further development; resampling must exclude every prior manifest and all characterized pools.

The release tools enforce a local procedure, not access control. They cannot stop another local agent from reading `sealed/`, prove that no human saw outcomes, or recover missing historical logs. A genuine confirmation claim still needs those operational controls. Do not rerun `seed_sets.py build` in this bundle: its entropy and selections are already frozen.
''')
    save('sealed/README.md',f'''
# Sealed holdout

`holdout.json` contains 256 independently drawn, unevaluated seeds. SHA-256: `{held['sha256']}`.

`generator.json` stores its separate reproduction key. Do not copy either file into development prompts, policy packages or public reports. Numeric count, bounds, disjointness, checksum and reproduction checks are allowed; simulator or economic-characteristic checks require candidate freeze and release.

This directory has no cryptographic or operating-system secrecy. Local agents can read it. Follow the release and retirement procedure in the evaluation contract. No actual holdout release or job plan has been created in this task.
''')
    repo=Path(provenance['source_root']);frozen={}
    for c in candidates:
        frozen[c['id']]={str(repo/rel if c['id']=='original-afs-r2' else repo/c['path']/rel):digest for rel,digest in c['runtime_files'].items()}
    dump(root/'freeze_record_template.json',dict(finalists_frozen=False,frozen_utc='REPLACE_AFTER_DEVELOPMENT_WITH_ISO_UTC',candidates=frozen,
        opponents_sha256=sha(root/'opponents.json'),evaluator_files={'REPLACE_WITH_ABSOLUTE_RUNNER_AND_ALL_RUNTIME_FILE_PATHS':'REPLACE_WITH_SHA256'},
        analysis_files={'REPLACE_WITH_ABSOLUTE_ANALYSIS_PLAN_AND_IMPLEMENTATION_PATHS':'REPLACE_WITH_SHA256'}))


if __name__=='__main__':write()
