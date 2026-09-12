# Kaggriculture evaluation seed sets

Created 256 representative seeds, 128 stress seeds and 256 sealed holdout seeds. All are valid, unique and mutually disjoint. No candidate panel ran. All work used the local CPU; the source checkout, campaign processes, champion and Git state were left untouched.

| Set | Seeds | Games per candidate: 16 opponents × 2 seats | Manifest |
|---|---:|---:|---|
| Representative | 256 | 8,192 | [manifests/representative.json](manifests/representative.json) |
| Stress | 128 | 4,096 | [manifests/stress.json](manifests/stress.json) |
| Sealed holdout | 256 | 8,192 | Separate file named in [holdout_receipt.json](holdout_receipt.json) |

The representative set is a uniform pseudorandom sample without replacement from the eligible default 31-bit seed domain. It was drawn before any characterization. The holdout uses an independent random key and rejects the entire 4,096-seed characterization pool as well as the representative set and prior exclusions. Its economic characteristics remain uninspected. Disjointness necessarily conditions these independently generated streams; the sets are not statistically independent without that condition.

The stress set takes both coordinate extremes on 26 frozen dimensions, a witness for each of eight reference first-shop types, all 12 planned contrast cells, and then fills to 128 by farthest-first distance on feature ranks. All extrema refer to the 4,096-seed pool, not the full seed space. No candidate win, draw, cash, reward or action trace influenced selection. [selection_protocol.json](selection_protocol.json) records the draw rule and original code hashes; [selection_result.json](selection_result.json) records every selection reason.

## What the economic labels mean

Shop paths are action-dependent in this engine. Each day, weed draws consume the RNG before the shop draw; changing the number of empty farm tiles can change the resulting shop. Seed-only potential draws are fixed, but realized shops and prices are not fixed by the seed alone. The stress set therefore combines latent random-draw features with shop demand under a declared PASS/PASS controller. These conditional labels do not promise the same conditions in candidate-versus-opponent games.

| Feature | Representative range | Stress range |
|---|---:|---:|
| PASS/PASS shop milk units | 0–630 | 0–792 |
| PASS/PASS shop wool units | 0–1116 | 0–1296 |
| PASS/PASS maximum shop copies | 1–5 | 1–6 |
| Latent weed hits in 5,800 draws | 17–48 | 11–57 |

Both sets include all eight reference first-shop types. The stress set has 128 distinct reference shop sequences. All 12 low/high combinations for reference milk/wool, early/late milk-minus-wool, and potential milk/wool demand have witnesses. No shop consumes melon; its town-center demand is fixed. No seed-only glut, oversupply, price-collapse or sale-timing label is claimed. See [FEATURE_DEFINITIONS.md](FEATURE_DEFINITIONS.md) for formulas and limits.

## Audit coverage and gaps

The full audit ran from **2026-09-12T18:57:14Z** to **2026-09-12T19:02:51Z**. It inventoried 4,288 files (8,236,321,105 bytes), scanned 4,000 text or compressed files, and processed 9,193,909,169 expanded bytes. Coverage includes all supplied `research/robust90` artifacts plus the source tree's `nt`, tests, benchmarks, scripts and other text. It covers saved training, tuning, diagnostic and evaluation seed fields, including future-continuation seeds, filenames, literal Python ranges and declared CLI blocks. The seed array in the NPZ training artifact was also decoded; its seeds were already excluded.

The frozen exclusion union contains **11,320** values, of which **8,039** fall in the sampling domain. This is a conservative superset, not a count of proven played episode seeds: seed counts and constant source ranges can add false positives. Each file has a saved hash, byte limit, extraction method and recovered seed list in [audit/files.jsonl](audit/files.jsonl); [audit/summary.json](audit/summary.json) and [audit/supplement.json](audit/supplement.json) give the audit details.

The active campaign changed one row file and added two files during that full scan. A final supplement at **2026-09-12T19:19:19Z** checked 22 changed/new files and brought the union to **11,412**. It found **zero overlap** with any delivered set or the characterization pool. A newly copied opponent binary matched bytes already inventoried; its adjacent metadata was scanned. The supplement trusts unchanged files by size and modification time; its evidence is in [audit_refresh_final.json](audit_refresh_final.json). The earlier supplement is retained as [audit_refresh.json](audit_refresh.json).

Coverage does not include Git history, deleted or missing logs, sibling checkouts, remote training stores, or unrecorded dynamic seeds. Compiled binaries were hashed, not decoded for embedded seeds. Literal seed arrays over 65,000 bytes can escape the text extractor unless their individual seeds also appear in rows. No text read or parse errors occurred. Later campaign activity can add exclusions. Re-audit after candidate freeze and before use; complete historical non-overlap beyond the audited records is not claimed.

## Validation and versions

[validation.json](validation.json) records passing counts, bounds, uniqueness, exclusion and disjointness checks. All 4,352 characterization rows and all four sampled/selected manifests reproduced exactly; holdout reproduction checked numbers only. The official-interpreter checks used 16 fixed-controller seasons, 11,504 transitions and 59,264 raw RNG/choice checks. PASS/PASS demand and weeds matched in all eight reference cases. Buying NE once changed the shop path in all eight paired cases. Four synthetic tests cover seed recovery, rejection sampling, frozen-file drift and holdout contamination. Six unexecuted job plans contain 36,864 development jobs across the three candidates; no holdout job file was made.

A first reproduction check failed only because JSON reordered an unordered list of constant feature names. The verifier now compares that list without order. The selected seeds and feature values did not change. Original selection code is retained under `source_snapshot/selection_tools`; validation checks that the sampling, feature, selection and build code remains unchanged.

Source: `C:\Users\Owner\Documents\Codex\2026-09-05\cont\lzhang-kaggriculture` at HEAD `a94b286fa14344d25e54d316210705485d07a70e`. HEAD alone does not describe uncommitted campaign work; [provenance.json](provenance.json) pins the actual source bytes. The frozen CPU host identifies its interpreter as Kaggriculture 1.32.7. All 16 opponent runtime file sets and all three candidate versions matched their saved hashes. See [OPPONENTS.md](OPPONENTS.md) and [candidates.json](candidates.json).

| Manifest | SHA-256 |
|---|---|
| Representative | `2e9b9c651f8d776bade17c68df406d15b38529fcdefdd0c533344a9a818d7651` |
| Stress | `9275977ccd595498c7b31086bcc9b5ad208e3ff5b988fba88573a37a67d22e54` |
| Sealed holdout | `d91ca06fe13a20beb973dfdbccf5894409eee3ec35dd1f731a2e1af3ac3fcef9` |

The holdout is procedurally separated, not encrypted or inaccessible to local agents. Its release rule is in [EVALUATION_CONTRACT.md](EVALUATION_CONTRACT.md). This task ends with these artifacts; it makes no strength, promotion or submission claim.
