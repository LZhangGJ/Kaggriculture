# Actual build record — TRI_A08_r11_r1

The root `BUILD.json` is copied from the receipt emitted by the actual successful final build. It is not the parent build manifest. The identical receipt remains next to the library. All 44 production source hashes match.

## Actual command

Driver:
```sh
python -B /mnt/data/TRI_A08_work/candidate/build.py
```

Compiler: `g++ (Debian 14.2.0-19) 14.2.0`.

```sh
g++ -DA08_SALE_FLOOR_DP=1 -DA08_SALE_SCHEDULE_DP=1 -DA08_PAID_CONTINUATION=1 -std=c++20 -O3 -DNDEBUG -march=x86-64 -ffp-contract=off -DR2_STARTUP_SUPPLY_MODE=2 -DR2_LOCAL_SALE_TIMING=1 -DR2_FINITE_FERTILIZER=1 -DR2_CROP_CLOCK_MODE=1 -DR2_OBSERVE_PUBLIC_TRADES=1 -DR2_MARKET_INTEGRAL=1 -DR2_SALE_CLOCK_MODE=2 -DA08_COMPLETE_CROP_STOPS=1 -DA08_LAND_DP_MODE=2 -DA08_REALIZATION_MODE=0 -DA08_CROP_PORTFOLIO_MODE=1 -DA08_CASH_LEDGER=0 -DA08_PREFIX_DAYS=5 -fPIC -shared -Wl,-Bsymbolic /mnt/data/TRI_A08_work/candidate/policy/bridge.cpp /mnt/data/TRI_A08_work/candidate/policy/executor/vendor/simulator.cpp -o /mnt/data/TRI_A08_work/candidate/policy/tri_a08_r11_r1.so
```

The absolute paths document where this build happened. The equivalent relocatable command is `python -B build.py` from the extracted release. C++20, `-O3`, `-DNDEBUG`, baseline x86-64, `-ffp-contract=off`, and every feature define above were used. No package download, runtime compilation, other policy binary, or network fetch is required for the production build.

## Actual outputs and measurements

- Library: `policy/tri_a08_r11_r1.so`, **1,248,560 bytes**.
- SHA256: `bb2f248f689b7a75042c7bf4897700099f1f6d1980b1abc38442db0a1bbadc4c`.
- Compiler-driver receipt duration: 26.936783 seconds.
- GNU time wall clock: 27.52 seconds; maximum RSS: 594,820 KiB; exit 0.
- Clean production-only rebuild: 28.15 seconds; maximum RSS 594,728 KiB; exit 0; native bytes identical.
- Root default entry smoke in that clean tree: two seats ×48 legal inputs, 96/96 saved-prefix actions match; no binary override.

Raw records: `validation/logs/margin_build.*`, `clean_rebuild.*`, `CLEAN_REBUILD_CHECK.json`, `CLEAN_DEFAULT_ENTRY.json`, `native_abi.txt`, and `parent_full_source_identity.json`. `CLEAN_REBUILD_CHECK.json` includes the independent clean build receipt. Whole-tree checksums are in `MANIFEST.sha256`; this file is not part of the 44 production-source compilation manifest.

## Runtime dependencies and bounds of verification

ELF64 Linux x86-64 shared object, dynamically linked to libstdc++, libm, libgcc_s, and libc. Inspected symbol requirements include GLIBCXX_3.4.31, CXXABI_1.3.9, and GLIBC_2.32. The recorded environment used Python 3.13.5 and Debian GCC 14.2.0. This confirms successful local loading, not compatibility with every Linux image or Kaggle resource-limit harness. Recompile offline with the target toolchain as needed; preserve the new source/native receipt rather than claiming a different-toolchain binary has this SHA.

## Ablation

`python -B build.py --sale-floor-dp 0 --out /tmp/tri_floor_off.so` disables only this added branch. The retained actual ablation binary is `research/ablations/floor_off.so` (SHA256 `dd3b27502f12ef5c770c3ed6171c5e87b425c1e435cf9decfad7fd3fe9ad249c`) with its matching receipt. On all seven supplied legal observation paths it reproduces 5,033/5,033 original A08_r11 actions. It is not the root production binary.
