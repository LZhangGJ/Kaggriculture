# A06 R14-F5L3 — local improvement candidate

Primary frozen panel: 95/160 (59.375%), versus original R12 68/160.
Additional check: 37/40, only 4 games per opponent.
Neither the per-opponent 80% nor 90% target was met.

Keep main.py and the entire policy/ directory together.
Linux x86-64 native is included; no ML or RL framework is needed.
Rebuild when needed: python build.py --unit
API checks: python tests/run_units.py
Forecast invariants: python tests/run_forecast.py

Full report, original-source hashes, raw match records, and reproduction tools
are in the accompanying experiment archive. Platform time/memory constraints
have not been fully enforced.
