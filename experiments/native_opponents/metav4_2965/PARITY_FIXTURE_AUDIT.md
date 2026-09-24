# Meta-family native parity fixture audit (2026-09-23)

The route-tape asset is **not** the Python table immediately after import.
`generate_assets.py:load_source()` explicitly calls
`module._alt_install(module._ALT_MODE)` before serializing routes. The public
agent performs the same lazy install on its first step-0 call. Consequently a
valid standalone comparison must drive both sides with the post-install route
table. `check_parity.py` now calls `_alt_install` and freezes a deepcopy of the
rival route **before** the first agent action; this prevents later mutation of
the fixture while matching the exported C++ asset.

A temporary diagnostic froze the table *before* `_alt_install`. It made all
variants fail, sometimes at step 144, because Python's rival opening lacked
orders that the C++ asset contained. That diagnostic was invalid; its failure
does not establish a C++ policy regression. It has been superseded by the
post-install fixture gate below. The previous dynamic Python fixture happened
to acquire the installed opening when the agent ran step 0, so its successful
results were not contradicted, but the new explicit freeze is auditable.

Precondition checked: for the Meta source, all 41 routes x 719 frames = 29,479
post-install Python actions match the C++ `fixture_route_action()` exactly,
including empty market slots. Soil, V57, and V15 asset payloads are byte-for-
byte equal to Meta's after the 44-byte magic/version/source-hash header.

Post-install frozen-fixture complete-game parity (each seed runs both seats,
719 steps, and compares emitted action at every step):

| Variant | Rival route | Seeds | Exact games | Report |
| --- | ---: | --- | ---: | --- |
| Meta | 0 | 2609500300..303 | 8/8 | `work/new_public_opponents/soil-current/meta-postinstall-fixture-4seed.json` |
| Soil current | 0 | 2609500300..303 | 8/8 | `work/new_public_opponents/soil-current/soil-postinstall-fixture-4seed.json` |
| Soil current | 105 | 2609500400..403 | 8/8 | `work/new_public_opponents/soil-current/soil-postinstall-route105-4seed.json` |
| V57 | 0 | 2609500600..603 | 8/8 | `work/new_public_opponents/kaggriculture-v57-funding-order-invariant/v57-postinstall-fixture-4seed.json` |
| V57 | 105 | 2609500700..703 | 8/8 | `work/new_public_opponents/kaggriculture-v57-funding-order-invariant/v57-postinstall-route105-4seed.json` |
| V15 experimental | 0 | 2609500600..603 | 0/8 | `work/new_public_opponents/kaggriculture-v15stack-submit/v15-postinstall-fixture-4seed.json` |

These are standalone calibration fixtures, not the production JobBatch path.
Soil has a separate isolated JobBatch student-vs-Python action/terminal parity
and scalar old-policy replay gate in `SOIL_CURRENT.md`. V57 is **not** wired
to JobBatch; V15 is incomplete. No new variant has been added to the live RL
opponent pool. The running default Meta `.so` was accidentally rebuilt during
this investigation; its prior binary was not preserved, so prior-SHA direct
binary equivalence cannot be claimed. The current default and later isolated
build each passed the same Meta Python oracle on seed 2609500300..303, but this
is an indirect, sample-limited check.
