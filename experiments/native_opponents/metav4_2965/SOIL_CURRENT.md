# Soil Remembers Rain: opt-in native opponent

Public source: `prvsiyan/kaggriculture-frontier-the-soil-remembers-rain`, Kaggle version 35 as downloaded on 2026-09-23, Apache-2.0. The downloaded `output/main.py` SHA-256 is `178ae0f727641cf4b618ebb98ade7aa1a1bed7517281aab9849de82a59d8ed3a`. Its extracted `soil_current.assets.bin` SHA-256 is `32b220d8d9c622d36afeee1b672073ecd7ef820114821bf222e993843a41dc98`. This is a distinct version from the older local Soil snapshot; do not label the old snapshot as the leaderboard version.

The public Python script remains the action oracle. `metav4_2965::Opponent(asset, true)` reuses the Meta chassis with Soil-specific layers. Standalone complete-game active-route parity was 16/16 games for seeds `2609500300..307` versus route 0 and 8/8 for seeds `2609500400..403` versus route 105; each game compared all 719 actions. Reports are in `work/new_public_opponents/soil-current/soil-after-r97-8seed.json` and `soil-route105-4seed.json` on the research host. Reproduce with `check_parity.py --source work/new_public_opponents/soil-current/output/main.py --assets experiments/native_opponents/metav4_2965/soil_current.assets.bin --soil-variant --steps 719 --rival-route 0` and an explicit fresh seed range.

JobBatch code 6 is optional, never part of the continuous runner's default opponent mix. The isolated build is made without replacing the live module:

```bash
BUILD_DIR=work/new_public_opponents/soil-current/jobbuild \
  bash experiments/native_student_rollout/build.sh
```

In a separate 2-game JobBatch smoke (`seed=2609500500`, both seats, policy RNG `111/222`, v45 actor), all 1,438 Soil actions matched the Python oracle along the JobBatch's actual student actions; both terminal cash pairs matched exactly: `71599:71287` and `67834:64376`. There were 17 actor days per game, 971 events total, and no runner error.

The same two games exposed an actor-kernel numerical gate, not an opponent action defect. Default aarch64 NEON JobBatch old-policy replay on 517 actionable events had max/mean absolute log-prob drift `6.51e-5 / 6.06e-7`, KL `6.80e-12`: **FAIL** at the existing mean tolerance `5e-7`. A scalar-only isolated build of the same source and inputs had `2.15e-5 / 2.34e-7`, KL `8.36e-13`: **PASS**. The two builds had identical prefix/action/actor hashes and terminal cash. Build the scalar control with:

```bash
BUILD_DIR=work/new_public_opponents/soil-current/jobbuild_scalar \
  CXX_BIN="$PWD/experiments/native_opponents/metav4_2965/scalar_cxx.sh" \
  bash experiments/native_student_rollout/build.sh
```

`STUDENT_SCALAR_LINEAR` only disables the optional NEON linear kernel; normal builds are unchanged. Do not admit Soil to live training until the actual actor build used there passes the unchanged old-policy replay gate. The source file and isolated build directories in `work/` are local diagnostics, not repository artifacts.
