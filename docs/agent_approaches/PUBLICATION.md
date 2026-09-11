# Publication record and artifact identities

Date: 2026-09-11. Branch: `docs/two-agent-approaches-20260911`.

## Source snapshots

The branch starts from P16 research publication commit **`4ef9875968f7104f26fc3140bd28b988101579f8`** on `agent/add-nt-simulator-orbit-migrations`. It retains that branch's source packages, compiled agents, tests and compressed evidence.

The route-clustering package is imported from `main` commit **`eec409775bc538cf65b3676172ee3ce741ca2469`**. Its original source package had 74 files. This handoff retains 66 of those files byte-for-byte, edits four documentation files and two offline packaging scripts, and replaces/omits two older method documents. It adds an English method guide, a focused export test and byte-preserving Git attributes.

The route-clustering frozen `main.py` and all `runtime/` files match the source commit's Git blobs. Native simulator and policy algorithm source are also preserved. The package's earlier server recovery provenance remains in [SOURCE_PROVENANCE.md](../../agents/route_clustering_switch_agent/SOURCE_PROVENANCE.md).

## What changed in this publication

- Added the [two-approach overview](README.md), the [route-clustering method](../../agents/route_clustering_switch_agent/docs/METHOD.md) and the [P16 method](P16_METHOD.md), with components, construction steps, execution behavior, evidence and limits.
- Added navigation from the repository README and development guide.
- Translated the imported route package's README, maintenance instructions and omissions record into English; extended its provenance record.
- Replaced the imported Chinese method Markdown with the English guide and omitted the duplicate Chinese PDF. Their originals remain in the source `main` commit. Existing SVG charts remain included.
- Fixed `export_teammate_meta_submission.py` to resolve development modules under `src/meta_agent`, and to accept/record the selected forced opening.
- Fixed `run_pipeline.sh` to pass the final selected opening to both exporters. Previously, re-export could silently use the Nash opening mixture although the frozen deployment and final evaluation used `G001`.
- Added a focused export check that copies frozen assets, imports both newly exported forms, and compares opening, route count, checkpoint behavior and reset behavior against the two frozen deployment forms.

No trained trees, route tapes, frozen submission programs, P16 native libraries or economic parameters were changed. No new Kaggle submission, training run or tournament was performed.

## Exact identities of the two original submissions

| Artifact | SHA-256 | Meaning |
|---|---|---|
| Original R1 portable archive | `c421a416b538df4eb4ad2da2d7598504c52dedd691537ab6fc246c3dec6c8d8d` | Submission `56146577`; archive in the R1 release matches the downloaded file tested locally |
| Original R1 native library | `7a8783c776ab8c01e3370685f6de6f8b6a43ea409ab710fbe9330f165560950f` | Library inside that portable archive |
| User-provided original R2 archive | `363101251f64cd1967c0203812d30782c57daf39c117e7852811b26b8f6db804` | `submission (6).tar.gz`; original archive is retained on the source machine |
| Original R2 native library | `1ead09a9bd48b20b512fb8fe57bbbbd87c86bb12fc9b1b42553e5c5b5bec121c` | Exact tested library, published with its runtime wrapper in the evidence package |
| Later R1 workflow-repair default library | `e285b58c4c35ea95ff522d8848a3f91afc1bb1156d1d637f585a03bb5deb1ca8` | Separate workflow-on R1 build with inspectable C++ source |
| Later R2 workflow-repair default library | `179b204db64a32e08af3afb34ae1e6687737c71e4d506f37c4dd3ebe0ec591f2` | Separate workflow-on R2 build; not the opponent in the local 100-seed R2 panel |

Original R2 is associated with `56149565` by acquisition context and its own entry label, not an independently authenticated Kaggle archive hash. Its archive contains a compiled policy and Python wrapper; complete exact C++ source is not supplied. The related later R2 implementation is included under its own identity. Different binaries must not be treated as the same tested program merely because both names contain “R2.”

The dated acquisition and identity receipts are [SUBMISSIONS.json](../../evidence/r2p16_route_repair_20260911/SUBMISSIONS.json) and [IDENTITY.json](../../evidence/r2p16_route_repair_20260911/IDENTITY.json). Public leaderboard scores in historical records are snapshots, not current ratings.

## Verification

Run these commands from the repository root:

```bash
python docs/agent_approaches/verify.py
python evidence/r2p16_route_repair_20260911/verify.py
python evidence/r2p16_route_agents_20260911/verify.py
```

The new verifier checks [ARTIFACTS.json](ARTIFACTS.json): package bytes, imported unchanged Git blobs, selected P16 runtime identities and the library inside the original R1 archive. It requires neither Git credentials nor native-library loading. Git attributes preserve the recorded bytes on fresh checkouts.

The existing evidence verifiers passed during this publication: 256 checked files and 3,150 recorded outcomes in the latest package, plus 132 checked files in the earlier package. The latest package includes receipts for 4,536,000 observations; this verification reads archived evidence and does not replay all transitions.

The focused route package tests cover one-switch/defer behavior and multi-file/single-file export. The four policy forms are imported in isolated Python processes, including the original frozen entries. This checks packaging and controller behavior on a small synthetic observation sequence; it is not full-game action equivalence or runtime-budget certification.

Validation details and executed-check results are recorded in [CHECKS.json](CHECKS.json). English text and local document targets are also checked before publication.

## Boundaries of the handoff

The full route-clustering data corpus and search matrices are not shipped. Its 89.615% holdout score remains an archived author result. The P16 full replay collection and intermediate builds remain on the source workspace; compact per-game records, receipts and exact runtime artifacts are included.

The route-clustering native simulator targets default rules version 1.32.7. The general repository lab describes an older 1.32.6 setup, and the P16 experiments use their recorded compiled host. The handoff does not collapse these into an assumed identical environment or rerun their native differential suites.

No common head-to-head panel between the two approaches was run here. The documents explain both methods and their respective evidence; they do not declare one universally strongest or promote the locally regressing procurement-repair candidate.
