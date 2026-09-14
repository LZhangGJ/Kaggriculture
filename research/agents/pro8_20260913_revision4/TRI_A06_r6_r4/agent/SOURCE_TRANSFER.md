# TRI_A06_r6_r4 source transport (not an optimization)

Apply only to a clean extraction of the exact fix1 parent archive:
SHA256 ed0723c90fba40808b597b042800e90dbf6e45a6b5d2b7918412267b1d479dd5.
The delivered JSON uses UTF-8, unified diffs with a/ and b/ paths for existing
text, and complete text for new files. Verify every before/after hash and every
source_inputs entry. No native binary, trace payload, historical ZIP or log is
transmitted. All unmodified parent files remain in place; deleted_paths is empty.

Production identity is EXACTLY the frozen r4: 42 policy inputs, the unchanged
root main.py, build.py, build_a06.py and compiler flags. Only the existing
policy/executor/policy.hpp changes; delivery_commitment.hpp is added. No other
production input changes. SOURCE_HASHES.json, IDENTITY.json and the development
seed inventory are copied from the frozen r4. No new seeds or matches.
Expected rebuilt policy/a06.so SHA256:
9c37d066f024d9f385390808552783246db22001f555102db9bb0e323756994c.
This bit identity is checked with the recorded GNU 14.2.0 toolchain; other
compiler versions, including central GCC13.3, need not emit identical bytes.

The parent native remains in the extracted tree until overwritten by the build.
DO NOT run the patched entry against that old library. Rebuild first:
    python build.py --unit

The original build.py and flags are unchanged. The complete parent default
tests/run_units.py suite is retained verbatim with a final invocation of the r4
test runner appended; no duplicate inherited test script is required.
All r4 C++ tests, terminal rule tests, audit adapter and verification logic are
unchanged. Only tests/r4/run_units.py has a transport-specific fallback: without
the four round4 replays, it executes synthetic delivery tests and the artificial
terminal/official-rule cases using an already present parent trace as a schema
shell. The official rule files in the parent and r4 were byte-compared identical.
Warm round4 branches and round4 common-prefix replay checks are explicitly
reported as NOT RERUN, never passed or replaced with different match results.
They run as originally delivered when the complete evidence/r4_feedback is
present. Large replays are not required for this text package's default --unit.
The original focused tests/build_fix1.py include-path repair (-I<root>/policy)
is included exactly, and its inherited dependencies remain available.

source_inputs is a superset of the 46 production/build inputs: it also hashes
all supplied test scripts, reused test fixtures and official rules, plus small
identity/transfer metadata. Test-only fallback hashes deliberately differ from
the full evidence package, not the production sources or fixed configuration.
Original BUILD, README and historical manifests left in the parent remain
historical records, not the transport's new verification receipts.
