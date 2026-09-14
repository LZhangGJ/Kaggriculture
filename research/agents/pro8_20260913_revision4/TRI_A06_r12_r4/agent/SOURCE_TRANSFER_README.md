# TRI_A06_r12_r4 source transfer

This payload reconstructs the frozen r4 production strategy, not an economic fix1.
Parent ZIP SHA256: 0c15b55c7cf7c29e57ad632432586a18f089df8a6f642d07d354d19c6a24f7c4.
Original r4 ZIP SHA256: ad7b2da0244254dd9be24365fe33880053995275702b4dd32b2e32beaea3aef4.
Original r4 native SHA256: 70f1fd2f8cc5957ea3b57091a81f8c868e2725ec1a7aabb3d0e425d08386ac84.

All 51 production/config/build inputs and all six inputs needed by the five
self-contained C++ unit suites are hashed in source_inputs. No binary, historical
archive or large replay fixture is embedded. New text files are complete; modified
text files use a/ and b/ unified diffs. Configuration and compiler flags are unchanged.

After applying the payload, verify every source_inputs SHA256. Remove the old
policy/a06.so and policy/a06.BUILD.json before building so the parent library cannot
be mistaken for the candidate. Run: python build.py --unit
Only a C++20 g++ compiler and Python standard library are required; no network.
The receiving controller must identify its rebuilt library separately from receipt
of the original native, even if its local build happens to be byte-identical.

The retained r4 correction closes terminal rolling harvest tasks through a real
return, DROP and market sale, preserving input/precedence/capacity/time constraints.
Calendar planning, ongoing fertilizer procurement, survival-wage ordering and the
previous terminal procurement cash admission are retained. No economic prototype
is promoted by this transfer. Full 1536-game acceptance has not been established.
The complete r4 evidence archive is preserved separately; this is a source-only
transport with the required offline tests, not a replacement for historical logs.
