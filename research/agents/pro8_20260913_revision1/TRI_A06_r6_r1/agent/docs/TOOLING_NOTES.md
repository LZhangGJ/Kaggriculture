# Tooling notes and honest measurement boundaries

The attachment was verified against the user-supplied SHA256 before extraction.
The initial cgroup/affinity/memory/disk/compiler receipt was written at
2026-09-13T02:18:23.287461Z, within one minute of the dispatch timestamp.

The first baseline compile used a shell `timeout` plus `/usr/bin/time -v` inside a
container tool call. That outer tool call returned a timeout before the compiler
finished, while the child compiler continued and completed. The baseline
builder's own receipt reports 72.72245355 seconds and a native hash identical to
the supplied parent. The empty `parent_short_compile.time` is NOT a peak-memory
measurement. No successful `/usr/bin/time` measurement is claimed for that call.
The later bounded resource monitor measured every listed validation run and the
actual revised compile, with exit status, wall time, sampled process-tree RSS,
cgroup memory usage, and timeout status in separate JSON receipts.

A first exploratory manifest-key probe found zero entries under a key named
`sources` because the supplied top-level parent BUILD uses
`compiled_source_checks`. It was not a source-integrity validation. The subsequent
`parent_source_verified.json` checks all 40 entries under the actual key. The
revised production integrity receipt separately verifies all 40 source hashes
against the compiler's real native build receipt. This schema discovery did not
change the parent or candidate source.

`*.running.json` files record how a monitored process started, not a claim that it
is still running. The matching `*.receipt.json` is the completion record. All
final success statements use completion receipts. Historical test JSON under the
unaltered input agent remains historical and is not counted as a new test run.

All time limits refer to this original dispatch: 2026-09-13T02:17:41.523Z, with
hard deadline 2026-09-13T04:17:41.523Z. No follow-up clock or reset was introduced.
