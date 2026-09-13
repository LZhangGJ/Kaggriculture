# Actual resource and timing record

Original dispatch: 2026-09-13T15:20:22.689Z. Hard deadline: 17:20:22.689Z.
First container resource probe: 15:21:06.695681182Z (44.007 seconds after dispatch).
Packaging checkpoint: 2026-09-13T15:58:14.546640+00:00, 37.86 minutes after the original dispatch.
All work, failures, rebuilds and packaging belong to that single unchanged budget.

## Allocation, not host capacity

- Affinity and effective cpuset: CPUs 0-4 (5 visible).
- Cgroup CPU quota: 400000 / 100000 = 4 effective CPU equivalents.
- Memory limit: 4294967296 bytes (4 GiB).
- Initial memory.current: 311156736 bytes.
- Initial host MemAvailable: 5130528 kB, explicitly NOT treated as allocation.
- Peak cgroup memory observed by packaging: 1342181376 bytes (1.250 GiB).
- Memory headroom exceeded the requested 30%; no OOM/max events were observed.
- Disk available at final snapshot: 31803379712 bytes.
- Python 3.13.5; g++ (Debian 14.2.0-19) 14.2.0; Linux x86-64.

One compile/probe worker was used at a time. Initial timeout-bounded compilation
and 48-observation parent probe were measured before scaling to seven saved
histories and the focused tests. No full opponent games or new seed panel were
launched.

## Measured commands

| Work | Wall time | Peak process-tree RSS |
|---|---:|---:|
| Initial parent rebuild | 26.99 s | 570160 KiB |
| Selected final rebuild | 26.80 s | 571980 KiB |
| Selected root probes (5033 calls) | 26.42 s | 248544 KiB |
| Unpacked rebuild plus default --unit | 80.51 s | 589576 KiB |
| Unpacked root-entry recheck (5033 calls) | 27.02 s | 248448 KiB |

These are local container measurements, not Kaggle sandbox timing guarantees.
Raw GNU time output, stdout/stderr, command/exit receipts and cgroup samples are
in validation/logs. Initial outer-tool timeouts and their successful independent
reruns are kept, rather than silently relabeled as successful tool invocations.
