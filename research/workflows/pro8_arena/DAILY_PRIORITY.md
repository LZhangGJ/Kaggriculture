# Daily tournaments take priority

Daily Bradley–Terry tournaments use WRX90, the mini PC and Vast. Each worker finishes its current batch, runs its assigned daily games ahead of other queued work, then returns to its ordinary queue. Existing continuous Elo rounds remain queued and retain their completed results.

The coordinator assigns unfinished daily games once, at a batch boundary. It keeps swapped-seat pairs together and splits work in proportion to configured full-game throughput. Current target weights are 1,834 for WRX90, 1,764 for the mini PC and 6,956 for Vast. These are relative weights, not worker counts or guaranteed speeds.

Ownership lives in private `daily-assignments` files. Remote manifests retain the original game IDs, seeds, agent versions and execution contract. Returned results go into the original daily tournament. They do not enter continuous Elo; they do enter the cumulative active-agent BT view. Repeated syncs cannot count a game twice.

New tournaments receive the same treatment automatically. The current partially completed tournament sends only its remaining games. No active game is killed to switch queues. A worker that finishes its share returns to continuous Elo while other machines finish theirs.

At the recent combined rate of roughly 10,554 games/hour, a fresh 23,296-game tournament has a capacity estimate of 2.2 hours, plus switching and sync overhead. This is a forecast, not a measured distributed tournament duration. Runtime differences can leave one host finishing later than the others.

If a host disappears, its game ownership remains reserved. Do not blindly reassign work that might still be running. Reconcile or drain the old worker before moving its unfinished assignments. Connection errors remain visible in arena status. Infrastructure retry limits remain in force.

Tests cover complete and disjoint assignment, seat-pair preservation, stable ownership on resume, coordinator exclusion of remote games, daily priority and automatic return to continuous Elo.
