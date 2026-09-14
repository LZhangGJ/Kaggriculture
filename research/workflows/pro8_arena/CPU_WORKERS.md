# Multiple CPU workers

WRX90 owns the seed ledger, round manifests, results, and Elo ratings. The mini PC keeps its existing queue. Extra workers receive distinct complete rounds; they never share a round with the mini PC.

Configure extra workers in `.arena/private/continuous-extra-hosts.json`, keyed by a stable name such as `vast`. Each entry uses the same fields as `continuous-host.json`: `enabled`, `ssh_host`, `root`, `python`, `repository_path`, and `seeds_per_round`. Keep this file and SSH credentials off GitHub.

Use the exact validated Docker image and the same game limits on every worker. Benchmark completed games before increasing `workers`. Each game runs two isolated agent processes and a referee; thread count alone is not a safe concurrency setting.

## Replace a rental

1. Stop its executor and run a final coordinator sync. Keep the mini PC running.
2. Copy the retiring worker's arena state to the replacement over SSH, including its partial game results. WRX90 already retains every imported result.
3. Validate Docker image identity and sandbox limits on the replacement. Update only its private SSH endpoint.
4. Start one executor on the replacement and verify the next sync imports new results.

If a rental disappears before migration, restore its assigned manifests, agent packages, and saved results from WRX90. Clear the corresponding `sent.json` receipts to resend unfinished rounds, after verifying the old executor has stopped. Game IDs and central result validation prevent double counting; unreturned games may need to run again. Do not change frozen manifests or seeds.
