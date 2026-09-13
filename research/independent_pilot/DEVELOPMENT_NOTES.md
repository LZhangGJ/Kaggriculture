# Development checks and repairs

Before freezing the pilot:

- The first policy replay check differed by 0.00048828125 in log probability. Large-support Binomial log probabilities used float32. Computing them in float64 reduced the checked maximum to below 0.000002. This was a failed correctness check, not a training result.
- The old full-action market menu could reuse optimistic cash after an uncertain sale. The new decoder carries a guaranteed cash floor and bounds product prices against the rival's maximum possible purchases per market slot. It keeps within-turn commands jointly affordable under those bounds.
- The first external-opponent smoke failed to load a binary because the default Ubuntu 22.04 libstdc++ lacked GLIBCXX_3.4.31. The pilot uses the existing Ubuntu 24.04 WSL distribution with an isolated Python environment. Opponent files remain unchanged.
- Two input-preparation attempts stopped before training: WSL Git could not resolve the Windows worktree path, then a receipt path remained relative. The launcher now accepts the verified base commit explicitly and resolves the output directory. Both abandoned preparation directories remain; `run-v3` is the first frozen trial run.

Early checks and throughput measurements used Ubuntu 22.04. Final checks and the pilot repeat on Ubuntu 24.04, where the frozen opponent binaries can load. Neither environment uses a GPU. The existing evaluation processes keep running.
