# Arena priority evaluations

PPO's CPU controller sends evaluation seed pairs to the existing Vast and mini-PC
arena executors. The GPU trainer and its checkpoint contract stay unchanged.

Each executor checks this queue before its next regular arena batch. Games already
running finish first. Evaluation holds the same scheduler lock, so ordinary arena
games cannot compete for those CPU slots. When the queue drains, ordinary games
resume. Evaluation results never enter the leaderboard or its rating files.

Vast uses 32 single-threaded neural workers; the mini PC uses six. Both use a
separate PyTorch 2.11 CPU installation and the pinned official engine. Vast keeps
its guarded chroot sandbox for public agents; the mini PC keeps Docker. The policy,
features, action decoder, seeds, greedy decisions, and promotion rules do not change.

Promotion head-to-head and history-neural games stay on Vast. In the cross-host
check, Vast matched all six WRX90 games exactly. The mini matched all outcomes and
five final-cash pairs; one neural matchup had different cash. Its public-agent and
PASS checks matched exactly, so it handles those panel jobs. Do not claim bitwise
neural-policy equivalence across CPU types.

The dispatcher copies each checkpoint/archive once, checks its SHA, and pins the
model source and engine hashes. Each result records both model identities, seed,
seat, terminal status, cash, and outcome. The coordinator rejects mismatched or
duplicate assignments and independently checks terminal results and scores.

`dispatch-plan.json` preserves host assignments across restarts. Each completed
seed pair has an atomic receipt. A controller restart reattaches to existing work
instead of changing ownership or rerunning finished games. Queue failures leave
receipts and return CPU capacity to normal arena work.

## Paths

- Coordinator: `/home/keith/kaggriculture-ppo-production-20260919/arena-eval`
- Vast: `/opt/ppo-eval`
- Mini PC: `/home/keith/ppo-eval`
- Host scheduling hook: `tools/arena/priority_eval.py`
- Host hook configuration: `.arena/priority-eval.json`

`run_controller.py` wraps the reviewed controller with this evaluation transport.
The controller still owns snapshot admission, screening, confirmation, and
promotion. Its active launch record points to the wrapper. Do not launch a second
controller or restart the stopped parent campaign.

`controller-migration.json` records the switch. The temporary controller-only STOP
marker drains its old local evaluation workers; it does not stop the GPU trainer.
Remove only this maintenance-owned marker when the old controller has exited and
the checked replacement is ready. Never clear a user-created stop marker.

## Checks

Run `python -m unittest test_dispatcher` with the pinned model source on PYTHONPATH.
This checks receipt rejection, priority queue return to normal work, and stable
assignments on resume. Full-game checks compare both seats against PASS, a public
agent, and a neural checkpoint on each remote host with existing WRX90 receipts.
Keep those diagnostic games out of production evaluation totals.
