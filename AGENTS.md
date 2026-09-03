# Repository instructions for coding agents

Before changing this repository, read [`agent.md`](agent.md) and the nearest
module or experiment `agent.md`.

A code contribution is incomplete until its documentation is updated:

1. New modules go under `agents/<module>/` and must contain `agent.md`.
2. New experiments go under `research/experiments/<experiment>/` and must contain
   `agent.md` plus the code/configuration needed to understand the run.
3. Update the module, experiment, and data indexes in the root `agent.md`.
4. Record executed results separately from plans or unverified claims.
5. Keep large Replay files, caches, native builds, and raw matrices outside Git;
   document their location, content, size, and hash in the experiment `agent.md`.
6. Run `python scripts/check_agent_docs.py` before committing.

Do not overwrite unrelated local changes. Generated submission files must be
regenerated from their source rather than edited by hand.
