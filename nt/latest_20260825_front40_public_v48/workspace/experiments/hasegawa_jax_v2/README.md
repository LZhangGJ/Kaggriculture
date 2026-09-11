# Hasegawa JAX V2

V2 is a behavioral reconstruction from public Hasegawa Replays, not a source-code
port. It replaces V1's daily nearest-Replay stitching with one coherent atomic
action program selected once from visible first-shop state.

Runtime path:

```text
visible state -> lock first-shop branch once -> coherent 719-step atomic program
              -> check current action preconditions
              -> execute valid atomic action OR repair only the affected action
              -> preserve market transaction order and cap impossible quantities
              -> official JAX simulator -> observe actual result -> continue
```

V2 does not call the old compound-task execution core. Its local guards check:

- current tile and unit position before every operation;
- current seed, carried item, shed stock and structure requirement;
- market sell availability and shed capacity in original transaction order;
- weed interruption and replay-action resumption;
- unit-position deviation and local movement recovery;
- terminal inventory liquidation.

V2 deliberately does **not**:

- select a new Replay route every day;
- clear all unit work at midnight;
- rebase completed-operation ledgers to another Replay;
- claim exact Hasegawa reproduction before milestone validation passes.

The trace is evidence for Hasegawa's interleaved task ordering. Runtime repairs
use only the current visible/own state. No future shop or opponent action is an
input to branch selection.
