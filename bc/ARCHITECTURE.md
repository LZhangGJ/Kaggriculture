# Kaggriculture behavior cloning architecture

Snapshot: September 18, 2026. Based on the implemented `exact-bc-v1` code.

The system learns from recorded games, saves a policy checkpoint, and uses that
policy to choose worker actions and market requests. One model covers the whole
season. It has **5.31 million parameters**.

## Replay to trained agent

```mermaid
flowchart TD
    A["Kaggle replay archives<br/>22,185 current-engine games"] --> B["Preserved observations and raw actions<br/>44,370 player trajectories"]
    B --> C["Audit against original archives<br/>Check rules, configurations and action alignment"]
    C --> D["Regenerate exact action labels<br/>Worker commands, market requests and quantities"]
    D --> E["Numeric feature cache<br/>Build once using 32 CPU workers on WRX90"]
    E --> F["Recurrent behavior cloning<br/>Two GPUs learn from the recorded actions"]
    O["Previous model checkpoint<br/>Compatible encoder weights only"] --> F
    F --> G["New policy checkpoint<br/>Encoder, memory and action decoders"]
    G --> H["Full-game evaluation<br/>Measure actual play against opponents"]
    H -. "After evaluation and selection" .-> I["Internal arena<br/>Continuous Elo and Bradley–Terry results"]
    H -. "Future stage" .-> J["PPO improvement<br/>Learn from rewards against an opponent league"]

    classDef data fill:#edf3fc,stroke:#5274a3,color:#162a42;
    classDef active fill:#fff1ce,stroke:#b88216,color:#513600;
    classDef ready fill:#e9f4ee,stroke:#4d8366,color:#183d28;
    classDef later fill:#f3f3f3,stroke:#999,color:#444,stroke-dasharray:5 5;
    class A,B,C,D data;
    class E active;
    class F,G,H ready;
    class I,J later;
```

**Snapshot:** seven completed BC epochs and full-game evaluations are available. The numeric cache is complete. PPO integration remains future work.

All 44,370 trajectories are included, with no development or test holdouts.
Unsupported action labels are masked and counted without removing their games.
The vocabularies contain 34 worker quantity values and 233 market quantity values;
context and grammar determine which quantity choices apply.

## How the policy makes one turn's decisions

```mermaid
flowchart TD
    A["Current observation<br/>Board, workers, inventory, prices,<br/>season clock and public history"] --> B["Shared game encoder<br/>Board CNN + entity features<br/>3 Transformer layers, width 192"]
    B --> C["Actor memory · GRU 256<br/>Carries information between turns"]
    P["Previous turn's actor memory"] --> C
    B --> V["Outcome memory · separate GRU 256<br/>Predicts win / draw / loss during training"]
    VP["Previous turn's outcome memory"] --> V
    B --> R["Board-cell, worker and item representations"]
    C --> W["Worker decoder<br/>Farmer first, then each hired worker<br/>PASS or command, then quantity if needed"]
    R --> W
    W --> X["Resolve the chosen worker prefix<br/>Using the pinned game rules"]
    X -->|"Updated provisional state for the next worker"| W
    X -->|"All worker requests chosen"| Y["Resolve the complete worker phase<br/>Including whole-turn planting cancellation"]
    Y --> M["Market decoder · fresh request memory<br/>END or request, then quantity if needed<br/>Up to 10 market requests"]
    C --> M
    R --> M
    M --> L["Update the market request ledger<br/>Track requests and estimated costs"]
    L -->|"Next request"| M
    W --> Z["Submit one complete action<br/>Farmer + hands + market requests"]
    M --> Z
    Z --> E["Actual game engine<br/>Resolves both players and advances the game"]
    E -->|"Next observed turn"| A

    classDef input fill:#edf3fc,stroke:#5274a3,color:#162a42;
    classDef model fill:#e9f4ee,stroke:#4d8366,color:#183d28;
    classDef rules fill:#fff1ce,stroke:#b88216,color:#513600;
    class A,P,VP input;
    class B,C,V,R,W,M model;
    class X,Y,L,Z,E rules;
```

The encoder runs once per turn. The smaller decoders reuse its representations
while local features track changes from requests already chosen. They do not see
future recorded actions.

Worker states remain provisional until every worker request is known: an excess
of planting requests can cancel earlier planting of that crop. The market
decoder starts with the fully resolved worker state. Requested purchases do not
become inventory until the real game engine fills them.

During behavior cloning, recorded actions supply each decoder's preceding
choices. During play, the model's own choices supply them. Both paths use the
same action builders and game-rule logic.

## Training and hardware

| Part | Implementation |
|---|---|
| Replay preparation | WRX90 CPU; 32 parallel workers |
| Workstation | Threadripper PRO 9965WX; 24 cores / 48 threads; about 183 GiB system RAM |
| Neural training | Two RTX PRO 6000 GPUs, 96 GB each; PyTorch distributed data parallel |
| Proposed production batch | 64 player trajectories globally; chunks of 16 turns; BF16 |
| Season memory | Persists across all 719 turns; gradients stop at each chunk boundary |
| Training objective | Separate means for worker and market gate, command and quantity losses; outcome loss weighted 0.05 |
| Initialization | Compatible encoder weights may be reused; recurrent states, action heads and optimizer start fresh |
| Evaluation | Fresh full games first; internal arena results after an agent is added |

The current policy learns **both worker actions and economic market requests**.
Behavior cloning teaches it to reproduce recorded decisions. PPO is a later
improvement stage, not part of the current cache build or BC training job.

The arena is a separate evaluation service, previously configured to run on
the mini PC and Vast. Its live hardware status was not checked for this diagram.
Passing integration tests or reducing training loss does not establish playing
strength; full-game evaluation supplies that evidence.

## Code map

- `exact_features.py` and `exact_actions.py`: observation and action features.
- `exact_model.py`: shared encoder, recurrent states and action heads.
- `exact_decoder.py`: replay supervision and live decisions.
- `worker_phase.py`: pinned worker-phase game rules.
- `cache_exact.py`: regenerated labels and numeric cache.
- `exact_training.py` and `train_exact.py`: recurrent training and two-GPU execution.
- `exact_identity.py`: cache, source, engine and vocabulary compatibility.
