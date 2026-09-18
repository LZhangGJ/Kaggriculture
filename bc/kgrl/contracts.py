"""Integrator-owned wire contracts. Workers must not edit this file.

Only JSON-compatible values cross executable boundaries. Raw official observations
preserve ordered inventory entries. Simulator snapshots are privileged and never
appear in ActorFrame. Tensor annotations are specified in torch_api.py.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Literal, Protocol, TypeAlias, TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np

JSON: TypeAlias = None | bool | int | float | str | list['JSON'] | dict[str, 'JSON']
JSONObject: TypeAlias = dict[str, JSON]
Observation: TypeAlias = JSONObject
RawAction: TypeAlias = JSONObject
Split: TypeAlias = Literal['correctness', 'train', 'dev', 'holdout']

API_VERSION = 'kgrl-v1'
HORIZON = 719
WIDTH = 256
LEGACY_TILE_CHANNELS = 48
LEGACY_GLOBAL_DIM = 128
ENTITY_DIM = 64
CONTEXT_DIM = 32
FEATURE_DIMS = {'legacy59-v1': 59, 'program64-v1': 64, 'latent8-v1': 8}
# Explicit v1 domain, not a claim that the official engine cannot accept more.
MAX_SEED_EXCLUSIVE = 2**40  # seed*1_000_003 remains within native uint64.
ENGINE_SHA256 = 'bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e'
SPEC_SHA256 = 'a82c89c1a2315b93f39775d8e025471a01b738647c9772658368ee6b1b6f4867'
HISTORICAL_CHECKPOINT_SHA256 = 'd26dd5d28f3ea4b36296d9554306752542efe759913763c79d292df07a3f9a5f'
ITEMS = ('WHEAT','CARROT','TOMATO','STRAWBERRY','MELON','EGG','MILK','WOOL',
         'FERTILIZER','GOOSE','COW','SHEEP')
VERBS = ('WAIT','CONTINUE','CANCEL','NORTH','SOUTH','EAST','WEST','PLANT','WATER',
         'HARVEST','FERTILIZE','DIG','BUILD_COOP','BUILD_PASTURE','PLACE','FEED',
         'CARE','COLLECT_FERTILIZER','DROP','PICKUP','STOP','HIRE','BUY_LAND',
         'BUY_SEED','BUY_PRODUCT','BUY_ANIMAL','SELL')
SHOPS = ('BAKERY','BRUNCH_SPOT','FARMERS_MARKET','ICE_CREAM_SHOP',
         'PET_CAFE','PIZZA_SHOP','SMOOTHIE_SHOP','YARN_STORE')
DEFAULT_CONFIG: JSONObject = {
    'episodeSteps':720, 'actTimeout':1, 'boardSize':10, 'startingMoney':3000,
    'maxMarketOrdersPerTurn':10, 'turnsPerDay':24, 'shedCapacity':100,
    'weedSpawnChance':0.005, 'townShopUnlockInterval':3,
    'townShopSellInterval':4, 'townCenterSellInterval':24,
    'farmHandCostMult':1, 'marketParams':{},
}

class ContractError(ValueError): pass
class UnsupportedConfiguration(ContractError): pass
class CapacityError(ContractError): pass
class ParityError(ContractError): pass
class EpisodeFault(ContractError): pass

@dataclass(frozen=True, slots=True)
class RuleSpec:
    engine_sha256: str
    specification_sha256: str
    configuration: JSONObject
    package_version: str = '1.32.7'
    api_version: str = API_VERSION

@dataclass(frozen=True, slots=True)
class DecodeConfig:
    mode: Literal['sample','greedy'] = 'sample'
    temperature: float = 1.0

@dataclass(frozen=True, slots=True)
class PolicyConfig:
    kind: Literal['primitive_v1','commitment_v1']
    width: int = WIDTH
    max_hands: int | None = 16
    max_quantity: int = 99_999
    max_active_commitments: int = 32
    max_edits_per_turn: int = 8
    decoder_guard: int = 4096
    latent_plans: int = 8
    plan_turns: int = 24
    entity_layers: int = 3
    entity_heads: int = 4
    # All limits belong to the policy grammar, never to official/native physics.

@dataclass(frozen=True, slots=True)
class Command:
    op: str
    item: str | None = None
    quantity: int | None = None
    def to_list(self) -> list[JSON]:
        out: list[JSON] = [self.op]
        if self.item is not None: out.append(self.item)
        if self.quantity is not None:
            if self.item is None: raise ContractError('quantity requires item')
            out.append(self.quantity)
        return out

@dataclass(frozen=True, slots=True)
class GameAction:
    farmer: Command = field(default_factory=lambda: Command('PASS'))
    hands: tuple[Command, ...] = ()
    market: tuple[Command, ...] = ()
    def to_wire(self) -> RawAction:
        return {'farmer':self.farmer.to_list(),
                'hands':[c.to_list() for c in self.hands],
                'market':[c.to_list() for c in self.market]}

@dataclass(frozen=True, slots=True)
class ActorFrame:
    observation: Observation
    previous_observation: Observation | None
    previous_action: GameAction | None
    control: JSONObject  # own serializable controller state BEFORE this turn
    reset: bool

@dataclass(frozen=True, slots=True)
class Choice:
    key: str                    # unique within a Menu; stable ordering required
    payload: JSONObject         # local executable decision, never future outcome
    features: tuple[float, ...]
    quantity_low: int = 1
    quantity_high: int = 1

@dataclass(frozen=True, slots=True)
class Menu:
    stage: str
    feature_schema: str
    choices: tuple[Choice, ...]

@dataclass(frozen=True, slots=True)
class FactorTrace:
    menu: Menu
    selected: int
    quantity: int
    old_log_prob: float          # conditional selection PLUS quantity density
    forced: bool = False         # deterministic executor step, zero PPO density

@dataclass(frozen=True, slots=True)
class TurnTrace:
    factors: tuple[FactorTrace, ...]
    action: GameAction          # actual issued action, including final turn
    control_after: JSONObject
    # Actor replay must rebuild/recheck menus under recorded selected prefixes.
    # No rewards, final cash, opponent identity or simulator snapshots here.

class CommitmentStatus(IntEnum):
    PENDING=0
    ACTIVE=1
    BLOCKED=2
    COMPLETE=3
    CANCELLED=4
    FAILED=5

@dataclass(frozen=True, slots=True)
class Commitment:
    id: int
    kind: Literal['crop','animal','expand','transport','maintenance','liquidate']
    targets: tuple[tuple[int,int], ...]
    item: str | None
    quantity: int
    worker_budget: int
    cash_budget: int
    reserve_items: tuple[int, ...]   # 12, ITEMS order; actual owned reservations
    reserve_seeds: tuple[int, ...]   # 5, ITEMS[:5] order
    dependencies: tuple[int, ...]
    created_turn: int
    deadline_turn: int
    priority: int
    status: CommitmentStatus
    progress: JSONObject
    # PENDING may express contingent finance. cash_budget is a maximum, NOT
    # proof that cash exists. Cannot reserve the same cash/item twice.

@dataclass(frozen=True, slots=True)
class ExecutionEvent:
    turn: int
    commitment_id: int | None
    code: str
    detail: JSONObject
    observable: bool = True

class DecisionSession(Protocol):
    def menu(self) -> Menu | None: ...
    def choose(self, selected: int, quantity: int) -> None: ...
    def finish(self) -> tuple[GameAction, JSONObject, tuple[ExecutionEvent,...]]: ...

@dataclass(frozen=True, slots=True)
class SimStep:
    observations: tuple[tuple[Observation,Observation], ...] # [E][2]
    cash: 'np.ndarray'             # float64 [E,2], exact reference bank cash
    terminated: 'np.ndarray'       # bool [E], terminal transition at 719
    truncated: 'np.ndarray'        # bool [E], NOT equivalent to game termination
    valid: 'np.ndarray'            # bool [E], fault-free official transition
    turn: 'np.ndarray'             # int64 [E], post-action turn 1..719
    errors: tuple[str | None, ...]

class Simulator(Protocol):
    rules: RuleSpec
    def reset(self, seeds: tuple[int,...]) -> tuple[tuple[Observation,Observation],...]: ...
    def step(self, actions: tuple[tuple[RawAction,RawAction],...]) -> SimStep: ...
    def snapshot(self) -> JSONObject: ...   # privileged; lossless ordered state
    def restore(self, snapshot: JSONObject) -> None: ...
    def close(self) -> None: ...

@dataclass(frozen=True, slots=True)
class AgentSpec:
    id: str
    family: str
    artifact_sha256: str
    kind: Literal['current','bundle','executable','fixture','tape','clone']
    split: Split
    role: Literal['original','history','self','counter','fixture']
    argv: tuple[str,...] = ()
    cwd: str | None = None
    bundle: str | None = None
    provenance: JSONObject = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class MatchJob:
    id: str
    seed: int
    split: Split
    players: tuple[AgentSpec,AgentSpec]
    learn: tuple[bool,bool]
    policy_rng_seeds: tuple[int,int]  # independent of environment seed
    sampling_probability: float
    loss_weights: tuple[float,float] # target seat-game exposure, NOT event counts

@dataclass(frozen=True, slots=True)
class MatchResult:
    id: str
    seed: int
    split: Split
    players: tuple[str,str]
    families: tuple[str,str]
    final_cash: tuple[float,float] | None
    outcome: tuple[int,int] | None
    transitions: int
    status: Literal['complete','agent_fault','simulator_fault','aborted']
    failure_reason: str | None
    timings: dict[str,float]
    sampling_probability: float
    evidence_path: str | None = None

@dataclass(frozen=True, slots=True)
class PPOConfig:
    gamma: float = 1.0
    gae_lambda: float = 0.97
    mc_weight: float = 1.0       # 0.5 available for historical-credit ablation
    clip: float = 0.2
    epochs: int = 2
    minibatches: int = 4        # complete seat-trajectories, not time slices
    microbatch_lanes: int = 2
    actor_lr: float = 5e-5
    critic_lr: float = 3e-4
    adam_eps: float = 1e-5
    actor_grad_norm: float = 0.5
    critic_grad_norm: float = 1.0
    entropy_coef: float = 0.005
    target_kl: float = 0.01
    stop_kl: float = 0.02
    cash_alpha: float = 0.0     # smoke default pure WDL; bounded aid opt-in
    bc_weight: float = 0.0

@dataclass(frozen=True, slots=True)
class UpdateReport:
    actor_steps: int
    critic_steps: int
    learner_turns: int
    decision_turns: int
    nonforced_factors: int
    complete_games: int
    metrics: dict[str,float]
    stopped_early: bool

@dataclass(frozen=True, slots=True)
class PanelSpec:
    id: str
    split: Split
    opponents: tuple[AgentSpec,...]
    seeds: tuple[int,...]
    policy_rng_seeds: tuple[int,...]
    decode: DecodeConfig
    both_seats: bool = True

@dataclass(frozen=True, slots=True)
class PanelReport:
    id: str
    split: Split
    results: tuple[MatchResult,...]
    metrics: JSONObject
    complete: bool
