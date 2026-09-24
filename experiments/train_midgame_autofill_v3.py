#!/usr/bin/env python3
"""Train the step-288 per-cell actor from final executable action events."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import numpy as np

from experiments.build_midgame_action_events_v3 import PACKED_OBSERVATION_CAPACITY
from experiments.train_midgame_student_v1 import _sha256


EVENT_CLASSES = (
    "STOP", "NONE_OR_KEEP", "RELEASE", "WHEAT", "CARROT", "TOMATO",
    "STRAWBERRY", "MELON", "GOOSE", "COW", "SHEEP",
)
MODEL_INPUTS = {
    "causal_context", "packed_observation", "observation_length",
    "token_continuous", "token_type", "token_category_a", "token_category_b",
    "token_category_c", "token_x", "token_y", "token_owner", "token_count",
    "state_event_offsets", "event_stage", "event_cell", "event_resources",
    "event_legal_mask",
}


def load_shard(directory: Path):
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    validation = manifest.get("validation", {})
    schema = manifest.get("schema", {})
    schema_name = schema.get("schema_name")
    required_validation = ({
        "execution_contract": "PASS", "scope": "first_handoff_only_step288",
        "controlled_job_projection": "PASS", "stop_terminal_exact": "PASS",
        "actor_owned_placement": "PASS",
        "label_fixed_point": "PASS", "resource_replay_exact": "PASS",
        "execution_semantics": "PASS", "packed_observation_capacity": "PASS",
    } if schema_name == "autoregressive-action-event-bc-v3" else {
        "execution_contract": "PASS", "scope": "student_state_dagger_day_boundary",
        "controlled_job_projection": "PASS", "stop_terminal_exact": "PASS",
        "actor_owned_placement": "PASS",
        "label_fixed_point": "PASS", "resource_replay_exact": "PASS",
        "clone_isolation": "PASS", "stage_mask_contract": "PASS",
        "rollout_fingerprint": "PASS", "no_identity_in_model_inputs": "PASS",
        "packed_observation_capacity": "PASS",
        "student_step_exactly_once": "PASS", "student_failures_zero": "PASS",
        "student_fallbacks_zero": "PASS", "student_illegal_zero": "PASS",
    })
    required_validation["label_source"] = "final_jobs_actions_plus_executable_STOP"
    if (manifest.get("validation_status") != "accepted" or
            schema_name not in ("autoregressive-action-event-bc-v3",
                                "student_state_dagger_action_event_v3") or
            tuple(schema.get("class_names", ())) != EVENT_CLASSES or
            set(schema.get("model_input_arrays", ())) != MODEL_INPUTS or
            schema.get("target_only_arrays") != ["event_label_class"] or
            any(validation.get(key) != value
                for key, value in required_validation.items())):
        raise ValueError("shard lacks the accepted v3 final-action execution contract")
    if schema_name == "autoregressive-action-event-bc-v3":
        semantic = validation.get("semantic_smoke", {})
        semantic_path = Path(semantic.get("path", ""))
        expected_checks = {
            "release_to_none_harvest_without_plant",
            "release_to_stop_harvest_without_plant",
            "empty_to_none_without_plant",
            "two_land_plan_exposes_third_land_slots",
        }
        if (not semantic_path.is_file() or
                _sha256(semantic_path) != semantic.get("sha256") or
                semantic.get("binary_sha256") != manifest.get("teacher_binary_sha256") or
                set(semantic.get("checks", ())) != expected_checks):
            raise ValueError("v3 semantic smoke does not attest the teacher binary")

    arrays = {}
    for name, spec in manifest["arrays"].items():
        path = directory / spec["path"]
        if _sha256(path) != spec["sha256"]:
            raise ValueError(f"{name}: digest mismatch")
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(value.shape) != spec["shape"] or value.dtype.str != spec["dtype"]:
            raise ValueError(f"{name}: dtype/shape mismatch")
        arrays[name] = value
    missing = (MODEL_INPUTS | {"event_label_class", "split", "state_step"}) - arrays.keys()
    if missing:
        raise ValueError(f"missing v3 arrays: {sorted(missing)}")
    if (arrays["packed_observation"].ndim != 2 or
            arrays["packed_observation"].shape[1] != PACKED_OBSERVATION_CAPACITY or
            manifest.get("dimensions", {}).get("packed_observation") !=
            PACKED_OBSERVATION_CAPACITY):
        raise ValueError("v3 shard does not use the fixed hour-zero observation ABI")

    offsets = np.asarray(arrays["state_event_offsets"], dtype=np.int64)
    state_count = len(arrays["split"])
    event_count = len(arrays["event_cell"])
    state_steps = np.asarray(arrays["state_step"], dtype=np.int64)
    expected_steps = ({288} if schema_name == "autoregressive-action-event-bc-v3"
                      else set(map(int, validation.get("state_steps", (
                          validation.get("state_step", -1),)))))
    if (not expected_steps or any(step < 288 or step >= 719 or step % 24
                                  for step in expected_steps) or
            set(map(int, np.unique(state_steps))) != expected_steps):
        raise ValueError("v3 shard state_step disagrees with its execution scope")
    if (offsets.shape != (state_count + 1,) or offsets[0] != 0 or
            offsets[-1] != event_count or np.any(offsets[1:] < offsets[:-1])):
        raise ValueError("invalid state-event offsets")
    for name in ("event_stage", "event_resources", "event_legal_mask",
                 "event_label_class"):
        if len(arrays[name]) != event_count:
            raise ValueError(f"{name}: event count mismatch")
    stage = np.asarray(arrays["event_stage"], dtype=np.int64)
    labels = np.asarray(arrays["event_label_class"], dtype=np.int64)
    masks = np.asarray(arrays["event_legal_mask"], dtype=np.int64)
    if (np.any((stage < 0) | (stage > 1)) or
            np.any((labels < 0) | (labels >= len(EVENT_CLASSES))) or
            np.any(masks <= 0) or np.any(masks >> len(EVENT_CLASSES)) or
            np.any(((masks >> labels) & 1) == 0)):
        raise ValueError("invalid event stage, label, or legal mask")
    release = stage == 0
    placement = ~release
    if (np.any(masks[release] & ~0b110) or
            np.any((masks[release] & 0b110) == 0) or
            np.any(labels[release] > 2) or np.any(labels[release] < 1) or
            np.any(masks[placement] & (1 << 2)) or
            np.any((masks[placement] & (1 << 1)) == 0)):
        raise ValueError("stage-specific v3 legal-mask contract failed")
    split_values = set(np.unique(np.asarray(arrays["split"], dtype=np.int64)))
    if not split_values or not split_values <= {0, 1}:
        raise ValueError("train/held-out split contains an invalid group")
    return manifest, arrays


def load_training_data(base_directory: Path, dagger_directories: list[Path]):
    base_manifest, base_arrays = load_shard(base_directory)
    if base_manifest["schema"]["schema_name"] != "autoregressive-action-event-bc-v3":
        raise ValueError("the primary shard must be the accepted step-288 BC corpus")
    base_sha = _sha256(base_directory / "manifest.json")
    sources = [(base_directory, base_manifest, base_arrays)]
    for directory in dagger_directories:
        manifest, arrays = load_shard(directory)
        rollout = manifest.get("rollout", {})
        if (manifest["schema"]["schema_name"] !=
                "student_state_dagger_action_event_v3" or
                rollout.get("manifest_sha256") != base_sha or
                rollout.get("binary_sha256") !=
                base_manifest.get("teacher_binary_sha256") or
                manifest.get("causal_context_names") !=
                base_manifest.get("causal_context_names") or
                manifest.get("event_resource_names") !=
                base_manifest.get("event_resource_names")):
            raise ValueError("DAgger shard is not bound to the primary BC contract")
        sources.append((directory, manifest, arrays))
    if len(sources) == 1:
        return base_manifest, base_arrays, sources

    state_arrays = (
        "causal_context", "packed_observation", "observation_length",
        "token_continuous", "token_type", "token_category_a",
        "token_category_b", "token_category_c", "token_x", "token_y",
        "token_owner", "token_count", "split", "state_step",
    )
    event_arrays = (
        "event_stage", "event_cell", "event_resources", "event_legal_mask",
        "event_label_class",
    )
    arrays = {
        name: np.concatenate([source[2][name] for source in sources], axis=0)
        for name in state_arrays + event_arrays
    }
    counts = np.concatenate([
        np.diff(np.asarray(source[2]["state_event_offsets"], dtype=np.int64))
        for source in sources
    ])
    arrays["state_event_offsets"] = np.concatenate((
        np.asarray([0], dtype="<i8"), np.cumsum(counts, dtype="<i8")))
    if set(np.unique(np.asarray(arrays["split"], dtype=np.int64))) != {0, 1}:
        raise ValueError("combined training data must contain train and held-out groups")
    return base_manifest, arrays, sources


def build_model(context_width: int, observation_width: int, resource_width: int,
                scale: int = 1, *, shop_action_head: bool = False,
                event_classes=EVENT_CLASSES):
    import torch

    if not 1 <= scale <= 4:
        raise ValueError("model scale must be in [1, 4]")
    projection, scalar = 16 * scale, 4 * scale
    embedding, hidden = 4 * scale, 64 * scale
    resource_hidden, event_embedding = 64 * scale, 8 * scale

    class ActionEventStudent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.context = torch.nn.Linear(context_width, projection)
            self.observation = torch.nn.Linear(observation_width, projection)
            self.observation_length = torch.nn.Linear(1, scalar)
            self.token_embeddings = torch.nn.ModuleList(
                torch.nn.Embedding(size, embedding)
                for size in (7, 32, 32, 32, 64, 64, 4))
            self.token = torch.nn.Linear(24 + 7 * embedding, projection)
            self.begin = torch.nn.Linear(3 * projection + scalar, hidden)
            self.resource = torch.nn.Linear(resource_width, resource_hidden)
            self.cell = torch.nn.Embedding(100, event_embedding)
            self.stage = torch.nn.Embedding(2, event_embedding)
            self.previous = torch.nn.Embedding(
                len(event_classes) + 1, event_embedding)
            self.gru = torch.nn.GRUCell(
                resource_hidden + 3 * event_embedding, hidden)
            self.head = torch.nn.Linear(hidden, len(event_classes))
            if shop_action_head:
                if resource_width != 383:
                    raise ValueError("shop action head requires the 383D resource ABI")
                self.shop_gate = torch.nn.Linear(hidden, 8)
                torch.nn.init.zeros_(self.shop_gate.weight)
                torch.nn.init.zeros_(self.shop_gate.bias)

        def initial_hidden(self, context, observation, observation_length,
                           token_continuous, token_categories, token_count):
            positions = torch.arange(token_continuous.shape[1], device=context.device)[None]
            mask = positions < token_count[:, None]
            denominator = token_count.clamp_min(1).float()[:, None]
            continuous = (token_continuous * mask[:, :, None]).sum(1) / denominator
            embedded = [(table(values) * mask[:, :, None]).sum(1) / denominator
                        for table, values in zip(self.token_embeddings, token_categories)]
            token_state = torch.cat((continuous, *embedded), dim=1)
            state = torch.cat((
                torch.relu(self.context(context)),
                torch.relu(self.observation(observation)),
                torch.relu(self.observation_length(observation_length[:, None])),
                torch.relu(self.token(token_state)),
            ), dim=1)
            return torch.tanh(self.begin(state))

        def step(self, hidden, resources, cells, stages, previous, legal):
            inputs = torch.cat((torch.relu(self.resource(resources)), self.cell(cells),
                                self.stage(stages), self.previous(previous)), dim=1)
            hidden = self.gru(inputs, hidden)
            logits = self.head(hidden)
            if shop_action_head:
                # These normalized columns equal 32 * exact demand per tick / 8.
                # Actions 3..10 produce products 0..7 respectively.
                logits = torch.cat((logits[:, :3], logits[:, 3:] +
                    self.shop_gate(hidden) * resources[:, 374:382] / 32), dim=1)
            return logits.masked_fill(~legal, -1e9), hidden

    return ActionEventStudent()


def train(args) -> None:
    import torch
    import torch.nn.functional as functional

    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401
        if not torch.npu.is_available():
            raise RuntimeError("requested NPU device is unavailable")
        torch.npu.set_device(args.device)
    elif args.device != "cpu":
        raise ValueError("device must be cpu or npu[:index]")
    device = torch.device(args.device)
    manifest, arrays, sources = load_training_data(
        args.shard, args.dagger_shard)
    initial_checkpoint = (torch.load(
        args.init_checkpoint, map_location="cpu", weights_only=False)
        if args.init_checkpoint else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    if device.type == "npu":
        torch.npu.manual_seed_all(args.seed)
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)

    split = np.asarray(arrays["split"], dtype=np.int64)
    offsets = np.asarray(arrays["state_event_offsets"], dtype=np.int64)
    event_counts = np.diff(offsets)
    train_states = np.flatnonzero((split == 0) & (event_counts > 0))
    heldout_states = np.flatnonzero((split == 1) & (event_counts > 0))
    if not len(train_states) or not len(heldout_states):
        raise ValueError("v3 shard needs non-empty train and held-out event states")
    event_state = np.repeat(np.arange(len(split)), np.diff(offsets))
    train_events = np.isin(event_state, train_states)

    def standardize(value, rows):
        sample = np.asarray(value[rows], dtype=np.float32)
        mean = sample.mean(0, dtype=np.float64).astype(np.float32)
        std = sample.std(0, dtype=np.float64).astype(np.float32)
        std = np.where(std < 1e-5, np.float32(1), std).astype(np.float32)
        return mean, std

    if initial_checkpoint:
        normalization = initial_checkpoint["normalization"]
        context_mean, context_std = normalization["context_mean"], normalization["context_std"]
        observation_mean = normalization["observation_mean"]
        observation_std = normalization["observation_std"]
        length_mean = normalization["observation_length_mean"]
        length_std = normalization["observation_length_std"]
        resource_mean, resource_std = normalization["resource_mean"], normalization["resource_std"]
    else:
        context_mean, context_std = standardize(arrays["causal_context"], train_states)
        observation_mean, observation_std = standardize(
            arrays["packed_observation"], train_states)
        length_mean, length_std = standardize(arrays["observation_length"], train_states)
        resource_mean, resource_std = standardize(arrays["event_resources"], train_events)

    def tensor(value, dtype=None):
        return torch.as_tensor(np.asarray(value).copy(), dtype=dtype, device=device)

    masks = np.asarray(arrays["event_legal_mask"], dtype=np.int64)
    legal = ((masks[:, None] >> np.arange(len(EVENT_CLASSES))) & 1).astype(bool)
    inputs = {
        "context": (tensor(arrays["causal_context"], torch.float32) - tensor(context_mean)) / tensor(context_std),
        "observation": (tensor(arrays["packed_observation"], torch.float32) - tensor(observation_mean)) / tensor(observation_std),
        "observation_length": (tensor(arrays["observation_length"], torch.float32) - tensor(length_mean)) / tensor(length_std),
        "token_continuous": tensor(arrays["token_continuous"], torch.float32),
        "token_categories": [tensor(arrays[name], torch.long) for name in
                             ("token_type", "token_category_a", "token_category_b",
                              "token_category_c", "token_x", "token_y", "token_owner")],
        "token_count": tensor(arrays["token_count"], torch.long),
        "resources": (tensor(arrays["event_resources"], torch.float32) - tensor(resource_mean)) / tensor(resource_std),
        "cell": tensor(arrays["event_cell"], torch.long),
        "stage": tensor(arrays["event_stage"], torch.long),
        "label": tensor(arrays["event_label_class"], torch.long),
        "legal": tensor(legal, torch.bool),
    }
    dimensions = {
        "causal_context": int(arrays["causal_context"].shape[1]),
        "packed_observation": int(arrays["packed_observation"].shape[1]),
        "event_resources": int(arrays["event_resources"].shape[1]),
    }

    def make_model():
        return build_model(dimensions["causal_context"],
                           dimensions["packed_observation"],
                           dimensions["event_resources"],
                           args.model_scale).to(device)

    def forward(model, states):
        state_tensor = tensor(states, torch.long)
        hidden = model.initial_hidden(
            inputs["context"][state_tensor], inputs["observation"][state_tensor],
            inputs["observation_length"][state_tensor],
            inputs["token_continuous"][state_tensor],
            [value[state_tensor] for value in inputs["token_categories"]],
            inputs["token_count"][state_tensor])
        previous = torch.full((len(states),), len(EVENT_CLASSES),
                              dtype=torch.long, device=device)
        losses, predicted, expected, legal_rows, stages, local_states = [], [], [], [], [], []
        lengths = offsets[np.asarray(states) + 1] - offsets[np.asarray(states)]
        for event_index in range(int(lengths.max())):
            active_np = np.flatnonzero(event_index < lengths)
            active = tensor(active_np, torch.long)
            rows_np = offsets[np.asarray(states)[active_np]] + event_index
            rows = tensor(rows_np, torch.long)
            event_legal = inputs["legal"][rows]
            logits, next_hidden = model.step(
                hidden[active], inputs["resources"][rows], inputs["cell"][rows],
                inputs["stage"][rows], previous[active], event_legal)
            labels = inputs["label"][rows]
            losses.append(functional.cross_entropy(logits, labels, reduction="none"))
            predicted.append(logits.argmax(1))
            expected.append(labels)
            legal_rows.append(event_legal)
            stages.append(inputs["stage"][rows])
            local_states.append(active)
            hidden = hidden.index_copy(0, active, next_hidden)
            previous = previous.index_copy(0, active, labels)
        return tuple(torch.cat(values) for values in
                     (losses, predicted, expected, legal_rows, stages, local_states))

    def batches(states, size, shuffle=False):
        values = np.asarray(states, dtype=np.int64)
        if shuffle:
            values = values[rng.permutation(len(values))]
        for start in range(0, len(values), size):
            yield values[start:start + size]

    def metrics(model, states):
        was_training = model.training
        model.eval()
        total_loss = total = correct = illegal = exact = 0
        support = np.zeros(len(EVENT_CLASSES), dtype=np.int64)
        hits = np.zeros(len(EVENT_CLASSES), dtype=np.int64)
        stage_support = np.zeros(2, dtype=np.int64)
        stage_hits = np.zeros(2, dtype=np.int64)
        with torch.no_grad():
            for batch in batches(states, args.eval_batch_states):
                losses, predicted, labels, event_legal, stages, local_states = forward(model, batch)
                total_loss += float(losses.sum().item())
                total += len(labels)
                matches = predicted == labels
                correct += int(matches.sum().item())
                rows = torch.arange(len(predicted), device=device)
                illegal += int((~event_legal[rows, predicted]).sum().item())
                exact += len(batch) - int(torch.unique(local_states[~matches]).numel())
                for index in range(len(EVENT_CLASSES)):
                    selected = labels == index
                    support[index] += int(selected.sum().item())
                    hits[index] += int((predicted[selected] == index).sum().item())
                for index in range(2):
                    selected = stages == index
                    stage_support[index] += int(selected.sum().item())
                    stage_hits[index] += int(matches[selected].sum().item())
        model.train(was_training)
        return {
            "loss": total_loss / total,
            "masked_accuracy": correct / total,
            "teacher_forced_state_exact": exact / len(states),
            "illegal_argmax": illegal,
            "stage_accuracy": {
                "release": float(stage_hits[0] / stage_support[0]) if stage_support[0] else None,
                "placement": float(stage_hits[1] / stage_support[1]) if stage_support[1] else None,
            },
            "class_recall": {name: {
                "support": int(support[index]),
                "recall": float(hits[index] / support[index]) if support[index] else None,
            } for index, name in enumerate(EVENT_CLASSES)},
        }

    probe = make_model()
    parameters = sum(value.numel() for value in probe.parameters())
    if not 80_000 <= parameters <= 1_500_000:
        raise RuntimeError(f"model size {parameters} outside v3 budget")
    overfit_states = train_states[:min(8, len(train_states))]
    overfit = make_model()
    optimizer = torch.optim.Adam(
        overfit.parameters(), lr=0.02 / args.model_scale)
    first_overfit = None
    for _ in range(args.overfit_steps):
        optimizer.zero_grad(set_to_none=True)
        losses, *_ = forward(overfit, overfit_states)
        loss = losses.mean()
        if first_overfit is None:
            first_overfit = float(loss.detach().item())
        loss.backward()
        optimizer.step()
    overfit_result = metrics(overfit, overfit_states)
    overfit_pass = (overfit_result["loss"] < first_overfit and
                    overfit_result["masked_accuracy"] >= 0.999 and
                    not overfit_result["illegal_argmax"])
    print(json.dumps({"event": "overfit_check",
                      "status": "PASS" if overfit_pass else "FAIL",
                      **overfit_result}), flush=True)
    if not overfit_pass:
        raise RuntimeError("v3 one-batch autoregressive overfit failed")

    torch.manual_seed(args.seed)
    model = make_model()
    if initial_checkpoint:
        if (initial_checkpoint.get("model_dimensions") != dimensions or
                initial_checkpoint.get("model_scale") != args.model_scale or
                tuple(initial_checkpoint.get("event_classes", ())) != EVENT_CLASSES):
            raise ValueError("initial checkpoint architecture disagrees with training data")
        model.load_state_dict(initial_checkpoint["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=1e-4)
    initial_train = metrics(model, train_states)
    initial_heldout = metrics(model, heldout_states)
    started = time.perf_counter()
    optimizer_steps = 0
    for epoch in range(args.epochs):
        epoch_loss = epoch_events = 0
        for batch in batches(train_states, args.batch_states, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            losses, *_ = forward(model, batch)
            loss = losses.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            optimizer_steps += 1
            epoch_loss += float(loss.detach().item()) * len(losses)
            epoch_events += len(losses)
        print(json.dumps({"event": "epoch", "epoch": epoch + 1,
                          "train_loss": epoch_loss / epoch_events}), flush=True)
    elapsed = time.perf_counter() - started
    final_train = metrics(model, train_states)
    final_heldout = metrics(model, heldout_states)
    state_steps = np.asarray(arrays["state_step"], dtype=np.int64)
    training_state_steps = [
        int(step) for step in np.unique(state_steps[train_states])]
    by_state_step = {}
    for step in np.unique(state_steps):
        step_train = train_states[state_steps[train_states] == step]
        step_heldout = heldout_states[state_steps[heldout_states] == step]
        by_state_step[str(int(step))] = {
            "train": metrics(model, step_train) if len(step_train) else None,
            "heldout": metrics(model, step_heldout) if len(step_heldout) else None,
        }
    if final_train["loss"] >= initial_train["loss"]:
        raise RuntimeError("v3 training loss did not decrease")
    if final_train["illegal_argmax"] or final_heldout["illegal_argmax"]:
        raise RuntimeError("masked v3 actor emitted an illegal argmax")

    checkpoint = {
        "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "model_dimensions": dimensions,
        "model_scale": args.model_scale,
        "training_state_steps": training_state_steps,
        "event_classes": EVENT_CLASSES,
        "normalization": {
            "context_mean": context_mean, "context_std": context_std,
            "observation_mean": observation_mean, "observation_std": observation_std,
            "observation_length_mean": length_mean, "observation_length_std": length_std,
            "resource_mean": resource_mean, "resource_std": resource_std,
        },
        "shard_manifest_sha256": _sha256(args.shard / "manifest.json"),
        "training_shard_manifest_sha256s": [
            _sha256(directory / "manifest.json")
            for directory, _manifest, _arrays in sources
        ],
        "model_inputs": sorted(MODEL_INPUTS),
    }
    torch.save(checkpoint, args.output)
    result = {
        "status": "PASS", "training_task": "autoregressive_action_event_actor_v3",
        "schema_name": manifest["schema"]["schema_name"],
        "shard_manifest_sha256": checkpoint["shard_manifest_sha256"],
        "device": str(device), "states": len(split), "events": len(event_state),
        "zero_event_states": int(np.sum(event_counts == 0)),
        "train_states": len(train_states), "heldout_states": len(heldout_states),
        "parameters": parameters,
        "model_scale": args.model_scale,
        "training_state_steps": checkpoint["training_state_steps"],
        "training_shards": [str(directory) for directory, _manifest, _arrays in sources],
        "one_batch_overfit": {"states": len(overfit_states),
                              "initial_loss": first_overfit, **overfit_result},
        "train": {"initial": initial_train, "final": final_train},
        "heldout": {"initial": initial_heldout, "final": final_heldout},
        "by_state_step": by_state_step,
        "timing": {"epochs": args.epochs, "optimizer_steps": optimizer_steps,
                   "seconds": elapsed,
                   "event_epochs_per_second": args.epochs * int(train_events.sum()) / elapsed},
        "checkpoint": str(args.output),
        "limitations": [
            "metrics are teacher-forced; a native free-running closed-loop evaluation is still required",
            "BC imitates final executable R1 actions and is not yet optimized for match outcome",
        ],
    }
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


def check(args) -> None:
    manifest, arrays, sources = load_training_data(args.shard, args.dagger_shard)
    print(json.dumps({
        "status": "PASS", "schema_name": manifest["schema"]["schema_name"],
        "states": len(arrays["split"]), "events": len(arrays["event_cell"]),
        "release_events": int(np.sum(np.asarray(arrays["event_stage"]) == 0)),
        "placement_events": int(np.sum(np.asarray(arrays["event_stage"]) == 1)),
        "training_shards": [str(directory) for directory, _manifest, _arrays in sources],
    }))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("check")
    verify.add_argument("--shard", type=Path, required=True)
    verify.add_argument("--dagger-shard", type=Path, action="append", default=[])
    verify.set_defaults(function=check)
    fit = commands.add_parser("train")
    fit.add_argument("--shard", type=Path, required=True)
    fit.add_argument("--dagger-shard", type=Path, action="append", default=[])
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--init-checkpoint", type=Path)
    fit.add_argument("--epochs", type=int, default=150)
    fit.add_argument("--overfit-steps", type=int, default=250)
    fit.add_argument("--learning-rate", type=float, default=0.003)
    fit.add_argument("--threads", type=int, default=8)
    fit.add_argument("--device", default="cpu")
    fit.add_argument("--batch-states", type=int, default=1024)
    fit.add_argument("--eval-batch-states", type=int, default=2048)
    fit.add_argument("--seed", type=int, default=20260923)
    fit.add_argument("--model-scale", type=int, choices=range(1, 5), default=1)
    fit.set_defaults(function=train)
    args = parser.parse_args()
    if args.command == "train" and (args.epochs < 1 or args.overfit_steps < 1 or
                                    args.batch_states < 1 or args.eval_batch_states < 1):
        parser.error("training counts must be positive")
    args.function(args)


if __name__ == "__main__":
    main()
