#!/usr/bin/env python3
"""Isolated R1-label warm-start for the two explicit economic input paths."""

import argparse
import copy
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.student_economic_features_v1 import (
    economic_features, prefix_flow_features, shop_rate_features)
from experiments.train_midgame_autofill_v3 import build_model, load_shard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--evaluate-checkpoint", type=Path)
    parser.add_argument("--train-states", type=int, default=512)
    parser.add_argument("--heldout-states", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    if bool(args.output) == bool(args.evaluate_checkpoint):
        parser.error("provide exactly one of --output or --evaluate-checkpoint")
    if args.output and args.output.exists():
        parser.error("refusing to overwrite output")
    torch.set_num_threads(8)
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (source.get("economic_features_semantics") != 2 or
            source.get("shop_token_semantics") != 2):
        raise ValueError("requires the isolated mature-stored/explicit-shop fork")
    manifest, shard = load_shard(args.shard)
    if (manifest["validation_status"] != "accepted" or
            manifest.get("rollout", {}).get("manifest_sha256") !=
            source["shard_manifest_sha256"]):
        raise ValueError("teacher shard is unaccepted or bound to another base corpus")
    norm = source["normalization"]
    offsets = np.asarray(shard["state_event_offsets"], dtype=np.int64)
    states = len(offsets) - 1
    rng = np.random.default_rng(20260924)
    train = rng.choice(np.flatnonzero(shard["split"] == 0),
                       args.train_states, replace=False)
    heldout = rng.choice(np.flatnonzero(shard["split"] == 1),
                         args.heldout_states, replace=False)
    selected = np.concatenate((train, heldout))
    features = np.zeros((states, 71), dtype=np.float32)
    rates = np.zeros((states, 9), dtype=np.float32)
    tokens = np.asarray(shard["token_continuous"]).copy()
    for state in selected:
        raw = shard["packed_observation"][state, :int(shard["observation_length"][state])]
        features[state] = economic_features(raw, mature_stored=True)
        rates[state] = shop_rate_features(raw)
        # Old town tokens lose ID/count association under separate mean pools.
        tokens[state, 0, 8:16] = (features[state, :8] *
                                  int(shard["token_count"][state]) *
                                  source.get("shop_token_gain", 1.))
    observation = np.concatenate((np.asarray(shard["packed_observation"], dtype=np.float32),
                                  features), axis=1)
    shop_resource = source.get("shop_resource_semantics") == 1
    event_features = np.zeros((len(shard["event_cell"]),
                               36 if shop_resource else 27), dtype=np.float32)
    for state in selected:
        day = int(shard["state_step"][state]) // 24
        for event in range(int(offsets[state]), int(offsets[state + 1])):
            event_features[event, :27] = prefix_flow_features(
                shard["event_resources"][event], day)
            if shop_resource:
                event_features[event, 27:36] = rates[state]
    resources = np.concatenate((np.asarray(shard["event_resources"], dtype=np.float32),
                                event_features), axis=1)
    dimensions = source["model_dimensions"]
    if dimensions["event_resources"] != (383 if shop_resource else 374):
        raise ValueError("shop-resource metadata/dimensions mismatch")
    teacher = build_model(2233, 3145, dimensions["event_resources"],
                          source["model_scale"])
    teacher.load_state_dict(source["model"])
    teacher.eval().requires_grad_(False)
    student = copy.deepcopy(teacher)
    student.observation.weight.requires_grad_(True)
    student.token.weight.requires_grad_(True)
    if shop_resource:
        student.resource.weight.requires_grad_(True)
    observation_mask = torch.zeros_like(student.observation.weight)
    observation_mask[:, 3074:3145] = 1
    token_mask = torch.zeros_like(student.token.weight)
    token_mask[:, 8:16] = 1
    student.observation.weight.register_hook(lambda gradient: gradient * observation_mask)
    student.token.weight.register_hook(lambda gradient: gradient * token_mask)
    parameters = [student.observation.weight, student.token.weight]
    if shop_resource:
        resource_mask = torch.zeros_like(student.resource.weight)
        resource_mask[:, 374:383] = 1
        student.resource.weight.register_hook(lambda gradient: gradient * resource_mask)
        parameters.append(student.resource.weight)
    optimizer = torch.optim.AdamW(parameters,
                                  lr=args.learning_rate, weight_decay=0)
    categories = [shard[name] for name in (
        "token_type", "token_category_a", "token_category_b",
        "token_category_c", "token_x", "token_y", "token_owner")]
    legal_mask = np.asarray(shard["event_legal_mask"], dtype=np.int64)
    legal = ((legal_mask[:, None] >> np.arange(11)) & 1).astype(bool)
    labels = np.asarray(shard["event_label_class"], dtype=np.int64)

    def tensor(value, dtype=torch.float32):
        return torch.as_tensor(np.asarray(value).copy(), dtype=dtype)

    def forward(model, batch):
        batch = np.asarray(batch, dtype=np.int64)
        hidden = model.initial_hidden(
            tensor((shard["causal_context"][batch] - norm["context_mean"]) /
                   norm["context_std"]),
            tensor((observation[batch] - norm["observation_mean"]) /
                   norm["observation_std"]),
            tensor((shard["observation_length"][batch] - norm["observation_length_mean"]) /
                   norm["observation_length_std"]),
            tensor(tokens[batch]),
            [tensor(value[batch], torch.long) for value in categories],
            tensor(shard["token_count"][batch], torch.long))
        previous = torch.full((len(batch),), 11, dtype=torch.long)
        event_logits, event_labels = [], []
        lengths = offsets[batch + 1] - offsets[batch]
        for position in range(int(lengths.max())):
            active = np.flatnonzero(position < lengths)
            rows = offsets[batch[active]] + position
            active_tensor = tensor(active, torch.long)
            masks = tensor(legal[rows], torch.bool)
            logits, updated = model.step(
                hidden.index_select(0, active_tensor),
                tensor((resources[rows] - norm["resource_mean"]) / norm["resource_std"]),
                tensor(shard["event_cell"][rows], torch.long),
                tensor(shard["event_stage"][rows], torch.long),
                previous.index_select(0, active_tensor), masks)
            hidden = hidden.index_copy(0, active_tensor, updated)
            previous = previous.index_copy(0, active_tensor,
                                           tensor(labels[rows], torch.long))
            actionable = (legal[rows].sum(1) > 1) & (np.asarray(shard["event_stage"][rows]) == 1)
            if actionable.any():
                event_logits.append(logits[actionable])
                event_labels.append(tensor(labels[rows][actionable], torch.long))
        return torch.cat(event_logits), torch.cat(event_labels)

    def report(subset):
        losses, kl, correct, total = [], [], 0, 0
        student.eval()
        with torch.no_grad():
            for batch in np.array_split(subset, max(1, (len(subset) + 31) // 32)):
                now, label = forward(student, batch)
                old, _ = forward(teacher, batch)
                losses.append(F.cross_entropy(now, label).item() * len(label))
                kl.append(F.kl_div(F.log_softmax(now, -1),
                                   F.softmax(old, -1), reduction="sum").item())
                correct += int((now.argmax(-1) == label).sum())
                total += len(label)
        return {"ce": sum(losses) / total, "kl": sum(kl) / total,
                "top1": correct / total, "events": total}

    before = report(heldout)
    if args.evaluate_checkpoint:
        candidate = torch.load(args.evaluate_checkpoint, map_location="cpu",
                               weights_only=False)
        if (candidate["model_dimensions"] != source["model_dimensions"] or
                candidate.get("aux_bc_shops", {}).get("teacher_shard_sha256") ==
                hashlib.sha256((args.shard / "manifest.json").read_bytes()).hexdigest()):
            raise ValueError("candidate dimensions disagree or evaluation shard was trained on")
        student.load_state_dict(candidate["model"])
        print({"evaluation_shard": str(args.shard), "source": before,
               "candidate": report(heldout)}, flush=True)
        return
    for epoch in range(args.epochs):
        student.train()
        for batch in np.array_split(rng.permutation(train),
                                    max(1, (len(train) + 31) // 32)):
            optimizer.zero_grad(set_to_none=True)
            now, label = forward(student, batch)
            with torch.no_grad():
                old, _ = forward(teacher, batch)
            weights = torch.where(label >= 3, 2.0, 1.0)
            ce = (F.cross_entropy(now, label, reduction="none") * weights).sum() / weights.sum()
            kl = F.kl_div(F.log_softmax(now, -1), F.softmax(old, -1), reduction="batchmean")
            loss = ce + kl
            loss.backward()
            optimizer.step()
        print({"epoch": epoch + 1, "train": report(train),
               "heldout": report(heldout)}, flush=True)
    after = report(heldout)
    output = copy.deepcopy(source)
    output["model"] = {key: value.detach().clone() for key, value in student.state_dict().items()}
    ids = output["rl_optimizer"]["param_groups"][0]["params"]
    keys = dict((name, key) for (name, _), key in zip(student.named_parameters(), ids))
    for name, columns in (("observation.weight", slice(3074, 3145)),
                          ("token.weight", slice(8, 16)),
                          ("resource.weight", slice(374, 383))):
        if name == "resource.weight" and not shop_resource:
            continue
        moments = output["rl_optimizer"]["state"][keys[name]]
        for moment in ("exp_avg", "exp_avg_sq"):
            moments[moment][:, columns] = 0
    output["aux_bc_shops"] = {
        "teacher_shard_sha256": hashlib.sha256((args.shard / "manifest.json").read_bytes()).hexdigest(),
        "teacher_scope": "accepted R1 action-event labels; not optimality claims",
        "train_states": len(train), "heldout_states": len(heldout),
        "heldout_before": before, "heldout_after": after,
        "epochs": args.epochs, "learning_rate": args.learning_rate,
    }
    output["rl"]["deployment_eligible"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print({"output": str(args.output), "heldout_before": before,
           "heldout_after": after}, flush=True)


if __name__ == "__main__":
    main()
