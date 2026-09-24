#!/usr/bin/env python3
"""Policy change at actionable placements under shop counterfactuals."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.train_midgame_autofill_v3 import build_model
from experiments.student_economic_features_v1 import (
    _unpack, economic_features, shop_rate_features)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("rollout", type=Path)
    parser.add_argument("--assert-old-policy", action="store_true")
    parser.add_argument("--all-placements", action="store_true")
    parser.add_argument("--step288-output", type=Path)
    parser.add_argument("--teacher-handoff", type=Path)
    args = parser.parse_args()
    if args.step288_output and (not args.all_placements or
                                args.step288_output.exists()):
        parser.error("step288 output needs --all-placements and a new path")
    if args.teacher_handoff and not args.step288_output:
        parser.error("teacher handoff requires --step288-output")
    torch.set_num_threads(8)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    norm = checkpoint["normalization"]
    z = np.load(args.rollout)
    offsets = z["day_event_offsets"]
    events = [next((j for j in range(int(a), int(b))
                    if z["event_stage"][j] == 1 and
                    z["event_legal"][j].sum() > 1), None)
              for a, b in zip(offsets[:-1], offsets[1:])]
    days = np.asarray([i for i, j in enumerate(events) if j is not None])
    events = np.asarray([j for j in events if j is not None])
    dims = checkpoint["model_dimensions"]
    model = build_model(dims["causal_context"], dims["packed_observation"],
                        dims["event_resources"], checkpoint["model_scale"],
                        shop_action_head=checkpoint.get("shop_action_head_semantics") == 1)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    observation = np.asarray(z["observation"][days], dtype=np.float32)
    starts = offsets[days]
    lengths = (offsets[days + 1] - starts if args.all_placements
               else events - starts + 1)
    selected_events = []
    for position in range(int(lengths.max())):
        active = np.flatnonzero(position < lengths)
        ids = starts[active] + position
        selected = ((z["event_stage"][ids] == 1) &
                    (z["event_legal"][ids].sum(1) > 1))
        selected_events.extend(ids[selected] if args.all_placements else
                               ids[selected & (position == lengths[active] - 1)])
    selected_events = np.asarray(selected_events, dtype=np.int64)
    resources = torch.from_numpy(z["event_resources"][:, :dims["event_resources"]])
    common = (torch.from_numpy(z["context"][days]),
              torch.from_numpy(z["observation_length"][days]),
              torch.from_numpy(z["token_continuous"][days]),
              [torch.from_numpy(x.astype(np.int64)) for x in z["token_categories"][days].transpose(1, 0, 2)],
              torch.from_numpy(z["token_count"][days].astype(np.int64)))
    event = (torch.from_numpy(z["event_cell"].astype(np.int64)),
             torch.from_numpy(z["event_stage"].astype(np.int64)),
             torch.from_numpy(z["event_previous"].astype(np.int64)),
             torch.from_numpy(z["event_legal"]))

    def forward(obs: torch.Tensor, tokens: torch.Tensor | None = None,
                event_resources: torch.Tensor | None = None) -> torch.Tensor:
        hidden = model.initial_hidden(common[0], obs,
                                      common[1], common[2] if tokens is None else tokens,
                                      common[3], common[4])
        result = []
        for position in range(int(lengths.max())):
            active = np.flatnonzero(position < lengths)
            active_ids = torch.from_numpy(active)
            ids = torch.from_numpy((starts[active] + position).astype(np.int64))
            logits, next_hidden = model.step(
                hidden.index_select(0, active_ids),
                (resources if event_resources is None else event_resources)[ids],
                *(field[ids] for field in event))
            hidden = hidden.index_copy(0, active_ids, next_hidden)
            selected = ((event[1][ids] == 1) &
                        (event[3][ids].sum(1) > 1))
            if not args.all_placements:
                selected &= torch.from_numpy(position == lengths[active] - 1)
            if selected.any():
                result.append(logits[selected])
        return torch.cat(result)

    with torch.no_grad():
        base_logits = forward(torch.from_numpy(observation))
        base = base_logits.softmax(-1)
    obs_gradient = torch.from_numpy(observation.copy()).requires_grad_(True)
    resource_gradient_input = resources.clone().requires_grad_(True)
    logits_gradient = forward(obs_gradient, event_resources=resource_gradient_input)
    winner = base.argmax(-1)
    runner_up = base_logits.scatter(1, winner[:, None], -1e9).argmax(-1)
    row = torch.arange(len(selected_events))
    base_gap = base_logits[row, winner] - base_logits[row, runner_up]
    margin = logits_gradient[row, winner] - logits_gradient[row, runner_up]
    gradient, resource_gradient = torch.autograd.grad(
        margin.sum(), (obs_gradient, resource_gradient_input))
    gradient = gradient.detach()
    if checkpoint.get("shop_resource_semantics") == 1:
        print({"group": "shop_resource_gradient",
               "gradient_l1_mean": float(resource_gradient[
                   selected_events, 374:383].abs().sum(-1).mean())})
    if checkpoint.get("shop_action_head_semantics") == 1:
        print({"group": "shop_action_head",
               "weight_l2": float(model.shop_gate.weight.norm()),
               "bias_l2": float(model.shop_gate.bias.norm())})
    entropy = -(base * base.clamp_min(1e-30).log()).sum(-1)
    chosen = torch.from_numpy(z["event_action"][selected_events].astype(np.int64))
    replay_error = (base.gather(1, chosen[:, None]).clamp_min(1e-30).log().flatten() -
                    torch.from_numpy(z["old_logprob"][selected_events])).abs()
    if args.assert_old_policy and float(replay_error.max()) > 7e-4:
        raise AssertionError(f"old-policy logprob mismatch: {float(replay_error.max())}")
    print({"days": len(days), "actionable_events": len(selected_events),
           "entropy_mean": float(entropy.mean()),
           "max_probability_mean": float(base.max(-1).values.mean()),
           "top_two_logit_gap_median": float(base_gap.median()),
           "top_two_logit_gap_p10": float(torch.quantile(base_gap, .1))})
    print({"old_logprob_max_abs": float(replay_error.max())})
    for name, subset in (("shops", slice(0, 8)),
                         ("pressure", slice(8, 35)),
                         ("rival_assets", slice(35, 71)),
                         ("all_economic", slice(0, 71))):
        changed = observation.copy()
        indices = np.arange(3074, 3145)[subset]
        changed[:, indices] = -np.asarray(norm["observation_mean"])[indices] / np.asarray(norm["observation_std"])[indices]
        with torch.no_grad():
            candidate_logits = forward(torch.from_numpy(changed))
            candidate = candidate_logits.softmax(-1)
        tv = (base - candidate).abs().sum(-1) * .5
        margin_change = ((base_logits[row, winner] - base_logits[row, runner_up]) -
                         (candidate_logits[row, winner] - candidate_logits[row, runner_up])).abs()
        print({"group": name, "argmax_flips": int((base.argmax(-1) != candidate.argmax(-1)).sum()),
               "tv_mean": float(tv.mean()), "tv_p95": float(torch.quantile(tv, .95)),
               "margin_shift_mean": float(margin_change.mean()),
               "gradient_l1_mean": float(gradient[:, indices].abs().sum(-1).mean())})

    # Replace one unlocked shop with another; update every input path that
    # represents it (packed observation, economic features, and town tokens).
    means = np.asarray(norm["observation_mean"])
    stds = np.asarray(norm["observation_std"])
    selected_steps = np.rint(
        z["observation"][:, 0] * stds[0] + means[0]
    ).astype(np.int64)[z["event_day_index"][selected_events]]
    selected_day_indices = z["event_day_index"][selected_events]
    raw_lengths = np.rint(z["observation_length"][days] *
                           norm["observation_length_std"] +
                           norm["observation_length_mean"]).astype(int)
    swapped = observation.copy()
    town_tokens = common[2].clone()
    shop_resources = resources.clone()
    categories = z["token_categories"][days]
    first_shop = np.full(len(offsets) - 1, -1, dtype=np.int64)
    for row_index, raw_length in enumerate(raw_lengths):
        raw = (observation[row_index] * stds + means).copy()
        shop_list = _unpack(raw[:raw_length])[3]
        if not len(shop_list):
            continue
        first_shop[days[row_index]] = shop_list[0]
        raw[raw_length - len(shop_list)] = 7 if shop_list[0] == 4 else 4
        changed_shops = _unpack(raw[:raw_length])[3]
        if checkpoint.get("shop_resource_semantics") == 1:
            rate = shop_rate_features(raw[:raw_length])
            normalized_rate = (rate - norm["resource_mean"][374:383]) / norm["resource_std"][374:383]
            shop_resources[int(offsets[days[row_index]]):int(offsets[days[row_index] + 1]),
                           374:383] = torch.from_numpy(normalized_rate)
        raw[3074:3145] = economic_features(
            raw[:raw_length],
            mature_stored=checkpoint.get("economic_features_semantics") == 2)
        swapped[row_index] = (raw - means) / stds
        for shop in range(8):
            token = np.flatnonzero((categories[row_index, 0] == 6) &
                                   (categories[row_index, 1] == shop + 1))
            assert len(token) == 1
            town_tokens[row_index, token[0], 0] = np.count_nonzero(changed_shops == shop) / 8
        if checkpoint.get("shop_token_semantics") == 2:
            town_tokens[row_index, 0, 8:16] = torch.from_numpy(
                np.bincount(changed_shops, minlength=8).astype(np.float32) / 8 *
                int(z["token_count"][days[row_index]]) *
                checkpoint.get("shop_token_gain", 1.))
    raw_only = observation.copy()
    raw_only[:, :3074] = swapped[:, :3074]
    extra_only = observation.copy()
    extra_only[:, 3074:3145] = swapped[:, 3074:3145]
    paths = [("shop_raw_path", raw_only, common[2], resources),
             ("shop_token_path", observation, town_tokens, resources),
             ("shop_economic_path", extra_only, common[2], resources)]
    if checkpoint.get("shop_resource_semantics") == 1:
        paths.append(("shop_resource_path", observation, common[2], shop_resources))
    paths.append(("consistent_one_shop_swap", swapped, town_tokens, shop_resources))
    for name, obs, tokens, slot_resources in paths:
        with torch.no_grad():
            swapped_logits = forward(torch.from_numpy(obs), tokens, slot_resources)
            swapped_policy = swapped_logits.softmax(-1)
        swapped_tv = (base - swapped_policy).abs().sum(-1) * .5
        shifted_gap = swapped_logits[row, winner] - swapped_logits[row, runner_up]
        gap_shift = (shifted_gap - base_gap).abs()
        report = {"group": name,
               "argmax_flips": int((base.argmax(-1) != swapped_policy.argmax(-1)).sum()),
               "tv_mean": float(swapped_tv.mean()),
               "margin_shift_mean": float(gap_shift.mean()),
               "margin_shift_p95": float(torch.quantile(gap_shift, .95)),
               "top_two_gap_crossings": int((shifted_gap < 0).sum()),
               "shift_exceeds_base_gap": int((gap_shift >= base_gap).sum())}
        if name in ("shop_resource_path", "consistent_one_shop_swap"):
            flips = base.argmax(-1) != swapped_policy.argmax(-1)
            first_handoff = torch.from_numpy(selected_steps == 288)
            report["step288_events"] = int(first_handoff.sum())
            report["step288_argmax_flips"] = int(flips[first_handoff].sum())
            if name == "consistent_one_shop_swap":
                report["step288_tv_mean"] = float(swapped_tv[first_handoff].mean())
                report["later_tv_mean"] = float(swapped_tv[~first_handoff].mean())
        if name in ("shop_resource_path", "consistent_one_shop_swap"):
            direction = torch.from_numpy(np.where(
                first_shop[z["event_day_index"][selected_events]] == 4, -1, 1))
            carrot = (swapped_policy[:, 4] - base[:, 4]) * direction
            report["pet_cafe_aligned_carrot_delta_mean"] = float(carrot.mean())
            report["pet_cafe_aligned_carrot_positive_rate"] = float((carrot > 0).float().mean())
        print(report)
        if name == "consistent_one_shop_swap" and args.step288_output:
            rows = []
            for day_index in np.unique(selected_day_indices[selected_steps == 288]):
                day_row = int(np.searchsorted(days, day_index))
                session = int(z["day_session_index"][day_index])
                selected = np.flatnonzero(selected_day_indices == day_index)
                raw = observation[day_row] * stds + means
                packed_hash = hashlib.sha256(np.rint(
                    raw[:raw_lengths[day_row]]).astype("<i4").tobytes()).hexdigest()
                rows.append({
                    "seed": int(z["seed"][session]),
                    "opponent_code": int(z["opponent"][session]),
                    "seat": int(z["seat"][session]),
                    "actor_packed_integer_sha256": packed_hash,
                    "shop_before": int(first_shop[day_index]),
                    "shop_after": 7 if first_shop[day_index] == 4 else 4,
                    "events": [{
                        "cell": int(z["event_cell"][selected_events[i]]),
                        "original_action": int(base[i].argmax()),
                        "counterfactual_action": int(swapped_policy[i].argmax()),
                        "tv": float(swapped_tv[i]),
                    } for i in selected],
                })
            result = {
                "scope": "fixed_behavior_prefix_step288_shop_swap",
                "checkpoint": str(args.checkpoint), "rollout": str(args.rollout),
                "rows": rows}
            if args.teacher_handoff:
                teacher = json.loads(args.teacher_handoff.read_text())
                names = {1: "thomas_2945_cpp", 2: "metav4_2965",
                         5: "fieldcraft_2887"}
                by_key = {(row["seed"], row["opponent"], row["seat"]): row
                          for row in teacher["teacher_handoff"]}
                if len(by_key) != len(teacher["teacher_handoff"]):
                    raise RuntimeError("duplicate teacher handoff identity")
                for row in rows:
                    key = (row["seed"], names[row["opponent_code"]], row["seat"])
                    target = by_key.pop(key)
                    if any(row[field] != target[field] for field in (
                            "actor_packed_integer_sha256", "shop_before",
                            "shop_after")):
                        raise RuntimeError(f"teacher/student state mismatch: {key}")
                    row["teacher_placement_changed"] = (
                        target["placements"] !=
                        target["counterfactual"]["placements"])
                    row["student_argmax_flips"] = sum(
                        event["original_action"] != event["counterfactual_action"]
                        for event in row["events"])
                if by_key:
                    raise RuntimeError("teacher handoff has unmatched states")
                result["matched_states"] = len(rows)
                result["independent_seeds"] = len({row["seed"] for row in rows})
                result["teacher_placement_changed"] = sum(
                    row["teacher_placement_changed"] for row in rows)
                result["student_states_with_argmax_flip"] = sum(
                    row["student_argmax_flips"] > 0 for row in rows)
                print({key: result[key] for key in (
                    "matched_states", "independent_seeds",
                    "teacher_placement_changed",
                    "student_states_with_argmax_flip")})
            args.step288_output.parent.mkdir(parents=True, exist_ok=True)
            args.step288_output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
