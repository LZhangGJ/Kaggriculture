#!/usr/bin/env python3
"""Diagnostic decoder scaffold for post-step288 autoregressive portfolio filling.

The 192/32 inputs are encoded latent widths, not a completed raw-feature
contract. Real training remains disabled until the C++ teacher exports the
state, calendars, masks and labels described by the feature audit.
"""

import argparse
import json


KINDS = ("SKIP", "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
         "GOOSE", "COW", "SHEEP")
RESOURCE_NAMES = ("cash", "labor_d0", "labor_d1", "labor_d2", "seed_wheat",
                  "seed_carrot", "seed_tomato", "seed_strawberry", "seed_melon",
                  "unassigned_slots")
DEPOTS = (44, 45, 54, 55)


def slot_key(cell):
    x, y = cell % 10, cell // 10
    return min(abs(x - depot % 10) + abs(y - depot // 10) for depot in DEPOTS), cell


def contains_private_identity(value):
    if isinstance(value, dict):
        if set(value) & {"bot", "opponent_id", "opponent_name", "source_family", "opponent_private"}:
            return True
        return any(contains_private_identity(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_private_identity(item) for item in value)
    return False


def validate_record(row):
    if row.get("format") != "kaggriculture-autofill-v1" or row.get("step", -1) < 288:
        raise ValueError("autofill record must be post-step288 kaggriculture-autofill-v1")
    provenance = row.get("provenance", {})
    if not provenance.get("policy_source") or not isinstance(provenance.get("diagnostic"), bool):
        raise ValueError("policy_source and explicit diagnostic flag are required")
    if contains_private_identity(row.get("state", {})):
        raise ValueError("opponent identity is forbidden in student state")
    slots = row.get("slots", [])
    cells = [slot.get("cell") for slot in slots]
    if cells != sorted(cells, key=slot_key) or len(cells) != len(set(cells)):
        raise ValueError("slots must use unique planner (near-depot, cell) ordering")
    for slot in slots:
        mask, label = slot.get("legal_mask"), slot.get("label")
        if not isinstance(mask, list) or len(mask) != len(KINDS) or any(x not in (0, 1, False, True) for x in mask):
            raise ValueError("each legal_mask must contain nine booleans")
        if not isinstance(label, int) or not 0 <= label < len(KINDS) or not mask[label]:
            raise ValueError("teacher label must be legal")
        remaining = slot.get("remaining_resources", {})
        if set(remaining) != set(RESOURCE_NAMES):
            raise ValueError(f"remaining_resources must contain {RESOURCE_NAMES}")
    return row


def autoregressive_context(row):
    """Yield each slot with the virtual prefix that must be applied before it."""
    validate_record(row)
    counts = [0] * len(KINDS)
    virtual = {}
    for slot in row["slots"]:
        yield {"cell": slot["cell"], "selected_portfolio": dict(virtual),
               "portfolio_counts": list(counts),
               "remaining_resources": slot["remaining_resources"],
               "legal_mask": slot["legal_mask"], "label": slot["label"]}
        label = slot["label"]
        virtual[slot["cell"]] = label
        counts[label] += 1


def build_model(context_dim=192, slot_dim=32, hidden=64, label_embedding=8):
    import torch

    class AutofillStudent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.context_encoder = torch.nn.Sequential(
                torch.nn.Linear(context_dim, hidden), torch.nn.ReLU())
            self.label_embedding = torch.nn.Embedding(len(KINDS) + 1, label_embedding)
            width = hidden + slot_dim + len(RESOURCE_NAMES) + len(KINDS) + label_embedding
            self.cell = torch.nn.GRUCell(width, hidden)
            self.head = torch.nn.Linear(hidden, len(KINDS))

        def begin(self, encoded_context):
            context = self.context_encoder(encoded_context)
            return context, context

        def decode_step(self, context, hidden, slot_features, remaining_resources,
                        portfolio_counts, previous_label, legal_mask):
            inputs = torch.cat((context, slot_features, remaining_resources, portfolio_counts,
                                self.label_embedding(previous_label)), dim=-1)
            hidden = self.cell(inputs, hidden)
            return self.head(hidden).masked_fill(~legal_mask.bool(), -1e9), hidden

        def forward(self, encoded_context, slot_features, remaining_resources,
                    portfolio_counts, previous_labels, legal_mask):
            context, hidden = self.begin(encoded_context)
            outputs = []
            for index in range(slot_features.shape[1]):
                logits, hidden = self.decode_step(
                    context, hidden, slot_features[:, index], remaining_resources[:, index],
                    portfolio_counts[:, index], previous_labels[:, index], legal_mask[:, index])
                outputs.append(logits)
            return torch.stack(outputs, dim=1)

    return AutofillStudent()


def synthetic_record():
    remaining = dict.fromkeys(RESOURCE_NAMES, 0.0)
    remaining["cash"] = 5000.0
    return {
        "type": "autofill", "format": "kaggriculture-autofill-v1", "step": 288,
        "state": {"encoded_context": [0.0] * 192,
                  "feature_schema": "synthetic-placeholder"},
        "slots": [
            {"cell": 31, "slot_features": [0.0] * 32,
             "remaining_resources": remaining,
             "legal_mask": [1, 1, 1, 1, 1, 1, 0, 0, 0], "label": 1},
            {"cell": 12, "slot_features": [0.0] * 32,
             "remaining_resources": remaining,
             "legal_mask": [1] * 9, "label": 7},
        ],
        "teacher": {"source": "closed_loop_bundle", "candidate_id": 3,
                    "objective": "paired_terminal_margin"},
        "provenance": {"policy_source": "synthetic-teacher", "diagnostic": True},
        "split_only": {"group_id": "7:synthetic", "seed": 7,
                       "opponent_group": "synthetic"},
    }


def self_check():
    import torch

    row = synthetic_record()
    contexts = list(autoregressive_context(row))
    assert contexts[0]["selected_portfolio"] == {}
    assert contexts[1]["selected_portfolio"] == {31: 1}
    assert contexts[1]["portfolio_counts"][1] == 1
    model = build_model()
    batch, slots = 2, 3
    legal = torch.ones(batch, slots, len(KINDS), dtype=torch.bool)
    legal[:, :, 8] = False
    logits = model(torch.zeros(batch, 192), torch.zeros(batch, slots, 32),
                   torch.zeros(batch, slots, len(RESOURCE_NAMES)),
                   torch.zeros(batch, slots, len(KINDS)),
                   torch.full((batch, slots), len(KINDS), dtype=torch.long),
                   legal)
    parameters = sum(value.numel() for value in model.parameters())
    assert logits.shape == (batch, slots, len(KINDS)) and parameters < 100_000
    assert torch.all(logits[:, :, 8] == -1e9)
    print(json.dumps({"status": "PASS", "classes": list(KINDS),
                      "parameters": parameters, "logits_shape": list(logits.shape)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if not args.self_check:
        parser.error("only --self-check is implemented; teacher export must precede training")
    self_check()


if __name__ == "__main__":
    main()
