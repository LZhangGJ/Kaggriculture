#!/usr/bin/env python3
"""Small contract tests for the training-only day-state control variate."""

from __future__ import annotations

import unittest

import numpy as np

from experiments.day_state_control_variate import (
    CONTROL_VARIATE_FEATURE_NAMES,
    CONTROL_VARIATE_FEATURE_SCHEMA_SHA256,
    DAY_STATE_FEATURE_NAMES,
    DAY_STATE_FEATURE_SCHEMA_SHA256,
    STUDENT_STEPS,
    _names_sha256,
    crossfit_global_hgb,
    extract_day_state_features,
    same_seed_peer_baseline,
    whole_seed_folds,
)


def packed_day(step=288):
    raw = np.zeros(3074, dtype=np.float64)
    day = step // 24
    raw[:4] = (step, day, 0, 0)
    for base, money in ((4, 100), (1510, 70)):
        raw[base:base + 6] = (money, 4, 5, 0, 3, 0)
        board = raw[base + 6:base + 1506].reshape(100, 15)
        board[:, 1:3] = -1
        board[:, 8] = -1
        board[:, 10] = -1
    own = raw[10:1510].reshape(100, 15)
    own[0] = (3, 1, -1, 10, 0, 2, 1, 0, 14, 0, -1, 1, 0, 0, 0)
    own[1] = (6, -1, 9, 0, 8, 3, 0, 2, -1, 1, -1, 0, 1, 1, 1)
    raw[3016:3028] = np.arange(12)
    raw[3028:3033] = np.arange(5) + 20
    raw[3033] = 1
    raw[3034:3046] = np.arange(12) + 40
    raw[3046] = 0
    raw[3047:3056] = np.arange(9) + 60
    raw[3056:3065] = np.arange(9) + 100
    raw[3065] = 2
    raw[3066:3068] = (2, 5)
    return raw, 3068


class DayStateControlVariateTest(unittest.TestCase):
    def test_schema_hash_is_frozen(self):
        self.assertEqual(len(DAY_STATE_FEATURE_NAMES), 163)
        self.assertEqual(
            _names_sha256(DAY_STATE_FEATURE_NAMES),
            DAY_STATE_FEATURE_SCHEMA_SHA256)
        self.assertEqual(
            CONTROL_VARIATE_FEATURE_SCHEMA_SHA256,
            "8d69d599415553cffb5c37a1f5d46a9ae5a6396d432a584fbb929569830d7a99")
        forbidden = {
            "seed", "environment_seed", "opponent", "seat", "route",
            "policy_seed",
        }
        self.assertFalse(any(
            name.lower().split(".", 1)[0] in forbidden
            for name in CONTROL_VARIATE_FEATURE_NAMES))

    def test_vectorized_extractor_uses_behavior_normalization_and_offsets(self):
        raw, length = packed_day()
        mean = np.linspace(-2.0, 3.0, 3074, dtype=np.float32)
        std = np.linspace(0.5, 1.5, 3074, dtype=np.float32)
        normalized = ((raw - mean) / std).astype(np.float32)[None, :]
        length_mean, length_std = 3060.0, 5.0
        normalized_length = np.asarray(
            [(length - length_mean) / length_std], dtype=np.float32)
        features = extract_day_state_features(
            normalized, normalized_length, np.asarray([288]), {
                "observation_mean": mean,
                "observation_std": std,
                "observation_length_mean": np.asarray(length_mean),
                "observation_length_std": np.asarray(length_std),
            }, chunk_size=1)
        economic = np.concatenate((normalized, np.full((1, 71), 7, dtype=np.float32)), axis=1)
        augmented = extract_day_state_features(
            economic, normalized_length, np.asarray([288]), {
                "observation_mean": np.concatenate((mean, np.zeros(71, dtype=np.float32))),
                "observation_std": np.concatenate((std, np.ones(71, dtype=np.float32))),
                "observation_length_mean": np.asarray(length_mean),
                "observation_length_std": np.asarray(length_std),
            }, chunk_size=1)
        np.testing.assert_array_equal(augmented, features)
        values = dict(zip(DAY_STATE_FEATURE_NAMES, features[0]))
        self.assertEqual(values["day"], 12)
        self.assertEqual(values["remaining_day"], 17)
        self.assertEqual(values["cash_difference"], 30)
        self.assertEqual(values["cash_total"], 170)
        self.assertEqual(values["own.kind_0.count"], 98)
        self.assertEqual(values["own.kind_3.count"], 1)
        self.assertEqual(values["own.kind_6.count"], 1)
        self.assertEqual(values["own.crop.CARROT.count"], 1)
        self.assertEqual(values["own.crop.CARROT.yield_sum"], 2)
        self.assertEqual(values["own.crop.CARROT.age_sum"], 2)
        self.assertEqual(values["own.crop.CARROT.unwatered_sum"], 1)
        self.assertEqual(values["own.animal.GOOSE.count"], 1)
        self.assertEqual(values["own.animal.GOOSE.yield_sum"], 3)
        self.assertEqual(values["own.animal.GOOSE.age_sum"], 4)
        self.assertEqual(values["own.animal.GOOSE.unfed_sum"], 2)
        self.assertEqual(values["own.animal.GOOSE.care_sum"], 1)
        self.assertEqual(values["own.status.fertilized_count"], 1)
        self.assertEqual(values["own.status.watered_count"], 1)
        self.assertEqual(values["own.status.fed_count"], 1)
        self.assertEqual(values["own.status.cared_count"], 1)
        self.assertEqual(values["own.status.fertilizer_available_count"], 1)
        self.assertEqual(values["private.shed.SHEEP"], 11)
        self.assertEqual(values["private.seed.MELON"], 24)
        self.assertEqual(values["private.hand.SHEEP"], 51)
        self.assertEqual(values["market.inventory.FERTILIZER"], 68)
        self.assertEqual(values["market.price.FERTILIZER"], 108)
        self.assertEqual(values["town.shop.FARMERS_MARKET"], 1)
        self.assertEqual(values["town.shop.PIZZA_SHOP"], 1)
        self.assertEqual(sum(
            values[f"town.shop.{name}"] for name in (
                "BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET",
                "ICE_CREAM_SHOP", "PET_CAFE", "PIZZA_SHOP",
                "SMOOTHIE_SHOP", "YARN_STORE")), 2)

    def test_whole_seed_fold_and_peer_leave_current_reward_out(self):
        seeds = np.asarray([10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60])
        rewards = np.linspace(-1.05, 1.05, len(seeds))
        folds = whole_seed_folds(seeds, 6)
        for seed in np.unique(seeds):
            self.assertEqual(len(np.unique(folds[seeds == seed])), 1)
        peer = same_seed_peer_baseline(seeds, rewards)
        changed = rewards.copy()
        changed[0] += 0.01
        changed_peer = same_seed_peer_baseline(seeds, changed)
        self.assertEqual(peer[0], changed_peer[0])
        self.assertNotEqual(peer[1], changed_peer[1])

    def test_crossfit_prediction_excludes_entire_current_seed(self):
        rng = np.random.default_rng(7)
        seed_count, games_per_seed = 12, 2
        game_count = seed_count * games_per_seed
        seeds = np.repeat(np.arange(100, 100 + seed_count), games_per_seed)
        rewards = np.asarray([
            (1.02 if (seed + game) % 2 else -1.02)
            for seed in range(seed_count) for game in range(games_per_seed)
        ])
        sessions = np.repeat(np.arange(game_count), len(STUDENT_STEPS))
        steps = np.tile(STUDENT_STEPS, game_count)
        features = rng.normal(size=(len(sessions), 163)).astype(np.float32)
        counts = np.ones(len(sessions), dtype=np.int32)
        first = crossfit_global_hgb(
            features, sessions, steps, seeds, rewards,
            actionable_counts=counts, max_workers=6)
        changed = rewards.copy()
        changed[0] = -changed[0]
        second = crossfit_global_hgb(
            features, sessions, steps, seeds, changed,
            actionable_counts=counts, max_workers=6)
        current_days = sessions == 0
        np.testing.assert_array_equal(
            first.predictions[current_days], second.predictions[current_days])
        self.assertEqual(first.metrics["folds"], 6)
        self.assertTrue(np.all(np.isfinite(first.advantages)))


if __name__ == "__main__":
    unittest.main()
