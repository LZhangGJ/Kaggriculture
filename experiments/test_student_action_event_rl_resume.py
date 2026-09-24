#!/usr/bin/env python3
"""Small regression check for the PPO rollout resume boundary."""

from __future__ import annotations

import copy
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.native_student_actor.native_job_batch import ppo_games

from experiments.train_student_action_event_rl_v3 import (
    ROLLOUT_SCHEMA,
    STUDENT_STEPS,
    _attach_native_day_advantages,
    _approx_kl,
    _batch_terms,
    _day_bundle_objective,
    _jobs,
    _load_native_rollout,
    _optimizer_state_for_resume,
    _paired_seed_loo_advantages,
    _restore_native_games,
    _resident_rollout,
    _save_native_rollout,
    _stratified_loo_advantages,
    _validate_native_job_routes,
    _validate_inputs,
    _validate_resume_metadata,
)


class ResumeContractTest(unittest.TestCase):
    def test_resident_batch_matches_cpu_source(self):
        arrays = {
            "context": np.ones((2, 2), dtype=np.float32),
            "observation": np.ones((2, 2), dtype=np.float32),
            "observation_length": np.ones((2, 1), dtype=np.float32),
            "token_continuous": np.ones((2, 2), dtype=np.float32),
            "token_categories": np.ones((2, 7), dtype=np.int64),
            "token_count": np.ones((2,), dtype=np.int64),
            "day_event_offsets": np.array([0, 2, 3]),
            "event_resources": np.ones((3, 2), dtype=np.float32),
            "event_cell": np.array([0, 1, 2]),
            "event_stage": np.array([0, 1, 1]),
            "event_legal": np.ones((3, 9), dtype=np.bool_),
            "event_action": np.array([0, 1, 2]),
            "old_logprob": np.zeros(3, dtype=np.float32),
        }
        games = [{"days": [
            {"native_index": index, "native_arrays": arrays}
            for index in range(2)]}]

        class Actor:
            def initial_hidden(self, context, *_args):
                return context

            def step(self, hidden, resources, _cells, _stages, _previous,
                     _legal):
                return torch.zeros((len(hidden), 9)), hidden + resources

        device = torch.device("cpu")
        with torch.no_grad():
            source = _batch_terms(Actor(), games, device, 1.0)
            resident = _batch_terms(
                Actor(), games, device, 1.0,
                resident=_resident_rollout(arrays, device))
        for left, right in zip(source, resident):
            self.assertTrue(torch.equal(left, right))

    def setUp(self):
        self.args = SimpleNamespace(
            opponent_backend="native_cpp", margin_weight=0.1,
            margin_scale=10000.0, temperature=1.0)
        self.fingerprints = {
            "checkpoint_sha256": "checkpoint",
            "binary_sha256": "binary",
            "manifest_sha256": "manifest",
        }
        self.artifacts = {
            "opponent": {"module_sha256": "module", "asset_sha256": "asset"},
        }
        self.payload = {
            "schema": ROLLOUT_SCHEMA,
            "action_unit": "day_bundle",
            "policy_version": "checkpoint",
            "binary_sha256": "binary",
            "manifest_sha256": "manifest",
            "opponent_backend": "native_cpp",
            "student_steps": STUDENT_STEPS,
            "temperature": 1.0,
            "reward": {
                "formula": "sign(margin)+weight*tanh(margin/scale)",
                "margin_weight": 0.1,
                "margin_scale": 10000.0,
            },
            "native_opponent_artifacts": copy.deepcopy(self.artifacts),
        }

    def validate(self, payload=None):
        _validate_resume_metadata(
            payload or self.payload, self.args, self.fingerprints,
            self.artifacts)

    def test_exact_and_legacy_default_temperature_pass(self):
        self.validate()
        legacy = copy.deepcopy(self.payload)
        del legacy["temperature"]
        self.validate(legacy)

    def test_temperature_or_artifact_drift_fails(self):
        legacy = copy.deepcopy(self.payload)
        del legacy["temperature"]
        self.args.temperature = 0.5
        with self.assertRaises(RuntimeError):
            self.validate(legacy)
        self.args.temperature = 1.0
        drifted = copy.deepcopy(self.payload)
        drifted["native_opponent_artifacts"]["opponent"]["module_sha256"] = "changed"
        with self.assertRaises(RuntimeError):
            self.validate(drifted)

    def test_replay_allowlist_drift_fails(self):
        artifacts = {
            "replay_clean": {
                "module_sha256": "module", "asset_sha256": "asset",
                "metadata_sha256": "metadata", "pool_routes": 1,
                "entries": [{
                    "clean_index": 65, "bundle_index": 237,
                    "family": "G397", "route_id": "110795524:1",
                }],
            },
        }
        payload = copy.deepcopy(self.payload)
        payload["native_opponent_artifacts"] = copy.deepcopy(artifacts)
        _validate_resume_metadata(
            payload, self.args, self.fingerprints, artifacts)
        artifacts["replay_clean"]["entries"][0]["bundle_index"] = 236
        with self.assertRaises(RuntimeError):
            _validate_resume_metadata(
                payload, self.args, self.fingerprints, artifacts)

    def test_native_job_input_drift_fails(self):
        args = copy.deepcopy(self.args)
        args.native_job_rollout = True
        fingerprints = {
            **self.fingerprints,
            "native_weights_sha256": "weights",
            "native_job_module_sha256": "job-module",
            "native_job_inputs_sha256": {"fast_env": "fast", "r1_config": "config"},
        }
        payload = {
            **copy.deepcopy(self.payload),
            "rollout_engine": "pure_cpp_job_batch",
            "native_weights_sha256": "weights",
            "native_job_module_sha256": "job-module",
            "native_job_inputs_sha256": {"fast_env": "fast", "r1_config": "config"},
        }
        _validate_resume_metadata(payload, args, fingerprints, self.artifacts)
        payload["native_job_inputs_sha256"]["r1_config"] = "changed"
        with self.assertRaises(RuntimeError):
            _validate_resume_metadata(payload, args, fingerprints, self.artifacts)

    def test_approx_kl_is_zero_or_positive(self):
        active = torch.tensor([True, True, False])
        old = torch.tensor([-1.0, -2.0, -3.0])
        self.assertEqual(float(_approx_kl(old, old, active)), 0.0)
        changed = _approx_kl(
            torch.tensor([-0.8, -2.3, -30.0]), old, active)
        self.assertGreater(float(changed), 0.0)
        tiny = _approx_kl(old + torch.finfo(torch.float32).eps, old, active)
        self.assertGreaterEqual(float(tiny), 0.0)

    def test_rl_checkpoint_requires_optimizer_state(self):
        self.assertIsNone(_optimizer_state_for_resume({}))
        state = {"state": {}}
        self.assertIs(_optimizer_state_for_resume({"rl_optimizer": state}), state)
        with self.assertRaisesRegex(RuntimeError, "missing optimizer state"):
            _optimizer_state_for_resume({"rl": {"gradient_steps": 1}})

    def test_day_bundle_sum_length_normalization_and_clip(self):
        # day0: ratio=1.5, A=4 -> clipped objective 4.8.
        # day1: ratio=0.5, A=-3 -> clipped objective -2.4.
        old = torch.zeros(4)
        new = torch.tensor([
            torch.log(torch.tensor(1.25)),
            torch.log(torch.tensor(1.2)),
            99.0,  # forced slot: excluded from the bundle likelihood
            torch.log(torch.tensor(0.5)),
        ])
        entropy = torch.tensor([0.2, 0.6, 99.0, 0.9])
        active = torch.tensor([True, True, False, True])
        event_days = torch.tensor([0, 0, 0, 1])
        day_games = torch.tensor([0, 1, 2])  # third day has no policy choice
        loss, mean_entropy, bundle = _day_bundle_objective(
            new, old, entropy, active, event_days, day_games,
            torch.tensor([4.0, -3.0, 999.0]), 0.2)
        torch.testing.assert_close(
            bundle["day_ratio"], torch.tensor([1.5, 0.5, 1.0]))
        torch.testing.assert_close(
            bundle["day_actionable"], torch.tensor([2.0, 1.0, 0.0]))
        torch.testing.assert_close(
            bundle["day_advantage"], torch.tensor([4.0, -3.0, 999.0]))
        torch.testing.assert_close(
            bundle["valid_days"], torch.tensor([True, True, False]))
        torch.testing.assert_close(loss, torch.tensor(-1.2))
        torch.testing.assert_close(mean_entropy, torch.tensor(0.65))

    def test_day_aligned_advantage_matches_old_loss_and_gradient_exactly(self):
        old = torch.zeros(4)
        active = torch.tensor([True, True, False, True])
        event_days = torch.tensor([0, 0, 0, 1])
        day_games = torch.tensor([0, 1])
        game_advantages = torch.tensor([4.0, -3.0])
        entropy = torch.tensor([0.2, 0.6, 99.0, 0.9])
        first = torch.tensor([0.1, -0.2, 7.0, 0.3], requires_grad=True)
        first_loss, first_entropy, _ = _day_bundle_objective(
            first, old, entropy, active, event_days, day_games,
            game_advantages, 0.2)
        first_loss.backward()
        first_gradient = first.grad.detach().clone()
        second = first.detach().clone().requires_grad_(True)
        second_loss, second_entropy, _ = _day_bundle_objective(
            second, old, entropy, active, event_days, day_games,
            game_advantages, 0.2,
            day_advantages=game_advantages.index_select(0, day_games))
        second_loss.backward()
        torch.testing.assert_close(first_loss, second_loss, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            first_entropy, second_entropy, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            first_gradient, second.grad, rtol=0.0, atol=0.0)

    def test_native_day_advantage_join_uses_identity_not_sorted_position(self):
        arrays = {
            "seed": np.asarray([7, 7]),
            "opponent": np.asarray([1, 2]),
            "seat": np.asarray([0, 1]),
            "route": np.asarray([-1, -1]),
            "policy_seed": np.asarray([101, 202]),
            "day_session_index": np.asarray([0, 0, 1, 1]),
            "day_step": np.asarray([288, 312, 288, 312]),
        }
        results = [{
            "seed": 7, "opponent": "metav4_2965", "seat": 1,
            "route": -1, "policy_seed": 202,
            "days": [{"step": 288}, {"step": 312}],
        }, {
            "seed": 7, "opponent": "thomas_2945_cpp", "seat": 0,
            "route": -1, "policy_seed": 101,
            "days": [{"step": 288}, {"step": 312}],
        }]
        _attach_native_day_advantages(
            results, arrays, np.asarray([10.0, 11.0, 20.0, 21.0]))
        self.assertEqual([
            day["day_state_advantage"] for day in results[0]["days"]
        ], [20.0, 21.0])
        self.assertEqual([
            day["day_state_advantage"] for day in results[1]["days"]
        ], [10.0, 11.0])

    def test_stratified_baseline_removes_opponent_and_seat_level(self):
        results = [
            {"opponent": opponent, "seat": seat}
            for opponent in ("easy", "hard") for seat in (0, 1)
            for _ in range(2)
        ]
        rewards = torch.tensor([
            1.1, 0.9, 0.8, 0.6, -0.7, -0.9, -0.8, -1.0,
        ]).numpy()
        advantages = _stratified_loo_advantages(results, rewards)
        for offset in range(0, len(results), 2):
            self.assertAlmostEqual(
                float(advantages[offset:offset + 2].mean()), 0.0, delta=1e-6)
        self.assertGreater(float(advantages.std()), 0.0)

    def test_paired_seed_baseline_removes_common_seed_shift(self):
        results = [
            {"seed": seed, "opponent": opponent, "seat": seat}
            for seed in (10, 20)
            for opponent in ("meta", "thomas") for seat in (0, 1)
        ]
        rewards = np.asarray([
            11.0, 10.0, 9.0, 8.0,
            -2.0, -3.0, -4.0, -5.0,
        ], dtype=np.float32)
        advantages = _paired_seed_loo_advantages(results, rewards)
        shifted = rewards.copy()
        shifted[:4] += 100.0
        shifted[4:] -= 50.0
        np.testing.assert_allclose(
            advantages,
            _paired_seed_loo_advantages(results, shifted),
            rtol=0.0, atol=5e-6)
        self.assertAlmostEqual(float(advantages[:4].mean()), 0.0, delta=1e-6)
        self.assertAlmostEqual(float(advantages[4:].mean()), 0.0, delta=1e-6)

    def test_paired_seed_jobs_reject_incomplete_block(self):
        args = SimpleNamespace(games=3, opponents=("meta", "thomas"))
        with self.assertRaisesRegex(ValueError, "complete opponent-by-seat"):
            _jobs(args, {}, {})

    def test_day_state_baseline_rejects_reward_support_drift_first(self):
        args = SimpleNamespace(
            day_state_crossfit_baseline=True, native_job_rollout=True,
            day_state_critic_workers=6, margin_weight=0.2,
            rollout_output=Path("unused.npz"))
        with self.assertRaisesRegex(ValueError, "margin_weight=0.1"):
            _validate_inputs(args)

    def test_native_npz_round_trip_rebuilds_games_and_rejects_drift(self):
        arrays = {
            "seed": np.asarray([7], dtype=np.uint64),
            "seat": np.asarray([0], dtype=np.int32),
            "opponent": np.asarray([1], dtype=np.int32),
            "route": np.asarray([-1], dtype=np.int32),
            "policy_seed": np.asarray([11], dtype=np.uint64),
            "own_cash": np.asarray([120.0]),
            "rival_cash": np.asarray([100.0]),
            "day_session_index": np.asarray([0], dtype=np.int32),
            "day_step": np.asarray([288], dtype=np.int32),
            "day_event_offsets": np.asarray([0, 1], dtype=np.int64),
            "context": np.zeros((1, 2), dtype=np.float32),
            "observation": np.zeros((1, 3), dtype=np.float32),
            "observation_length": np.ones(1, dtype=np.float32),
            "token_continuous": np.zeros((1, 1, 2), dtype=np.float32),
            "token_categories": np.zeros((1, 7, 1), dtype=np.uint32),
            "token_count": np.ones(1, dtype=np.uint32),
            "event_resources": np.zeros((1, 4), dtype=np.float32),
            "event_cell": np.asarray([3], dtype=np.int32),
            "event_stage": np.asarray([1], dtype=np.int32),
            "event_previous": np.asarray([11], dtype=np.int32),
            "event_legal_mask": np.asarray([3], dtype=np.uint32),
            "event_legal": np.asarray(
                [[True, True] + [False] * 9], dtype=np.bool_),
            "event_action": np.asarray([1], dtype=np.int32),
            "old_logprob": np.asarray([-0.5], dtype=np.float32),
            "old_entropy": np.asarray([0.6], dtype=np.float32),
        }
        games = ppo_games(arrays)
        games[0].update({
            "opponent_backend": "native_cpp",
            "opponent_module_sha256": "module",
            "opponent_asset_sha256": "asset",
            "policy_sha256": "checkpoint",
            "binary_sha256": "binary",
            "manifest_sha256": "manifest",
            "elapsed_seconds": 1.0,
        })
        payload = {**self.payload, "rollout_engine": "pure_cpp_job_batch",
                   "games": games}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollout.npz"
            _save_native_rollout(path, payload, arrays)
            with zipfile.ZipFile(path) as archive:
                self.assertEqual(
                    {entry.compress_type for entry in archive.infolist()},
                    {zipfile.ZIP_STORED})
            loaded, loaded_arrays, game_metadata = _load_native_rollout(path)
            restored = _restore_native_games(
                loaded_arrays, game_metadata, 0.1, 10000.0)
        self.assertEqual(loaded["schema"], ROLLOUT_SCHEMA)
        self.assertEqual(restored[0]["seed"], 7)
        self.assertEqual(restored[0]["route"], -1)
        self.assertEqual(restored[0]["policy_sha256"], "checkpoint")
        self.assertEqual(restored[0]["days"][0]["step"], 288)
        drifted_arrays = dict(loaded_arrays)
        drifted_arrays["route"] = np.asarray([0], dtype=np.int32)
        with self.assertRaises(RuntimeError):
            _restore_native_games(
                drifted_arrays, game_metadata, 0.1, 10000.0)
        game_metadata[0]["seed"] = 8
        with self.assertRaises(RuntimeError):
            _restore_native_games(
                loaded_arrays, game_metadata, 0.1, 10000.0)

    def test_native_replay_route_and_detailed_variant_round_trip(self):
        arrays = {
            "seed": np.asarray([7], dtype=np.uint64),
            "seat": np.asarray([1], dtype=np.int32),
            "opponent": np.asarray([3], dtype=np.int32),
            "route": np.asarray([237], dtype=np.int32),
            "policy_seed": np.asarray([11], dtype=np.uint64),
            "own_cash": np.asarray([120.0]),
            "rival_cash": np.asarray([100.0]),
            "day_session_index": np.asarray([0], dtype=np.int32),
            "day_step": np.asarray([288], dtype=np.int32),
            "day_event_offsets": np.asarray([0, 1], dtype=np.int64),
            "context": np.zeros((1, 2), dtype=np.float32),
            "observation": np.zeros((1, 3), dtype=np.float32),
            "observation_length": np.ones(1, dtype=np.float32),
            "token_continuous": np.zeros((1, 1, 2), dtype=np.float32),
            "token_categories": np.zeros((1, 7, 1), dtype=np.uint32),
            "token_count": np.ones(1, dtype=np.uint32),
            "event_resources": np.zeros((1, 4), dtype=np.float32),
            "event_cell": np.asarray([3], dtype=np.int32),
            "event_stage": np.asarray([1], dtype=np.int32),
            "event_previous": np.asarray([11], dtype=np.int32),
            "event_legal_mask": np.asarray([3], dtype=np.uint32),
            "event_legal": np.asarray(
                [[True, True] + [False] * 9], dtype=np.bool_),
            "event_action": np.asarray([1], dtype=np.int32),
            "old_logprob": np.asarray([-0.5], dtype=np.float32),
            "old_entropy": np.asarray([0.6], dtype=np.float32),
        }
        _validate_native_job_routes(arrays, 1)
        games = ppo_games(arrays)
        games[0].update({
            "opponent_variant": "replay:G397:110795524:1",
            "opponent_backend": "native_cpp",
            "opponent_module_sha256": "module",
            "opponent_asset_sha256": "asset",
            "policy_sha256": "checkpoint",
            "binary_sha256": "binary",
            "manifest_sha256": "manifest",
            "elapsed_seconds": 1.0,
        })
        payload = {**self.payload, "rollout_engine": "pure_cpp_job_batch",
                   "games": games}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollout.npz"
            _save_native_rollout(path, payload, arrays)
            _, loaded_arrays, game_metadata = _load_native_rollout(path)
            restored = _restore_native_games(
                loaded_arrays, game_metadata, 0.1, 10000.0)
        self.assertEqual(restored[0]["route"], 237)
        self.assertEqual(
            restored[0]["opponent_variant"], "replay:G397:110795524:1")
        invalid = dict(arrays)
        invalid["route"] = np.asarray([-1], dtype=np.int32)
        with self.assertRaises(RuntimeError):
            _validate_native_job_routes(invalid, 1)
        drifted = dict(loaded_arrays)
        drifted["route"] = np.asarray([236], dtype=np.int32)
        with self.assertRaises(RuntimeError):
            _restore_native_games(
                drifted, game_metadata, 0.1, 10000.0)

    def test_native_public_opponent_four_requires_no_route(self):
        arrays = {
            "route": np.asarray([-1], dtype=np.int32),
            "opponent": np.asarray([4], dtype=np.int32),
        }
        _validate_native_job_routes(arrays, 1)
        arrays["route"] = np.asarray([237], dtype=np.int32)
        with self.assertRaisesRegex(RuntimeError, "identity is invalid"):
            _validate_native_job_routes(arrays, 1)


if __name__ == "__main__":
    unittest.main()
