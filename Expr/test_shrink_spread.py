"""CPU regression tests: python3 -m unittest discover -s Expr -v."""
import math
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from shrink_spread import (
    ChunkEdit, Recall, checkpoint_spec, chunked_recurrence, matches_target,
    newton_schulz5, query_pair_positions, settings_grid, validate_replay, write_pair_recall_csv,
)
from state_geometry import json_safe, recurrence


class ShrinkSpreadTests(unittest.TestCase):
    def inputs(self, length=65):
        generator = torch.Generator().manual_seed(12)
        q, k = [torch.nn.functional.normalize(
            torch.randn(2, length, 2, 4, generator=generator), dim=-1) for _ in range(2)]
        v = torch.randn(2, length, 2, 4, generator=generator)
        alpha = torch.full_like(q, 0.95)
        return q, k, v, alpha, 0.2 * k

    def edit(self, kind="iso_kdn", rank=1, gamma=0.1, eta=0.5, directions="keys"):
        return ChunkEdit(kind, 256, rank, gamma, eta, 2, 7, directions)

    def test_unedited_and_identity_edits_match_token_recurrence(self):
        tensors = self.inputs()
        expected, _ = recurrence(*tensors, 0.5)
        for chunk in (1, 16, 32, 64, 100):
            for edit in (None, self.edit(gamma=1., eta=0.)):
                actual = chunked_recurrence(*tensors, 0.5, chunk, edit)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_first_edit_affects_short_sequences_only_after_boundary(self):
        tensors = self.inputs(64)
        expected, _ = recurrence(*tensors, 0.5)
        edit = self.edit()
        actual = chunked_recurrence(*tensors, 0.5, 16, edit)
        self.assertEqual(edit.first_effect_token, 32)
        torch.testing.assert_close(actual[:, :32], expected[:, :32], rtol=0, atol=0)
        self.assertGreater((actual[:, 32:] - expected[:, 32:]).abs().max().item(), 1e-4)

    def test_past_window_does_not_use_current_or_future_keys(self):
        _, keys, _, _, _ = self.inputs()
        update = torch.eye(4).expand(2, 2, 4, 4).clone()
        expected = self.edit()(keys, 16, update)
        changed = keys.clone()
        changed[:, 16:] = 100
        torch.testing.assert_close(self.edit()(changed, 16, update), expected, rtol=0, atol=0)

    def test_future_inputs_do_not_change_past_outputs(self):
        tensors = self.inputs()
        expected = chunked_recurrence(*tensors, 0.5, 16, self.edit())
        changed = [x.clone() for x in tensors]
        changed[1][:, 41:] *= -1
        changed[2][:, 41:] += 10
        actual = chunked_recurrence(*changed, 0.5, 16, self.edit())
        torch.testing.assert_close(actual[:, :41], expected[:, :41], rtol=0, atol=0)

    def test_common_key_axis_is_shrunk_on_left(self):
        keys = torch.tensor([1., 0.]).expand(1, 8, 1, 2)
        update = torch.tensor([[[[2., 3.], [5., 7.]]]])
        actual = self.edit(eta=0.)(keys, 8, update)
        expected = update.clone()
        expected[..., 0, :] *= 0.1
        torch.testing.assert_close(actual, expected)

    def test_randomized_iteration_does_not_lock_onto_mean(self):
        keys = torch.tensor([[[[0.8, 0.6]], [[-0.8, 0.6]]]]).repeat(1, 8, 1, 1)
        edit = self.edit(eta=0.)
        actual = edit(keys, 16, torch.eye(2)[None, None])
        torch.testing.assert_close(actual, torch.diag(torch.tensor([0.1, 1.]))[None, None])
        self.assertAlmostEqual(edit.diagnostics()["power_quality"], 1.0, places=5)

    def test_zero_covariance_and_rank_deficiency_do_not_shrink_null_axes(self):
        keys = torch.tensor([0.6, 0.8, 0., 0.]).expand(1, 32, 1, 4)
        update = torch.eye(4)[None, None]
        edit = self.edit(kind="diag_kdn", rank=2, eta=0.)
        torch.testing.assert_close(edit(keys, 32, update), update, rtol=0, atol=0)
        self.assertEqual(edit.diagnostics()["supported_rank"], 0)
        iso = self.edit(rank=2, eta=0.)
        actual = iso(keys, 32, update)
        direction = keys[0, 0, 0]
        expected = update - 0.9 * torch.outer(direction, direction)
        torch.testing.assert_close(actual, expected)
        self.assertAlmostEqual(iso.diagnostics()["supported_rank"], 1., places=5)

    def test_spread_targets_whole_shrunk_update(self):
        keys = torch.tensor([1., 0.]).expand(1, 8, 1, 2)
        update = torch.eye(2)[None, None]
        edit = self.edit(eta=0.5)
        actual = edit(keys, 8, update)
        shrunk = torch.diag(torch.tensor([0.1, 1.]))[None, None]
        torch.testing.assert_close(actual, 0.5 * shrunk + 0.5 * newton_schulz5(shrunk))
        metrics = edit.diagnostics()
        self.assertLess(metrics["common_shrunk"], metrics["common_before"])
        self.assertGreater(metrics["common_after"], metrics["common_shrunk"])

    def test_spread_preserves_norm_at_small_and_large_scales(self):
        matrix = torch.tensor([[1., 0.2], [-0.3, 2.]])
        expected = newton_schulz5(matrix)
        for scale in (1., 1e-20, 1e20):
            actual = newton_schulz5(matrix * scale) / scale
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
            self.assertAlmostEqual(actual.norm().item(), matrix.norm().item(), places=5)
        torch.testing.assert_close(newton_schulz5(torch.zeros_like(matrix)), torch.zeros_like(matrix))

    def test_only_new_write_is_edited_and_next_residual_uses_edited_state(self):
        q = k = alpha = torch.ones(1, 3, 1, 1)
        v = torch.tensor([2., 4., 0.]).view(1, 3, 1, 1)
        write = torch.tensor([1., 1., 0.5]).view(1, 3, 1, 1)
        actual = chunked_recurrence(q, k, v, alpha, write, 1., 1, self.edit(gamma=0., eta=0.))
        # Chunk 1 reads 4, then its new write of +2 is removed, retaining 2.
        # The next correction is 0.5*(0-2), producing 1, not 0 or 2.
        torch.testing.assert_close(actual.flatten(), torch.tensor([2., 4., 1.]))

    def test_tiny_update_is_not_lost_by_subtracting_large_states(self):
        class Recorder:
            count = 0
            first_effect_token = None
            def __init__(self):
                self.updates = []
            def __call__(self, keys, start, update, write=None):
                self.updates.append(update.clone())
                self.count += 1
                return update
        q = alpha = torch.ones(1, 3, 1, 1)
        k = torch.zeros_like(q)
        v = torch.tensor([1e20, 1., 0.]).view_as(q)
        write = torch.ones_like(q)
        edit = Recorder()
        chunked_recurrence(q, k, v, alpha, write, 1., 1, edit)
        self.assertEqual(edit.updates[1].item(), 1.)

    def test_outer_autocast_does_not_change_intervention(self):
        tensors = self.inputs()
        expected = chunked_recurrence(*tensors, 0.5, 16, self.edit())
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            actual = chunked_recurrence(*tensors, 0.5, 16, self.edit())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_chunk_update_matches_explicit_decay_weighted_writes(self):
        class Recorder:
            count = 0
            first_effect_token = None
            def __init__(self):
                self.updates = []
            def __call__(self, keys, start, update, write=None):
                self.updates.append((start, update.clone()))
                self.count += 1
                return update
        q, k, v, _, write = [x.double() for x in self.inputs(17)]
        alpha = torch.linspace(0.6, 0.99, q.numel(), dtype=q.dtype).reshape_as(q)
        _, deltas = recurrence(q, k, v, alpha, write, 0.5)
        edit = Recorder()
        chunked_recurrence(q, k, v, alpha, write, 0.5, 4, edit)
        for start, actual in edit.updates:
            expected = torch.zeros_like(actual)
            for t in range(start, start + 4):
                retention = alpha[:, t + 1:start + 4].prod(1)
                expected += retention[..., None] * write[:, t, ..., None] * deltas[:, t, ..., None, :]
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_replay_validation_rejects_nonfinite_or_incorrect_outputs(self):
        q, k, v, alpha, write = self.inputs()
        out, _ = recurrence(q, k, v, alpha, write, 0.5)
        call = dict(kind="iso_kdn", q=q, k=k, v=v, write=write,
                    g_cumsum=alpha.log2().cumsum(1), scale=0.5, out=out)
        self.assertLess(validate_replay(call, (2., None), 16), 1e-5)
        for bad in (out * 2, torch.full_like(out, math.nan)):
            with self.assertRaises(RuntimeError):
                validate_replay(dict(call, out=bad), (2., None), 16)

    def test_invalid_parameters_fail(self):
        for kwargs in (dict(rank=0), dict(gamma=-0.1), dict(eta=1.1)):
            with self.assertRaises(ValueError):
                self.edit(**kwargs)

    def test_gain_directions_use_raw_past_writes_not_keys(self):
        keys = torch.tensor([[[[0.8, 0.6]], [[-0.8, 0.6]]]]).repeat(1, 8, 1, 1)
        gains = torch.tensor([[[[1., 3.]], [[1., -3.]]]]).repeat(1, 8, 1, 1)
        update = torch.eye(2)[None, None]
        from_keys = self.edit(kind="diag_kdn", eta=0.)(keys, 8, update, write=gains)
        from_gains = self.edit(kind="diag_kdn", eta=0., directions="gains")(keys, 8, update, write=gains)
        torch.testing.assert_close(from_keys, torch.diag(torch.tensor([0.1, 1.]))[None, None])
        torch.testing.assert_close(from_gains, torch.diag(torch.tensor([1., 0.1]))[None, None])
        gains[:, 8:] = 10000
        changed_keys = torch.randn_like(keys)
        actual = self.edit(kind="diag_kdn", eta=0., directions="gains")(changed_keys, 8, update, write=gains)
        torch.testing.assert_close(actual, from_gains, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            self.edit(directions="gains")

    def test_gain_vectors_are_passed_through_recurrence(self):
        tensors = self.inputs()
        class GainRecorder(ChunkEdit):
            def __call__(self, keys, start, update, write=None):
                torch.testing.assert_close(write, tensors[-1], rtol=0, atol=0)
                return super().__call__(keys, start, update, write=write)
        edit = GainRecorder("diag_kdn", 256, 1, 0.1, 0., 2, 7, "gains")
        actual = chunked_recurrence(*tensors, 0.5, 16, edit)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertGreater(edit.count, 0)

    def test_random_control_is_orthogonal_and_independent_of_data(self):
        _, keys, _, _, _ = self.inputs()
        update = torch.eye(4).expand(2, 2, 4, 4).clone()
        expected = self.edit(rank=2, eta=0., directions="random")(keys, 16, update)
        actual = self.edit(rank=2, eta=0., directions="random")(keys * 0, 16, update)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        projector = (update - actual) / 0.9
        torch.testing.assert_close(projector @ projector, projector)
        torch.testing.assert_close(projector.mT, projector)
        torch.testing.assert_close(projector.diagonal(dim1=-2, dim2=-1).sum(-1), torch.full((2, 2), 2.))
        one = self.edit(rank=2, eta=0., directions="random")(keys[:1], 16, update[:1])
        torch.testing.assert_close(one, expected[:1], rtol=0, atol=0)

    def test_spread_only_has_matched_schedule_and_no_shrink(self):
        tensors = self.inputs()
        edit = self.edit(rank=0, gamma=1., eta=0.5, directions="none")
        keys, update = tensors[1], torch.diag(torch.tensor([0.1, 0.2, 1., 2.])).expand(2, 2, 4, 4)
        actual = edit(keys, 16, update)
        torch.testing.assert_close(actual, 0.5 * update + 0.5 * newton_schulz5(update))
        self.assertIsNone(json_safe(edit.diagnostics())["captured"])
        fresh = self.edit(rank=0, gamma=1., eta=0.5, directions="none")
        chunked_recurrence(*tensors, 0.5, 16, fresh)
        self.assertEqual(fresh.first_effect_token, 32)

    def test_sweep_contains_all_requested_controls_once(self):
        args = SimpleNamespace(windows=[256], ranks=[1, 2], gammas=[0.1, 0.3], etas=[0., 0.5])
        for kind, expected_count in (("iso_kdn", 18), ("diag_kdn", 26)):
            grid = settings_grid(args, kind)
            self.assertEqual(len(grid), expected_count)
            self.assertEqual(len({s["name"] for s in grid}), len(grid))
            self.assertEqual(sum(s["directions"] == "gains" for s in grid), 8 if kind == "diag_kdn" else 0)
            self.assertEqual(sum(s["directions"] == "random" for s in grid), 8)
            spread = [s for s in grid if s["name"] == "spread-only"]
            self.assertEqual(len(spread), 1)
            self.assertEqual((spread[0]["gamma"], spread[0]["eta"]), (1., 0.5))

    def test_pair_recall_uses_original_context_order_and_ignores_fillers(self):
        inputs = torch.tensor([[2, 10, 4, 11, 3, 12, 3, 2, 4, 2],
                               [4, 11, 2, 10, 3, 12, 2, 4, 3, 4]])
        labels = torch.full_like(inputs, -100)
        labels[:, 6:9] = torch.tensor([[12, 10, 11], [10, 11, 12]])
        predictions = labels.clone()
        predictions[0, 7] = 999  # First context pair, but second query.
        positions = query_pair_positions(inputs, labels, 3)
        torch.testing.assert_close(positions[:, 6:9], torch.tensor([[2, 0, 1], [1, 0, 2]]))
        self.assertTrue((positions[:, 9] == -1).all())
        recall = Recall(3)
        for i in range(2):
            recall.update(predictions[i:i+1], labels[i:i+1], positions[i:i+1])
        report = recall.report()
        self.assertEqual(report["accuracy"], 5 / 6)
        self.assertEqual([p["accuracy"] for p in report["recall_by_pair_position"]], [0.5, 1., 1.])
        self.assertEqual([p["total"] for p in report["recall_by_pair_position"]], [2, 2, 2])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.csv"
            result = {"slices": [dict(seq_len=10, num_kv_pairs=3, settings={"kernel": report})]}
            write_pair_recall_csv(result, path)
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["pair_position"], "1")
            self.assertEqual(rows[0]["accuracy"], "0.5")
            self.assertEqual(rows[2]["key_token_index"], "4")
        json.dumps(report, allow_nan=False)

    def test_pair_mapping_rejects_missing_query_keys(self):
        with self.assertRaises(ValueError):
            query_pair_positions(torch.tensor([[2, 10, 7, 0]]), torch.tensor([[-100, -100, 10, -100]]), 1)

    def test_checkpoint_selection_and_cpu_plan(self):
        config = {"learning_rate": 0.01, "model": {"sequence_mixer": {"name": "hybrid", "kwargs": {"configs": [
            {"name": "zoology_mixers.DiagKDNMixer", "kwargs": {"num_heads": 8, "head_dim": 64}}]}}}}
        checkpoint = {"config": config}
        spec = checkpoint_spec(checkpoint)
        target = SimpleNamespace(heads=8, head_dim=64, lr=1e-2)
        self.assertTrue(matches_target(spec, target))
        self.assertFalse(matches_target(dict(spec, learning_rate=4e-4), target))
        self.assertFalse(matches_target(dict(spec, head_dim=128), target))
        with tempfile.TemporaryDirectory() as directory:
            # Deliberately misleading filename: selection must use the config.
            path = Path(directory) / "h1-d128-lr4e-4.pt"
            torch.save(checkpoint, path)
            run = subprocess.run([sys.executable, str(Path(__file__).with_name("shrink_spread.py")),
                                  directory, "--plan-only"], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("27 settings including kernel/replay", run.stdout)


if __name__ == "__main__":
    unittest.main()
