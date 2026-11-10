"""CPU checks for causal boundary schedules, read-only probes and paired reporting."""
import csv
import io
import json
import math
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from boundary_diagnostics import BoundaryProbes, paired_sequence_delta, summarize_positions, write_diagnostic_csvs
from shrink_spread import ChunkEdit, Recall, chunk_ranges, chunked_recurrence, run_checkpoint, settings_grid
from state_geometry import json_safe


class BoundaryDiagnosticTests(unittest.TestCase):
    def tensors(self, length=97):
        rng = torch.Generator().manual_seed(17)
        q, k = [torch.nn.functional.normalize(torch.randn(3, length, 2, 4, generator=rng), dim=-1)
                for _ in range(2)]
        v = torch.randn(3, length, 2, 4, generator=rng)
        return q, k, v, torch.full_like(k, 0.95), 0.2 * k

    def edit(self, gamma=0.1):
        return ChunkEdit("iso_kdn", 256, 1, gamma, 0., 2, 1234)

    def test_shifted_schedule_is_identity_without_edits_and_causal_with_edits(self):
        tensors = self.tensors()
        baseline = chunked_recurrence(*tensors, 0.5, 16)
        for chunk, offset, first in ((16, 0, 32), (16, 2, 34), (32, 0, 64)):
            plain = chunked_recurrence(*tensors, 0.5, chunk, offset=offset)
            identity = chunked_recurrence(*tensors, 0.5, chunk, self.edit(1.), offset)
            torch.testing.assert_close(plain, baseline, rtol=0, atol=0)
            torch.testing.assert_close(identity, baseline, rtol=0, atol=0)
            edit = self.edit()
            actual = chunked_recurrence(*tensors, 0.5, chunk, edit, offset)
            self.assertEqual(edit.first_effect_token, first)
            torch.testing.assert_close(actual[:, :first], baseline[:, :first], rtol=0, atol=0)
            self.assertGreater((actual[:, first:] - baseline[:, first:]).abs().max().item(), 0)
            changed = [t.clone() for t in tensors]
            changed[1][:, first:] *= -1
            changed[2][:, first:] += 10
            future = chunked_recurrence(*changed, 0.5, chunk, self.edit(), offset)
            torch.testing.assert_close(future[:, :first], actual[:, :first], rtol=0, atol=0)

    def test_ranges_cover_partial_tail_and_reject_invalid_phase(self):
        self.assertEqual(list(chunk_ranges(39, 16, 2)), [(0, 18), (18, 34), (34, 39)])
        self.assertEqual(list(chunk_ranges(12, 16, 2)), [(0, 12)])
        for chunk, offset in ((0, 0), (16, -1), (16, 16)):
            with self.assertRaises(ValueError):
                list(chunk_ranges(64, chunk, offset))

    def test_probe_observes_actual_state_edit_without_altering_outputs(self):
        # State at token 1 is 4. Removing its +2 write leaves 2. Probe error
        # changes 0 -> (2-4)^2=4 before the next residual update can correct it.
        q = k = alpha = torch.ones(1, 3, 1, 1)
        v = torch.tensor([2., 4., 0.]).view_as(q)
        write = torch.tensor([1., 1., 0.5]).view_as(q)
        probes = BoundaryProbes(1)
        def observe(start, end, before, after, basis):
            probes.observe(0, start, end, k, v, write, before, after, basis)
        expected = chunked_recurrence(q, k, v, alpha, write, 1., 1, self.edit(0.))
        actual = chunked_recurrence(q, k, v, alpha, write, 1., 1, self.edit(0.), probe=observe)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        row, = probes.report()
        self.assertEqual(row["squared_error_before"], 0.)
        self.assertEqual(row["squared_error_after"], 4.)
        self.assertEqual(row["relative_squared_error_after"], 0.25)
        self.assertEqual(row["key_projection_share"], 1.)
        self.assertEqual(row["gain_projection_share"], 1.)
        self.assertTrue(row["terminal_pair"])

    def test_probe_preserves_layer_pair_offsets_and_pools_unequal_batches(self):
        q, k, v, alpha, write = self.tensors(64)
        before = torch.eye(4).expand(3, 2, 4, 4).clone()
        after = before * 0.5
        basis = before[..., :1]
        whole, split = BoundaryProbes(32), BoundaryProbes(32)
        whole.observe(1, 18, 34, k, v, write, before, after, basis)
        for a, b in ((0, 2), (2, 3)):
            split.observe(1, 18, 34, k[a:b], v[a:b], write[a:b], before[a:b], after[a:b], basis[a:b])
        for expected, actual in zip(whole.report(), split.report()):
            for key in expected:
                self.assertAlmostEqual(expected[key], actual[key], places=5)
        rows = whole.report()
        self.assertEqual([r["pair_position"] for r in rows], list(range(10, 18)))
        self.assertEqual(rows[-1]["tokens_before_boundary"], 0)
        self.assertEqual(rows[0]["tokens_before_boundary"], 14)
        self.assertEqual(rows[0]["example_heads"], 6)
        self.assertEqual(rows[0]["layer"], 1)
        # A chunk crossing into queries is excluded, even if it has odd tokens.
        whole.observe(1, 60, 64, k, v, write, before, after, None)
        empty = BoundaryProbes(16)
        empty.observe(1, 18, 34, k, v, write, before, after, None)
        self.assertEqual(empty.report(), [])

    def test_zero_target_and_absent_basis_export_as_null(self):
        vectors = torch.zeros(1, 4, 1, 2)
        state = torch.zeros(1, 1, 2, 2)
        probes = BoundaryProbes(2)
        probes.observe(0, 2, 4, vectors, vectors, vectors, state, state, None)
        row, = probes.report()
        self.assertIsNone(row["relative_squared_error_before"])
        self.assertIsNone(row["key_projection_share"])
        json.dumps(probes.report(), allow_nan=False)

    def test_terminal_groups_follow_shift_and_exclude_warmup(self):
        recall = Recall(128)
        recall.total[:] = 1
        recall.correct[:] = 1
        recall.correct[16] = 0  # Pair 17 belongs only to the shifted terminal set.
        report = recall.report()
        ordinary = summarize_positions(report, 512, 128, 16, 0, chunk_ranges)
        shifted = summarize_positions(report, 512, 128, 16, 2, chunk_ranges)
        larger = summarize_positions(report, 512, 128, 32, 0, chunk_ranges)
        self.assertEqual(ordinary["terminal_pair_positions"], list(range(16, 129, 8)))
        self.assertEqual(shifted["terminal_pair_positions"], list(range(17, 122, 8)))
        self.assertEqual(larger["terminal_pair_positions"], list(range(32, 129, 16)))
        self.assertEqual(ordinary["terminal"]["accuracy"], 1.)
        self.assertEqual(shifted["terminal"]["correct"], shifted["terminal"]["total"] - 1)
        self.assertEqual(sum(q["total"] for q in shifted["quarters"]), 128)

    def test_per_example_counts_and_sequence_uncertainty_preserve_pairing(self):
        labels = torch.tensor([[10, 11, -100], [10, 11, -100]])
        positions = torch.tensor([[0, 1, -1], [0, 1, -1]])
        replay, edited = Recall(2, True), Recall(2, True)
        replay.update(labels, labels, positions)
        predictions = labels.clone()
        predictions[0, 0] = 999
        for i in range(2):
            edited.update(predictions[i:i+1], labels[i:i+1], positions[i:i+1])
        report = edited.report()
        self.assertEqual(report["per_example_correct_by_pair"], [[0, 1], [1, 1]])
        self.assertEqual(report["per_example_total_by_pair"], [[1, 1], [1, 1]])
        delta = paired_sequence_delta(report, replay.report())
        self.assertEqual(delta["mean_accuracy_delta"], -0.25)
        self.assertEqual(delta["standard_error"], 0.25)
        bad = replay.report()
        bad["per_example_total_by_pair"] = [[2, 0], [1, 1]]
        with self.assertRaises(ValueError):
            paired_sequence_delta(report, bad)

    def test_focused_presets_are_small_and_include_matched_controls(self):
        boundary = settings_grid(SimpleNamespace(preset="boundary"), "iso_kdn")
        self.assertEqual(len(boundary) + 1, 8)
        self.assertEqual(len({s["name"] for s in boundary}), len(boundary))
        for chunk, offset in ((16, 0), (16, 2), (32, 0)):
            selected = [s for s in boundary if s.get("chunk") == chunk and s.get("chunk_offset") == offset]
            self.assertEqual({s["directions"] for s in selected}, {"keys", "random"})
        gentle = settings_grid(SimpleNamespace(preset="gentle"), "diag_kdn")
        self.assertEqual(len(gentle) + 1, 10)
        self.assertEqual({s["eta"] for s in gentle}, {0., 0.05, 0.1})
        self.assertEqual(sum(s["directions"] == "none" for s in gentle), 3)  # Replay plus two spread controls.

    def test_checkpoint_loop_exports_shifted_probes_with_stable_layer_ids(self):
        # Exercise orchestration with real CPU recurrence/probes and synthetic
        # model/data adapters; CUDA kernels and actual checkpoints are HPC checks.
        q, k, v, alpha, write = self.tensors(64)
        inputs = torch.zeros(3, 64, dtype=torch.long)
        inputs[:, :32:2] = torch.arange(1, 17)
        inputs[:, 32:48] = torch.arange(1, 17)
        labels = torch.full_like(inputs, -100)
        labels[:, 32:48] = torch.arange(16) % 4
        active = dict(run=None, calls=None)

        class Model:
            frozen = False
            def requires_grad_(self, flag):
                self.frozen = not flag
            def __call__(self, x):
                self_assert.assertFalse(torch.is_grad_enabled())
                for layer in range(2):
                    call = dict(kind="iso_kdn", q=q[:len(x)], k=k[:len(x)], v=v[:len(x)],
                                write=write[:len(x)], scale=0.5,
                                g_cumsum=alpha[:len(x)].log2().cumsum(1))
                    output = (active["run"](call) if active["run"] else
                              chunked_recurrence(call["q"], call["k"], call["v"], alpha[:len(x)],
                                                 call["write"], 0.5, 16))
                    if active["calls"] is not None:
                        active["calls"].append(dict(call, out=output))
                return output.mean(2)

        @contextmanager
        def capture(calls):
            active["calls"] = calls
            try:
                yield
            finally:
                active["calls"] = None

        @contextmanager
        def replace(run):
            active["run"] = run
            try:
                yield
            finally:
                active["run"] = None

        self_assert = self
        model = Model()
        data = SimpleNamespace(inputs=inputs, labels=labels)
        module = SimpleNamespace(MQARConfig=lambda **kwargs: SimpleNamespace(build=lambda seed: data))
        checkpoint = dict(epoch=1, config=dict(learning_rate=.01, model=dict(vocab_size=64,
            sequence_mixer=dict(name="zoology_mixers.IsoKDNMixer", kwargs=dict(num_heads=8, head_dim=64)))))
        args = SimpleNamespace(preset="boundary", chunk=16, chunk_offset=0, power_steps=2,
                               seed=1234, examples=3, batch_size=2)
        with patch.dict("sys.modules", {"zoology.data.multiquery_ar": module}), \
             patch("shrink_spread.build_model", return_value=model), \
             patch("shrink_spread.capture_kdn_calls", capture), \
             patch("shrink_spread.replace_kdn", replace), \
             patch("shrink_spread.choose_gate", return_value=(2., None, 0.)), \
             patch.object(torch.Tensor, "cuda", lambda x: x), redirect_stdout(io.StringIO()):
            result = run_checkpoint(Path("fake.pt"), [(64, 16)], settings_grid(args, "iso_kdn"), args, checkpoint)
        self.assertTrue(model.frozen)
        settings = result["slices"][0]["settings"]
        ordinary = settings["keys-w256-r1-g0.1-e0-c16-o0"]
        shifted = settings["keys-w256-r1-g0.1-e0-c16-o2"]
        self.assertEqual({r["layer"] for r in ordinary["boundary_probes"]}, {0, 1})
        self.assertEqual({r["example_heads"] for r in ordinary["boundary_probes"]}, {6})
        self.assertEqual(shifted["boundary_probes"], [])  # First edit is after this context ends.
        self.assertEqual(ordinary["position_summary"]["terminal"]["total"], 3)
        self.assertEqual(shifted["replay_position_summary"]["terminal"]["total"], 0)
        self.assertEqual(len(ordinary["per_example_correct_by_pair"]), 3)
        json.dumps(json_safe(result), allow_nan=False)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_diagnostic_csvs(result, Path(directory) / "fake")
            self.assertEqual(len(paths), 2)
            with paths[0].open() as stream:
                summary = list(csv.DictReader(stream))
            self.assertEqual(len(summary), 7)
            self.assertEqual(summary[0]["setting"], "replay")


if __name__ == "__main__":
    unittest.main()
