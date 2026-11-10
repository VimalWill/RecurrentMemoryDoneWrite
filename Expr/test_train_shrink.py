"""CPU autograd/causality checks and an optional CUDA fused-path parity check."""
import copy
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "RecurrentMemory" / "src"))
from kdn_shrink import iso_shrink_forward, past_key_basis, shrink_recurrence
from train_mqar_shrink import experiment_plan, parse_args


class TrainShrinkTests(unittest.TestCase):
    def inputs(self, length=13, dtype=torch.float64):
        gen = torch.Generator().manual_seed(8)
        q, k = [torch.nn.functional.normalize(torch.randn(1, length, 2, 3,
                 generator=gen, dtype=dtype), dim=-1) for _ in range(2)]
        v = torch.randn(1, length, 2, 2, generator=gen, dtype=dtype)
        alpha = torch.rand(q.shape, generator=gen, dtype=dtype) * .2 + .7
        beta = torch.rand(q.shape[:-1], generator=gen, dtype=dtype) * .3
        return q, k, v, alpha, beta

    def test_control_matches_independent_recurrence_and_gradients(self):
        inputs = [x.requires_grad_() for x in self.inputs()]
        q, k, v, alpha, beta = inputs
        state = q.new_zeros(1, 2, 3, 2)
        outputs = []
        for t in range(q.shape[1]):
            state = state * alpha[:, t, :, :, None]
            delta = v[:, t] - torch.einsum('bhk,bhkv->bhv', k[:, t], state)
            state = state + torch.einsum('bhk,bhv->bhkv', k[:, t] * beta[:, t, :, None], delta)
            outputs.append(torch.einsum('bhk,bhkv->bhv', q[:, t], state) / 3**.5)
        reference = torch.stack(outputs, 1)
        actual = shrink_recurrence(*inputs, gamma=1., chunk=4)
        torch.testing.assert_close(actual, reference)
        expected_grad = torch.autograd.grad(reference.square().sum(), inputs, retain_graph=True)
        actual_grad = torch.autograd.grad(actual.square().sum(), inputs)
        for actual, expected in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual, expected)

    def test_shrink_gradcheck_across_boundary(self):
        q, k, v, alpha, beta = self.inputs(length=9)
        # The axis is intentionally stop-gradient, so finite differences in k
        # would test a different derivative. Check all state-update inputs.
        self.assertTrue(torch.autograd.gradcheck(
            lambda v, alpha, beta: shrink_recurrence(q, k, v, alpha, beta, gamma=.3, chunk=3),
            tuple(x.requires_grad_() for x in (v, alpha, beta))))

    def test_basis_detached_and_null_keys_no_edit(self):
        q, k, v, alpha, beta = self.inputs()
        self.assertFalse(past_key_basis(k.requires_grad_(), 1).requires_grad)
        self.assertEqual(past_key_basis(torch.zeros_like(k), 1).abs().sum().item(), 0.)

    def test_causality_and_first_boundary(self):
        inputs = list(self.inputs())
        original = shrink_recurrence(*inputs, gamma=.3, chunk=4)
        control = shrink_recurrence(*inputs, gamma=1., chunk=4)
        torch.testing.assert_close(original[:, :8], control[:, :8], rtol=0, atol=0)
        self.assertGreater((original[:, 8:] - control[:, 8:]).abs().max().item(), 1e-6)
        for index in (0, 1, 2, 3, 4):
            changed = [x.clone() for x in inputs]
            changed[index][:, 9:] *= 2
            result = shrink_recurrence(*changed, gamma=.3, chunk=4)
            torch.testing.assert_close(result[:, :9], original[:, :9], rtol=0, atol=0)

    def test_full_gate_path_has_finite_nonzero_gradients(self):
        torch.manual_seed(9)
        shapes = [(1, 13, 2, 3)] * 4 + [(1, 13, 4), (2,), (2,), (2,), (6,), (2,)]
        values = [torch.randn(shape, requires_grad=True) for shape in shapes]
        output = iso_shrink_forward(*values, omega_min=0., r_min=.01,
                                   info_scale=3., gamma=.9, chunk=4, window=8, rank=1)
        output.square().sum().backward()
        for value in values:
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(), 0.)

    def test_plan_pairs_seeds_and_includes_length_extrapolation(self):
        args = parse_args(["--seeds", "1234", "1235", "--plan-only"])
        plan = experiment_plan(args)
        self.assertEqual([(r['seed'], r['gamma']) for r in plan['runs']],
                         [(1234, 1.), (1234, .9), (1235, 1.), (1235, .9)])
        self.assertEqual(plan['layers'], 2)
        self.assertEqual(plan['d_model'], 8 * 64)
        self.assertEqual(max(length for length, _ in plan['train_slices']), 256)
        self.assertIn((1024, 256), plan['test_slices'])


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA + FLA environment")
class CUDAShrinkParityTests(unittest.TestCase):
    def test_control_forward_and_backward_against_fused_layer(self):
        from LinearAttn.iso_kdn import IsotropicKalmanDeltaNetwork
        torch.manual_seed(10)
        fused = IsotropicKalmanDeltaNetwork(hidden_size=64, head_dim=64, num_heads=1).cuda().train()
        replay = copy.deepcopy(fused)
        replay.shrink_gamma = 1.
        x = torch.randn(2, 65, 64, device='cuda') * .2
        with torch.autocast('cuda', dtype=torch.bfloat16):
            expected = fused(x)[0]
            actual = replay(x)[0]
        torch.testing.assert_close(actual.float(), expected.float(), rtol=.08, atol=.01)
        expected.float().square().mean().backward()
        actual.float().square().mean().backward()
        for (name, p), (_, other) in zip(fused.named_parameters(), replay.named_parameters()):
            if p.grad is None:
                self.assertIsNone(other.grad, name)
                continue
            self.assertIsNotNone(other.grad, name)
            self.assertTrue(torch.isfinite(other.grad).all(), name)
            relative = (p.grad.float()-other.grad.float()).norm() / p.grad.float().norm().clamp_min(1e-8)
            self.assertLess(relative.item(), .15, name)


if __name__ == '__main__':
    unittest.main()
