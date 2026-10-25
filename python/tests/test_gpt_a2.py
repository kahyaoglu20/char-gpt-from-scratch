"""
test_gpt_a2.py -- unit tests of the complete GPT (standard library `unittest`, no extra packages).

MAT 451E, Assignment 2 (character-level GPT: training and generation)
Author: Ömer Kahyaoğlu <kahyaoglu20@itu.edu.tr>, student no. 090240741

    python -m unittest discover -s python/tests -v          (from the submission root, a few seconds)

`run_assignment2.py` is the evidence for the report; these tests are the regression net underneath it.
Each one is small, deterministic (seed 451) and names the property it protects.
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpt_a1.autograd import Tensor, parameter                     # noqa: E402
from gpt_a1.data import build_vocab, encode, split_contiguous     # noqa: E402
from gpt_a1.params import count_params, init_params               # noqa: E402
from gpt_a2 import model as M                                     # noqa: E402
from gpt_a2 import ops                                            # noqa: E402
from gpt_a2.optim import Adam                                     # noqa: E402
from gpt_a2.sample import generate, next_distribution, sample_index  # noqa: E402
from gpt_a2.train import schedule, train                          # noqa: E402

SEED = 451


def leaf(a):
    return parameter(np.asarray(a, dtype=np.float64))


def const(a):
    return Tensor(np.asarray(a, dtype=np.float64))


def small_model(V=7, T=5, d=6, L=2, seed=SEED):
    rng = np.random.default_rng(seed)
    P = init_params(V, T, d, L, 0.3, rng=rng)
    for k, p in P.items():
        if "gamma" in k or "beta" in k or k.split(".")[-1].startswith("b_"):
            p.data[...] = p.data + 0.1 * rng.standard_normal(p.data.shape)
    return P, rng


class GeluTests(unittest.TestCase):
    def test_values_against_the_exact_gelu(self):
        z = np.array([-1.0, 0.0, 1.0])
        self.assertLess(np.max(np.abs(ops.gelu_np(z) - ops.gelu_exact_np(z))), 2e-4)
        self.assertEqual(ops.gelu_np(np.array([0.0]))[0], 0.0)

    def test_derivative_formula_matches_the_engine(self):
        z = leaf(np.linspace(-5, 5, 101))
        ops.gelu(z).sum().backward()
        np.testing.assert_allclose(z.grad, ops.gelu_grad_np(z.data), atol=1e-14)


class CrossEntropyTests(unittest.TestCase):
    def test_two_by_three_example(self):
        Z = leaf([[2.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        L = M.cross_entropy(Z, [0, 1])
        ref = 0.5 * (math.log(math.e ** 2 + 1 + math.e ** -1) - 2 + math.log(2 + math.e) - 1)
        self.assertAlmostEqual(float(L.data), ref, places=14)
        L.backward()
        np.testing.assert_allclose(Z.grad.sum(axis=1), 0.0, atol=1e-16)       # every gradient row sums to zero

    def test_uniform_logits_give_log_v_and_shifts_change_nothing(self):
        self.assertAlmostEqual(float(M.cross_entropy(const(np.zeros((4, 65))), [0, 1, 2, 3]).data), math.log(65), places=14)
        Z = np.random.default_rng(SEED).standard_normal((5, 9))
        a = float(M.cross_entropy(const(Z), [1, 2, 3, 4, 5]).data)
        b = float(M.cross_entropy(const(Z + 1000.0), [1, 2, 3, 4, 5]).data)
        self.assertAlmostEqual(a, b, places=11)

    def test_case_b1(self):
        Z = [[0, 0, 0], [math.log(2), 0, 0], [0, math.log(3), 0]]
        ref = (math.log(3) + math.log(2) + math.log(5 / 3)) / 3
        self.assertAlmostEqual(float(M.cross_entropy(const(Z), [0, 0, 1]).data), ref, places=14)


class ModelTests(unittest.TestCase):
    def test_hand_derived_gradients_match_autodiff_for_every_parameter(self):
        P, rng = small_model()
        X, Y = rng.integers(0, 7, (3, 5)), rng.integers(0, 7, (3, 5))
        L, Z, cache = M.loss_fn(X, Y, P)
        L.backward()
        g, _ = M.gpt_backward(M.cross_entropy_backward(Z.data, Y.reshape(-1)), cache, P)
        for k, p in P.items():
            np.testing.assert_allclose(p.grad, g[k], atol=1e-14, err_msg=k)

    def test_parameter_count_of_the_assignment_configuration(self):
        self.assertEqual(count_params(init_params(65, 64, 64, 2)), 128 * 65 + 103680)

    def test_future_tokens_do_not_change_earlier_logits(self):
        P, rng = small_model()
        Pc = M.as_constants(P)
        X = rng.integers(0, 7, (2, 5))
        X2 = X.copy()
        X2[:, -1] = (X2[:, -1] + 1) % 7
        Z1 = M.gpt_forward(X, Pc)[0].data.reshape(2, 5, 7)
        Z2 = M.gpt_forward(X2, Pc)[0].data.reshape(2, 5, 7)
        self.assertEqual(np.abs(Z1[:, :-1] - Z2[:, :-1]).max(), 0.0)
        self.assertGreater(np.abs(Z1[:, -1] - Z2[:, -1]).max(), 0.0)

    def test_residual_block_with_zero_branches_is_the_identity(self):
        H = np.eye(2)[None]
        D, _ = M.A1.layer_norm(const(H), const([[1, 1]]), const([[0, 0]]))
        G, _ = M.feed_forward(D, const(np.zeros((2, 8))), const(np.zeros((1, 8))), const(np.zeros((8, 2))), const(np.zeros((1, 2))))
        np.testing.assert_array_equal(H + G.data, H)


class AdamTests(unittest.TestCase):
    def test_scalar_exercise(self):
        th = np.array([0.0])
        Adam({"x": th}, 1e-3).step({"x": th}, {"x": np.array([0.2])})
        self.assertAlmostEqual(th[0], -1e-3 * 0.2 / (0.2 + 1e-8), places=17)

    def test_first_step_is_minus_lr_times_sign(self):
        th = np.zeros((2, 3))
        g = np.array([[1 / 6, -1 / 3, 1 / 6], [1 / 6, 1 / 6, -1 / 3]])
        Adam({"W": th}, 1e-3).step({"W": th}, {"W": g})
        np.testing.assert_allclose(th, -1e-3 * np.sign(g), atol=1e-10)


class TrainingAndSamplingTests(unittest.TestCase):
    def test_schedule(self):
        self.assertEqual(schedule(1003854, 32, 64), (490, 24500))
        self.assertEqual(schedule(1003854, 32, 64, s_max=100), (490, 100))

    def test_the_model_learns_an_alternating_sequence(self):
        itos, stoi = build_vocab("AB" * 600)
        ids = encode("AB" * 600, stoi)
        tr, va, _ = split_contiguous(ids)
        P = init_params(2, 16, 16, 1, 0.02, rng=np.random.default_rng(SEED))
        cfg = {"B": 8, "T": 16, "L": 1, "lr": 3e-3, "beta1": 0.9, "beta2": 0.999, "eps_adam": 1e-8,
               "epochs": 100, "s_max": 120, "eval_every": 40}
        rec = train(P, tr, va, cfg, np.random.default_rng(1), np.random.default_rng(2), log=lambda *a: None)
        self.assertLess(rec["train_curve"][-1], 0.05 * rec["train_curve"][0])
        self.assertTrue(rec["val_only_heldout_and_no_update"])
        out = generate(P, "A", stoi, itos, 20, 16, None, L=1, greedy=True)
        self.assertEqual(out, "AB" * 10 + "A")

    def test_context_is_cropped_to_the_last_t_tokens(self):
        P, rng = small_model(T=5)
        Pc = M.as_constants(P)
        ctx = rng.integers(0, 7, 9)
        p = next_distribution(ctx, Pc, 5)
        Z, _ = M.gpt_forward(ctx[-5:][None], Pc)
        np.testing.assert_array_equal(p, M.softmax_np(Z.data[-1][None])[0])

    def test_categorical_sampler_frequencies(self):
        p = np.array([0.665, 0.245, 0.09])
        rng = np.random.default_rng(SEED)
        f = np.bincount([sample_index(p, rng) for _ in range(20000)], minlength=3) / 20000
        self.assertLess(np.abs(f - p).max(), 0.015)


if __name__ == "__main__":
    unittest.main(verbosity=2)
