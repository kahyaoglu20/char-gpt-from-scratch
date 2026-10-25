"""
test_gpt_a1.py -- unit tests of the foundational forward model (standard library `unittest`, no extra packages).

MAT 451E, Assignment 1 (character-level GPT, foundations)
Author: Ömer Kahyaoğlu <kahyaoglu20@itu.edu.tr>, student no. 090240741

    python -m unittest discover -s python/tests -v          (from the submission root, about 2 s)

`run_assignment1.py` is the *evidence* for the report: it records 82 measured checks.  This file is the
*regression net* for the library underneath it.  Every test is small, deterministic (seed 451) and states
the mathematical property it protects, so a future edit that breaks an identity fails here in milliseconds,
before anybody regenerates the report.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))        # make `gpt_a1` importable from anywhere

from gpt_a1 import model as M                                        # noqa: E402
from gpt_a1.autograd import Tensor, parameter                        # noqa: E402
from gpt_a1.checks import central_difference, rel_err                # noqa: E402
from gpt_a1.data import (build_vocab, decode, encode, get_batch,     # noqa: E402
                         num_starts, split_contiguous)
from gpt_a1.params import count_formula, count_params, init_params   # noqa: E402

SEED = 451


def const(a):
    return Tensor(np.asarray(a, dtype=np.float64))


def leaf(a):
    return parameter(np.asarray(a, dtype=np.float64))


def objective(F, R):
    """Scalar test objective J = sum(R * F), so that dJ/dF = R exactly."""
    return (F * R).sum()


class DataTests(unittest.TestCase):
    """Q1-Q3: vocabulary bijection, contiguous split, shifted windows."""

    def test_vocabulary_is_a_sorted_bijection(self):
        text = "hello, world\n"
        itos, stoi = build_vocab(text)
        self.assertEqual(itos, sorted(set(text)))
        self.assertEqual(decode(encode(text, stoi), itos), text)
        self.assertEqual(sorted(stoi.values()), list(range(len(itos))))

    def test_split_partitions_the_sequence_in_order(self):
        for N in (10, 11, 99, 1000, 1115394):
            ids = np.arange(N)
            train, val, n_tr = split_contiguous(ids)
            self.assertEqual(n_tr, (9 * N) // 10)                    # floor(0.9 N), exact integer arithmetic
            self.assertEqual(len(train) + len(val), N)
            np.testing.assert_array_equal(np.concatenate([train, val]), ids)

    def test_number_of_legal_starts_matches_brute_force(self):
        for n in range(1, 30):
            for T in range(1, 8):
                brute = sum(1 for r in range(n) if r + T <= n - 1)   # last target index r+T must exist
                self.assertEqual(num_starts(n, T), brute)

    def test_windows_are_shifted_by_one_and_stay_inside_the_split(self):
        ids = np.arange(500)
        rng = np.random.default_rng(SEED)
        X, Y, r = get_batch(ids, 32, 16, rng)
        self.assertEqual(X.shape, (32, 16))
        np.testing.assert_array_equal(Y, X + 1)                      # on arange: target = next token
        self.assertTrue(np.all(Y.max(axis=1) <= 499) and np.all(r >= 0))

    def test_split_shorter_than_context_is_refused(self):
        with self.assertRaises(ValueError):
            get_batch(np.arange(8), 2, 8, np.random.default_rng(SEED))


class ParameterTests(unittest.TestCase):
    """Q4: the parameter count, the initialization and the absence of missing entries."""

    def test_count_matches_the_assignment_formula(self):
        V, T, d, L = 65, 64, 64, 2
        P = init_params(V, T, d, L)
        self.assertEqual(count_params(P), count_formula(V, T, d, L))
        self.assertEqual(count_params(P), 128 * V + 103680)

    def test_initial_values(self):
        P = init_params(7, 5, 4, 2, sigma=0.02, rng=np.random.default_rng(SEED))
        for name, p in P.items():
            self.assertTrue(np.all(np.isfinite(p.data)), name)
            if "gamma" in name:
                self.assertTrue(np.all(p.data == 1.0), name)
            elif "beta" in name or name.split(".")[-1].startswith("b_"):
                self.assertTrue(np.all(p.data == 0.0), name)
        W = np.concatenate([p.data.ravel() for n, p in P.items() if "gamma" not in n and "beta" not in n
                            and not n.split(".")[-1].startswith("b_")])
        self.assertLess(abs(W.mean()), 5 * 0.02 / np.sqrt(W.size))   # within five standard errors
        self.assertAlmostEqual(W.std(), 0.02, delta=0.02 * 0.1)


class AutogradTests(unittest.TestCase):
    """The reverse-mode engine against centred finite differences."""

    def test_broadcast_and_reuse_accumulate(self):
        a = leaf([[1.0, 2.0, 3.0]])
        b = leaf([[2.0], [3.0]])
        J = ((a * b) + a).sum()                                       # `a` is used twice and broadcast over rows
        J.backward()
        np.testing.assert_allclose(a.grad, (b.data.sum() + 2) * np.ones((1, 3)))
        np.testing.assert_allclose(b.grad, a.data.sum() * np.ones((2, 1)))

    def test_matmul_exp_sum_chain_against_finite_differences(self):
        rng = np.random.default_rng(SEED)
        A, B = rng.standard_normal((3, 4)), rng.standard_normal((4, 2))
        a, b = leaf(A), leaf(B)
        (((a @ b) * 0.3).exp()).sum().backward()

        def J():
            return float(np.exp(0.3 * (A @ B)).sum())
        for theta, grad in ((A, a.grad), (B, b.grad)):
            E = rng.standard_normal(theta.shape)
            fd = central_difference(J, theta, E, 1e-5)
            self.assertLess(rel_err(fd, float(np.sum(grad * E))), 1e-7)

    def test_getitem_scatter_adds_repeated_indices(self):
        w = leaf(np.arange(6.0).reshape(3, 2))
        w[np.array([0, 2, 0])].sum().backward()
        np.testing.assert_array_equal(w.grad, [[2, 2], [0, 0], [1, 1]])

    def test_detach_blocks_the_gradient(self):
        w = leaf([[1.0, 2.0]])
        (w.detach() * 3.0 + leaf([[0.0, 0.0]])).sum().backward()
        self.assertIsNone(w.grad)


class EmbeddingTests(unittest.TestCase):
    """Q5 and Case A1."""

    def test_case_a1_matrices(self):
        W_E = [[0.12, -0.07], [-0.03, 0.11], [0.08, 0.04], [-0.09, 0.02]]
        W_P = [[0.01, 0.02], [-0.02, 0.03], [0.04, -0.01]]
        H, _ = M.embed(np.array([[0, 2, 1]]), leaf(W_E), leaf(W_P))
        np.testing.assert_allclose(H.data[0], [[0.13, -0.05], [0.06, 0.07], [0.01, 0.10]], atol=1e-15)

    def test_one_hot_product_equals_row_lookup(self):
        rng = np.random.default_rng(SEED)
        W_E, W_P = leaf(rng.standard_normal((9, 4))), leaf(rng.standard_normal((6, 4)))
        X = rng.integers(0, 9, size=(3, 6))
        H, _ = M.embed(X, W_E, W_P)
        np.testing.assert_array_equal(H.data, M.embed_lookup(X, W_E.data, W_P.data))

    def test_gradients_are_sums_over_the_batch(self):
        rng = np.random.default_rng(SEED)
        B, T, V, d = 4, 5, 7, 3
        X = rng.integers(0, V, size=(B, T))
        W_E, W_P = leaf(rng.standard_normal((V, d))), leaf(rng.standard_normal((T + 2, d)))
        D = rng.standard_normal((B, T, d))
        H, O = M.embed(X, W_E, W_P)
        objective(H, D).backward()
        G_WE, G_WP = M.embedding_backward(D, O, T + 2)
        np.testing.assert_allclose(W_E.grad, G_WE, atol=1e-13)
        np.testing.assert_allclose(W_P.grad, G_WP, atol=1e-13)
        np.testing.assert_array_equal(W_P.grad[T:], 0.0)             # unused positions get exactly zero


class LayerNormTests(unittest.TestCase):
    """Q6."""

    def test_worked_example(self):
        one, zero = const([[1.0, 1.0]]), const([[0.0, 0.0]])
        y, c = M.layer_norm(const([[1.0, 3.0]]), one, zero, eps=0.0)
        np.testing.assert_allclose(y.data, [[-1.0, 1.0]])
        self.assertEqual((c["mu"].data.item(), c["v"].data.item()), (2.0, 1.0))   # population variance
        ya, _ = M.layer_norm(const([[1.0, 3.0]]), const([[2.0, 0.5]]), const([[1.0, -1.0]]), eps=0.0)
        np.testing.assert_allclose(ya.data, [[-1.0, -0.5]])

    def test_rows_are_standardized_and_a_constant_row_is_finite(self):
        rng = np.random.default_rng(SEED)
        H = 3.0 + 2.0 * rng.standard_normal((4, 5, 16))
        y, _ = M.layer_norm(const(H), const(np.ones((1, 16))), const(np.zeros((1, 16))))
        np.testing.assert_allclose(y.data.mean(-1), 0.0, atol=1e-12)
        np.testing.assert_allclose(y.data.var(-1), 1.0, atol=1e-4)
        yc, _ = M.layer_norm(const(np.full((1, 4), 7.0)), const(np.ones((1, 4))), const([[0.5, -0.5, 0.25, 0.0]]))
        np.testing.assert_allclose(yc.data, [[0.5, -0.5, 0.25, 0.0]])             # eps keeps 0/0 finite

    def test_backward_formula_matches_autograd(self):
        rng = np.random.default_rng(SEED)
        B, T, d = 3, 4, 5
        H, g, b = leaf(rng.standard_normal((B, T, d))), leaf(1 + 0.3 * rng.standard_normal((1, d))), leaf(rng.standard_normal((1, d)))
        R = rng.standard_normal((B, T, d))
        Y, c = M.layer_norm(H, g, b)
        objective(Y, R).backward()
        G_X, G_g, G_b = M.layer_norm_backward(R, c["xhat"].data, c["s"].data, g.data)
        for auto, hand in ((H.grad, G_X), (g.grad, G_g), (b.grad, G_b)):
            np.testing.assert_allclose(auto, hand, atol=1e-12)


class AttentionTests(unittest.TestCase):
    """Q7, Benchmark A and the acceptance properties."""

    def test_benchmark_a_matches_the_displayed_matrices(self):
        a = (1 + 1e-5) ** -0.5
        d = 2
        W_E = leaf([[1, -1], [2, 0], [0, 2]])
        W_QKV = leaf(np.hstack([np.zeros((d, d)), np.zeros((d, d)), np.eye(d)]))
        F, c = M.forward_a1(np.array([[0, 1, 2]]), W_E, leaf(np.zeros((3, 2))), leaf([[1, 1]]), leaf([[0, 0]]),
                            W_QKV, leaf(np.eye(d)))
        np.testing.assert_allclose(c["A"].data[0], [[1, 0, 0], [.5, .5, 0], [1 / 3, 1 / 3, 1 / 3]], atol=1e-15)
        np.testing.assert_allclose(F.data[0], [[a, -a], [a, -a], [a / 3, -a / 3]], atol=1e-15)
        self.assertEqual(c["A"].data[0][np.triu_indices(3, 1)].max(), 0.0)         # forbidden weights are exactly 0

    def test_explicit_three_by_three_example(self):
        Q = const([[0.0, 2.0], [2.0, 1.0], [1.0, 0.0]])
        V = const([[0.0, 2.0], [2.0, 3.0], [1.0, 1.0]])
        _, A, Z = M.attend(Q, Q, V)
        np.testing.assert_allclose(Z.data[0], [0.0, 2.0])                           # the first query sees only itself
        np.testing.assert_allclose(A.data.sum(axis=1), 1.0, atol=1e-15)

    def test_future_tokens_do_not_influence_earlier_outputs(self):
        rng = np.random.default_rng(SEED)
        V, T, d = 11, 8, 6
        W_E, W_P = leaf(rng.standard_normal((V, d))), leaf(rng.standard_normal((T, d)))
        g, b = leaf(np.ones((1, d))), leaf(np.zeros((1, d)))
        W_QKV, W_O = leaf(rng.standard_normal((d, 3 * d)) / np.sqrt(d)), leaf(rng.standard_normal((d, d)) / np.sqrt(d))
        X = rng.integers(0, V, size=(2, T))
        F1, _ = M.forward_a1(X, W_E, W_P, g, b, W_QKV, W_O)
        X2 = X.copy()
        X2[:, -1] = (X2[:, -1] + 1) % V
        F2, _ = M.forward_a1(X2, W_E, W_P, g, b, W_QKV, W_O)
        self.assertEqual(np.abs(F2.data[:, :-1] - F1.data[:, :-1]).max(), 0.0)      # bit-for-bit unchanged
        self.assertGreater(np.abs(F2.data[:, -1] - F1.data[:, -1]).max(), 0.0)      # the perturbation was real

    def test_vectorized_attention_equals_the_per_sample_loop(self):
        rng = np.random.default_rng(SEED)
        U = rng.standard_normal((5, 9, 4))
        W_QKV, W_O = rng.standard_normal((4, 12)) / 2, rng.standard_normal((4, 4)) / 2
        F, _ = M.causal_self_attention(const(U), const(W_QKV), const(W_O))
        np.testing.assert_allclose(F.data, M.attention_per_sample(U, W_QKV, W_O), atol=1e-13)

    def test_hand_derived_backward_matches_autograd_term_by_term(self):
        rng = np.random.default_rng(SEED)
        B, T, d = 3, 5, 4
        U = leaf(rng.standard_normal((B, T, d)))
        W_QKV, W_O = leaf(rng.standard_normal((d, 3 * d)) / 2), leaf(rng.standard_normal((d, d)) / 2)
        R = rng.standard_normal((B, T, d))
        F, c = M.causal_self_attention(U, W_QKV, W_O)
        objective(F, R).backward()
        g = M.attention_backward(R, U.data, c["Q"].data, c["K"].data, c["V"].data, c["A"].data, c["Z"].data,
                                 W_QKV.data, W_O.data)
        auto = {"W_O": W_O.grad, "Z": c["Z"].grad, "A": c["A"].grad, "V": c["V"].grad, "S": c["S"].grad,
                "Q": c["Q"].grad, "K": c["K"].grad, "W_QKV": W_QKV.grad, "U": U.grad}
        for key, value in auto.items():
            np.testing.assert_allclose(value, g[key], atol=1e-12, err_msg=key)

    def test_edge_cases_stay_finite(self):
        rng = np.random.default_rng(SEED)
        for X, V, d in ((np.array([[2], [0], [1]]), 5, 4),           # one-token context
                        (np.array([[1, 3, 0, 2]]), 5, 4),            # batch of one
                        (np.zeros((1, 7), dtype=int), 3, 4),         # repeated character
                        (np.array([[0, 1, 1, 0]]), 2, 3)):           # tiny vocabulary
            T = X.shape[1]
            W_E, W_P = leaf(rng.standard_normal((V, d))), leaf(rng.standard_normal((T, d)))
            W_QKV, W_O = leaf(rng.standard_normal((d, 3 * d))), leaf(rng.standard_normal((d, d)))
            g, b = leaf(np.ones((1, d))), leaf(np.zeros((1, d)))
            F, _ = M.forward_a1(X, W_E, W_P, g, b, W_QKV, W_O)
            objective(F, rng.standard_normal(F.shape)).backward()
            self.assertTrue(np.all(np.isfinite(F.data)))
            for p in (W_E, W_P, g, b, W_QKV, W_O):
                self.assertTrue(np.all(np.isfinite(p.grad)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
