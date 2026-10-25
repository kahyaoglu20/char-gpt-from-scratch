#!/usr/bin/env python3
"""
run_assignment2.py -- MAT 451E Assignment 2 (character-level GPT: training and generation), Python.

Author: Ömer Kahyaoğlu <kahyaoglu20@itu.edu.tr>, student no. 090240741

Runs every question (Q1-Q6), Benchmark B, Cases B1-B5 and the minimum acceptance tests, trains the
model on Tiny Shakespeare for 50 epochs, generates text, and records every measured number.

    python python/run_assignment2.py                      everything (1-1.5 h on a CPU, almost all of it training)
    python python/run_assignment2.py --steps 500          a short training run (S_max = 500)
    python python/run_assignment2.py --reuse-training     re-run every check and figure on the saved model
    python -m unittest discover -s python/tests           the unit tests (about 2 s)

Organisation: utilities (log, check, show), then one function per question in the assignment's order
(q1_ffn ... q6_integration), the benchmark and cases (benchmark_b, case_b1 ... case_b5), the
training experiment (case_b4), generation (q5_generation), figures, and main.  Every verification
goes through check(id, ...), so each row of results/python/checks.csv has a stable identifier.

Outputs (relative paths only): results/python/{log.txt, results.json, checks.csv, training_log.csv,
training.json, samples.txt, model_final.npz, config.json, figures/*.png}.  Exit code 0 iff every check passes.
"""
import argparse
import csv
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))                       # also works under "python -I"

from gpt_a1 import model as A1                      # noqa: E402  (reused Assignment 1 code)
from gpt_a1.autograd import Tensor, parameter       # noqa: E402
from gpt_a1.checks import central_difference, max_abs, rel_err  # noqa: E402
from gpt_a1.data import (build_vocab, encode, get_batch, load_corpus, sha256_hex,  # noqa: E402
                         split_contiguous)
from gpt_a1.params import count_formula, count_params, init_params  # noqa: E402
from gpt_a2 import model as M                       # noqa: E402
from gpt_a2 import ops                              # noqa: E402
from gpt_a2.optim import Adam                       # noqa: E402
from gpt_a2.sample import generate, next_distribution, sample_index  # noqa: E402
from gpt_a2.train import evaluate_fixed, fixed_windows, schedule, train  # noqa: E402

DATA = ROOT / "data" / "tiny_shakespeare.txt"
A1DIR = ROOT / "data" / "assignment1"
OUT = ROOT / "results" / "python"
FIG = OUT / "figures"
SEED = 451
CFG = {"B": 32, "T": 64, "d": 64, "L": 2, "d_ff": 256, "sigma": 0.02, "eps": 1e-5, "mask_value": -1e9,
       "lr": 1e-3, "beta1": 0.9, "beta2": 0.999, "eps_adam": 1e-8, "epochs": 50, "s_max": 0, "eval_every": 50,
       "B_val": 8, "preview_every": 2450, "preview_len": 80, "sample_len": 200, "prompt": "ROMEO: ",
       "temperature": 1.0, "seed_init": "Assignment 1 params_init.npz (seed 451)", "seed_train_batches": SEED,
       "seed_val_batches": SEED + 1, "seed_previews": SEED + 2, "seed_final_sample": SEED + 3,
       "fixed_eval_windows": 512, "grad_clip": "none", "weight_decay": 0, "dtype": "float64", "index_base": 0}
H_VALUES = [1e-3, 1e-4, 1e-5, 1e-6]

LOG, RESULTS, CHECKS = [], {}, []


# ------------------------------------------------------------------------------- utilities
def log(*parts):
    line = " ".join(str(p) for p in parts)
    print(line, flush=True)
    LOG.append(line)


def section(title):
    log("")
    log("=" * 78)
    log(title)
    log("=" * 78)


def mat(A, fmt="{:10.6f}"):
    A = np.atleast_2d(np.asarray(A, dtype=np.float64))
    return "\n".join("   [" + " ".join(fmt.format(x) for x in row) + " ]" for row in A)


def show(name, A, fmt="{:10.6f}"):
    log(f"{name}  shape={tuple(np.shape(A))}")
    log(mat(A, fmt))


def check(cid, question, description, observed, threshold, expected="0", rule="<="):
    """Record one verification row: observed value against a pass threshold."""
    obs = float(observed)
    passed = {"<=": obs <= threshold, ">=": obs >= threshold, "==": obs == threshold}[rule]
    CHECKS.append({"id": cid, "question": question, "description": description, "expected": expected,
                   "observed": obs, "threshold": threshold, "rule": rule, "passed": bool(passed)})
    log(f"[{'PASS' if passed else 'FAIL'}] {cid}: {description} = {obs:.3e} (expected {expected}, pass if {rule} {threshold:g})")
    return passed


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if np.isfinite(x) else str(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def leaf(a):
    return parameter(np.asarray(a, dtype=np.float64))


def const(a):
    return Tensor(np.asarray(a, dtype=np.float64))


def zero_grads(P):
    for p in P.values():
        p.grad = None


def load_a1_params():
    """Assignment 1's saved initialization, as trainable leaves (the handoff of the two assignments)."""
    with np.load(A1DIR / "params_init.npz") as z:
        return {k: parameter(z[k]) for k in z.files}


def small_params(V, T, d, L, rng, sigma=0.3, jitter=0.1):
    """A small random model with non-trivial gains and biases (for derivative tests)."""
    P = init_params(V, T, d, L, sigma, rng=rng)
    for k, p in P.items():
        if "gamma" in k or "beta" in k or k.split(".")[-1].startswith("b_"):
            p.data[...] = p.data + jitter * rng.standard_normal(p.data.shape)
    return P


def group_norms(g):
    """Gradient norms of the four parameter groups the assignment names."""
    groups = {"embeddings (W_E, W_P)": ["W_E", "W_P"],
              "attention projections (W_QKV, W_O)": [k for k in g if k.endswith("W_QKV") or k.endswith("W_O")],
              "feed-forward weights (W_1, W_2)": [k for k in g if k.endswith("W_1") or k.endswith("W_2")],
              "vocabulary projection (W_voc)": ["W_voc"]}
    return {name: float(np.sqrt(sum(np.sum(g[k] ** 2) for k in ks))) for name, ks in groups.items()}


# ========================================================================== data and handoff
def setup_data():
    section("Data, vocabulary and the Assignment 1 handoff")
    text, raw = load_corpus(DATA)
    itos, stoi = build_vocab(text)
    ids = encode(text, stoi)
    train_ids, val_ids, n_tr = split_contiguous(ids)
    a1_vocab = json.loads((A1DIR / "vocab.json").read_text(encoding="utf-8"))["itos"]
    a1_cfg = json.loads((A1DIR / "config.json").read_text(encoding="utf-8"))
    log(f"file {DATA.name}: N = {len(ids)} characters, V = {len(itos)}, SHA-256 {sha256_hex(raw)}")
    log(f"split rule floor(0.9 N): N_tr = {len(train_ids)}, N_va = {len(val_ids)} (first character of validation at {n_tr})")
    check("D.1", "Q6", "vocabulary differs from Assignment 1's vocab.json (characters)",
          sum(a != b for a, b in zip(itos, a1_vocab)) + abs(len(itos) - len(a1_vocab)), 0, "0", "==")
    P = load_a1_params()
    P_count = count_params(P)
    V = len(itos)
    log(f"loaded {len(P)} tensors from data/assignment1/params_init.npz: P = {P_count} = 128V + 103680 = {128 * V + 103680}")
    check("D.2", "Q6", "P(loaded) - (128V + 103680)", P_count - (128 * V + 103680), 0, "0", "==")
    fresh = init_params(V, CFG["T"], CFG["d"], CFG["L"], CFG["sigma"], rng=np.random.default_rng(SEED))
    diff = max(max_abs(P[k].data, fresh[k].data) for k in P)
    check("D.3", "Q6", "saved Assignment 1 weights vs re-running its init (seed 451)", diff, 0, "0", "==")
    # fixed-input logits from the saved and the re-created weights must agree before any training
    Xf = encode("ROMEO: But soft, what light", stoi)[None, :]
    Za, _ = M.gpt_forward(Xf, M.as_constants(P), CFG["L"])
    Zb, _ = M.gpt_forward(Xf, M.as_constants(fresh), CFG["L"])
    check("D.4", "Q6", "fixed-input logits: saved vs re-created weights", max_abs(Za.data, Zb.data), 0, "0", "==")
    RESULTS["data"] = {"file": DATA.name, "sha256": sha256_hex(raw), "N": len(ids), "V": V, "N_tr": len(train_ids),
                       "N_va": len(val_ids), "vocab": "".join(itos), "a1_config": a1_cfg, "P": P_count}
    return {"text": text, "itos": itos, "stoi": stoi, "ids": ids, "train": train_ids, "val": val_ids, "n_tr": n_tr,
            "V": V, "P0": P}


# ========================================================================== Q1: residuals and FFN
def q1_ffn(D):
    section("Q1  Residual connections and the feed-forward network")
    out = {}
    # ---- the small case d = 2: dimensions and one self-chosen token row, entry by entry
    d, f = 2, 8
    Drow = np.array([[0.5, -1.0]])
    W1 = np.array([[0.4, -0.3, 0.2, 0.1, -0.5, 0.6, 0.0, 0.3], [0.1, 0.2, -0.4, 0.5, 0.3, -0.1, 0.7, -0.2]])
    b1 = np.array([[0.0, 0.1, -0.1, 0.2, 0.0, -0.2, 0.1, 0.0]])
    W2 = np.array([[0.3, -0.2], [0.1, 0.4], [-0.5, 0.2], [0.2, 0.1], [0.0, -0.3], [0.4, 0.0], [-0.1, 0.5], [0.2, 0.2]])
    b2 = np.array([[0.05, -0.05]])
    G, c = M.feed_forward(const(Drow[None]), const(W1), const(b1), const(W2), const(b2))
    J_loop = np.array([[sum(Drow[0, j] * W1[j, k] for j in range(d)) + b1[0, k] for k in range(f)]])
    g_loop = np.array([[0.5 * z * (1 + math.tanh(math.sqrt(2 / math.pi) * (z + 0.044715 * z ** 3))) for z in J_loop[0]]])
    G_loop = np.array([[sum(g_loop[0, k] * W2[k, j] for k in range(f)) + b2[0, j] for j in range(d)]])
    log(f"d = 2: W1 {W1.shape}, b1 {b1.shape}, W2 {W2.shape}, b2 {b2.shape}; token row D = {Drow.ravel()}")
    show("J = D W1 + b1 (1 x 4d)", J_loop, "{:8.4f}")
    show("g(J)", g_loop, "{:8.4f}")
    show("G = g(J) W2 + b2 (1 x d)", G_loop, "{:9.5f}")
    check("Q1.1", "Q1", "d=2 example: vectorized FFN vs the elementwise sums",
          max(max_abs(c["J"].data, J_loop), max_abs(G.data[0], G_loop)), 1e-15)
    out["small"] = {"D": Drow, "W1": W1, "b1": b1, "W2": W2, "b2": b2, "J": J_loop, "gJ": g_loop, "G": G_loop}
    # ---- tanh approximation against the exact GELU z Phi(z)
    z = np.array([-1.0, 0.0, 1.0])
    approx, exact = ops.gelu_np(z), ops.gelu_exact_np(z)
    for zi, a, e in zip(z, approx, exact):
        log(f"z = {zi:+.0f}: tanh approximation {a:+.9f}, exact z*Phi(z) {e:+.9f}, difference {a - e:+.3e}")
    zz = np.linspace(-6, 6, 2401)
    worst = float(np.max(np.abs(ops.gelu_np(zz) - ops.gelu_exact_np(zz))))
    log(f"largest |approximation - exact| on [-6, 6]: {worst:.3e}")
    check("Q1.2", "Q1", "max |tanh-GELU - z Phi(z)| at z = -1, 0, 1", np.max(np.abs(approx - exact)), 1e-3, "~1.6e-4")
    out["gelu_points"] = {"z": z, "approx": approx, "exact": exact, "max_on_grid": worst}
    # ---- g'(z): the assignment's formula vs the engine's derivative of the composed GELU vs FD
    zt = leaf(zz)
    ops.gelu(zt).sum().backward()
    e_auto = max_abs(zt.grad, ops.gelu_grad_np(zz))
    h = 1e-5
    e_fd = float(np.max(np.abs((ops.gelu_np(zz + h) - ops.gelu_np(zz - h)) / (2 * h) - ops.gelu_grad_np(zz))))
    log(f"g'(z) on [-6, 6]: formula vs autodiff {e_auto:.2e}; formula vs centred difference (h=1e-5) {e_fd:.2e}")
    check("Q1.3", "Q1", "g'(z) formula vs autodiff of the composed GELU (2401 points)", e_auto, 1e-14)
    out["gelu_grad"] = {"auto": e_auto, "fd": e_fd}
    # ---- residual Jacobian dR/dH = I + J_F(H), and feature vs token mixing
    rng = np.random.default_rng(SEED)
    T, dd = 4, 3
    Pm = small_params(5, T, dd, 1, rng)
    H = rng.standard_normal((1, T, dd))
    k = "block1."

    def residual(Hin):
        U, _ = A1.layer_norm(const(Hin), Pm[k + "ln1_gamma"], Pm[k + "ln1_beta"])
        F, _ = M.attention(U, Pm[k + "W_QKV"], Pm[k + "W_O"])
        return Hin + F.data, F.data
    n = T * dd
    JR, JF = np.zeros((n, n)), np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1e-6
        Rp, Fp = residual(H + e.reshape(1, T, dd))
        Rm, Fm = residual(H - e.reshape(1, T, dd))
        JR[:, j] = ((Rp - Rm) / 2e-6).ravel()
        JF[:, j] = ((Fp - Fm) / 2e-6).ravel()
    e_id = float(np.max(np.abs(JR - JF - np.eye(n))))
    log(f"dR/dH - J_F(H) - I over all {n}x{n} entries (central differences): {e_id:.2e}; ||J_F|| = {np.linalg.norm(JF, 2):.3f}")
    check("Q1.4", "Q1", "residual Jacobian: max |dR/dH - J_F - I|", e_id, 1e-8)
    # the attention block of T x d is lower block-triangular in tokens; the FFN Jacobian is block-diagonal
    blockF = np.array([[np.abs(JF[i * dd:(i + 1) * dd, j * dd:(j + 1) * dd]).max() for j in range(T)] for i in range(T)])
    show("attention: max |dF_i / dH_j| (token i, token j)", blockF, "{:9.2e}")
    Dm = rng.standard_normal((1, T, dd))

    def ffn(Din):
        G, _ = M.feed_forward(const(Din), Pm[k + "W_1"], Pm[k + "b_1"], Pm[k + "W_2"], Pm[k + "b_2"])
        return G.data
    blockG = np.zeros((T, T))
    for j in range(T):
        Dp = Dm.copy()
        Dp[0, j] += 0.5
        blockG[:, j] = np.abs(ffn(Dp) - ffn(Dm))[0].max(axis=1)
    show("feed-forward: max |change of G_i| after changing token j", blockG, "{:9.2e}")
    off = blockG[~np.eye(T, dtype=bool)].max()
    check("Q1.5", "Q1", "FFN mixes features only: change of other tokens' outputs", off, 0, "0", "==")
    check("Q1.6", "Q1", "attention mixes tokens causally: max |dF_i/dH_j| for j > i", blockF[np.triu_indices(T, 1)].max(), 1e-8)
    out["jacobian"] = {"identity_err": e_id, "attention_blocks": blockF, "ffn_blocks": blockG}
    # ---- the feed-forward backward formulas, term by term against autodiff
    B, T, dd = 3, 4, 5
    Pm = small_params(7, T, dd, 1, rng)
    Dn = leaf(rng.standard_normal((B, T, dd)))
    W1, b1, W2, b2 = (Pm[k + n_] for n_ in ("W_1", "b_1", "W_2", "b_2"))
    Gq, cq = M.feed_forward(Dn, W1, b1, W2, b2)
    Rg = rng.standard_normal((B, T, dd))
    zero_grads(Pm)
    (Gq * Rg).sum().backward()
    gm = M.feed_forward_backward(Rg, Dn.data, cq["J"].data, cq["Ga"].data, W1.data, W2.data)
    auto = {"W_2": W2.grad, "b_2": b2.grad, "Ga": cq["Ga"].grad, "J": cq["J"].grad, "W_1": W1.grad, "b_1": b1.grad, "D": Dn.grad}
    terms = {key: max_abs(auto[key], gm[key]) for key in auto}
    for key, v in terms.items():
        log(f"   G_{key:<4} formula vs autodiff: {v:.2e}")
    check("Q1.7", "Q1", "feed-forward backward: every term vs autodiff", max(terms.values()), 1e-13)
    out["ffn_terms"] = terms
    RESULTS["Q1"] = out


def case_b2():
    section("Case B2  Random feed-forward and residual (seed 451, B=2, T=5, d=4, d_ff=16)")
    rng = np.random.default_rng(SEED)
    B, T, d, f = 2, 5, 4, 16
    H = rng.normal(0, 0.02, (B, T, d))
    W1, W2 = rng.normal(0, 0.02, (d, f)), rng.normal(0, 0.02, (f, d))
    b1, b2 = np.zeros((1, f)), np.zeros((1, d))
    g1, z1 = np.ones((1, d)), np.zeros((1, d))
    W1t, W2t = leaf(W1), leaf(W2)

    def run(W1x, W2x):
        Dn, _ = A1.layer_norm(const(H), const(g1), const(z1))
        G, c = M.feed_forward(Dn, W1x, const(b1), W2x, const(b2))
        return const(H) + G, G, c
    Hn, G, c = run(W1t, W2t)
    J = c["J"].data.reshape(B, T, f)
    shapes = {"J": J.shape[1:], "G": G.shape[1:], "H_next": Hn.shape[1:]}
    log(f"per-sample shapes: J {shapes['J']} (T x 4d), G {shapes['G']} (T x d), H_next {shapes['H_next']}")
    check("B2.1", "Q1", "shape mismatches (J 5x16, G 5x4, H_next 5x4)", int(shapes != {"J": (5, 16), "G": (5, 4), "H_next": (5, 4)}), 0, "0", "==")
    check("B2.2", "Q1", "max |H_next - (H + G)|", max_abs(Hn.data, H + G.data), 0, "0", "==")
    (Hn * Hn).sum().backward()                        # J = sum_{b,i,j} (H_next)_{bij}^2
    picks = {"W_1": [tuple(rng.integers(0, s) for s in W1.shape) for _ in range(5)],
             "W_2": [tuple(rng.integers(0, s) for s in W2.shape) for _ in range(5)]}
    rows = []
    for name, theta, grad in (("W_1", W1, W1t.grad), ("W_2", W2, W2t.grad)):
        for idx in picks[name]:
            def J_():
                return float(np.sum(run(const(W1), const(W2))[0].data ** 2))
            row = {"param": name, "entry": list(map(int, idx)), "autodiff": float(grad[idx])}
            for h in H_VALUES:
                E = np.zeros_like(theta)
                E[idx] = 1.0
                fd = central_difference(J_, theta, E, h)
                row[f"fd_{h:g}"] = fd
                row[f"rel_{h:g}"] = rel_err(fd, grad[idx])
                row[f"relstrict_{h:g}"] = abs(fd - grad[idx]) / max(abs(fd), abs(grad[idx]), 1e-300)
            rows.append(row)
            log(f"   {name}{list(map(int, idx))}: autodiff {row['autodiff']:+.6e}  " +
                "  ".join(f"h={h:g}: rel {row[f'rel_{h:g}']:.1e} (strict {row[f'relstrict_{h:g}']:.1e})" for h in H_VALUES))
    best = {h: max(r[f"rel_{h:g}"] for r in rows) for h in H_VALUES}
    strict = {h: max(r[f"relstrict_{h:g}"] for r in rows) for h in H_VALUES}
    check("B2.3", "Q1", "FD vs autodiff on 10 entries of W1, W2 (assignment measure, h=1e-5)", best[1e-5], 1e-9)
    check("B2.4", "Q1", "FD vs autodiff, scale-free relative error at h=1e-5", strict[1e-5], 1e-5)
    RESULTS["B2"] = {"shapes": {k: list(v) for k, v in shapes.items()}, "rows": rows, "worst_rel": best, "worst_strict": strict,
                     "H": H, "G0": G.data[0]}


# ========================================================================== Q2: logits and cross-entropy
def q2_cross_entropy(D):
    section("Q2  Vocabulary projection and cross-entropy")
    out = {}
    Z = np.array([[2.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    y = np.array([0, 1])                                  # labels (1, 2) in one-based form
    a = Z.max(axis=1)
    lse = a + np.log(np.exp(Z - a[:, None]).sum(axis=1))
    nll = lse - Z[np.arange(2), y]
    ref = 0.5 * (math.log(math.e ** 2 + 1 + math.e ** -1) - 2 + math.log(2 + math.e) - 1)
    Zt = leaf(Z)
    Lt = M.cross_entropy(Zt, y)
    Lt.backward()
    log(f"row maxima a = {a}; LSE = {lse}; NLL = {nll}; L = {nll.mean():.15f}; closed form {ref:.15f}")
    check("Q2.1", "Q2", "2x3 example: |L - closed form| (engine and by hand)", max(abs(float(Lt.data) - ref), abs(nll.mean() - ref)), 1e-15)
    Gz = M.cross_entropy_backward(Z, y)
    show("dL/dZ = (P - E) / 2", Gz, "{:10.6f}")
    check("Q2.2", "Q2", "dL/dZ: |formula - autodiff| and max |row sum|", max(max_abs(Gz, Zt.grad), np.abs(Gz.sum(1)).max()), 1e-16)
    out["example"] = {"Z": Z, "y1": y + 1, "a": a, "lse": lse, "nll": nll, "L": float(Lt.data), "ref": ref, "grad": Gz}
    # ---- shift invariance and why the shift is needed
    shift = np.array([[1000.0], [-1000.0]])
    L_shift = float(M.cross_entropy(const(Z + shift), y).data)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        naive = float(np.mean(np.log(np.exp(Z + shift).sum(1)) - (Z + shift)[np.arange(2), y]))
    log(f"adding (+1000, -1000) to the rows: stable loss {L_shift:.15f}; without the max shift: {naive}")
    check("Q2.3", "Q2", "row shift by +-1000: |L(Z + c) - L(Z)| (stable LSE)", abs(L_shift - ref), 1e-12)
    out["shift"] = {"stable": L_shift, "naive": str(naive)}
    # ---- the ordering requirement: 2 x 3 labels, row-major (Python) vs column-major (MATLAB)
    lab = np.array([[1, 2, 3], [4, 5, 6]])
    py_order, ml_order = lab.reshape(-1).tolist(), lab.reshape(-1, order="F").tolist()
    log(f"labels {lab.tolist()}: Python row-major flatten {py_order}; MATLAB column-major Y(:) {ml_order}")
    rng = np.random.default_rng(SEED)
    Hd = rng.standard_normal((2, 3, 4))
    Wv = rng.standard_normal((4, 7))
    labels = rng.integers(0, 7, (2, 3))
    L_py = M.cross_entropy_np(Hd.reshape(6, 4) @ Wv, labels.reshape(-1))[0]
    L_ml = M.cross_entropy_np(Hd.reshape(6, 4, order="F") @ Wv, labels.reshape(-1, order="F"))[0]
    L_mix = M.cross_entropy_np(Hd.reshape(6, 4) @ Wv, labels.reshape(-1, order="F"))[0]
    perm = rng.permutation(6)
    L_perm = M.cross_entropy_np((Hd.reshape(6, 4) @ Wv)[perm], labels.reshape(-1)[perm])[0]
    log(f"same rule for rows and labels: row-major L = {L_py:.12f}, column-major L = {L_ml:.12f}; "
        f"rows row-major but labels column-major: L = {L_mix:.12f} (wrong pairing)")
    check("Q2.4", "Q2", "same flattening rule (row- or column-major) or a joint permutation: max |delta L|",
          max(abs(L_py - L_ml), abs(L_py - L_perm)), 1e-14)
    check("Q2.5", "Q2", "mismatched flattening changes the loss: |L_mix - L|", abs(L_mix - L_py), 1e-3, "> 0", ">=")
    out["ordering"] = {"labels": lab, "python": py_order, "matlab": ml_order, "L_py": L_py, "L_ml": L_ml, "L_mix": L_mix, "L_perm": L_perm}
    # ---- uniform logits, and one target probability of 0.7
    V = D["V"]
    L0 = float(M.cross_entropy(const(np.zeros((5, V))), np.arange(5)).data)
    check("Q2.6", "Q2", f"all-zero logits: |L - log V| (V = {V})", abs(L0 - math.log(V)), 1e-15)
    z7 = np.log(np.array([[0.7, 0.2, 0.1]]))
    check("Q2.7", "Q2", "target probability 0.7: |NLL + log 0.7|", abs(float(M.cross_entropy(const(z7), [0]).data) + math.log(0.7)), 1e-15)
    out["uniform"] = {"L": L0, "logV": math.log(V)}
    # ---- head gradients: dL/dW_voc = Hbar^T (P - E)/BT, dL/dHbar = (P - E)/BT W_voc^T
    Hb, Wv2 = leaf(rng.standard_normal((12, 6))), leaf(rng.standard_normal((6, V)) * 0.3)
    yy = rng.integers(0, V, 12)
    Zh = Hb @ Wv2
    M.cross_entropy(Zh, yy).backward()
    G = M.cross_entropy_backward(Zh.data, yy)
    e_head = max(max_abs(Wv2.grad, Hb.data.T @ G), max_abs(Hb.grad, G @ Wv2.data.T))
    check("Q2.8", "Q2", "head gradients Hbar^T G and G W_voc^T vs autodiff", e_head, 1e-15)
    RESULTS["Q2"] = out


def case_b1():
    section("Case B1  Fixed logit matrix and exact cross-entropy")
    Z = np.array([[0.0, 0.0, 0.0], [math.log(2), 0.0, 0.0], [0.0, math.log(3), 0.0]])
    y = np.array([0, 0, 1])                               # (1, 1, 2) in one-based form
    Zt = leaf(Z)
    Lt = M.cross_entropy(Zt, y)
    Lt.backward()
    P = M.softmax_np(Z)
    E = np.eye(3)[y]
    ref = (math.log(3) + math.log(2) + math.log(5 / 3)) / 3
    show("P = softmax(Z)", P, "{:10.7f}")
    log(f"correct-class probabilities {P[np.arange(3), y]} (expected 1/3, 1/2, 3/5); L = {float(Lt.data):.12f}, L_ref = {ref:.12f}")
    check("B1.1", "Q2", "|L - L_ref|", abs(float(Lt.data) - ref), 1e-6)
    check("B1.2", "Q2", "max |dL/dZ - (P - E)/3| (autodiff)", max_abs(Zt.grad, (P - E) / 3), 1e-6)
    check("B1.3", "Q2", "correct-class probabilities vs (1/3, 1/2, 3/5)", max_abs(P[np.arange(3), y], [1 / 3, 1 / 2, 3 / 5]), 1e-15)
    shifted = float(M.cross_entropy(const(Z + np.array([[5.0], [-3.0], [100.0]])), y).data)
    check("B1.4", "Q2", "shifting each row by a constant: |delta L|", abs(shifted - float(Lt.data)), 1e-13)
    RESULTS["B1"] = {"Z": Z, "P": P, "E": E, "L": float(Lt.data), "ref": ref, "grad": Zt.grad, "shifted": shifted}


# ========================================================================== Q3: autodiff and Adam
def q3_autodiff_adam(D):
    section("Q3  Automatic differentiation and Adam")
    out = {}
    # ---- the scalar Adam exercise
    g1, b1, b2, eta, eps = 0.2, 0.9, 0.999, 1e-3, 1e-8
    m1, v1 = (1 - b1) * g1, (1 - b2) * g1 ** 2
    mh, vh = m1 / (1 - b1), v1 / (1 - b2)
    upd = -eta * mh / (math.sqrt(vh) + eps)
    th = np.array([0.0])
    opt = Adam({"x": th}, eta, b1, b2, eps)
    opt.step({"x": th}, {"x": np.array([g1])})
    log(f"g1 = 0.2: m1 = {m1:.6g}, v1 = {v1:.6g}, m^1 = {mh:.6g}, v^1 = {vh:.6g}, update = {upd:.12f}")
    check("Q3.1", "Q3", "scalar Adam exercise: |optimizer update - hand computation|", abs(th[0] - upd), 1e-18)
    out["adam_scalar"] = {"m1": m1, "v1": v1, "m_hat": mh, "v_hat": vh, "update": upd}
    # ---- full model: every hand-derived gradient against the autodiff engine, on a real batch
    P = load_a1_params()
    X, Y, _ = get_batch(D["train"], CFG["B"], CFG["T"], np.random.default_rng(SEED))
    zero_grads(P)
    Lt, Z, cache = M.loss_fn(X, Y, P, CFG["L"])
    Lt.backward()
    GZ = M.cross_entropy_backward(Z.data, Y.reshape(-1))
    gm, _ = M.gpt_backward(GZ, cache, P, CFG["L"])
    errs = {k: max_abs(P[k].grad, gm[k]) for k in P}
    scale = {k: float(np.max(np.abs(P[k].grad))) for k in P}
    worst = max(errs, key=errs.get)
    log(f"real batch (B=32, T=64) at the Assignment 1 initialization: loss {float(Lt.data):.6f}")
    log(f"hand-derived vs autodiff, 25 tensors: worst {worst} {errs[worst]:.2e} (its gradient scale {scale[worst]:.2e})")
    check("Q3.2", "Q3", "every parameter gradient: hand-derived chain rule vs autodiff (max abs)", max(errs.values()), 1e-15)
    # the unused FFN/LN2 parameters of the stage are now on the path: none of the 25 tensors is zero
    zero_tensors = [k for k in P if not np.any(P[k].grad != 0)]
    log(f"tensors with an all-zero gradient: {zero_tensors}")
    check("Q3.3", "Q3", "tensors without any gradient (every parameter is on a path from L)", len(zero_tensors), 0, "0", "==")
    Hbar = cache["Hbar"].data
    G_voc = Hbar.T @ GZ
    check("Q3.4", "Q3", "dL/dW_voc = Hbar^T dL/dZ: shape d x V and value vs autodiff",
          max_abs(G_voc, P["W_voc"].grad) + (G_voc.shape != (CFG["d"], D["V"])), 1e-16)
    gE, gP = P["W_E"].grad, P["W_P"].grad
    check("Q3.5", "Q3", "W_E and W_P gradients: zero or non-finite (count of tensors)",
          int(not (np.all(np.isfinite(gE)) and np.any(gE != 0))) + int(not (np.all(np.isfinite(gP)) and np.any(gP != 0))), 0, "0", "==")
    out["full"] = {"loss": float(Lt.data), "errors": errs, "scale": scale, "norms": group_norms({k: P[k].grad for k in P}),
                   "W_E_rows_hit": int(np.sum(np.any(gE != 0, axis=1)))}
    # ---- removing the embeddings from the graph
    P2 = load_a1_params()
    P2d = dict(P2)
    P2d["W_E"] = P2["W_E"].detach()                         # a numerical copy without history
    M.loss_fn(X, Y, P2d, CFG["L"])[0].backward()
    log(f"embeddings detached: W_E.grad = {P2['W_E'].grad} -> Adam would never move the token table; "
        f"W_P still receives a gradient: {P2['W_P'].grad is not None}")
    check("Q3.6", "Q3", "gradient reaching W_E through a detached copy (1 = present)", int(P2["W_E"].grad is not None), 0, "0 (absent)", "==")
    RESULTS["Q3"] = out


def case_b3_adam():
    section("Case B3 (part 1)  One Adam step on a random 2 x 3 matrix (seed 451)")
    rng = np.random.default_rng(SEED)
    theta0 = rng.standard_normal((2, 3))
    g = np.array([[0.5, -0.2, 0.01], [-1.5, 0.3, 2.0]])
    b1, b2, eta, eps = 0.9, 0.999, 1e-3, 1e-8
    m1 = (1 - b1) * g
    v1 = (1 - b2) * g * g
    mh, vh = m1 / (1 - b1), v1 / (1 - b2)
    th_hand = theta0 - eta * mh / (np.sqrt(vh) + eps)
    th = theta0.copy()
    opt = Adam({"W": th}, eta, b1, b2, eps)
    opt.step({"W": th}, {"W": g})
    show("theta_0", theta0)
    show("m_1", m1, "{:10.6f}")
    show("v_1", v1, "{:12.8f}")
    show("theta_1 (optimizer)", th, "{:12.8f}")
    e = max(max_abs(opt.m["W"], m1), max_abs(opt.v["W"], v1), max_abs(th, th_hand))
    check("B3.1", "Q4", "Adam step: max |optimizer - matrix equations| (m1, v1, theta1)", e, 1e-6)
    RESULTS["B3_adam"] = {"theta0": theta0, "g": g, "m1": m1, "v1": v1, "m_hat": mh, "v_hat": vh, "theta1": th, "err": e}


# ========================================================================== Benchmark B
def benchmark_b():
    section("Benchmark B  Deterministic logits, loss, gradients and Adam (B=1, T=2, d=2, V=3)")
    H = np.eye(2)[None]
    F = np.zeros((1, 2, 2))
    R = H + F
    G, _ = M.feed_forward(A1.layer_norm(const(R), const([[1.0, 1.0]]), const([[0.0, 0.0]]))[0],
                          const(np.zeros((2, 8))), const(np.zeros((1, 8))), const(np.zeros((8, 2))), const(np.zeros((1, 2))))
    Hn = R + G.data
    show("R = H + F", R[0], "{:5.1f}")
    show("FFN output G", G.data[0], "{:5.1f}")
    show("H' = R + G", Hn[0], "{:5.1f}")
    check("BB.1", "Bench", "max |R - H| + |G| + |H' - H|", max_abs(R, H) + np.abs(G.data).max() + max_abs(Hn, H), 0, "0", "==")
    Hbar = leaf(np.eye(2))
    W = leaf(np.zeros((2, 3)))
    y = np.array([1, 2])                                  # (2, 3) in one-based form
    Z = Hbar @ W
    Lt = M.cross_entropy(Z, y)
    Lt.backward()
    P = M.softmax_np(Z.data)
    E = np.eye(3)[y]
    GZ = M.cross_entropy_backward(Z.data, y)
    GW_ref = np.array([[1 / 6, -1 / 3, 1 / 6], [1 / 6, 1 / 6, -1 / 3]])
    show("Z", Z.data, "{:6.2f}")
    show("P", P, "{:10.7f}")
    show("E", E, "{:4.0f}")
    log(f"L = {float(Lt.data):.12f}; log 3 = {math.log(3):.12f}")
    show("dL/dZ = (P - E)/2", GZ, "{:10.6f}")
    show("dL/dW_voc", W.grad, "{:10.6f}")
    check("BB.2", "Bench", "logits, probabilities, loss: max abs error", max(np.abs(Z.data).max(), max_abs(P, np.full((2, 3), 1 / 3)),
          abs(float(Lt.data) - math.log(3))), 1e-6)
    check("BB.3", "Bench", "dL/dZ and dL/dW_voc vs the displayed matrices", max(max_abs(GZ, GW_ref), max_abs(W.grad, GW_ref)), 1e-6)
    Wn = W.data.copy()
    opt = Adam({"W": Wn}, 1e-3, 0.9, 0.999, 1e-8)
    opt.step({"W": Wn}, {"W": W.grad})
    g = W.grad
    expected = -1e-3 * g / (np.abs(g) + 1e-8)
    show("W_voc after one Adam step", Wn, "{:12.9f}")
    check("BB.4", "Bench", "first Adam update vs -eta g/(|g| + eps)", max_abs(Wn, expected), 1e-7)
    check("BB.5", "Bench", "first Adam update vs the rounded display (+-0.001)", max_abs(Wn, 1e-3 * np.array([[-1, 1, -1], [-1, -1, 1]])), 1e-7)
    RESULTS["BenchB"] = {"R": R[0], "G": G.data[0], "Hn": Hn[0], "Z": Z.data, "P": P, "E": E, "L": float(Lt.data),
                         "GZ": GZ, "GW": W.grad, "W_new": Wn, "W_expected": expected}


# ========================================================================== Q4: schedule and gradient checks
def q4_gradient_check(D):
    section("Q4  Training schedule and gradient checks")
    s_epoch, s_total = schedule(len(D["train"]), CFG["B"], CFG["T"], CFG["epochs"], CFG["s_max"])
    log(f"S_epoch = floor(N_train / (B T)) = floor({len(D['train'])} / {CFG['B'] * CFG['T']}) = {s_epoch}; "
        f"S_total = 50 S_epoch = {50 * s_epoch}")
    log(f"each step sees B T = {CFG['B'] * CFG['T']} windows-positions drawn at random, so an epoch is a budget of "
        f"{s_epoch * CFG['B'] * CFG['T']} predictions, not a pass over every character")
    check("Q4.1", "Q4", "S_epoch - floor(N_train / (B T))", s_epoch - len(D["train"]) // (CFG["B"] * CFG["T"]), 0, "0", "==")
    # centred differences on the full model, at one fixed batch, for one scalar of every tensor kind
    P = load_a1_params()
    rng = np.random.default_rng(SEED)
    X, Y, _ = get_batch(D["train"], 4, CFG["T"], rng)          # fixed batch, no resampling during the test
    zero_grads(P)
    M.loss_fn(X, Y, P, CFG["L"])[0].backward()
    Pc = M.as_constants(P)

    def J():
        return float(M.cross_entropy(M.gpt_forward(X, Pc, CFG["L"])[0], Y.reshape(-1)).data)
    names = ["W_E", "W_P", "block1.W_QKV", "block1.W_O", "block1.W_1", "block1.b_1", "block1.W_2", "block2.b_2",
             "block1.ln1_gamma", "block2.ln2_beta", "lnf_gamma", "W_voc"]
    deltas = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    rows = []
    for name in names:
        g = P[name].grad
        idx = np.unravel_index(int(np.argmax(np.abs(g))), g.shape)  # the entry with the largest gradient
        row = {"param": name, "entry": [int(i) for i in idx], "autodiff": float(g[idx])}
        for dl in deltas:
            E = np.zeros_like(Pc[name].data)
            E[idx] = 1.0
            fd = central_difference(J, Pc[name].data, E, dl)
            row[f"{dl:g}"] = abs(fd - g[idx]) / max(1.0, abs(fd), abs(g[idx]))
        rows.append(row)
        log(f"   {name:<17}{str(row['entry']):<10} AD {row['autodiff']:+.4e}  " + " ".join(f"{row[f'{dl:g}']:.0e}" for dl in deltas))
    per_delta = {dl: max(r[f"{dl:g}"] for r in rows) for dl in deltas}
    best = min(per_delta, key=per_delta.get)
    log("worst rho over the 12 tensors per delta: " + "  ".join(f"{dl:g}: {v:.1e}" for dl, v in per_delta.items()) + f"  -> best delta {best:g}")
    check("Q4.2", "Q4", "full-model rho_k at delta = 1e-5 (12 tensors, fixed batch)", per_delta[1e-5], 1e-8)
    check("Q4.3", "Q4", f"full-model rho_k at the best delta ({best:g})", per_delta[best], 1e-9)
    finite = all(np.all(np.isfinite(p.data)) and np.all(np.isfinite(p.grad)) for p in P.values())
    check("Q4.4", "Q4", "non-finite parameters or gradients at the fixed batch (0 = none)", int(not finite), 0, "0", "==")
    RESULTS["Q4"] = {"s_epoch": s_epoch, "s_total_full": 50 * s_epoch, "fd": rows, "per_delta": per_delta, "best_delta": best,
                     "norms_init": group_norms({k: P[k].grad for k in P})}


def case_b3_synthetic():
    section("Case B3 (part 2)  The complete model on a repeated synthetic sequence ABAB...")
    text = "AB" * 2000
    itos, stoi = build_vocab(text)
    ids = encode(text, stoi)
    tr, va, _ = split_contiguous(ids)
    V, T, d, L = len(itos), 32, 32, 2
    P = init_params(V, T, d, L, 0.02, rng=np.random.default_rng(SEED))
    cfg = dict(CFG, B=16, T=T, d=d, L=L, epochs=100, s_max=300, eval_every=10)
    rec = train(P, tr, va, cfg, np.random.default_rng(SEED + 1), np.random.default_rng(SEED + 2), log=lambda *a: None)
    first, last = rec["train_curve"][0], rec["train_curve"][-1]
    sample = generate(P, "A", stoi, itos, 40, T, np.random.default_rng(SEED + 3), L)
    greedy = generate(P, "A", stoi, itos, 40, T, None, L, greedy=True)
    log(f"V = {V}, log V = {math.log(V):.4f}; training loss {first:.4f} -> {last:.6f} in {rec['s_total']} steps; "
        f"validation {rec['evals'][-1][2]:.6f}")
    log(f"after training: categorical sample {sample!r}; greedy {greedy!r}")
    check("B3.2", "Q4", "ABAB: final training loss / initial", last / first, 0.05)
    check("B3.3", "Q4", "ABAB greedy continuation: characters that break the alternation", sum(a == b for a, b in zip(greedy, greedy[1:])), 0, "0", "==")
    RESULTS["B3_synthetic"] = {"V": V, "logV": math.log(V), "first": first, "last": last, "steps": rec["s_total"],
                               "curve": rec["train_curve"], "val_last": rec["evals"][-1][2], "sample": sample, "greedy": greedy}


# ========================================================================== Case B4: Tiny Shakespeare training
def case_b4(D, reuse=False):
    section("Case B4  Tiny Shakespeare end-to-end training (B=32, T=64, d=64, L=2, lr 1e-3, 50 epochs)")
    vocab = {"stoi": D["stoi"], "itos": D["itos"]}
    if reuse and (OUT / "model_final.npz").exists():
        with np.load(OUT / "model_final.npz") as z:
            P = {k: parameter(z[k]) for k in z.files}
        rec = json.loads((OUT / "training.json").read_text(encoding="utf-8"))
        rec["train_curve"] = np.array(rec["train_curve"])
        log(f"reusing the saved model ({rec['s_total']} steps, {rec['seconds']:.0f} s of training)")
    else:
        P = load_a1_params()
        rec = train(P, D["train"], D["val"], CFG, np.random.default_rng(SEED), np.random.default_rng(SEED + 1),
                    np.random.default_rng(SEED + 2), vocab, log, CFG["preview_every"], CFG["preview_len"], CFG["prompt"])
        np.savez(OUT / "model_final.npz", **{k: p.data for k, p in P.items()})
        rec_json = dict(rec, train_curve=rec["train_curve"].tolist())
        (OUT / "training.json").write_text(json.dumps(jsonable(rec_json)), encoding="utf-8")
        with open(OUT / "training_log.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["step", "train_loss", "val_loss"])
            for s, lt, lv in rec["evals"]:
                w.writerow([s, f"{lt:.6f}", f"{lv:.6f}"])
    evals = np.array(rec["evals"], dtype=float)
    s_epoch, s_total = rec["s_epoch"], rec["s_total"]
    log(f"S_epoch = {s_epoch}, S_total = {s_total}; Adam steps taken {rec['adam_steps']}; {rec['seconds'] / 60:.1f} min")
    # low-noise final estimates on fixed, documented windows (never trained on as a set; train windows may overlap)
    starts_va = fixed_windows(len(D["val"]), CFG["T"], CFG["fixed_eval_windows"])
    starts_tr = fixed_windows(len(D["train"]), CFG["T"], CFG["fixed_eval_windows"])
    L_va = evaluate_fixed(P, D["val"], CFG["T"], starts_va, CFG["L"])
    L_tr = evaluate_fixed(P, D["train"], CFG["T"], starts_tr, CFG["L"])
    P0 = load_a1_params()
    L_va0 = evaluate_fixed(P0, D["val"], CFG["T"], starts_va, CFG["L"])
    log(f"fixed {len(starts_va)} validation windows: initial {L_va0:.4f} (log V = {math.log(D['V']):.4f}) -> final {L_va:.4f}")
    log(f"fixed {len(starts_tr)} training windows: final {L_tr:.4f}; generalization gap (val - train) {L_va - L_tr:+.4f}")
    check("B4.1", "Q4", "training steps taken - S_total", rec["adam_steps"] - s_total, 0, "0", "==")
    check("B4.2", "Q4", "validation batches only from the held-out part and no update during validation (1 = ok)",
          int(rec["val_only_heldout_and_no_update"]), 1, "1", "==")
    check("B4.3", "Q4", "final fixed-window validation loss below log V (nats)", math.log(D["V"]) - L_va, 0.5, "> 0.5", ">=")
    finite = all(np.all(np.isfinite(p.data)) for p in P.values())
    check("B4.4", "Q4", "non-finite trained parameters (0 = none)", int(not finite), 0, "0", "==")
    log(f"loss records: {len(evals)} evaluation points (step 1 and every {CFG['eval_every']} steps); first "
        f"train {evals[0, 1]:.4f} / val {evals[0, 2]:.4f}; last train {evals[-1, 1]:.4f} / val {evals[-1, 2]:.4f}")
    norms_first, norms_last = rec["norms"][0][1], rec["norms"][-1][1]
    for k in ("W_E", "W_P", "block1.W_QKV", "block1.W_1", "W_voc"):
        log(f"   ||grad {k}||: step 1 {norms_first[k]:.3e}, last evaluation {norms_last[k]:.3e}")
    RESULTS["B4"] = {"s_epoch": s_epoch, "s_total": s_total, "seconds": rec["seconds"], "evals": evals,
                     "L_val_fixed": L_va, "L_train_fixed": L_tr, "L_val_fixed_init": L_va0, "gap": L_va - L_tr,
                     "n_fixed": [len(starts_va), len(starts_tr)], "previews": rec["previews"],
                     "norms_first": norms_first, "norms_last": norms_last,
                     "group_norms_first": group_from_norms(norms_first), "group_norms_last": group_from_norms(norms_last),
                     "train_curve_smoothed": smooth(np.asarray(rec["train_curve"]), 50)[::25]}
    return P, rec


def group_from_norms(norms):
    """Group norms sqrt(sum ||g_k||^2) from per-tensor norms."""
    return group_norms({k: np.array([v]) for k, v in norms.items()})


def smooth(x, w):
    if len(x) < w:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[w:] - c[:-w]) / w


# ========================================================================== Q5: generation
def q5_generation(D, P):
    section("Q5  Autoregressive generation")
    out = {}
    z = np.array([2.0, 1.0, 0.0])
    for tau in (1.0, 0.5, 2.0):
        p = M.softmax_np((z / tau)[None])[0]
        H = float(-(p * np.log(p)).sum())
        log(f"tau = {tau}: p = {np.round(p, 6)}, sum = {p.sum():.15f}, entropy {H:.4f} nats")
        out[f"tau_{tau}"] = {"p": p, "entropy": H}
    check("Q5.1", "Q5", "z=(2,1,0): |p - (e^2, e, 1)/(e^2+e+1)| and |sum p - 1|",
          max(max_abs(out["tau_1.0"]["p"], np.exp([2, 1, 0]) / np.exp([2, 1, 0]).sum()), abs(out["tau_1.0"]["p"].sum() - 1)), 1e-15)
    rng = np.random.default_rng(SEED)
    draws = np.array([sample_index(out["tau_1.0"]["p"], rng) for _ in range(100_000)])
    freq = np.bincount(draws, minlength=3) / len(draws)
    log(f"100000 categorical draws: frequencies {freq} vs p {np.round(out['tau_1.0']['p'], 4)}")
    check("Q5.2", "Q5", "sampler: max |empirical frequency - p| over 100000 draws", np.abs(freq - out["tau_1.0"]["p"]).max(), 0.01)
    out["freq"] = freq
    # ---- context cropping: a prompt of length T + 5 keeps the last T tokens, positions re-indexed from 0
    T = CFG["T"]
    Pc = M.as_constants(P)
    prompt = D["ids"][1000:1000 + T + 5]
    p_full = next_distribution(prompt, Pc, T, CFG["L"])
    kept = prompt[5:]
    Z, _ = M.gpt_forward(kept[None], Pc, CFG["L"])
    p_crop = M.softmax_np(Z.data[-1][None])[0]
    log(f"prompt length T + 5 = {len(prompt)}: kept indices 6..{T + 5} (1-based), i.e. positions 1..{T} of W_P")
    check("Q5.3", "Q5", "cropped context: next_distribution vs explicit last-T forward", max_abs(p_full, p_crop), 0, "0", "==")
    # only the last row depends on the newest token; earlier rows ignore it (causality)
    Z2, _ = M.gpt_forward(np.r_[kept[:-1], (kept[-1] + 1) % D["V"]][None], Pc, CFG["L"])
    check("Q5.4", "Q5", "changing the newest token: change of all earlier rows' logits", max_abs(Z.data[:-1], Z2.data[:-1]), 0, "0", "==")
    out["crop"] = {"len": len(prompt), "kept_first_1based": 6, "kept_last_1based": T + 5}
    # ---- the required samples
    text = generate(P, CFG["prompt"], D["stoi"], D["itos"], CFG["sample_len"], T, np.random.default_rng(SEED + 3), CFG["L"])
    greedy = generate(P, CFG["prompt"], D["stoi"], D["itos"], CFG["sample_len"], T, None, CFG["L"], greedy=True)
    log("sampled continuation (200 characters, categorical, tau = 1):")
    log(text)
    log("greedy continuation (argmax, for contrast):")
    log(greedy)
    outside = sum(ch not in D["stoi"] for ch in text)
    check("Q5.5", "Q5", "generated characters outside the corpus vocabulary", outside, 0, "0", "==")
    check("Q5.6", "Q5", "sample length - (prompt + 200)", len(text) - (len(CFG["prompt"]) + CFG["sample_len"]), 0, "0", "==")
    temps = {}
    for tau in (0.5, 2.0):
        temps[tau] = generate(P, CFG["prompt"], D["stoi"], D["itos"], 120, T, np.random.default_rng(SEED + 3), CFG["L"], temperature=tau)
        log(f"optional tau = {tau}: {temps[tau]!r}")
    out.update({"sample": text, "greedy": greedy, "tau_samples": {str(k): v for k, v in temps.items()}})
    (OUT / "samples.txt").write_text(f"prompt: {CFG['prompt']!r}\n\n[categorical sampling, tau = 1, 200 characters]\n{text}\n\n"
                                     f"[greedy, for contrast]\n{greedy}\n\n" +
                                     "".join(f"[optional tau = {k}]\n{v}\n\n" for k, v in temps.items()), encoding="utf-8")
    RESULTS["Q5"] = out


# ========================================================================== Q6: integration, acceptance, B5
def q6_integration(D):
    section("Q6  Integration: shapes, residual input, masked weights, logits")
    P = load_a1_params()
    rows = []
    log(f"{'tensor':<18}{'shape':<12}{'entries':>9}")
    for k, p in P.items():
        rows.append({"name": k, "shape": list(p.shape), "numel": int(p.data.size)})
        log(f"{k:<18}{str(p.shape):<12}{p.data.size:>9}")
    V = D["V"]
    check("Q6.1", "Q6", "P - (128V + 103680) and P - formula", abs(count_params(P) - (128 * V + 103680)) +
          abs(count_params(P) - count_formula(V, CFG["T"], CFG["d"], CFG["L"])), 0, "0", "==")
    X, Y, _ = get_batch(D["train"], CFG["B"], CFG["T"], np.random.default_rng(SEED))
    Z, c = M.gpt_forward(X, M.as_constants(P), CFG["L"])
    b1 = c["blocks"][0]
    H0 = b1["H"].data
    check("Q6.2", "Q6", "embedding output and block input are B x T x d (mismatch count)",
          int(H0.shape != (CFG["B"], CFG["T"], CFG["d"])) + int(b1["U"].shape != H0.shape), 0, "0", "==")
    check("Q6.3", "Q6", "first residual: max |R - (H0 + F)| (attention output feeds the sum)",
          max_abs(b1["R"].data, H0 + b1["F"].data), 0, "0", "==")
    A = np.stack([blk["A"].data for blk in c["blocks"]])
    fut = np.triu(np.ones((CFG["T"], CFG["T"]), dtype=bool), 1)
    check("Q6.4", "Q6", "both blocks: largest masked future weight", A[:, :, fut].max(), 1e-7)
    check("Q6.5", "Q6", "both blocks: max_i |sum_j A_ij - 1|", np.abs(A.sum(-1) - 1).max(), 1e-12)
    L0 = float(M.cross_entropy(Z, Y.reshape(-1)).data)
    log(f"logits {Z.shape} = (BT x V) = ({CFG['B'] * CFG['T']} x {V}); initial loss {L0:.6f}; log V = {math.log(V):.6f}")
    check("Q6.6", "Q6", "logits shape (BT x V) mismatch + non-finite loss", int(Z.shape != (CFG["B"] * CFG["T"], V)) + int(not math.isfinite(L0)), 0, "0", "==")
    check("Q6.7", "Q6", "initial loss vs log V: |L0 - log V|", abs(L0 - math.log(V)), 0.05)
    RESULTS["Q6"] = {"table": rows, "L0": L0, "logV": math.log(V), "logits_shape": list(Z.shape)}


def acceptance(D):
    section("Minimum acceptance tests (T=1, B=1, repeated characters, tiny vocabulary, causality, per-sample)")
    out = []

    def run(name, X, V, d=8, L=2, seed=SEED):
        rng = np.random.default_rng(seed)
        B, T = X.shape
        P = small_params(V, T, d, L, rng, sigma=0.5)
        Y = rng.integers(0, V, (B, T))
        Lt, Z, c = M.loss_fn(X, Y, P, L)
        Lt.backward()
        finite = math.isfinite(float(Lt.data)) and all(np.all(np.isfinite(p.grad)) for p in P.values())
        Pc = M.as_constants(P)
        row = {"name": name, "B": B, "T": T, "V": V, "loss": float(Lt.data), "finite": bool(finite)}
        if T > 1:
            X2 = X.copy()
            X2[:, -1] = (X2[:, -1] + 1) % V
            Z1 = M.gpt_forward(X, Pc, L)[0].data.reshape(B, T, V)
            Z2 = M.gpt_forward(X2, Pc, L)[0].data.reshape(B, T, V)
            row["causal_delta"] = float(np.abs(Z2[:, :-1] - Z1[:, :-1]).max())
            row["last_delta"] = float(np.abs(Z2[:, -1] - Z1[:, -1]).max())
        U = c["blocks"][0]["U"].data
        F_loop = M.attention_per_sample(U, P["block1.W_QKV"].data, P["block1.W_O"].data)
        row["vec_vs_loop"] = max_abs(c["blocks"][0]["F"].data, F_loop)
        out.append(row)
        log(f"{name:<34} loss {row['loss']:.4f} finite {finite} vec-loop {row['vec_vs_loop']:.1e}"
            + (f" causal {row['causal_delta']:.1e} (last {row['last_delta']:.1e})" if "causal_delta" in row else ""))
        return row
    run("one-token context T=1 (B=3)", np.array([[2], [0], [1]]), 5)
    run("batch of size one B=1 (T=6)", np.array([[1, 3, 0, 2, 2, 4]]), 5)
    run("repeated character 'aaaaaaa'", np.zeros((1, 7), dtype=int), 3)
    run("tiny vocabulary V=2 'abba' x2", np.array([[0, 1, 1, 0], [1, 0, 0, 1]]), 2)
    run("short batch B=2, T=3, d=8", np.array([[0, 1, 2], [2, 2, 1]]), 3)
    check("ACC.1", "Q6", "non-finite losses or gradients in the five cases (count)", sum(not r["finite"] for r in out), 0, "0", "==")
    check("ACC.2", "Q6", "change the last token: max change of earlier positions' logits", max(r.get("causal_delta", 0.0) for r in out), 0, "0", "==")
    check("ACC.3", "Q6", "the change reaches the last position (min over cases with T > 1)",
          min(r["last_delta"] for r in out if "last_delta" in r), 1e-12, "> 0", ">=")
    check("ACC.4", "Q6", "vectorized vs per-sample attention (max over cases)", max(r["vec_vs_loop"] for r in out), 1e-14)
    RESULTS["acceptance"] = out


def case_b5(D):
    section("Case B5  Assignment 1 standalone attention vs the attention inside Assignment 2")
    P = load_a1_params()
    # the held-out batch of Assignment 1's protocol: B = 32 validation windows, seed 451
    Xv, _, rv = get_batch(D["val"], CFG["B"], CFG["T"], np.random.default_rng(SEED))
    H0, _ = A1.embed(Xv, Tensor(P["W_E"].data), Tensor(P["W_P"].data))
    U, _ = A1.layer_norm(H0, Tensor(P["block1.ln1_gamma"].data), Tensor(P["block1.ln1_beta"].data))
    F1, _ = A1.causal_self_attention(U, Tensor(P["block1.W_QKV"].data), Tensor(P["block1.W_O"].data))   # Assignment 1
    F2, _ = M.attention(U, Tensor(P["block1.W_QKV"].data), Tensor(P["block1.W_O"].data))               # Assignment 2
    Z, c = M.gpt_forward(Xv, M.as_constants(P), CFG["L"])
    F_in_model = c["blocks"][0]["F"].data
    e1, e2 = max_abs(F1.data, F2.data), max_abs(F1.data, F_in_model)
    log(f"held-out batch: validation starts {rv[:5].tolist()}...; F is {F1.shape}")
    log(f"max |F(A1) - F(A2 routine)| = {e1:.2e};  max |F(A1) - F inside gpt_forward| = {e2:.2e}")
    check("B5.1", "Q6", "standalone (A1) vs embedded (A2) attention, saved weights", max(e1, e2), 1e-5)
    cases = [("B1", "fixed logits, exact cross-entropy", "3 x 3", "L = (log 3 + log 2 + log 5/3)/3; dL/dZ = (P-E)/3", ["B1.1", "B1.2"]),
             ("B2", "random FFN and residual", "B=2, T=5, d=4, d_ff=16", "H_next = H + G; FD = autodiff", ["B2.1", "B2.2", "B2.3"]),
             ("B3", "Adam step and synthetic training", "2 x 3; V = 2", "matrix equations = optimizer; loss -> 0", ["B3.1", "B3.2", "B3.3"]),
             ("B4", "Tiny Shakespeare end to end", "B=32, T=64, d=64, L=2", "val loss well below log V; held-out only", ["B4.1", "B4.2", "B4.3", "B4.4"]),
             ("B5", "A1 vs A2 attention", "32 x 64 x 64", "max |F1 - F2| < 1e-5", ["B5.1"])]
    RESULTS["B5"] = {"err_routine": e1, "err_in_model": e2, "starts": rv[:8].tolist(), "cases": cases}


# ========================================================================== figures
def figures(D, P, rec):
    from gpt_a2 import plots
    made = plots.make_all(FIG, results=RESULTS, rec=rec, P=P, D=D, cfg=CFG)
    log(f"figures written: {made}" if made else "matplotlib not available: figures skipped")
    return made


# ========================================================================== main
def main(argv=None):
    ap = argparse.ArgumentParser(description="MAT 451E Assignment 2: train and sample a from-scratch character GPT.")
    ap.add_argument("--steps", type=int, default=0, help="S_max: cap the number of training steps (0 = 50 epochs)")
    ap.add_argument("--reuse-training", action="store_true", help="use results/python/model_final.npz instead of training")
    ap.add_argument("--no-figures", action="store_true", help="skip the matplotlib figures")
    args = ap.parse_args(argv)
    CFG["s_max"] = args.steps
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    section("Environment")
    env = {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
           "processor": platform.processor(), "float": "IEEE-754 float64"}
    try:
        import matplotlib
        env["matplotlib"] = matplotlib.__version__
    except ImportError:
        env["matplotlib"] = "not installed"
    for k, v in env.items():
        log(f"{k:>10}: {v}")
    log("configuration: " + json.dumps(CFG))
    D = setup_data()
    q1_ffn(D)
    case_b2()
    q2_cross_entropy(D)
    case_b1()
    q3_autodiff_adam(D)
    case_b3_adam()
    benchmark_b()
    q4_gradient_check(D)
    case_b3_synthetic()
    q6_integration(D)
    acceptance(D)
    case_b5(D)
    P, rec = case_b4(D, reuse=args.reuse_training)
    q5_generation(D, P)
    section("Q6  The numbered checks of the assignment")
    q6 = [("1", "every trainable shape printed and P = 128V + 103680", ["Q6.1", "D.2"]),
          ("2", "embedding output and residual block input dimensions", ["Q6.2"]),
          ("3", "causal attention output feeds the first residual sum", ["Q6.3"]),
          ("4", "masked weights ~ 0, attention rows sum to one", ["Q6.4", "Q6.5"]),
          ("5", "non-zero finite gradients for W_E and W_P", ["Q3.5"]),
          ("6", "logits BT x V, finite loss, initial loss vs log V", ["Q6.6", "Q6.7"]),
          ("7", "loss plot, 200-character sample, parameter count, run instructions", ["B4.1", "Q5.6", "Q6.1"])]
    status = {ch["id"]: ch["passed"] for ch in CHECKS}
    for num, text, ids in q6:
        log(f"check {num}: {text}: {'PASS' if all(status[i] for i in ids) else 'FAIL'} (evidence {', '.join(ids)})")
    RESULTS["Q6_list"] = [{"num": n, "text": t, "evidence": i, "passed": all(status[k] for k in i)} for n, t, i in q6]
    RESULTS["figures"] = [] if args.no_figures else figures(D, P, rec)
    n_pass = sum(ch["passed"] for ch in CHECKS)
    section("Summary")
    log(f"checks passed: {n_pass} / {len(CHECKS)}")
    log(f"wall time {time.time() - t_start:.1f} s")
    RESULTS.update({"env": env, "config": CFG, "checks": CHECKS,
                    "summary": {"passed": n_pass, "total": len(CHECKS), "seconds": time.time() - t_start}})
    (OUT / "config.json").write_text(json.dumps(jsonable({**CFG, "V": D["V"], "N": len(D["ids"]), "P": count_params(P)}), indent=2))
    (OUT / "results.json").write_text(json.dumps(jsonable(RESULTS), indent=1, ensure_ascii=False), encoding="utf-8")
    with open(OUT / "checks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(CHECKS[0].keys()))
        w.writeheader()
        w.writerows(CHECKS)
    (OUT / "log.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    return 0 if n_pass == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
