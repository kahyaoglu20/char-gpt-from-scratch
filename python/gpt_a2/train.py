"""
train.py -- the optimization loop: fresh random windows, cross-entropy, autodiff gradients, Adam,
periodic held-out validation and text previews.

MAT 451E, Assignment 2 (character-level GPT: training and generation)
Author: Ömer Kahyaoğlu <kahyaoglu20@itu.edu.tr>, student no. 090240741

    S_epoch = floor(N_train / (B T))           steps per "epoch" (a step budget, not a pass over the data)
    S_total = 50 S_epoch                       or min(50 S_epoch, S_max) when S_max > 0

Every step draws B fresh windows from the training part, records the training loss *before* the
update, back-propagates through the whole model with the autodiff engine and applies one Adam step.
At step 1 and every `eval_every` steps a fresh batch of min(8, B) windows is drawn from the
validation part and evaluated *after* the update, on constant tensors: no graph, no gradient, no
change to the Adam moments (checked on every evaluation).
"""
import time

import numpy as np

from gpt_a1.data import get_batch

from .model import as_constants, cross_entropy, gpt_forward, loss_fn
from .optim import Adam
from .sample import generate


def schedule(n_train, B, T, epochs=50, s_max=0):
    s_epoch = n_train // (B * T)
    s_total = epochs * s_epoch if s_max <= 0 else min(epochs * s_epoch, s_max)
    return s_epoch, s_total


def gradients(P):
    """The .grad arrays after backward(); a parameter the loss does not reach gets an explicit zero."""
    return {k: (p.grad if p.grad is not None else np.zeros_like(p.data)) for k, p in P.items()}


def evaluate(P, X, Y, L=2):
    """Loss on one batch with constant parameters (no graph is recorded)."""
    Z, _ = gpt_forward(X, as_constants(P), L)
    return float(cross_entropy(Z, np.asarray(Y).reshape(-1)).data)


def fixed_windows(n, T, count):
    """A documented, deterministic set of window starts spread evenly over a split of length n."""
    return np.unique(np.linspace(0, n - T - 1, count).round().astype(np.int64))


def evaluate_fixed(P, ids, T, starts, L=2, chunk=64):
    """Mean loss over fixed windows (equal weight per predicted character): a low-noise estimate."""
    total, count = 0.0, 0
    for i in range(0, len(starts), chunk):
        r = starts[i:i + chunk]
        X = ids[r[:, None] + np.arange(T)]
        Y = ids[r[:, None] + np.arange(T) + 1]
        total += evaluate(P, X, Y, L) * X.size
        count += X.size
    return total / count


def train(P, train_ids, val_ids, cfg, rng_train, rng_val, rng_sample=None, vocab=None, log=print,
          preview_every=0, preview_len=80, prompt="ROMEO: "):
    """Optimize the Tensors of P in place. Returns the full record of the run."""
    B, T, L = cfg["B"], cfg["T"], cfg["L"]
    s_epoch, s_total = schedule(len(train_ids), B, T, cfg.get("epochs", 50), cfg.get("s_max", 0))
    arrays = {k: p.data for k, p in P.items()}          # Adam updates these arrays in place
    opt = Adam(arrays, cfg["lr"], cfg["beta1"], cfg["beta2"], cfg["eps_adam"])
    B_val = min(8, B)
    train_curve = np.empty(s_total)
    evals, previews, norms = [], [], []
    val_ok = True
    t0 = time.perf_counter()
    for s in range(1, s_total + 1):
        X, Y, _ = get_batch(train_ids, B, T, rng_train)          # fresh random windows, training part only
        for p in P.values():
            p.grad = None
        L_t, _, _ = loss_fn(X, Y, P, L)
        L_t.backward()                                           # reverse-mode autodiff through the whole GPT
        g = gradients(P)
        if not all(np.all(np.isfinite(v)) for v in g.values()):
            raise FloatingPointError(f"non-finite gradient at step {s}")
        train_curve[s - 1] = float(L_t.data)                     # training loss, measured before the update
        opt.step(arrays, g)
        if s == 1 or s % cfg["eval_every"] == 0:
            before = opt.fingerprint()
            Xv, Yv, rv = get_batch(val_ids, B_val, T, rng_val)  # held-out part only
            L_v = evaluate(P, Xv, Yv, L)                         # after the update; no backward, no step
            val_ok &= opt.fingerprint() == before and int(rv.min()) >= 0 and int(rv.max()) <= len(val_ids) - T - 1
            gn = {k: float(np.linalg.norm(v)) for k, v in g.items()}
            evals.append((s, train_curve[s - 1], L_v))
            norms.append((s, gn))
            if s == 1 or s % (cfg["eval_every"] * 20) == 0 or s == s_total:
                rate = s / (time.perf_counter() - t0)
                log(f"step {s:6d}/{s_total}  epoch {s / s_epoch:5.2f}  train {train_curve[s - 1]:.4f}  "
                    f"val {L_v:.4f}  ({rate:.1f} steps/s)")
        if preview_every and vocab and (s % preview_every == 0 or s == s_total):
            text = generate(P, prompt, vocab["stoi"], vocab["itos"], preview_len, T, rng_sample, L)
            previews.append((s, text))
            log(f"   preview @ {s}: {text!r}")
    return {"s_epoch": s_epoch, "s_total": s_total, "train_curve": train_curve, "evals": evals,
            "previews": previews, "norms": norms, "val_only_heldout_and_no_update": bool(val_ok),
            "seconds": time.perf_counter() - t0, "adam_steps": opt.t}
