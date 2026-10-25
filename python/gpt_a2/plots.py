"""
plots.py -- the figures of the report, in the same style as Assignment 1.

MAT 451E, Assignment 2 (character-level GPT: training and generation)
Author: Ömer Kahyaoğlu <kahyaoglu20@itu.edu.tr>, student no. 090240741

Nothing here computes a result that is reported elsewhere: every function draws numbers the driver has
already measured (the training record, the gradient-check table, the trained parameters).  matplotlib is
imported inside `make_all`, so the package works without it.
"""
from pathlib import Path

import numpy as np

PALETTE = {"blue": "#2b6cb0", "orange": "#dd6b20", "green": "#2f9e6e", "purple": "#7c5cc4",
           "rose": "#d6336c", "gold": "#b7791f", "teal": "#17858a", "slate": "#5a6479"}
INK, SOFT, GRID, ZERO = "#1f2937", "#6b7280", "#d9dde5", "#dfe3ea"


def _style(plt):
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5, "axes.titlepad": 8,
        "axes.edgecolor": SOFT, "axes.linewidth": 0.8, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": SOFT, "ytick.color": SOFT, "text.color": INK,
        "xtick.labelcolor": INK, "ytick.labelcolor": INK, "legend.frameon": False, "legend.fontsize": 8.5,
        "lines.linewidth": 1.6, "lines.markersize": 4, "figure.facecolor": "white", "savefig.facecolor": "white",
        "savefig.dpi": 200, "savefig.bbox": "tight"})


def _smooth(x, w):
    if len(x) < w:
        return np.arange(len(x)) + 1, x
    c = np.cumsum(np.insert(np.asarray(x, dtype=float), 0, 0.0))
    return np.arange(w, len(x) + 1), (c[w:] - c[:-w]) / w


def _loss_curves(plt, out, rec, V, s_epoch, final):
    curve = np.asarray(rec["train_curve"], dtype=float)
    ev = np.asarray(rec["evals"], dtype=float)
    fig, ax = plt.subplots(figsize=(7.6, 4.1))
    ax.plot(np.arange(1, len(curve) + 1), curve, color=PALETTE["blue"], alpha=0.16, lw=0.6)
    xs, ys = _smooth(curve, 200)
    ax.plot(xs, ys, color=PALETTE["blue"], lw=1.6, label="training loss (every step, 200-step mean)")
    ax.plot(ev[:, 0], ev[:, 2], "o", ms=2.2, color=PALETTE["orange"], alpha=0.55, label="validation batch (8 windows, every 50 steps)")
    xv, yv = _smooth(ev[:, 2], 10)
    ax.plot(ev[xv - 1, 0], yv, color=PALETTE["orange"], lw=1.6, label="validation (mean of 10 evaluations)")
    ax.axhline(np.log(V), color=INK, ls="--", lw=1, label=rf"uniform prediction $\log V = {np.log(V):.3f}$")
    if final:
        ax.axhline(final["val"], color=PALETTE["rose"], ls=":", lw=1, label=f"final validation, fixed windows: {final['val']:.3f}")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy (nats per character)")
    ax.set_title("Tiny Shakespeare: training and held-out loss")
    ax.set_xlim(0, len(curve))
    ax.set_ylim(min(1.2, float(np.min(ys)) - 0.1), np.log(V) + 0.25)
    sec = ax.secondary_xaxis("top", functions=(lambda s: s / s_epoch, lambda e: e * s_epoch))
    sec.set_xlabel("epoch", color=SOFT)
    ax.legend(loc="upper right")
    fig.savefig(out / "loss_curves.png")
    plt.close(fig)


def _gelu(plt, out):
    from math import erf, sqrt
    z = np.linspace(-4, 4, 801)
    a = 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z ** 3)))
    e = z * 0.5 * (1 + np.vectorize(erf)(z / sqrt(2)))
    t = np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z ** 3))
    gp = 0.5 * (1 + t) + 0.5 * z * (1 - t * t) * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * z * z)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), constrained_layout=True)
    ax = axes[0]
    ax.plot(z, a, color=PALETTE["blue"], label=r"tanh approximation $g(z)$")
    ax.plot(z, e, color=PALETTE["orange"], ls="--", label=r"exact $z\,\Phi(z)$")
    ax.plot(z, gp, color=PALETTE["green"], label=r"$g'(z)$ (assignment's formula)")
    ax.plot(z, np.maximum(z, 0), color=SOFT, lw=0.8, ls=":", label="ReLU, for reference")
    ax.set_xlabel("$z$")
    ax.set_title("GELU and its derivative")
    ax.legend(loc="upper left")
    ax = axes[1]
    ax.plot(z, a - e, color=PALETTE["purple"])
    for zi in (-1, 0, 1):
        ai = 0.5 * zi * (1 + np.tanh(np.sqrt(2 / np.pi) * (zi + 0.044715 * zi ** 3)))
        ei = zi * 0.5 * (1 + erf(zi / sqrt(2)))
        ax.plot([zi], [ai - ei], "o", color=PALETTE["rose"], ms=4)
        ax.annotate(f"{ai - ei:+.1e}", (zi, ai - ei), textcoords="offset points", xytext=(4, 6), fontsize=8, color=PALETTE["rose"])
    ax.set_xlabel("$z$")
    ax.set_title(r"approximation error $g(z) - z\,\Phi(z)$")
    fig.savefig(out / "gelu.png")
    plt.close(fig)


def _grad_norms(plt, out, rec):
    steps = np.array([s for s, _ in rec["norms"]])
    groups = {"embeddings $W_E, W_P$": ["W_E", "W_P"], "attention $W_{QKV}, W_O$": ["W_QKV", "W_O"],
              "feed-forward $W_1, W_2$": ["W_1", "W_2"], "vocabulary $W_{voc}$": ["W_voc"]}
    colors = [PALETTE["gold"], PALETTE["teal"], PALETTE["purple"], PALETTE["rose"]]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for (name, keys), col in zip(groups.items(), colors):
        vals = np.array([np.sqrt(sum(v ** 2 for k, v in n.items() if any(k.endswith(x) for x in keys))) for _, n in rec["norms"]])
        xs, ys = _smooth(vals, 10)
        ax.semilogy(steps, vals, color=col, alpha=0.18, lw=0.6)
        ax.semilogy(steps[xs - 1], ys, color=col, label=name)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel(r"$\|\nabla\|_F$ (at the evaluation steps)")
    ax.set_title("Gradient norms of the four parameter groups during training")
    ax.legend(ncol=2, loc="upper right")
    fig.savefig(out / "grad_norms.png")
    plt.close(fig)


def _fd_delta(plt, out, rows):
    deltas = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    cmap = plt.get_cmap("viridis")
    for i, r in enumerate(rows):
        ax.loglog(deltas, [max(r[f"{d:g}"], 1e-17) for d in deltas], "o-", ms=3, lw=1, color=cmap(i / max(1, len(rows) - 1)),
                  label=r["param"])
    ax.loglog(deltas, 1e-4 * (np.array(deltas) / 1e-2) ** 2, color=INK, ls=":", lw=1, label=r"slope 2 ($\propto\delta^2$)")
    ax.loglog(deltas, 1e-16 / np.array(deltas), color=SOFT, ls="--", lw=1, label=r"round-off $\propto\varepsilon/\delta$")
    ax.set_xlabel(r"step $\delta$")
    ax.set_ylabel(r"$\rho_k = |g^{AD}-g^{FD}|/\max(1,|g^{AD}|,|g^{FD}|)$")
    ax.set_title("Full-model gradient check: truncation against round-off")
    ax.invert_xaxis()
    ax.set_ylim(1e-17, 1e-2)
    ax.legend(fontsize=6.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.savefig(out / "fd_delta.png")
    plt.close(fig)


def _attention(plt, out, P, D, cfg):
    from .model import as_constants, gpt_forward
    from gpt_a1.data import encode
    text = "ROMEO:\nBut soft, what light through yonder window breaks?"[:cfg["T"]]
    X = encode(text, D["stoi"])[None]
    _, c = gpt_forward(X, as_constants(P), cfg["L"])
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.9), constrained_layout=True)
    labels = [{"\n": "↵", " ": "␣"}.get(ch, ch) for ch in text]
    for ax, l in zip(axes, range(len(c["blocks"]))):
        A = np.ma.masked_equal(c["blocks"][l]["A"].data[0], 0.0)
        cmap = plt.get_cmap("magma").copy()
        cmap.set_bad(ZERO)
        ax.grid(False)
        im = ax.imshow(A, cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(len(text)), labels, fontsize=5.5, family="monospace")
        ax.set_yticks(range(len(text)), labels, fontsize=5.5, family="monospace")
        ax.set_title(f"block {l + 1}: trained attention weights $A_{{ij}}$")
        ax.set_xlabel("key (earlier character)")
        ax.set_ylabel("query (current character)")
    fig.colorbar(im, ax=list(axes), shrink=0.7, pad=0.01)
    fig.savefig(out / "attention_trained.png")
    plt.close(fig)


def _synthetic(plt, out, syn):
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    c = np.asarray(syn["curve"], dtype=float)
    ax.semilogy(np.arange(1, len(c) + 1), c, color=PALETTE["blue"], label="training loss")
    ax.axhline(syn["logV"], color=INK, ls="--", lw=1, label=r"$\log V = \log 2$")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy")
    ax.set_title("Case B3: the full model on ABAB…")
    ax.legend()
    fig.savefig(out / "synthetic_abab.png")
    plt.close(fig)


def make_all(out_dir, *, results, rec, P, D, cfg):
    """Draw every figure into out_dir; returns the file names (empty if matplotlib is missing)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _style(plt)
    b4 = results.get("B4", {})
    _loss_curves(plt, out, rec, D["V"], b4.get("s_epoch", 1), {"val": b4["L_val_fixed"]} if "L_val_fixed" in b4 else None)
    _gelu(plt, out)
    _grad_norms(plt, out, rec)
    _fd_delta(plt, out, results["Q4"]["fd"])
    _attention(plt, out, P, D, cfg)
    _synthetic(plt, out, results["B3_synthetic"])
    return ["loss_curves.png", "gelu.png", "grad_norms.png", "fd_delta.png", "attention_trained.png", "synthetic_abab.png"]
