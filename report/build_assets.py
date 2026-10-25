#!/usr/bin/env python3
"""
build_assets.py -- turn the *measured* results of Assignment 2 into LaTeX for the report.

Author: Ömer Kahyaoğlu <kahyaoglu20@itu.edu.tr>, student no. 090240741

Reads results/python/results.json (and results/matlab/results.json when the MATLAB program has been run)
and writes report/generated/:
    macros.tex       \\newcommand definitions of every number quoted in the text
    matrices.tex     measured matrices (Benchmark B, Case B1, the d = 2 example, ...)
    checkvals.tex    \\chkpy{ID} / \\chkml{ID}: the measured value of every check
    tab_*.tex        table bodies (verification errors, parameters, gradient checks, cases, ...)
    *.txt            the generated text (samples, previews), shown verbatim
    snippets/        the exact source of every function listed in the report
No number in the report is typed by hand.  Without MATLAB results the MATLAB columns show a dash and the
text says that the MATLAB program has not been run for this build.

    python report/build_assets.py        (from the submission root)
"""
import ast
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "report" / "generated"
SNIP = GEN / "snippets"
PY = json.loads((ROOT / "results" / "python" / "results.json").read_text(encoding="utf-8"))
_ml = ROOT / "results" / "matlab" / "results.json"
ML = json.loads(_ml.read_text(encoding="utf-8")) if _ml.exists() else None
DASH = "---"


# ------------------------------------------------------------------ formatting helpers
def tex_escape(s):
    rep = {"\\": r"\textbackslash{}", "_": r"\_", "^": r"\textasciicircum{}", "{": r"\{", "}": r"\}",
           "%": r"\%", "&": r"\&", "#": r"\#", "$": r"\$", "~": r"\textasciitilde{}"}
    return "".join(rep.get(ch, ch) for ch in str(s))


def num(x, digits=2):
    """Scientific notation for LaTeX; exact zeros print as 0."""
    if isinstance(x, str):
        return x
    x = float(x)
    if x == 0:
        return "0"
    if math.isnan(x):
        return "NaN"
    mant, e = f"{x:.{max(digits, 1) - 1}e}".split("e")
    e = int(e)
    if -2 <= e <= 4:
        if x.is_integer():
            return f"{int(x)}"
        return f"{x:.{max(digits, 1) + 1}g}"
    return rf"{mant}\times10^{{{e}}}"


def sci(x, digits=2):
    return rf"\ensuremath{{{num(x, digits)}}}"


def fix(x, d=4):
    return f"{float(x):.{d}f}"


def intc(n):
    return f"{int(round(float(n))):,}".replace(",", r"\,")


def macro(name, value):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def write(name, text):
    (GEN / name).write_text(text, encoding="utf-8")


def bmat(M, fmt="{:.6f}"):
    rows = [" & ".join(fmt.format(v) for v in row) for row in M]
    return r"\begin{bmatrix}" + r" \\ ".join(rows) + r"\end{bmatrix}"


def frac_mat(M):
    """Small rational matrices (Benchmark B) as fractions where exact."""
    from fractions import Fraction
    out = []
    for row in M:
        cells = []
        for v in row:
            f = Fraction(v).limit_denominator(12)
            if abs(float(f) - v) < 1e-12 and f.denominator != 1:
                cells.append(("-" if f < 0 else "") + rf"\tfrac{{{abs(f.numerator)}}}{{{f.denominator}}}")
            else:
                cells.append(f"{v:.0f}" if float(v).is_integer() else f"{v:.6f}")
        out.append(" & ".join(cells))
    return r"\begin{bmatrix}" + r" \\ ".join(out) + r"\end{bmatrix}"


# ------------------------------------------------------------------ code snippets
def py_function(path, name, cls=None):
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    nodes = tree.body
    if cls is not None:
        nodes = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    node = next(n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    seg = ast.get_source_segment(src, node, padded=True)
    lines = seg.splitlines()
    indent = min(len(l) - len(l.lstrip()) for l in lines if l.strip())
    return "\n".join(l[indent:] for l in lines)


def build_snippets():
    SNIP.mkdir(parents=True, exist_ok=True)
    for f in SNIP.iterdir():
        f.unlink()
    pkg = ROOT / "python" / "gpt_a2"
    groups = {
        "py_ops": (pkg / "ops.py", ["tanh", "log", "pick", "gelu"]),
        "py_gelu_grad": (pkg / "ops.py", ["gelu_grad_np"]),
        "py_ffn": (pkg / "model.py", ["feed_forward", "block"]),
        "py_attention": (pkg / "model.py", ["attention"]),
        "py_gpt": (pkg / "model.py", ["gpt_forward", "cross_entropy", "loss_fn"]),
        "py_ce_back": (pkg / "model.py", ["cross_entropy_backward"]),
        "py_ffn_back": (pkg / "model.py", ["feed_forward_backward"]),
        "py_block_back": (pkg / "model.py", ["block_backward", "gpt_backward"]),
        "py_adam": (pkg / "optim.py", ["Adam"]),
        "py_schedule": (pkg / "train.py", ["schedule", "evaluate", "fixed_windows", "evaluate_fixed"]),
        "py_train": (pkg / "train.py", ["train"]),
        "py_sample": (pkg / "sample.py", ["next_distribution", "sample_index", "generate"]),
    }
    for name, (path, funcs) in groups.items():
        (SNIP / f"{name}.py").write_text("\n\n".join(py_function(path, f) for f in funcs) + "\n", encoding="utf-8")
    for f in (ROOT / "matlab" / "src").glob("*.m"):
        lines = [l for l in f.read_text(encoding="utf-8").splitlines() if "Kahyao" not in l]
        (SNIP / f.name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    for f in SNIP.iterdir():
        bad = [ch for ch in f.read_text(encoding="utf-8") if ord(ch) > 126]
        assert not bad, (f.name, bad[:5])


# ------------------------------------------------------------------ tables
def ml_checks():
    return {c["id"]: c for c in ML["checks"]} if ML else {}


def table_checks():
    ml = ml_checks()
    out = []
    for c in PY["checks"]:
        m = ml.get(c["id"])
        desc = tex_escape(c["description"])
        rule = {"<=": r"$\le$", ">=": r"$\ge$", "==": "$=$"}[c["rule"]] + " " + sci(c["threshold"], 1)
        ok = c["passed"] and (m is None or m["passed"])
        out.append(f"{c['id']} & {desc} & {rule} & {sci(c['observed'])} & {sci(m['observed']) if m else DASH} & "
                   f"{'pass' if ok else r'\textbf{FAIL}'} \\\\")
    write("tab_checks.tex", "\n".join(out) + "\n")


MATH_NAMES = {"W_E": r"$W_E$", "W_P": r"$W_P$", "W_QKV": r"$W_{QKV}^{(\ell)}$", "W_O": r"$W_O^{(\ell)}$",
              "W_1": r"$W_1^{(\ell)}$", "b_1": r"$b_1^{(\ell)}$", "W_2": r"$W_2^{(\ell)}$", "b_2": r"$b_2^{(\ell)}$",
              "ln1_gamma": r"$\gamma_1^{(\ell)}$", "ln1_beta": r"$\beta_1^{(\ell)}$",
              "ln2_gamma": r"$\gamma_2^{(\ell)}$", "ln2_beta": r"$\beta_2^{(\ell)}$",
              "lnf_gamma": r"$\gamma_f$", "lnf_beta": r"$\beta_f$", "W_voc": r"$W_{\mathrm{voc}}$"}


def math_name(n):
    base = n.split(".")[-1]
    m = MATH_NAMES.get(base, tex_escape(n))
    if n.startswith("block"):
        m = m.replace(r"\ell", n[5])
    return m


def table_params():
    out = []
    for r in PY["Q6"]["table"]:
        out.append(f"{math_name(r['name'])} & \\code{{{r['name']}}} & ${r['shape'][0]}\\times{r['shape'][1]}$ & {intc(r['numel'])} \\\\")
    write("tab_params.tex", "\n".join(out) + "\n")


def table_fd():
    deltas = ["0.01", "0.001", "0.0001", "1e-05", "1e-06", "1e-07", "1e-08"]
    out = []
    for r in PY["Q4"]["fd"]:
        cells = " & ".join(sci(r[k], 1) for k in deltas)
        out.append(f"{math_name(r['param'])} & {sci(r['autodiff'], 3)} & {cells} \\\\")
    worst = " & ".join(sci(PY["Q4"]["per_delta"][k], 1) for k in deltas)
    out.append(r"\midrule \textbf{worst} & & " + worst + r" \\")
    write("tab_fd.tex", "\n".join(out) + "\n")


def table_b2():
    out = []
    for r in PY["B2"]["rows"]:
        out.append(f"$({r['param'][0]}_{r['param'][2]})_{{{r['entry'][0] + 1},{r['entry'][1] + 1}}}$ & {sci(r['autodiff'], 4)} & "
                   + " & ".join(sci(r[f"rel_{h}"], 1) for h in ("0.001", "0.0001", "1e-05", "1e-06")) + " \\\\")
    write("tab_b2.tex", "\n".join(out) + "\n")


def table_acceptance():
    out = []
    for r in PY["acceptance"]:
        causal = sci(r["causal_delta"]) if "causal_delta" in r else DASH
        last = sci(r["last_delta"]) if "last_delta" in r else DASH
        out.append(f"{tex_escape(r['name'])} & {fix(r['loss'], 4)} & {'yes' if r['finite'] else 'NO'} & {causal} & {last} & "
                   f"{sci(r['vec_vs_loop'])} \\\\")
    write("tab_acceptance.tex", "\n".join(out) + "\n")


def table_cases():
    status = {c["id"]: c["passed"] for c in PY["checks"]}
    out = []
    for cid, title, dims, prop, ids in PY["B5"]["cases"]:
        ok = all(status[i] for i in ids)
        obs = ", ".join(rf"\code{{{i}}} {sci(next(c['observed'] for c in PY['checks'] if c['id'] == i))}" for i in ids)
        out.append(f"{cid} & {'pass' if ok else r'\textbf{FAIL}'} & {tex_escape(title)} & {tex_escape(dims)} & "
                   f"{tex_escape(prop)} & {obs} \\\\")
    write("tab_cases.tex", "\n".join(out) + "\n")


def table_norms():
    first, last = PY["B4"]["group_norms_first"], PY["B4"]["group_norms_last"]
    names = {"embeddings (W_E, W_P)": r"embeddings $W_E, W_P$", "attention projections (W_QKV, W_O)": r"attention $W_{QKV}, W_O$",
             "feed-forward weights (W_1, W_2)": r"feed-forward $W_1, W_2$", "vocabulary projection (W_voc)": r"vocabulary $W_{\mathrm{voc}}$"}
    out = [f"{names[k]} & {sci(PY['Q4']['norms_init'][k], 3)} & {sci(first[k], 3)} & {sci(last[k], 3)} \\\\" for k in first]
    write("tab_norms.tex", "\n".join(out) + "\n")


def table_numbered():
    out = []
    for r in PY["Q6_list"]:
        ev = ", ".join(rf"\code{{{i}}}" for i in r["evidence"])
        out.append(f"{r['num']} & {tex_escape(r['text'])} & {ev} & {'pass' if r['passed'] else r'\textbf{FAIL}'} \\\\")
    write("tab_numbered.tex", "\n".join(out) + "\n")


def text_files():
    q5 = PY["Q5"]
    write("sample.txt", q5["sample"] + "\n")
    write("greedy.txt", q5["greedy"] + "\n")
    write("tau.txt", "".join(f"[tau = {k}]\n{v}\n\n" for k, v in q5["tau_samples"].items()))
    prev = PY["B4"]["previews"]
    write("previews.txt", "".join(f"[step {s:>6}]  {t}\n\n".replace("\n", "\n" + " " * 16, t.count("\n")) for s, t in prev))
    syn = PY["B3_synthetic"]
    write("synthetic.txt", f"categorical: {syn['sample']}\ngreedy:      {syn['greedy']}\n")
    if ML and "Q5" in ML:
        write("sample_ml.txt", ML["Q5"]["sample"] + "\n")


# ------------------------------------------------------------------ macros and matrices
def build_macros():
    m = ""
    d = PY["data"]
    m += macro("Ncorpus", intc(d["N"])) + macro("Vsize", str(d["V"])) + macro("Ntr", intc(d["N_tr"])) + macro("Nva", intc(d["N_va"]))
    m += macro("Shasum", d["sha256"][:16] + r"\dots") + macro("ShasumFull", d["sha256"]) + macro("Ptotal", intc(d["P"]))
    m += macro("logV", fix(math.log(d["V"]), 6))
    q1 = PY["Q1"]
    gp = q1["gelu_points"]
    m += macro("GeluApproxM", f"{gp['approx'][0]:+.9f}") + macro("GeluExactM", f"{gp['exact'][0]:+.9f}")
    m += macro("GeluApproxP", f"{gp['approx'][2]:+.9f}") + macro("GeluExactP", f"{gp['exact'][2]:+.9f}")
    m += macro("GeluDiff", sci(abs(gp["approx"][2] - gp["exact"][2]), 3)) + macro("GeluGridMax", sci(gp["max_on_grid"], 2))
    m += macro("GeluGradAuto", sci(q1["gelu_grad"]["auto"])) + macro("GeluGradFD", sci(q1["gelu_grad"]["fd"]))
    m += macro("ResJacErr", sci(q1["jacobian"]["identity_err"])) + macro("FFNTermsErr", sci(max(q1["ffn_terms"].values())))
    b2 = PY["B2"]
    m += macro("BtwoRel", sci(b2["worst_rel"]["1e-05"])) + macro("BtwoStrict", sci(b2["worst_strict"]["1e-05"]))
    q2 = PY["Q2"]
    ex = q2["example"]
    m += macro("CEexLSEa", fix(ex["lse"][0], 6)) + macro("CEexLSEb", fix(ex["lse"][1], 6))
    m += macro("CEexNLLa", fix(ex["nll"][0], 6)) + macro("CEexNLLb", fix(ex["nll"][1], 6)) + macro("CEexL", fix(ex["L"], 12))
    m += macro("CEexRef", fix(ex["ref"], 12)) + macro("ShiftStable", fix(q2["shift"]["stable"], 12))
    m += macro("ShiftNaive", tex_escape(q2["shift"]["naive"]))
    o = q2["ordering"]
    m += macro("OrdPy", "(" + ", ".join(map(str, o["python"])) + ")") + macro("OrdMl", "(" + ", ".join(map(str, o["matlab"])) + ")")
    m += macro("LOrdPy", fix(o["L_py"], 12)) + macro("LOrdMl", fix(o["L_ml"], 12)) + macro("LOrdMix", fix(o["L_mix"], 12))
    m += macro("LOrdPerm", fix(o["L_perm"], 12))
    b1 = PY["B1"]
    m += macro("BoneL", fix(b1["L"], 12)) + macro("BoneRef", fix(b1["ref"], 12))
    q3 = PY["Q3"]
    a = q3["adam_scalar"]
    m += macro("AdamMone", num(a["m1"], 3)) + macro("AdamVone", num(a["v1"], 3)) + macro("AdamMhat", num(a["m_hat"], 3))
    m += macro("AdamVhat", num(a["v_hat"], 3)) + macro("AdamUpd", f"{a['update']:.12f}")
    m += macro("FullLoss", fix(q3["full"]["loss"], 6)) + macro("FullErr", sci(max(q3["full"]["errors"].values())))
    m += macro("WErowsHit", str(q3["full"]["W_E_rows_hit"]))
    q4 = PY["Q4"]
    m += macro("Sepoch", intc(q4["s_epoch"])) + macro("StotalFull", intc(q4["s_total_full"]))
    m += macro("FDbestDelta", num(q4["best_delta"], 1)) + macro("FDatFive", sci(q4["per_delta"]["1e-05"]))
    m += macro("FDbest", sci(min(q4["per_delta"].values())))
    syn = PY["B3_synthetic"]
    m += macro("SynFirst", fix(syn["first"], 4)) + macro("SynLast", sci(syn["last"], 2)) + macro("SynSteps", str(syn["steps"]))
    m += macro("SynVal", sci(syn["val_last"], 2))
    b4 = PY["B4"]
    ev = b4["evals"]
    m += macro("Stotal", intc(b4["s_total"])) + macro("TrainMinutes", f"{b4['seconds'] / 60:.1f}")
    m += macro("Lvalfixed", fix(b4["L_val_fixed"], 4)) + macro("Ltrainfixed", fix(b4["L_train_fixed"], 4))
    m += macro("Lgap", f"{b4['gap']:+.4f}") + macro("Lvalinit", fix(b4["L_val_fixed_init"], 4))
    m += macro("NfixedVa", str(b4["n_fixed"][0])) + macro("NfixedTr", str(b4["n_fixed"][1]))
    m += macro("FirstTrain", fix(ev[0][1], 4)) + macro("FirstVal", fix(ev[0][2], 4))
    m += macro("LastTrain", fix(ev[-1][1], 4)) + macro("LastVal", fix(ev[-1][2], 4)) + macro("NumEvals", str(len(ev)))
    m += macro("Perplexity", f"{math.exp(b4['L_val_fixed']):.2f}") + macro("PerplexityInit", f"{math.exp(b4['L_val_fixed_init']):.2f}")
    m += macro("BitsPerChar", f"{b4['L_val_fixed'] / math.log(2):.3f}") + macro("NumPreviews", str(len(b4["previews"])))
    m += macro("EpochsRun", f"{b4['s_total'] / b4['s_epoch']:.0f}")
    vals = [e[2] for e in ev[-100:]]                     # measured noise of the 8-window validation batches
    diffs = [b - a for a, b in zip(vals, vals[1:])]
    noise = (sum(x * x for x in diffs) / max(1, len(diffs))) ** 0.5 / math.sqrt(2) if diffs else 0.0
    m += macro("ValNoise", f"{noise:.3f}")
    q5 = PY["Q5"]
    for tag, key in (("One", "tau_1.0"), ("Half", "tau_0.5"), ("Two", "tau_2.0")):
        p = q5[key]["p"]
        m += macro(f"Prob{tag}", "(" + ", ".join(f"{x:.6f}" for x in p) + ")") + macro(f"Ent{tag}", fix(q5[key]["entropy"], 4))
    m += macro("SamplerFreq", "(" + ", ".join(f"{x:.4f}" for x in q5["freq"]) + ")")
    m += macro("LzeroInit", fix(PY["Q6"]["L0"], 6)) + macro("BfiveErr", sci(max(PY["B5"]["err_routine"], PY["B5"]["err_in_model"])))
    e = PY["env"]
    m += macro("PyVersion", e["python"]) + macro("NumpyVersion", e["numpy"]) + macro("MplVersion", e["matplotlib"])
    m += macro("Platform", tex_escape(e["platform"])) + macro("PySeconds", f"{PY['summary']['seconds'] / 60:.1f}")
    m += macro("PyPassed", str(PY["summary"]["passed"])) + macro("NChecks", str(PY["summary"]["total"]))
    if ML:
        me = ML.get("env", {})
        m += macro("MlRun", "1") + macro("MlPassed", str(ML["summary"]["passed"])) + macro("MlVersion", tex_escape(me.get("matlab", "?")))
        m += macro("MlGrad", tex_escape(me.get("gradient", "?"))) + macro("MlToolbox", tex_escape(me.get("deep_learning_toolbox", "?")))
        m += macro("MlComputer", tex_escape(me.get("computer", "?")))
        mb = ML.get("B4", {})
        m += macro("MlLvalfixed", fix(mb.get("L_val_fixed", float("nan")), 4)) + macro("MlLtrainfixed", fix(mb.get("L_train_fixed", float("nan")), 4))
        m += macro("MlLvalinit", fix(mb.get("L_val_fixed_init", float("nan")), 4)) + macro("MlLgap", f"{mb.get('gap', float('nan')):+.4f}")
        m += macro("MlPerplexity", f"{math.exp(mb.get('L_val_fixed', float('nan'))):.2f}")
        m += macro("MlBitsPerChar", f"{mb.get('L_val_fixed', float('nan')) / math.log(2):.3f}")
        m += macro("MlStotal", intc(mb.get("s_total", 0))) + macro("MlMinutes", f"{mb.get('seconds', 0) / 60:.1f}")
        m += macro("MlTotalMinutes", f"{ML['summary']['seconds'] / 60:.1f}")
        m += macro("MlValDiff", fix(abs(mb.get("L_val_fixed", float("nan")) - b4["L_val_fixed"]), 4))
    else:
        m += macro("MlRun", "0") + macro("MlPassed", DASH) + macro("MlVersion", "not run") + macro("MlGrad", DASH)
        m += macro("MlToolbox", DASH) + macro("MlComputer", DASH)
        m += macro("MlLvalfixed", DASH) + macro("MlLtrainfixed", DASH) + macro("MlLvalinit", DASH) + macro("MlLgap", DASH)
        m += macro("MlPerplexity", DASH) + macro("MlBitsPerChar", DASH) + macro("MlStotal", DASH) + macro("MlMinutes", DASH)
        m += macro("MlTotalMinutes", DASH) + macro("MlValDiff", DASH)
    write("macros.tex", m)


def build_matrices():
    m = ""
    s = PY["Q1"]["small"]
    m += macro("SmallWone", bmat(s["W1"], "{:.1f}")) + macro("SmallWtwo", bmat(s["W2"], "{:.2f}"))
    m += macro("SmallJ", bmat(s["J"], "{:.3f}")) + macro("SmallgJ", bmat(s["gJ"], "{:.4f}")) + macro("SmallG", bmat(s["G"], "{:.5f}"))
    m += macro("SmallBone", bmat(s["b1"], "{:.1f}")) + macro("SmallBtwo", bmat(s["b2"], "{:.2f}"))
    jac = PY["Q1"]["jacobian"]
    m += macro("BlockF", bmat(jac["attention_blocks"], "{:.2f}"))
    m += macro("BlockG", bmat(jac["ffn_blocks"], "{:.2f}"))
    m += macro("CEexGrad", bmat(PY["Q2"]["example"]["grad"], "{:.6f}"))
    b1 = PY["B1"]
    m += macro("BoneP", bmat(b1["P"], "{:.6f}")) + macro("BoneGrad", bmat(b1["grad"], "{:.6f}"))
    bb = PY["BenchB"]
    m += macro("BenchP", bmat(bb["P"], "{:.9f}")) + macro("BenchGZ", frac_mat(bb["GZ"])) + macro("BenchGW", frac_mat(bb["GW"]))
    m += macro("BenchWnew", bmat(bb["W_new"], "{:+.9f}")) + macro("BenchL", fix(bb["L"], 12))
    b3 = PY["B3_adam"]
    m += macro("BthreeThetaZero", bmat(b3["theta0"], "{:.6f}")) + macro("BthreeG", bmat(b3["g"], "{:.2f}"))
    m += macro("BthreeM", bmat(b3["m1"], "{:.4f}")) + macro("BthreeV", bmat(b3["v1"], "{:.7f}"))
    m += macro("BthreeThetaOne", bmat(b3["theta1"], "{:.6f}")) + macro("BthreeErr", sci(b3["err"]))
    write("matrices.tex", m)


def build_check_values():
    out = ["% measured value of every verification check (generated)"]
    ml = ml_checks()
    for c in PY["checks"]:
        out.append(rf"\expandafter\def\csname chkpy@{c['id']}\endcsname{{{sci(c['observed'])}}}")
        mv = sci(ml[c["id"]]["observed"]) if c["id"] in ml else DASH
        out.append(rf"\expandafter\def\csname chkml@{c['id']}\endcsname{{{mv}}}")
    write("checkvals.tex", "\n".join(out) + "\n")


def main():
    GEN.mkdir(parents=True, exist_ok=True)
    build_snippets()
    table_checks()
    table_params()
    table_fd()
    table_b2()
    table_acceptance()
    table_cases()
    table_norms()
    table_numbered()
    text_files()
    build_macros()
    build_matrices()
    build_check_values()
    assert PY["summary"]["passed"] == PY["summary"]["total"], "Python checks failing: fix before building"
    if ML:
        assert ML["summary"]["passed"] == ML["summary"]["total"], "MATLAB checks failing: fix before building"
    print("generated:", sorted(p.name for p in GEN.iterdir()), "| MATLAB results:", "yes" if ML else "not present")


if __name__ == "__main__":
    main()
