import json, signal, numpy as np, sympy as sp
from invariant_generator.feynman import parse_pmlb_metadata

def round_sig(expr, digits=6, zero=1e-6):
    reps = {}
    for f in expr.atoms(sp.Float):
        v = float(f)
        reps[f] = sp.Integer(0) if abs(v) < zero else sp.Float(float(f"{v:.{digits}g}"), 15)
    return expr.xreplace(reps)

def is_exact(cand, truth):
    cand = round_sig(cand, digits=15)  # drop |c| < 1e-6 only
    r = sp.simplify(sp.powsimp(cand / truth, force=True))
    r = sp.simplify(round_sig(r.evalf(), digits=6))
    return (not r.free_symbols) and r != 0, r

class T(Exception): pass
def handler(*_): raise T()
signal.signal(signal.SIGALRM, handler)
rows = json.load(open("results/feynman_benchmark/pilot/benchmark_summary.json"))
for row in rows:
    pr = parse_pmlb_metadata(f"data/feynman_tables/pmlb_metadata/{row['equation_id']}.yaml")
    syms = pr.symbols(); truth = pr.expression()
    for arm in ("pipeline", "baseline"):
        x = row.get(arm, {})
        if "expanded_equation" not in x: continue
        err = x["fresh_sample_relative_l2"]
        if not (err is not None and err < 1e-2): continue
        cand = sp.sympify(x["expanded_equation"], locals=syms)
        signal.alarm(60)
        try: ok, r = is_exact(cand, truth)
        except T: ok, r = None, "timeout"
        signal.alarm(0)
        print(f"{row['equation_id']:18s} {arm:8s} err={err:.1e} old={x['recovered_within_tolerance']!s:5s} new={ok!s:5s} ratio={str(r)[:60]}")
