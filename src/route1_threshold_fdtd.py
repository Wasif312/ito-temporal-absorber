"""route1_threshold_fdtd.py -- full-wave margin over the Rozanov limit near the
threshold depth.

Designs from results/r1_threshold.json (decay time 210 fs): the 7.5 nm and
10 nm films at 75 deg and the 10 nm film at 70 deg, at drops of wp between
22.5 % and 27.5 %. For each: the limit from the FDTD incident spectrum, the best
static film (three fixed drops around the adiabatic optimum) and the switched
film (five pump delays around the adiabatic optimum). Results go to
results/route1_threshold_fdtd.json.
"""
import json
import os
import time
import numpy as np
from route1_fdtd import one, limit_from_ref, OUT
from runopts import Workers, n_workers

if __name__ == "__main__":
    workers = n_workers(__doc__)
    t0 = time.time()
    rows = json.load(open(os.path.join(OUT, "r1_threshold.json")))
    want = [(7.5e-9, 75.0, 0.225), (7.5e-9, 75.0, 0.25), (7.5e-9, 75.0, 0.275),
            (10e-9, 75.0, 0.225), (10e-9, 75.0, 0.25), (10e-9, 75.0, 0.275),
            (10e-9, 70.0, 0.25), (10e-9, 70.0, 0.275)]
    designs = {}
    for d, deg, depth in want:
        r = [x for x in rows if abs(x["tau_dec"] - 210e-15) < 1e-18 and abs(x["d"] - d) < 1e-12
             and x["deg"] == deg and abs(x["depth"] - depth) < 1e-9][0]
        designs[f"d{d*1e9:g}_{deg:g}_{depth:g}"] = r
    jobs = []
    for tag, des in designs.items():
        for dd in (-0.01, 0.0, 0.01):
            jobs.append((tag, des, True, "static", float(np.clip(des["static_drop"] + dd, 0, des["depth"])), False))
        for do in (-50e-15, -25e-15, 0.0, 25e-15, 50e-15):
            jobs.append((tag, des, True, "switched", float(des["pump"] + do), False))
    with Workers(workers) as pool:
        res = pool.map(one, jobs)
    summary = []
    for tag, des in designs.items():
        lim = limit_from_ref(tag, des, True)
        rs = [r for r in res if r["tag"] == tag]
        bs = max([r for r in rs if r["kind"] == "static"], key=lambda r: r["eta"])
        bw = max([r for r in rs if r["kind"] == "switched"], key=lambda r: r["eta"])
        bal = max(abs(r["balance"]) for r in rs)
        summary.append(dict(tag=tag, d=des["d"], deg=des["deg"], depth=des["depth"], eta_R=lim,
                            static=bs["eta"], switched=bw["eta"], pump=bw["x"], margin=bw["eta"] - lim,
                            margin_adiabatic=des["margin"], Wmod=bw["Wmod"], balance=bal))
        print(f"{tag}: limit {lim:.4f}, static {bs['eta']:.4f}, switched {bw['eta']:.4f} at pump "
              f"{bw['x']*1e15:+.0f} fs; margin {bw['eta'] - lim:+.4f} (adiabatic {des['margin']:+.4f}); "
              f"W_mod {bw['Wmod']:+.4f}; largest |balance| {bal:.1e}", flush=True)
    with open(os.path.join(OUT, "route1_threshold_fdtd.json"), "w") as fh:
        json.dump(dict(designs=designs, results=res, summary=summary), fh, indent=1)
    print(f"saved results/route1_threshold_fdtd.json ({time.time() - t0:.0f} s)")
