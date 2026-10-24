"""appendix_data.py -- tabulated field of a switched uniform medium.

Uniform ITO whose plasma frequency drops by 10 % at 20 fs (panel b of
fig_step3.py): FDTD field of the two switching forms against the exact
solution of the same mode, sampled every 2.5 fs from 10 to 45 fs for a table.
Results go to results/fig1b_trace.json.
"""
import json
import os
import numpy as np
from bulk import run_bulk, step_scale

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main():
    sc = step_scale(20e-15, 0.10)
    t_s = np.arange(10.0, 45.01, 2.5)
    out = dict(t_fs=t_s.tolist())
    for mom, name in ((False, "current"), (True, "momentum")):
        r = run_bulk(sc, mom, 0.0, courant=0.25, t_end=45e-15)
        t = r["t"] * 1e15
        idx = [int(np.argmin(np.abs(t - ts))) for ts in t_s]
        out[name] = dict(t_fs=[float(t[i]) for i in idx], E=[float(r["E"][i]) for i in idx],
                         E_exact=[float(r["E_exact"][i]) for i in idx],
                         max_err=float(np.max(np.abs(r["E"] - r["E_exact"]))))
        print(f"{name}: largest |E - E_exact| {out[name]['max_err']:.2e}")
    with open(os.path.join(OUT, "fig1b_trace.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("saved results/fig1b_trace.json")


if __name__ == "__main__":
    main()
