"""r1_fixed_wide.py -- the fixed-damping reference for r1_damping.py.

Same designs and search window as r1_damping.py (chirps up to 1000 nm), with the
damping fixed, so that the margin can be read against the strength of the
damping rise from no rise to the hot-electron model. Results go to
results/r1_fixed_wide.json.
"""
import json
import os
import time
import numpy as np
import r1_design as rd
import r1_real as rr
from runopts import Workers, n_workers

rd.SPAN_MAX = 1000e-9
rr.SPANS = np.arange(150e-9, 1001e-9, 50e-9)
rr.CENTRES = np.arange(1350e-9, 2001e-9, 50e-9)
rr.VARIANTS["pec_fixed"] = dict(ito="Tirole2022", g0=1.0, hot=False, back="pec")

if __name__ == "__main__":
    workers = n_workers(__doc__)
    t0 = time.time()
    jobs = [("pec_fixed", d, deg, depth) for d, deg in ((5e-9, 80.0), (7.5e-9, 75.0), (10e-9, 70.0))
            for depth in (0.30, 0.40)]
    with Workers(workers) as pool:
        rows = pool.map(rr.search, jobs, chunksize=1)
    for r in rows:
        print(f"{r['var']} d {r['d']*1e9:4.1f} nm, {r['deg']:.0f} deg, drop {r['depth']:.2f}: "
              f"margin {r['margin']:+.4f} (switched {r['switched']:.4f}, limit {r['eta_R']:.4f}, "
              f"static {r['static']:.4f}) span {r['span']*1e9:.0f} nm, centre {r['lam_c']*1e9:.0f} nm")
    with open(os.path.join(rr.OUT, "r1_fixed_wide.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"saved results/r1_fixed_wide.json ({time.time() - t0:.0f} s)")
