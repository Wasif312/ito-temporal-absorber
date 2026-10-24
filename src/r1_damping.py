"""r1_damping.py -- how far the damping may rise before the margin over the bound vanishes.

The hot-electron damping gamma0 (1 + Te/T0) of Baxter et al. uses T0 = 15 000 K.
This search repeats the route-1 design search (perfect conductor) for weaker
rises, T0 = 30 000, 60 000 and 120 000 K, for the designs closest to the bound
in r1_real_wide.py (5 nm at 80 deg, 7.5 nm at 75 deg) and for the 10 nm film
at 70 deg, at drops of 30 % and 40 %. Results go to results/r1_damping.json.
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
for T0 in (30000.0, 60000.0, 120000.0):
    rr.VARIANTS[f"pec_hot_T{int(T0/1000)}k"] = dict(ito="Tirole2022", g0=1.0, hot=True, back="pec", T0=T0)

if __name__ == "__main__":
    workers = n_workers(__doc__)
    t0 = time.time()
    jobs = [(var, d, deg, depth) for var in ("pec_hot_T30k", "pec_hot_T60k", "pec_hot_T120k")
            for d, deg in ((5e-9, 80.0), (7.5e-9, 75.0), (10e-9, 70.0)) for depth in (0.30, 0.40)]
    with Workers(workers) as pool:
        rows = pool.map(rr.search, jobs, chunksize=1)
    for r in rows:
        print(f"{r['var']:14s} d {r['d']*1e9:4.1f} nm, {r['deg']:.0f} deg, drop {r['depth']:.2f} (gamma x"
              f"{r['gamma_scale_max']:.2f}): margin {r['margin']:+.4f} (switched {r['switched']:.4f}, limit "
              f"{r['eta_R']:.4f}, static {r['static']:.4f}) span {r['span']*1e9:.0f} nm, "
              f"centre {r['lam_c']*1e9:.0f} nm")
    with open(os.path.join(rr.OUT, "r1_damping.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"saved results/r1_damping.json ({time.time() - t0:.0f} s)")
