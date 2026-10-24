"""r1_threshold.py -- route 1 with a chirped probe: the depth at which the bound breaks.

Adiabatic design search (r1_design.search) for 7.5 nm and 10 nm films at 70 and
75 deg in the glass, for drops of wp from 20 % to 30 %, with the decay time of
Tirole et al. (210 fs) and of Alam et al. (360 fs). Results go to
results/r1_threshold.json.
"""
import json
import os
import time
from r1_design import search, OUT
from runopts import Workers, n_workers


def job(a):
    return search(*a)


if __name__ == "__main__":
    workers = n_workers(__doc__)
    t0 = time.time()
    jobs = [(d, deg, depth, tau)
            for tau in (210e-15, 360e-15)
            for d in (7.5e-9, 10e-9)
            for deg in (70.0, 75.0)
            for depth in (0.20, 0.225, 0.25, 0.275, 0.30)]
    with Workers(workers) as pool:
        rows = pool.map(job, jobs)
    for r in rows:
        print(f"decay {r['tau_dec']*1e15:.0f} fs, d {r['d']*1e9:4.1f} nm, {r['deg']:.0f} deg, depth {r['depth']:.3f}: "
              f"margin {r['margin']:+.4f} (switched {r['switched']:.4f}, limit {r['eta_R']:.4f}, static "
              f"{r['static']:.4f}), centre {r['lam_c']*1e9:.0f} nm, span {r['span']*1e9:.0f} nm, "
              f"FWHM {r['T']*1e15:.0f} fs, pump {r['pump']*1e15:+.0f} fs")
    with open(os.path.join(OUT, "r1_threshold.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"saved results/r1_threshold.json ({time.time() - t0:.0f} s)")
