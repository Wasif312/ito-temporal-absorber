"""r1_real_wide.py -- wider search for the variant with rising damping.

With damping that rises with the electron temperature (variant pec_hot of
r1_real.py), no design at 70 and 75 deg beats the Rozanov limit. This search
checks that the conclusion does not come from the narrow window: films of 5 to
20 nm, angles of 65 to 80 deg, drops of 25, 30 and 40 %, and chirps up to
1000 nm across the FWHM. Results go to results/r1_real_wide.json.
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

if __name__ == "__main__":
    workers = n_workers(__doc__)
    t0 = time.time()
    jobs = [("pec_hot", d, deg, depth) for d in (5e-9, 7.5e-9, 10e-9, 15e-9, 20e-9)
            for deg in (65.0, 70.0, 75.0, 80.0) for depth in (0.25, 0.30, 0.40)]
    with Workers(workers) as pool:
        rows = pool.map(rr.search, jobs, chunksize=1)
    for r in rows:
        print(f"{r['var']} d {r['d']*1e9:4.1f} nm, {r['deg']:.0f} deg, drop {r['depth']:.2f}: "
              f"margin {r['margin']:+.4f} (switched {r['switched']:.4f}, limit {r['eta_R']:.4f}, "
              f"static {r['static']:.4f}) span {r['span']*1e9:.0f} nm, centre {r['lam_c']*1e9:.0f} nm")
    b = max(rows, key=lambda r: r["margin"])
    print(f"largest margin {b['margin']:+.4f} at d {b['d']*1e9:.1f} nm, {b['deg']:.0f} deg, drop {b['depth']:.2f}")
    with open(os.path.join(rr.OUT, "r1_real_wide.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"saved results/r1_real_wide.json ({time.time() - t0:.0f} s)")
