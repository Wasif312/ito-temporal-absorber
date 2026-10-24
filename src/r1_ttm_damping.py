"""r1_ttm_damping.py -- margin against the change of the damping, with the
time-dependent hot-electron model.

The damping law gamma(Te) = gamma0 (T0 + Te)/(T0 + 300 K) of Baxter et al. has
T0 = 15 000 K, a damping that rises with the electron temperature. Gurung et
al. (Sci. Adv. 11, eadu8850 (2025)) instead report damping that falls as the
electrons heat. This script spans both signs: rising damping with T0 = 15 000,
30 000, 60 000 and 120 000 K, fixed damping, and falling damping with
T0 = -120 000, -60 000 and -40 000 K, for the 7.5 nm film at 75 deg and the
5 nm film at 80 deg, at largest drops of 30 % and 40 %, with the drop profiles
of the two-temperature model (r1_ttm.py). Results go to
results/r1_ttm_damping_<coupling>.json.

Usage: python3 r1_ttm_damping.py [--coupling baxter|sarkar] [--workers N]
"""
import json
import os
import time

import r1_ttm as m              # registers the variants pec_fixed and pec_T0_<k>k
import r1_real as rr
from runopts import Workers, parser

T0S = m.T0S
GEOMS = [(7.5e-9, 75.0), (5e-9, 80.0)]
DEPTHS = [0.30, 0.40]


def main():
    ap = parser(__doc__)
    ap.add_argument("--coupling", choices=["baxter", "sarkar"], default="baxter")
    args = ap.parse_args()
    t0 = time.time()
    m.gas()
    variants = ["pec_fixed"] + [f"pec_T0_{int(T0/1000)}k" for T0 in T0S]
    jobs = [(v, d, deg, depth, args.coupling) for v in variants for d, deg in GEOMS for depth in DEPTHS]
    with Workers(args.workers) as pool:
        profs = pool.map(m.profile_for, jobs, chunksize=1)
        rows = pool.map(m.search, [(p["var"], p["d"], p["deg"], p["depth"], p) for p in profs], chunksize=1)
    for p, r in zip(profs, rows, strict=True):
        g = float(rr.gamma_scale(r["var"], r["depth"]))
        r.update(gamma_scale_max=g, I_GWcm2=p["I_GWcm2"], fluence_mJcm2=p["fluence_mJcm2"], decay_fs=p["decay_fs"])
        print(f"{r['var']:13s} d {r['d']*1e9:4.1f} nm, {r['deg']:.0f} deg, drop {r['depth']:.2f}: damping x{g:.2f}, "
              f"pump {p['I_GWcm2']:.0f} GW/cm^2, decay {p['decay_fs']:.0f} fs; margin {r['margin']:+.4f} "
              f"(switched {r['switched']:.4f}, limit {r['eta_R']:.4f}, static {r['static']:.4f})", flush=True)
    with open(os.path.join(m.OUT, f"r1_ttm_damping_{args.coupling}.json"), "w") as fh:
        json.dump(dict(coupling=args.coupling, T0s=T0S, profiles=profs, designs=rows), fh, indent=1)
    print(f"saved results/r1_ttm_damping_{args.coupling}.json ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
