"""convergence.py -- grid and time-step convergence of the key full-wave results.

Three designs, each at the pump delay and static drop that the production runs
found best, recomputed on finer grids and with a smaller time step:

  route1      7.5 nm ITO film at 75 deg on a perfect conductor, 30 % drop,
              chirped probe (route1_fdtd.py, design best_0.30)
  route3      ITO/Au mirror of Tirole et al. at 37.17 deg in the glass, 10 %
              drop, chirped probe (route3_fdtd.py)
  realistic   5 nm ITO film at 80 deg on a perfect conductor, 40 % drop,
              damping that rises with the electron temperature
              (route1_real_fdtd.py, design pec_hot_5nm_80_40)

Resolutions: the base grid of the production runs, the film cells halved
and quartered, the time step halved, and the cells outside the film halved.
For each resolution the script records the absorbed fraction of the best
static film, of the switched film (current-continuous form), the Rozanov limit
from the incident spectrum (route 1 designs), the margin and the largest
energy-balance residual. Results go to results/convergence.json. Each run is
saved to results/convergence_partial.json as it finishes, so an interrupted
study resumes where it stopped.

Usage: python3 convergence.py [--workers N] [--heavy-workers M] [--restart]
"""
import inspect
import json
import os
import time
import numpy as np

import fdtd1d
import angle1d
from mirror import stack, delta_profile
import route3_fdtd as r3
import route1_fdtd as r1
import route1_real_fdtd as rr_fd
import r1_real as rr
from runopts import Workers, parser

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CHECKPOINT = os.path.join(OUT, "convergence_partial.json")

# (label, refinement of the film cells, Courant number, refinement of the cells
# outside the film). The film cells set the time step. For route 3 the cells of
# the glass and the gold follow the film cells (10 nm / refine); for route 1 they
# keep the 20 nm of the production runs unless the last factor refines them.
LEVELS = [("base", 1, 0.5, 1), ("cells/2", 2, 0.5, 1), ("cells/4", 4, 0.5, 1), ("dt/2", 1, 0.25, 1),
          ("glass/2", 1, 0.5, 2)]
H_COARSE_ANGLE = inspect.signature(angle1d.simulate).parameters["h_coarse"].default    # 20 nm


def load(name):
    with open(os.path.join(OUT, name)) as fh:
        return json.load(fh)


def key(r):
    """Identity of a run: case, resolution level and static or switched film."""
    return (r["case"], r["level"], r["kind"])


def job_key(job):
    """key() of the run that a job will produce."""
    family, a = job
    return (a[0], a[1], a[5]) if family == "route1" else ("route3", a[0], a[4])


def job_cost(job):
    """Relative run time: cells times steps, the route-1 designs being longer."""
    family, a = job
    refine, courant, coarse = (a[2], a[3], a[4]) if family == "route1" else (a[1], a[2], a[3])
    return refine ** 2 * 0.5 / courant * coarse * (2.0 if family == "route1" else 1.0)


def is_heavy(job):
    family, a = job
    return family == "route1" and a[2] == max(lev[1] for lev in LEVELS)


def any_job(job):
    family, a = job
    return route1_job(a) if family == "route1" else route3_job(a)


# ------------------------------------------------------------------ route 3
def route3_job(args):
    level, refine, courant, coarse, kind, x = args
    kw = dict(h_fine=1e-9 / refine, h_coarse=10e-9 / (refine * coarse), courant=courant)
    ref = fdtd1d.simulate(stack(), r3.KX, r3.PROBE, r3.T_END, reference=True, **kw)
    if kind == "static":
        scale = [lambda t, x=x: np.full_like(np.asarray(t, float), 1 - x), None]
    else:
        dfun = delta_profile(r3.T_ARR + x, r3.DMAX)
        scale = [lambda t: 1 + dfun(t), None]
    run = fdtd1d.simulate(stack(), r3.KX, r3.PROBE, r3.T_END, wp_scale=scale, **kw)
    ui, ur, ut = fdtd1d.energies(run, ref)
    W, D = run["wmod"].sum(), run["diss"].sum()
    return dict(case="route3", level=level, kind=kind, x=x, eta=1 - ur / ui,
                balance=(ui + W - ur - ut - D - run["u_end"]) / ui)


# ------------------------------------------------------------------ route 1 (fixed or rising damping)
def route1_job(args):
    case, level, refine, courant, coarse, kind, x, des = args
    if case == "route1":
        s = r1.setup(des, True)
        stack_ = r1.film(des["d"])
        h = r1.h_for(des["d"]) / refine
        gs = None
        if kind == "static":
            wp = [lambda t, x=x: np.full_like(np.asarray(t, float), 1 - x)]
        else:
            dfun = delta_profile(s["t_arr"] + x, des["depth"])
            wp = [lambda t: 1 + dfun(t)]
    else:
        s = rr_fd.setup(des)
        stack_ = rr_fd.make_stack(des["var"], des["d"])
        h = s["h"] / refine
        var = des["var"]
        if kind == "static":
            g = float(rr.gamma_scale(var, x))
            wp = [lambda t, x=x: np.full_like(np.asarray(t, float), 1 - x)]
            gs = [lambda t, g=g: np.full_like(np.asarray(t, float), g)]
        else:
            dfun = delta_profile(s["t_arr"] + x, des["depth"])
            wp = [lambda t: 1 + dfun(t)]
            gs = [lambda t: rr.gamma_scale(var, -dfun(t))]
    common = dict(h_fine=h, h_coarse=H_COARSE_ANGLE / coarse, z_src=r1.Z_SRC, courant=courant)
    ref = angle1d.simulate(stack_, s["th"], s["probe"], s["t_end"], reference=True, **common)
    run = angle1d.simulate(stack_, s["th"], s["probe"], s["t_end"], wp_scale=wp, gamma_scale=gs, **common)
    ui, ur = angle1d.energies(run, ref)
    W, D = run["wmod"].sum(), run["diss"].sum()
    out = dict(case=case, level=level, kind=kind, x=x, eta=1 - ur / ui,
               balance=(ui + W - ur - D - run["u_end"]) / ui)
    if kind == "static":
        # Rozanov limit from the incident spectrum of this resolution
        dec = max(1, int(0.25e-15 / ref["dt"]))
        E = r1.spectrum(ref["recE"][0], ref["dt"], r1.C0 / r1.LAM_S, decimate=dec)
        w = 2 * np.pi * r1.C0 / r1.LAM_S
        S = np.abs(E) ** 2 * w ** 2
        B = 2 * np.pi ** 2 * r1.N0 * des["d"] / np.cos(np.deg2rad(des["deg"]))
        out["eta_R"] = float(r1.rozanov_max_absorption(S / S.max(), r1.LAM_S, B)[0])
    return out


def main():
    ap = parser(__doc__)
    ap.add_argument("--heavy-workers", type=int, default=1,
                    help="worker processes for the finest fixed-angle grid, about 3 GB each (default 1)")
    ap.add_argument("--restart", action="store_true", help="ignore the runs saved by an interrupted run")
    args = ap.parse_args()
    t0 = time.time()

    # production optima
    r3res = load("route3.json")["results"]
    r3_static = max([r for r in r3res if r["kind"] == "static"], key=lambda r: r["eta"])
    r3_switch = max([r for r in r3res if r["kind"] == "switched" and not r["momentum"]], key=lambda r: r["eta"])
    r1f = load("route1_fdtd.json")
    d1 = r1f["designs"]["best_0.30"]
    s1 = [s for s in r1f["summary"] if s["tag"] == "best_0.30"][0]
    rf = load("route1_real_fdtd.json")
    d2 = rf["designs"]["pec_hot_5nm_80_40"]
    s2 = [s for s in rf["summary"] if s["tag"] == "pec_hot_5nm_80_40"][0]
    static2 = max([r for r in rf["results"] if r["tag"] == "pec_hot_5nm_80_40" and r["kind"] == "static"],
                  key=lambda r: r["eta"])["x"]
    s1_static = max([r for r in r1f["results"] if r["tag"] == "best_0.30" and r["chirped"]
                     and r["kind"] == "static"], key=lambda r: r["eta"])["x"]

    jobs3, jobs1 = [], []
    for level, refine, courant, coarse in LEVELS:
        jobs3.append((level, refine, courant, coarse, "static", r3_static["x"]))
        jobs3.append((level, refine, courant, coarse, "switched", r3_switch["x"]))
        jobs1.append(("route1", level, refine, courant, coarse, "static", s1_static, d1))
        jobs1.append(("route1", level, refine, courant, coarse, "switched", s1["pump"], d1))
        jobs1.append(("realistic", level, refine, courant, coarse, "static", static2, d2))
        jobs1.append(("realistic", level, refine, courant, coarse, "switched", s2["pump"], d2))
    # Every finished run goes to a checkpoint file at once, and a restarted
    # script skips the runs it already holds (--restart discards them).
    done = {}
    if os.path.exists(CHECKPOINT) and not args.restart:
        with open(CHECKPOINT) as fh:
            done = {key(r): r for r in json.load(fh)}
        print(f"{len(done)} runs read from {os.path.relpath(CHECKPOINT, os.path.dirname(OUT))}", flush=True)

    # The finest fixed-angle grid holds about 3 GB per run: at most
    # --heavy-workers of those run at once, while the other workers take the
    # remaining runs, longest first.
    jobs = [("route1", j) for j in jobs1] + [("route3", j) for j in jobs3]
    todo = sorted([j for j in jobs if job_key(j) not in done], key=job_cost, reverse=True)
    with Workers(args.workers) as pool:
        for _, r in pool.imap_limited(any_job, todo, heavy=is_heavy, max_heavy=args.heavy_workers):
            done[key(r)] = r
            tmp = CHECKPOINT + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(list(done.values()), fh)
            os.replace(tmp, CHECKPOINT)
            print(f"   {r['case']:9s} {r['level']:8s} {r['kind']:8s} absorbed {r['eta']:.5f} "
                  f"({len(done)} of {len(jobs)} runs, {time.time() - t0:.0f} s)", flush=True)
    res = list(done.values())

    summary = []
    for case in ("route1", "realistic", "route3"):
        for level, refine, courant, coarse in LEVELS:
            st = [r for r in res if r["case"] == case and r["level"] == level and r["kind"] == "static"][0]
            sw = [r for r in res if r["case"] == case and r["level"] == level and r["kind"] == "switched"][0]
            row = dict(case=case, level=level, refine=refine, courant=courant, coarse=coarse, static=st["eta"],
                       switched=sw["eta"], balance=max(abs(st["balance"]), abs(sw["balance"])))
            if "eta_R" in st:
                row["eta_R"] = st["eta_R"]
                row["margin"] = sw["eta"] - st["eta_R"]
            summary.append(row)
            extra = f", limit {row['eta_R']:.5f}, margin {row['margin']:+.5f}" if "eta_R" in row else ""
            print(f"{case:9s} {level:8s} static {row['static']:.5f}, switched {row['switched']:.5f}{extra}, "
                  f"largest |balance| {row['balance']:.1e}", flush=True)
    with open(os.path.join(OUT, "convergence.json"), "w") as fh:
        json.dump(dict(levels=LEVELS, results=res, summary=summary), fh, indent=1)
    if os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)
    print(f"saved results/convergence.json ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
