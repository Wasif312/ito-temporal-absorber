"""route1_ttm_fdtd.py -- full-wave check of route 1 with the time-dependent
hot-electron model.

Designs from r1_ttm.py (drop profiles of the two-temperature model, perfect
conductor behind the film), run in the fixed-angle FDTD of angle1d.py. For each
design: the Rozanov limit from the incident spectrum, the best static film
(three fixed drops around the adiabatic optimum, with the damping that belongs
to each drop), the switched film at five pump delays around the adiabatic
optimum (current-continuous form), the momentum-continuous form at the best
delay, and the energy balance. Results go to
results/route1_ttm_fdtd_<coupling><tag>.json.

Usage: python3 route1_ttm_fdtd.py [--coupling baxter|sarkar] [--workers N]
           [--source r1_ttm|r1_ttm_damping] [--pick VAR,D_NM,DEG,DEPTH ...] [--tag TEXT]
"""
import json
import os
import time
import numpy as np

from angle1d import simulate, energies
import route1_real_fdtd as base
import r1_real as rr
from r1_ttm import OUT          # importing r1_ttm registers the variant pec_fixed
from runopts import Workers, parser

PICKS = [("pec_fixed", 7.5e-9, 75.0, 0.25), ("pec_fixed", 7.5e-9, 75.0, 0.30), ("pec_fixed", 10e-9, 70.0, 0.30),
         ("pec_hot", 5e-9, 80.0, 0.40), ("pec_hot", 7.5e-9, 75.0, 0.40)]


def profile_fun(prof, t_pump):
    tp, dp = np.array(prof["t"]), np.array(prof["drop"])
    return lambda t: np.interp(np.asarray(t, float) - t_pump, tp, dp, left=0.0, right=dp[-1])


def one(job):
    tag, des, prof, kind, x, mom = job
    s = base.setup(des)
    var = des["var"]
    if kind == "limit":
        return dict(tag=tag, kind=kind, eta_R=base.limit(tag, des))
    ref = base.get_ref(tag, des)
    if kind == "static":
        g = float(rr.gamma_scale(var, x))
        wp = [lambda t, x=x: np.full_like(np.asarray(t, float), 1 - x)]
        gs = [lambda t, g=g: np.full_like(np.asarray(t, float), g)]
    else:
        dfun = profile_fun(prof, s["t_arr"] + x)
        wp = [lambda t: 1 - dfun(t)]
        gs = [lambda t: rr.gamma_scale(var, dfun(t))]
    run = simulate(base.make_stack(var, des["d"]), s["th"], s["probe"], s["t_end"], h_fine=s["h"], z_src=base.Z_SRC,
                   wp_scale=wp, gamma_scale=gs, momentum=mom)
    ui, ur = energies(run, ref)
    W, D = run["wmod"].sum(), run["diss"].sum()
    return dict(tag=tag, kind=kind, x=x, momentum=mom, eta=1 - ur / ui, Wmod=W / ui,
                balance=(ui + W - ur - D - run["u_end"]) / ui)


def same_design(r, var, d, deg, depth):
    return r["var"] == var and abs(r["d"] - d) < 1e-12 and r["deg"] == deg and abs(r["depth"] - depth) < 1e-9


def parse_pick(text):
    var, d_nm, deg, depth = text.split(",")
    return var, float(d_nm) * 1e-9, float(deg), float(depth)


def main():
    ap = parser(__doc__)
    ap.add_argument("--coupling", choices=["baxter", "sarkar"], default="baxter")
    ap.add_argument("--source", default="r1_ttm", choices=["r1_ttm", "r1_ttm_damping"],
                    help="adiabatic search whose designs are checked (results/<source>_<coupling>.json)")
    ap.add_argument("--pick", action="append", type=parse_pick, metavar="VAR,D_NM,DEG,DEPTH",
                    help="design to check, e.g. pec_fixed,7.5,75,0.30 (repeatable; default: the five designs of PICKS)")
    ap.add_argument("--tag", default="", help="suffix of the output file name")
    args = ap.parse_args()
    t0 = time.time()
    with open(os.path.join(OUT, f"{args.source}_{args.coupling}.json")) as fh:
        data = json.load(fh)
    designs, profs = {}, {}
    for pick in (args.pick or PICKS):
        var, d, deg, depth = pick
        tag = f"{var}_{d*1e9:g}nm_{deg:.0f}_{int(round(depth*100))}"
        designs[tag] = [r for r in data["designs"] if same_design(r, *pick)][0]
        profs[tag] = [p for p in data["profiles"] if same_design(p, *pick)][0]
    jobs = []
    for tag, des in designs.items():
        jobs.append((tag, des, profs[tag], "limit", 0.0, False))
        for dd in (-0.01, 0.0, 0.01):
            x = float(np.clip(des["static_drop"] + dd, 0, des["depth"]))
            jobs.append((tag, des, profs[tag], "static", x, False))
        for do in (-50e-15, -25e-15, 0.0, 25e-15, 50e-15):
            jobs.append((tag, des, profs[tag], "switched", float(des["pump"] + do), False))
    with Workers(args.workers) as pool:
        res = pool.map(one, jobs, chunksize=1)
        extra = []
        for tag, des in designs.items():
            b = max([r for r in res if r["tag"] == tag and r["kind"] == "switched"], key=lambda r: r["eta"])
            extra.append((tag, des, profs[tag], "switched", b["x"], True))
        res += pool.map(one, extra, chunksize=1)
    summary = []
    for tag, des in designs.items():
        rs = [r for r in res if r["tag"] == tag]
        lim = [r for r in rs if r["kind"] == "limit"][0]["eta_R"]
        rs = [r for r in rs if r["kind"] != "limit"]
        bs = max([r for r in rs if r["kind"] == "static"], key=lambda r: r["eta"])
        bw = max([r for r in rs if r["kind"] == "switched" and not r["momentum"]], key=lambda r: r["eta"])
        bm = [r for r in rs if r["kind"] == "switched" and r["momentum"]][0]
        bal = max(abs(r["balance"]) for r in rs)
        summary.append(dict(tag=tag, var=des["var"], d=des["d"], deg=des["deg"], depth=des["depth"], eta_R=lim,
                            static=bs["eta"], switched=bw["eta"], momentum=bm["eta"], pump=bw["x"],
                            margin=bw["eta"] - lim, margin_adiabatic=des["margin"], Wmod=bw["Wmod"], balance=bal,
                            lam_c=des["lam_c"], span=des["span"], T=des["T"],
                            I_GWcm2=profs[tag]["I_GWcm2"], decay_fs=profs[tag]["decay_fs"]))
        print(f"{tag}: limit {lim:.4f}; static {bs['eta']:.4f}; switched {bw['eta']:.4f} at pump "
              f"{bw['x']*1e15:+.0f} fs, momentum form {bm['eta']:.4f}; margin {bw['eta'] - lim:+.4f} (adiabatic "
              f"{des['margin']:+.4f}); W_mod {bw['Wmod']:+.4f}; largest |balance| {bal:.1e}", flush=True)
    name = f"route1_ttm_fdtd_{args.coupling}{args.tag}.json"
    with open(os.path.join(OUT, name), "w") as fh:
        json.dump(dict(coupling=args.coupling, source=args.source, designs=designs, results=res, summary=summary),
                  fh, indent=1)
    print(f"saved results/{name} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
