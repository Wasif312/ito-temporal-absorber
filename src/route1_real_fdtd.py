"""route1_real_fdtd.py -- full-wave check of route 1 with realistic material choices.

Designs from the adiabatic searches r1_real.py, r1_real_wide.py and
r1_damping.py, run in the fixed-angle FDTD of angle1d.py:
  - damping that rises with the electron temperature, perfect conductor: the
    designs closest to the Rozanov limit;
  - gold back mirror (eps_inf = 5 refit of Olmon et al.), damping fixed and
    rising;
  - half the damping rise (T0 = 30 000 K), perfect conductor;
For each design: the Rozanov limit of the perfect conductor from the FDTD
incident spectrum, the best static film (three fixed drops around the adiabatic
optimum, with the damping that belongs to each drop), the switched film (five
pump delays around the adiabatic optimum), the energy balance, and the share
of the absorbed energy taken by the gold. Results go to
results/route1_real_fdtd.json.
"""
import json
import os
import time
import numpy as np
from angle1d import AngleStack, simulate, energies
from fdtd1d import Medium, C0
from ito_model import ITO_SETS
from mirror import gaussian_probe, delta_profile
from slab_drude import spectrum
from bound_scope import rozanov_max_absorption
from gold_fit import params as au_params
import r1_real as rr
from r1_design import sweep, N0
from runopts import Workers, n_workers

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
Z_SRC = -2.5e-6
LAM_S = np.linspace(700e-9, 4000e-9, 3301)
for T0 in (30000.0, 60000.0, 120000.0):
    rr.VARIANTS.setdefault(f"pec_hot_T{int(T0/1000)}k",
                           dict(ito="Tirole2022", g0=1.0, hot=True, back="pec", T0=T0))


def make_stack(var, d):
    v = rr.VARIANTS[var]
    ito = ITO_SETS[v["ito"]]
    layers = [(Medium(ito["eps_inf"], 0), d)]
    species = [(ito["wp"], ito["gamma"])]
    if v["back"] == "au":
        au = au_params()
        layers.append((Medium(au["eps_inf"], 1), 200e-9))
        species.append((au["wp"], au["gamma"]))
    return AngleStack(N0, layers, species)


def setup(des):
    wc, dw = sweep(des["lam_c"], des["span"])
    T = des["T"]
    tc = 3 * T + 50e-15
    th = np.deg2rad(des["deg"])
    t_arr = tc + abs(Z_SRC) * N0 * np.cos(th) / C0
    return dict(th=th, probe=gaussian_probe(T, tc, w0=wc, chirp=dw / T), t_arr=t_arr,
                t_end=t_arr + 3 * T + 500e-15, h=min(1e-9, des["d"] / 10))


REF = {}


def get_ref(key, des):
    if key not in REF:
        s = setup(des)
        REF[key] = simulate(make_stack(des["var"], des["d"]), s["th"], s["probe"], s["t_end"],
                            h_fine=s["h"], z_src=Z_SRC, reference=True)
    return REF[key]


def one(job):
    tag, des, kind, x, mom = job
    s = setup(des)
    var = des["var"]
    nsp = 2 if rr.VARIANTS[var]["back"] == "au" else 1
    ref = get_ref(tag, des)
    if kind == "static":
        g = float(rr.gamma_scale(var, x))
        wp = [lambda t, x=x: np.full_like(np.asarray(t, float), 1 - x)]
        gs = [lambda t, g=g: np.full_like(np.asarray(t, float), g)]
    else:
        dfun = delta_profile(s["t_arr"] + x, des["depth"])
        wp = [lambda t: 1 + dfun(t)]
        gs = [lambda t: rr.gamma_scale(var, -dfun(t))]
    wp += [None] * (nsp - 1)
    gs += [None] * (nsp - 1)
    run = simulate(make_stack(var, des["d"]), s["th"], s["probe"], s["t_end"], h_fine=s["h"], z_src=Z_SRC,
                   wp_scale=wp, gamma_scale=gs, momentum=mom)
    ui, ur = energies(run, ref)
    W, D = run["wmod"].sum(), run["diss"]
    return dict(tag=tag, kind=kind, x=x, momentum=mom, eta=1 - ur / ui, Wmod=W / ui,
                diss_ito=D[0] / ui, diss_au=(D[1] / ui if nsp == 2 else 0.0),
                balance=(ui + W - ur - D.sum() - run["u_end"]) / ui)


def limit(tag, des):
    ref = get_ref(tag, des)
    dec = max(1, int(0.25e-15 / ref["dt"]))
    E = spectrum(ref["recE"][0], ref["dt"], C0 / LAM_S, decimate=dec)
    w = 2 * np.pi * C0 / LAM_S
    S = np.abs(E) ** 2 * w ** 2
    B = 2 * np.pi ** 2 * N0 * des["d"] / np.cos(np.deg2rad(des["deg"]))
    return float(rozanov_max_absorption(S / S.max(), LAM_S, B)[0])


def pick(rows, var, d, deg, depth):
    return [r for r in rows if r["var"] == var and abs(r["d"] - d) < 1e-12 and r["deg"] == deg
            and abs(r["depth"] - depth) < 1e-9][0]


if __name__ == "__main__":
    workers = n_workers(__doc__)
    t0 = time.time()
    real = json.load(open(os.path.join(OUT, "r1_real.json")))
    wide = json.load(open(os.path.join(OUT, "r1_real_wide.json")))
    damp = json.load(open(os.path.join(OUT, "r1_damping.json")))
    designs = {
        "pec_hot_5nm_80_40": pick(wide, "pec_hot", 5e-9, 80.0, 0.40),
        "pec_hot_7.5nm_75_40": pick(wide, "pec_hot", 7.5e-9, 75.0, 0.40),
        "au_fixed_10nm_70_40": pick(real, "au_fixed", 10e-9, 70.0, 0.40),
        "au_fixed_7.5nm_75_40": pick(real, "au_fixed", 7.5e-9, 75.0, 0.40),
        "au_hot_7.5nm_75_30": pick(real, "au_hot", 7.5e-9, 75.0, 0.30),
    }
    # half the damping rise of the hot-electron model (T0 = 30 000 K)
    designs["pec_hot_T30k_7.5nm_75_40"] = pick(damp, "pec_hot_T30k", 7.5e-9, 75.0, 0.40)
    designs["pec_hot_T30k_5nm_80_30"] = pick(damp, "pec_hot_T30k", 5e-9, 80.0, 0.30)
    jobs = []
    for tag, des in designs.items():
        for dd in (-0.01, 0.0, 0.01):
            jobs.append((tag, des, "static", float(np.clip(des["static_drop"] + dd, 0, des["depth"])), False))
        for do in (-50e-15, -25e-15, 0.0, 25e-15, 50e-15):
            jobs.append((tag, des, "switched", float(des["pump"] + do), False))
    with Workers(workers) as pool:
        res = pool.map(one, jobs, chunksize=1)
        extra = []
        for tag, des in designs.items():
            b = max([r for r in res if r["tag"] == tag and r["kind"] == "switched"], key=lambda r: r["eta"])
            extra.append((tag, des, "switched", b["x"], True))
        res += pool.map(one, extra, chunksize=1)
    summary = []
    for tag, des in designs.items():
        lim = limit(tag, des)
        rs = [r for r in res if r["tag"] == tag]
        bs = max([r for r in rs if r["kind"] == "static"], key=lambda r: r["eta"])
        bw = max([r for r in rs if r["kind"] == "switched" and not r["momentum"]], key=lambda r: r["eta"])
        bm = [r for r in rs if r["kind"] == "switched" and r["momentum"]][0]
        bal = max(abs(r["balance"]) for r in rs)
        summary.append(dict(tag=tag, var=des["var"], d=des["d"], deg=des["deg"], depth=des["depth"], eta_R=lim,
                            static=bs["eta"], switched=bw["eta"], momentum=bm["eta"], pump=bw["x"],
                            margin=bw["eta"] - lim, margin_adiabatic=des["margin"], Wmod=bw["Wmod"],
                            diss_au=bw["diss_au"], static_diss_au=bs["diss_au"], balance=bal,
                            lam_c=des["lam_c"], span=des["span"], T=des["T"]))
        print(f"{tag}: limit {lim:.4f}; static {bs['eta']:.4f} (gold {bs['diss_au']:.4f}); switched {bw['eta']:.4f} "
              f"(gold {bw['diss_au']:.4f}) at pump {bw['x']*1e15:+.0f} fs, momentum form {bm['eta']:.4f}; margin "
              f"{bw['eta'] - lim:+.4f} (adiabatic {des['margin']:+.4f}); W_mod {bw['Wmod']:+.4f}; largest |balance| "
              f"{bal:.1e}", flush=True)
    with open(os.path.join(OUT, "route1_real_fdtd.json"), "w") as fh:
        json.dump(dict(designs=designs, results=res, summary=summary), fh, indent=1)
    print(f"saved results/route1_real_fdtd.json ({time.time() - t0:.0f} s)")
