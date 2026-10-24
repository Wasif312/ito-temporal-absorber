"""route1_fdtd.py -- route 1 with a chirped probe, full-wave at a fixed angle.

For the designs found by r1_design.py (thin ITO film on a perfect conductor,
p-polarised light at a fixed angle in the glass, chirped probe, pump-induced
drop of wp), the fixed-angle FDTD of angle1d.py computes:
  - the incident spectrum and from it the Rozanov limit eta_R (water filling,
    bound_scope.py) with B = 2 pi^2 n0 d / cos(theta) for a conducting film,
    the bound that every static non-magnetic absorber of the same thickness
    obeys at this angle;
  - the best static film (fixed drop of wp around the adiabatic optimum);
  - the switched film against the pump delay, current-continuous form, and the
    momentum-continuous form at the best delay;
  - the energy balance of every run;
  - for one design, the same spectrum as a transform-limited pulse, to show
    what the chirp contributes.
Results go to results/route1_fdtd.json.
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
from r1_design import Design, sweep, N0
from runopts import Workers, n_workers

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ITO = ITO_SETS["Tirole2022"]
M_ITO = Medium(ITO["eps_inf"], 0)
SPECIES = [(ITO["wp"], ITO["gamma"])]
Z_SRC = -2.5e-6
LAM_S = np.linspace(700e-9, 4000e-9, 3301)


def film(d):
    return AngleStack(N0, [(M_ITO, d)], SPECIES)


def h_for(d):
    return min(1e-9, d / 10)


def setup(des, chirped=True):
    wc, dw = sweep(des["lam_c"], des["span"])
    if chirped:
        T, chirp = des["T"], dw / des["T"]
    else:
        T, chirp = 4 * np.log(2) / dw, 0.0
    tc = 3 * max(T, 20e-15) + 50e-15
    th = np.deg2rad(des["deg"])
    t_arr = tc + abs(Z_SRC) * N0 * np.cos(th) / C0
    t_end = t_arr + 3 * max(T, 20e-15) + 500e-15
    return dict(th=th, probe=gaussian_probe(T, tc, w0=wc, chirp=chirp), t_end=t_end, t_arr=t_arr, T=T)


REF = {}


def get_ref(key, des, chirped):
    if key not in REF:
        s = setup(des, chirped)
        REF[key] = simulate(film(des["d"]), s["th"], s["probe"], s["t_end"], h_fine=h_for(des["d"]),
                            z_src=Z_SRC, reference=True)
    return REF[key]


def one(job):
    tag, des, chirped, kind, x, mom = job
    s = setup(des, chirped)
    ref = get_ref((tag, chirped), des, chirped)
    if kind == "static":
        scale = [lambda t, x=x: np.full_like(np.asarray(t, float), 1 - x)]
    else:
        dfun = delta_profile(s["t_arr"] + x, des["depth"])
        scale = [lambda t: 1 + dfun(t)]
    run = simulate(film(des["d"]), s["th"], s["probe"], s["t_end"], h_fine=h_for(des["d"]), z_src=Z_SRC,
                   wp_scale=scale, momentum=mom)
    ui, ur = energies(run, ref)
    W, D = run["wmod"].sum(), run["diss"].sum()
    return dict(tag=tag, chirped=chirped, kind=kind, x=x, momentum=mom, eta=1 - ur / ui, Wmod=W / ui,
                diss=D / ui, balance=(ui + W - ur - D - run["u_end"]) / ui)


def limit_from_ref(tag, des, chirped):
    ref = get_ref((tag, chirped), des, chirped)
    dec = max(1, int(0.25e-15 / ref["dt"]))
    E = spectrum(ref["recE"][0], ref["dt"], C0 / LAM_S, decimate=dec)
    w = 2 * np.pi * C0 / LAM_S
    S = np.abs(E) ** 2 * w ** 2                  # per unit wavelength
    B = 2 * np.pi ** 2 * N0 * des["d"] / np.cos(np.deg2rad(des["deg"]))
    return float(rozanov_max_absorption(S / S.max(), LAM_S, B)[0])


if __name__ == "__main__":
    workers = n_workers(__doc__)
    t0 = time.time()
    rows = json.load(open(os.path.join(OUT, "r1_design.json")))
    picks = {}
    for depth in (0.10, 0.20, 0.30):
        cand = [r for r in rows if abs(r["depth"] - depth) < 1e-9 and r["d"] >= 7.4e-9]
        picks[f"best_{depth:.2f}"] = max(cand, key=lambda r: r["margin"])
        c10 = [r for r in cand if abs(r["d"] - 10e-9) < 1e-12]
        picks[f"d10_{depth:.2f}"] = max(c10, key=lambda r: r["margin"])
    # drop duplicates
    seen, designs = set(), {}
    for tag, des in picks.items():
        key = (des["d"], des["deg"], des["depth"])
        if key not in seen:
            seen.add(key)
            designs[tag] = des
    jobs = []
    for tag, des in designs.items():
        for dd in (-0.02, -0.01, 0.0, 0.01, 0.02):
            x = float(np.clip(des["static_drop"] + dd, 0, des["depth"]))
            jobs.append((tag, des, True, "static", x, False))
        for do in (-50e-15, -25e-15, 0.0, 25e-15, 50e-15):
            jobs.append((tag, des, True, "switched", float(des["pump"] + do), False))
    # transform-limited contrast for the best 30 % design
    tl_tag = "best_0.30"
    tl = designs[tl_tag]
    for off in np.arange(-400e-15, 100.1e-15, 10e-15):
        jobs.append((tl_tag, tl, False, "switched", float(off), False))
    for dd in (-0.02, 0.0, 0.02):
        jobs.append((tl_tag, tl, False, "static", float(np.clip(tl["static_drop"] + dd, 0, tl["depth"])), False))
    uniq = []
    for j in jobs:
        if not any(j[0] == u[0] and j[2:] == u[2:] for u in uniq):
            uniq.append(j)
    with Workers(workers) as pool:
        res = pool.map(one, uniq)
        extra = []
        for tag in designs:
            b = max([r for r in res if r["tag"] == tag and r["chirped"] and r["kind"] == "switched"],
                    key=lambda r: r["eta"])
            extra.append((tag, designs[tag], True, "switched", b["x"], True))
        res += pool.map(one, extra)
        # refine the transform-limited pump scan around its best coarse delay
        rt = [r for r in res if r["tag"] == tl_tag and not r["chirped"] and r["kind"] == "switched"]
        x0 = max(rt, key=lambda r: r["eta"])["x"]
        fine = [(tl_tag, tl, False, "switched", float(x0 + o), False)
                for o in np.arange(-8e-15, 8.1e-15, 2e-15) if abs(o) > 1e-18]
        res += pool.map(one, fine)
    lims = {tag: limit_from_ref(tag, des, True) for tag, des in designs.items()}
    lims_tl = limit_from_ref(tl_tag, tl, False)
    summary = []
    for tag, des in designs.items():
        rs = [r for r in res if r["tag"] == tag and r["chirped"]]
        bs = max([r for r in rs if r["kind"] == "static"], key=lambda r: r["eta"])
        bw = max([r for r in rs if r["kind"] == "switched" and not r["momentum"]], key=lambda r: r["eta"])
        bm = [r for r in rs if r["kind"] == "switched" and r["momentum"]][0]
        bal = max(abs(r["balance"]) for r in rs)
        D = Design(des["d"], des["deg"], des["depth"])
        ad_sw = D.switched_at(des["lam_c"], des["span"], des["T"], bw["x"])
        summary.append(dict(tag=tag, d=des["d"], deg=des["deg"], depth=des["depth"], lam_c=des["lam_c"],
                            span=des["span"], T=des["T"], eta_R=lims[tag], eta_R_adiabatic=des["eta_R"],
                            static=bs["eta"], static_drop=bs["x"], switched=bw["eta"], pump=bw["x"],
                            switched_adiabatic=ad_sw, momentum=bm["eta"], margin=bw["eta"] - lims[tag],
                            margin_momentum=bm["eta"] - lims[tag], Wmod=bw["Wmod"], balance=bal))
        print(f"{tag}: d {des['d']*1e9:.1f} nm, {des['deg']:.0f} deg, depth {des['depth']:.2f}, centre "
              f"{des['lam_c']*1e9:.0f} nm, span {des['span']*1e9:.0f} nm, FWHM {des['T']*1e15:.0f} fs: limit "
              f"{lims[tag]:.4f} (adiabatic search {des['eta_R']:.4f}); best static {bs['eta']:.4f}; switched "
              f"{bw['eta']:.4f} at pump {bw['x']*1e15:+.0f} fs (adiabatic {ad_sw:.4f}), momentum form "
              f"{bm['eta']:.4f}; margin {bw['eta'] - lims[tag]:+.4f}; W_mod {bw['Wmod']:+.4f}; "
              f"largest |balance| {bal:.1e}", flush=True)
    rt = [r for r in res if r["tag"] == tl_tag and not r["chirped"]]
    tl_sw = max([r for r in rt if r["kind"] == "switched"], key=lambda r: r["eta"])
    tl_st = max([r for r in rt if r["kind"] == "static"], key=lambda r: r["eta"])
    print(f"transform-limited pulse with the spectrum of {tl_tag}: limit {lims_tl:.4f}, best static "
          f"{tl_st['eta']:.4f}, best switched {tl_sw['eta']:.4f} at pump {tl_sw['x']*1e15:+.0f} fs")
    out = dict(designs=designs, results=res, summary=summary, limits=lims,
               transform_limited=dict(tag=tl_tag, limit=lims_tl, static=tl_st["eta"], switched=tl_sw["eta"],
                                      pump=tl_sw["x"], rows=rt))
    with open(os.path.join(OUT, "route1_fdtd.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"saved results/route1_fdtd.json ({time.time() - t0:.0f} s)")
