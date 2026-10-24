"""route3_angles.py -- the chirp-matched absorber at prism angles in full-wave FDTD.

The adiabatic scan of angle_scan.py shows that both films absorb more beyond
the 37.17 deg of the Tirole et al. geometry, at angles that a prism reaches.
This script tests three angles inside the glass, 45, 50 and 57 deg, at the
in-plane wavevector of the 1370 nm centre wavelength. Probe: Gaussian intensity
envelope with a linear chirp of 150 nm across its FWHM, red first. Static
film: a fixed drop of wp. Switched film: the pump-induced drop of Tirole et al.
(220 fs pump, 30 fs rise, 210 fs decay, largest drop 10 %).

At each angle the adiabatic estimate at the same in-plane wavevector picks the
probe FWHM (300, 450 or 600 fs), the static drop and the pump delay; the FDTD
then scans the static drop and the pump delay around those values, and checks
the momentum-continuous form at the best delay. Results go to
results/route3_angles.json.
"""
import json
import os
import time
import numpy as np
from fdtd1d import simulate, energies, C0
from mirror import stack, gaussian_probe, delta_profile
from ito_model import eps_ito, eps_gold, tmm
from runopts import Workers, n_workers

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
LC = 1370e-9
WC = 2 * np.pi * C0 / LC
DW = 2 * np.pi * C0 * (1 / 1295e-9 - 1 / 1445e-9)          # sweep across the FWHM
DMAX = 0.10
Z_SRC = -2.0e-6
ANGLES = (45.0, 50.0, 57.0)
T_CHOICES = (300e-15, 450e-15, 600e-15)


def setup(deg, T):
    kx = 2 * np.pi * 1.5 / LC * np.sin(np.deg2rad(deg))
    v_z = C0 / 1.5 * np.cos(np.deg2rad(deg))
    tc = 3 * T
    return dict(kx=kx, T=T, chirp=DW / T, tc=tc, t_end=tc + 3 * T + 150e-15,
                t_arr=tc + abs(Z_SRC) / v_z)


def adiabatic(deg):
    """Instantaneous-frequency estimate at the in-plane wavevector of the
    centre wavelength. Returns the best static drop, and for each probe FWHM
    the best pump delay, with the absorbed fractions."""
    kx = 2 * np.pi * 1.5 / LC * np.sin(np.deg2rad(deg))
    lam_g = np.linspace(1150e-9, 1650e-9, 201)
    s = kx * lam_g / (2 * np.pi * 1.5)
    lam_g = lam_g[s < 0.995]
    dl = np.linspace(0, DMAX, 81)
    Rm = np.array([[tmm([2.25, eps_ito(wl, "Tirole2022", wp_scale=1 - x), eps_gold(wl), 1.0],
                        [40e-9, 100e-9], wl, np.arcsin(kx * wl / (2 * np.pi * 1.5)), "p")[0]
                    for wl in lam_g] for x in dl])

    def R_of(lam, drop):
        i = np.clip((lam - lam_g[0]) / (lam_g[1] - lam_g[0]), 0, len(lam_g) - 1.001)
        j = np.clip(drop / (dl[1] - dl[0]), 0, len(dl) - 1.001)
        i0, j0 = i.astype(int), j.astype(int)
        fi, fj = i - i0, j - j0
        return ((1 - fi) * (1 - fj) * Rm[j0, i0] + fi * (1 - fj) * Rm[j0, i0 + 1]
                + (1 - fi) * fj * Rm[j0 + 1, i0] + fi * fj * Rm[j0 + 1, i0 + 1])
    out = dict(static={}, switched={})
    for T in T_CHOICES:
        t = np.arange(-4 * T, 4 * T, 1e-15)
        I = np.exp(-4 * np.log(2) * (t / T) ** 2)
        lam_t = 2 * np.pi * C0 / (WC + DW / T * t)

        def eta(drop_t, I=I, lam_t=lam_t, t=t):
            return float(np.trapezoid(I * (1 - R_of(lam_t, drop_t)), t) / np.trapezoid(I, t))
        st = max((eta(np.full_like(t, x)), float(x)) for x in dl)
        sw = max((eta(-delta_profile(off, DMAX)(t)), float(off))
                 for off in np.arange(-900e-15, 301e-15, 25e-15))
        out["static"][T] = st
        out["switched"][T] = sw
    return out


REF = {}


def one(job):
    deg, T, kind, x, mom = job
    s = setup(deg, T)
    probe = gaussian_probe(T, s["tc"], w0=WC, chirp=s["chirp"])
    key = (deg, T)
    if key not in REF:
        REF[key] = simulate(stack(), s["kx"], probe, s["t_end"], reference=True)
    if kind == "static":
        scale = [lambda t, x=x: np.full_like(np.asarray(t, float), 1 - x), None]
    else:
        dfun = delta_profile(s["t_arr"] + x, DMAX)
        scale = [lambda t: 1 + dfun(t), None]
    run = simulate(stack(), s["kx"], probe, s["t_end"], wp_scale=scale, momentum=mom)
    U_inc, U_r, U_t = energies(run, REF[key])
    W, Dd = run["wmod"].sum(), run["diss"].sum()
    return dict(deg=deg, T=T, kind=kind, x=x, momentum=mom, eta=1 - U_r / U_inc, Wmod=W / U_inc,
                diss=Dd / U_inc, T_tr=U_t / U_inc,
                balance=(U_inc + W - U_r - U_t - Dd - run["u_end"]) / U_inc)


if __name__ == "__main__":
    workers = n_workers(__doc__)
    t0 = time.time()
    ad = {deg: adiabatic(deg) for deg in ANGLES}
    jobs, plan = [], {}
    for deg in ANGLES:
        a = ad[deg]
        # the switched film picks its probe FWHM; the static film gets the same probe
        T = max(T_CHOICES, key=lambda T: a["switched"][T][0])
        st_drop = a["static"][T][1]
        sw_off = a["switched"][T][1]
        plan[deg] = dict(T=T, static_ad=a["static"][T], switched_ad=a["switched"][T],
                         all_static=[(float(k), v) for k, v in a["static"].items()],
                         all_switched=[(float(k), v) for k, v in a["switched"].items()])
        print(f"{deg:4.1f} deg: adiabatic picks FWHM {T*1e15:.0f} fs; static {a['static'][T][0]:.3f} at "
              f"drop {st_drop*100:.2f} %, switched {a['switched'][T][0]:.3f} at pump {sw_off*1e15:+.0f} fs",
              flush=True)
        for dd in np.arange(-0.015, 0.0151, 0.005):
            x = float(np.clip(st_drop + dd, 0, DMAX))
            jobs.append((deg, T, "static", x, False))
        for do in np.arange(-100e-15, 100.1e-15, 25e-15):
            jobs.append((deg, T, "switched", float(sw_off + do), False))
    jobs = list(dict.fromkeys(jobs))
    with Workers(workers) as pool:
        res = pool.map(one, jobs)
        extra = []
        for deg in ANGLES:
            b = max([r for r in res if r["deg"] == deg and r["kind"] == "switched"], key=lambda r: r["eta"])
            extra.append((deg, b["T"], "switched", b["x"], True))
        res += pool.map(one, extra)
    summary = []
    for deg in ANGLES:
        rs = [r for r in res if r["deg"] == deg]
        bs = max([r for r in rs if r["kind"] == "static"], key=lambda r: r["eta"])
        bw = max([r for r in rs if r["kind"] == "switched" and not r["momentum"]], key=lambda r: r["eta"])
        bm = [r for r in rs if r["kind"] == "switched" and r["momentum"]][0]
        bal = max(abs(r["balance"]) for r in rs)
        summary.append(dict(deg=deg, T=bs["T"], static=bs["eta"], static_drop=bs["x"], switched=bw["eta"],
                            pump=bw["x"], momentum=bm["eta"], gain=bw["eta"] - bs["eta"],
                            Wmod=bw["Wmod"], Wmod_mom=bm["Wmod"], balance=bal,
                            static_ad=plan[deg]["static_ad"][0], switched_ad=plan[deg]["switched_ad"][0]))
        print(f"{deg:4.1f} deg, FWHM {bs['T']*1e15:.0f} fs: best static {bs['eta']:.3f} (drop {bs['x']*100:.2f} %; "
              f"adiabatic {plan[deg]['static_ad'][0]:.3f}); switched {bw['eta']:.3f} at pump "
              f"{bw['x']*1e15:+.0f} fs (adiabatic {plan[deg]['switched_ad'][0]:.3f}); momentum form "
              f"{bm['eta']:.3f}; gain {bw['eta'] - bs['eta']:+.3f}; W_mod {bw['Wmod']:+.4f}; "
              f"largest |balance| {bal:.1e}")
    with open(os.path.join(OUT, "route3_angles.json"), "w") as fh:
        json.dump(dict(results=res, summary=summary,
                       plan={str(k): v for k, v in plan.items()}), fh, indent=1)
    print(f"saved results/route3_angles.json ({time.time() - t0:.0f} s)")
