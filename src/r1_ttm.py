"""r1_ttm.py -- route 1 with the time-dependent hot-electron model of ttm.py.

The searches of r1_design.py and r1_real.py describe the pump-induced drop of
wp by the kernel of Tirole et al. (30 fs rise, 210 fs decay) at every pump
strength, and tie the damping to the drop through the electron temperature.
Here the two-temperature model gives the time course itself. A 220 fs pump,
p-polarised, at the probe angle (so that it sweeps along the film with the
probe footprint) and at the wavelength where the cold film absorbs most,
heats the electrons; the drop follows Te(t), and the electrons cool through
the electron-phonon coupling. The pump absorption follows the instantaneous
permittivity, so the film absorbs less as it heats.

For each film, angle and largest drop the script finds the pump intensity
that gives that drop, stores the drop profile, and repeats the adiabatic design
search of r1_real.py (perfect conductor behind the film) with fixed damping
and with the damping of Baxter et al. The coupling g_ep(Te) of Baxter et al. is
the default; --coupling sarkar uses the constant G = 3e16 W m^-3 K^-1 of
Sarkar, Un and Sivan. Results go to results/r1_ttm_<coupling>.json.

Usage: python3 r1_ttm.py [--coupling baxter|sarkar] [--workers N]
"""
import json
import os
import time
import numpy as np
from scipy.optimize import minimize

import ttm
import r1_design as rd
import r1_real as rr
from bound_scope import gamma_film_on_pec
from ito_model import eps_ito
from runopts import Workers, parser

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
GW = 1e13
FWHM_PUMP = 220e-15
T_RUN = np.arange(-4 * FWHM_PUMP, 2500e-15, 0.5e-15)
GEOMS = [(5e-9, 80.0), (7.5e-9, 75.0), (10e-9, 70.0), (10e-9, 75.0)]
DEPTHS = [0.20, 0.25, 0.30, 0.35, 0.40]
VARS = ["pec_fixed", "pec_hot"]
rd.SPAN_MAX = 1000e-9
rr.VARIANTS["pec_fixed"] = dict(ito="Tirole2022", g0=1.0, hot=False, back="pec")
# damping laws gamma0 (T0 + Te)/(T0 + 300 K): rising (T0 > 0) and falling (T0 < 0), used by r1_ttm_damping.py
T0S = [15000.0, 30000.0, 60000.0, 120000.0, -120000.0, -60000.0, -40000.0]
for _T0 in T0S:
    rr.VARIANTS[f"pec_T0_{int(_T0 / 1000)}k"] = dict(ito="Tirole2022", g0=1.0, hot=True, back="pec", T0=_T0)
CENTRES = np.arange(1350e-9, 2001e-9, 50e-9)
SPANS = np.arange(150e-9, 1001e-9, 50e-9)
DURATIONS = (200e-15, 300e-15, 450e-15, 600e-15, 800e-15)
G_SARKAR = 3e16
GAS = None


def gas():
    global GAS
    if GAS is None:
        GAS = ttm.load_gas(0.9)
    return GAS


def pump_setup(var, d, deg):
    """Pump wavelength (cold absorption maximum) and the absorbed fraction of
    the film against Te, for the damping law of the variant."""
    th = np.deg2rad(deg)
    lam = np.linspace(1000e-9, 2200e-9, 481)
    A_cold = 1 - np.abs(gamma_film_on_pec(2.25, eps_ito(lam, "Tirole2022"), d, lam, th, "p")) ** 2
    lam_p = float(lam[np.argmax(A_cold)])
    g = gas()
    Ts = np.geomspace(300.0, 59000.0, 200)
    hot = rr.VARIANTS[var]["hot"]
    T0 = rr.VARIANTS[var].get("T0", 15000.0)
    A = []
    for T in Ts:
        e = eps_ito(np.array([lam_p]), "Tirole2022", wp_scale=np.sqrt(g.wp2_ratio(T)),
                    gamma_scale=max(float(ttm.gamma_baxter(T, T0)), 0.05) if hot else 1.0)
        A.append(1 - abs(gamma_film_on_pec(2.25, e, d, np.array([lam_p]), th, "p")[0]) ** 2)
    A = np.array(A)
    return lam_p, (lambda T: np.interp(T, Ts, A)), float(A[0])


def profile_for(args):
    """Pump intensity for a largest drop `depth`, and the drop profile."""
    var, d, deg, depth, coupling = args
    lam_p, absorb, A0 = pump_setup(var, d, deg)
    G = G_SARKAR if coupling == "sarkar" else None
    g = gas()
    th = np.deg2rad(deg)

    def run(logI):
        return ttm.run(g, absorb, ttm.gaussian_pulse(10 ** logI * GW, FWHM_PUMP, 0.0), d, np.cos(th), T_RUN,
                       G_const=G)
    lo, hi = -1.0, 4.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if run(mid)["drop"].max() > depth:
            hi = mid
        else:
            lo = mid
        if hi - lo < 2e-4:
            break
    r = run(0.5 * (lo + hi))
    I0 = 10 ** (0.5 * (lo + hi))
    return dict(var=var, d=d, deg=deg, depth=depth, coupling=coupling, lam_pump=lam_p, A_cold=A0, I_GWcm2=I0,
                fluence_mJcm2=float(np.sqrt(np.pi / (4 * np.log(2))) * I0 * GW * FWHM_PUMP * 0.1),
                Te_max=float(r["Te"].max()), drop_max=float(r["drop"].max()),
                rise_fs=ttm.rise_time(T_RUN, r["drop"]) * 1e15, decay_fs=ttm.decay_time(T_RUN, r["drop"]) * 1e15,
                t=T_RUN[::4].tolist(), drop=r["drop"][::4].tolist(), Te=r["Te"][::4].tolist())


class TTMDesign(rr.Design):
    """r1_real.Design with the drop profile of the two-temperature model."""

    def __init__(self, var, d, deg, depth, prof):
        self.var, self.d, self.deg, self.depth = var, d, deg, depth
        self.R = rr.rmap(var, d, deg)
        self.B = 2 * np.pi ** 2 * rd.N0 * d / np.cos(np.deg2rad(deg))
        self.pt, self.pd = np.array(prof["t"]), np.array(prof["drop"])
        self.profiles = np.array([self.profile(tp) for tp in rd.PUMPS])

    def profile(self, tp):
        return np.interp(rd.t - tp, self.pt, self.pd, left=0.0, right=self.pd[-1])

    def switched_at(self, lam_c, span, T, tp):
        wc, dw = rd.sweep(lam_c, span)
        I = np.exp(-4 * np.log(2) * (rd.t / T) ** 2)
        lam_t = 2 * np.pi * rd.C0 / np.maximum(wc + dw / T * rd.t, 0.05 * wc)
        return float(np.trapezoid(I * (1 - rr.interp(self.R, lam_t, self.profile(tp))), rd.t)
                     / np.trapezoid(I, rd.t))


def search(args):
    var, d, deg, depth, prof = args
    D = TTMDesign(var, d, deg, depth, prof)
    best = None
    for lam_c in CENTRES:
        for span in SPANS:
            if not rd.allowed(lam_c, span, 300e-15):
                continue
            for T in DURATIONS:
                eR = D.eta_R(lam_c, span, T)
                sw = D.switched_all(lam_c, span, T)
                k = int(np.argmax(sw))
                if best is None or sw[k] - eR > best[0]:
                    best = (sw[k] - eR, lam_c, span, T, rd.PUMPS[k])

    def neg(x):
        lam_c, span, T, tp = x[0] * 1e-9, x[1] * 1e-9, x[2] * 1e-15, x[3] * 1e-15
        if not rd.allowed(lam_c, span, T):
            return 1.0
        return -(D.switched_at(lam_c, span, T, tp) - D.eta_R(lam_c, span, T))
    x0 = np.array([best[1] * 1e9, best[2] * 1e9, best[3] * 1e15, best[4] * 1e15])
    opt = minimize(neg, x0, method="Nelder-Mead", options=dict(xatol=1.0, fatol=1e-5, maxiter=400))
    lam_c, span, T, tp = opt.x[0] * 1e-9, opt.x[1] * 1e-9, opt.x[2] * 1e-15, opt.x[3] * 1e-15
    sw = D.switched_at(lam_c, span, T, tp)
    eR = D.eta_R(lam_c, span, T)
    st, st_drop = D.static(lam_c, span, T)
    return dict(var=var, d=d, deg=deg, depth=depth, B=D.B, lam_c=lam_c, span=span, T=T, pump=tp, switched=sw,
                eta_R=eR, static=st, static_drop=st_drop, margin=sw - eR, coarse_margin=float(best[0]))


def main():
    ap = parser(__doc__)
    ap.add_argument("--coupling", choices=["baxter", "sarkar"], default="baxter")
    args = ap.parse_args()
    t0 = time.time()
    gas()                     # build or load the electron tables once, before the workers start
    jobs = [(v, d, deg, depth, args.coupling) for v in VARS for d, deg in GEOMS for depth in DEPTHS]
    with Workers(args.workers) as pool:
        profs = pool.map(profile_for, jobs, chunksize=1)
        for p in profs:
            print(f"{p['var']:9s} d {p['d']*1e9:4.1f} nm, {p['deg']:.0f} deg, drop {p['depth']:.2f}: pump "
                  f"{p['lam_pump']*1e9:.0f} nm (cold absorption {p['A_cold']:.3f}), {p['I_GWcm2']:.1f} GW/cm^2, "
                  f"{p['fluence_mJcm2']:.1f} mJ/cm^2, Te {p['Te_max']:.0f} K, rise {p['rise_fs']:.0f} fs, decay "
                  f"{p['decay_fs']:.0f} fs", flush=True)
        rows = pool.map(search, [(p["var"], p["d"], p["deg"], p["depth"], p) for p in profs], chunksize=1)
    for r in rows:
        print(f"{r['var']:9s} d {r['d']*1e9:4.1f} nm, {r['deg']:.0f} deg, drop {r['depth']:.2f}: margin "
              f"{r['margin']:+.4f} (switched {r['switched']:.4f}, limit {r['eta_R']:.4f}, static {r['static']:.4f}), "
              f"sweep {(r['lam_c']+r['span']/2)*1e9:.0f} to {(r['lam_c']-r['span']/2)*1e9:.0f} nm, FWHM "
              f"{r['T']*1e15:.0f} fs, pump {r['pump']*1e15:+.0f} fs", flush=True)
    with open(os.path.join(OUT, f"r1_ttm_{args.coupling}.json"), "w") as fh:
        json.dump(dict(coupling=args.coupling, profiles=profs, designs=rows), fh, indent=1)
    print(f"saved results/r1_ttm_{args.coupling}.json ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
