"""r1_real.py -- route 1 with a chirped probe, with realistic material choices.

The adiabatic design search of r1_design.py, repeated for five variants of the
film and its back mirror:

    pec_hot       perfect conductor; damping rises with the electron temperature
                  that produces the drop of wp (hot_electron.py)
    au_hot        gold back mirror (gold_fit.py) and the same rising damping
    au_fixed      gold back mirror, damping fixed (isolates the gold)
    pec_hot_g15   as pec_hot with 1.5 times the damping of the 40 nm film, a
                  stand-in for the extra surface scattering of thinner films
    pec_hot_bohn  as pec_hot with the Drude set of Bohn et al. (2021), the most
                  damped published set (gamma = 0.078 wp)

The Rozanov limit is that of a perfect conductor behind the film,
B = 2 pi^2 n0 d / cos(theta); with gold the integral of the static structure
diverges, so for gold the limit serves as the reference of an ideal mirror.
Films of 7.5 and 10 nm at 70 and 75 deg in the glass, largest drops of wp from
20 % to 40 %. Results go to results/r1_real.json.
"""
import json
import os
import time
import numpy as np
from scipy.optimize import minimize
from bound_scope import gamma_film_on_pec, rozanov_max_absorption
from chirp_scope import drop_profile, t
from ito_model import eps_ito, C0
from hot_electron import load_map
from gold_fit import eps_gold_fit, gamma_film_on_metal
from r1_design import sweep, spectrum_lam, allowed, LAM, N0, PUMPS
from runopts import Workers, n_workers

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DROPS = np.linspace(0, 0.40, 161)
CENTRES = np.arange(1350e-9, 1951e-9, 50e-9)
SPANS = np.arange(150e-9, 701e-9, 50e-9)
HOT = load_map(0.9)
VARIANTS = {
    "pec_hot": dict(ito="Tirole2022", g0=1.0, hot=True, back="pec"),
    "au_hot": dict(ito="Tirole2022", g0=1.0, hot=True, back="au"),
    "au_fixed": dict(ito="Tirole2022", g0=1.0, hot=False, back="au"),
    "pec_hot_g15": dict(ito="Tirole2022", g0=1.5, hot=True, back="pec"),
    "pec_hot_bohn": dict(ito="Bohn2021", g0=1.0, hot=True, back="pec"),
}


def gamma_scale(var, drop):
    """Damping scale at a drop of wp: gamma0 (1 + Te/T0) / (1 + 300 K/T0), with
    T0 = 15 000 K unless the variant sets another value."""
    v = VARIANTS[var]
    if not v["hot"]:
        return v["g0"]
    T0 = v.get("T0", 15000.0)
    Te = HOT.te_of_drop(drop)
    return v["g0"] * (1 + Te / T0) / (1 + 300.0 / T0)


def eps_film(var, lam, drop):
    v = VARIANTS[var]
    return eps_ito(lam, v["ito"], wp_scale=1 - drop, gamma_scale=gamma_scale(var, drop))


def rmap(var, d, deg):
    th = np.deg2rad(deg)
    eps_au = eps_gold_fit(LAM)
    rows = []
    for x in DROPS:
        e = eps_film(var, LAM, x)
        if VARIANTS[var]["back"] == "pec":
            G = gamma_film_on_pec(N0 ** 2, e, d, LAM, th, "p")
        else:
            G = gamma_film_on_metal(N0 ** 2, e, eps_au, d, LAM, th)
        rows.append(np.abs(G) ** 2)
    return np.array(rows)


def interp(R, lam, drop):
    i = np.clip((lam - LAM[0]) / (LAM[1] - LAM[0]), 0, len(LAM) - 1.001)
    j = np.clip(drop / (DROPS[1] - DROPS[0]), 0, len(DROPS) - 1.001)
    i0, j0 = i.astype(int), j.astype(int)
    fi, fj = i - i0, j - j0
    return ((1 - fi) * (1 - fj) * R[j0, i0] + fi * (1 - fj) * R[j0, i0 + 1]
            + (1 - fi) * fj * R[j0 + 1, i0] + fi * fj * R[j0 + 1, i0 + 1])


class Design:
    def __init__(self, var, d, deg, depth, tau_dec=210e-15):
        self.var, self.d, self.deg, self.depth, self.tau_dec = var, d, deg, depth, tau_dec
        self.R = rmap(var, d, deg)
        self.B = 2 * np.pi ** 2 * N0 * d / np.cos(np.deg2rad(deg))
        self.profiles = np.array([drop_profile(tp, depth, tau_dec=tau_dec) for tp in PUMPS])

    def eta_R(self, lam_c, span, T):
        S = spectrum_lam(lam_c, span, T)
        return rozanov_max_absorption(S / S.max(), LAM, self.B)[0]

    def static(self, lam_c, span, T):
        S = spectrum_lam(lam_c, span, T)
        jmax = int(round(self.depth / (DROPS[1] - DROPS[0])))
        vals = np.trapezoid(S[None, :] * (1 - self.R[:jmax + 1]), LAM, axis=1) / np.trapezoid(S, LAM)
        j = int(np.argmax(vals))
        return float(vals[j]), float(DROPS[j])

    def switched_all(self, lam_c, span, T):
        wc, dw = sweep(lam_c, span)
        I = np.exp(-4 * np.log(2) * (t / T) ** 2)
        lam_t = 2 * np.pi * C0 / np.maximum(wc + dw / T * t, 0.05 * wc)
        R = interp(self.R, np.broadcast_to(lam_t, self.profiles.shape), self.profiles)
        return np.trapezoid(I[None, :] * (1 - R), t, axis=1) / np.trapezoid(I, t)

    def switched_at(self, lam_c, span, T, tp):
        wc, dw = sweep(lam_c, span)
        I = np.exp(-4 * np.log(2) * (t / T) ** 2)
        lam_t = 2 * np.pi * C0 / np.maximum(wc + dw / T * t, 0.05 * wc)
        prof = drop_profile(tp, self.depth, tau_dec=self.tau_dec)
        return float(np.trapezoid(I * (1 - interp(self.R, lam_t, prof)), t) / np.trapezoid(I, t))


def search(args):
    var, d, deg, depth = args
    D = Design(var, d, deg, depth)
    best = None
    for lam_c in CENTRES:
        for span in SPANS:
            if not allowed(lam_c, span, 300e-15):
                continue
            for T in (200e-15, 300e-15, 450e-15, 600e-15):
                eR = D.eta_R(lam_c, span, T)
                sw = D.switched_all(lam_c, span, T)
                k = int(np.argmax(sw))
                if best is None or sw[k] - eR > best[0]:
                    best = (sw[k] - eR, lam_c, span, T, PUMPS[k])

    def neg(x):
        lam_c, span, T, tp = x[0] * 1e-9, x[1] * 1e-9, x[2] * 1e-15, x[3] * 1e-15
        if not allowed(lam_c, span, T):
            return 1.0
        return -(D.switched_at(lam_c, span, T, tp) - D.eta_R(lam_c, span, T))
    x0 = np.array([best[1] * 1e9, best[2] * 1e9, best[3] * 1e15, best[4] * 1e15])
    opt = minimize(neg, x0, method="Nelder-Mead", options=dict(xatol=1.0, fatol=1e-5, maxiter=400))
    lam_c, span, T, tp = opt.x[0] * 1e-9, opt.x[1] * 1e-9, opt.x[2] * 1e-15, opt.x[3] * 1e-15
    sw = D.switched_at(lam_c, span, T, tp)
    eR = D.eta_R(lam_c, span, T)
    st, st_drop = D.static(lam_c, span, T)
    return dict(var=var, d=d, deg=deg, depth=depth, B=D.B, lam_c=lam_c, span=span, T=T, pump=tp,
                switched=sw, eta_R=eR, static=st, static_drop=st_drop, margin=sw - eR,
                gamma_scale_max=float(gamma_scale(var, depth)))


if __name__ == "__main__":
    workers = n_workers(__doc__)
    t0 = time.time()
    jobs = [(var, d, deg, depth) for var in VARIANTS for d in (7.5e-9, 10e-9) for deg in (70.0, 75.0)
            for depth in (0.20, 0.25, 0.30, 0.35, 0.40)]
    with Workers(workers) as pool:
        rows = pool.map(search, jobs, chunksize=1)
    for r in rows:
        print(f"{r['var']:13s} d {r['d']*1e9:4.1f} nm, {r['deg']:.0f} deg, drop {r['depth']:.2f} (gamma x"
              f"{r['gamma_scale_max']:.2f}): margin {r['margin']:+.4f} (switched {r['switched']:.4f}, limit "
              f"{r['eta_R']:.4f}, static {r['static']:.4f}) centre {r['lam_c']*1e9:.0f} nm, span "
              f"{r['span']*1e9:.0f} nm, FWHM {r['T']*1e15:.0f} fs, pump {r['pump']*1e15:+.0f} fs")
    with open(os.path.join(OUT, "r1_real.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"saved results/r1_real.json ({time.time() - t0:.0f} s)")
