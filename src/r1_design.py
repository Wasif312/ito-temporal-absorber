"""r1_design.py -- route 1 with a chirped probe: adiabatic design search.

Thin ITO film (Tirole et al. 2022 Drude set) on a perfect conductor, p-polarised
light at a fixed angle theta in the glass (n0 = 1.5). The probe has a Gaussian
intensity envelope of FWHM T and a frequency that rises linearly in time
(red first), sweeping the wavelength from lam_c + span/2 to lam_c - span/2
across the FWHM, as the FDTD source of mirror.gaussian_probe does. The pump
lowers wp through the kernel of Tirole et al. (220 fs pump, 30 fs rise,
210 fs decay), by at most `depth`.

For each film thickness d, angle theta and depth, the search maximises the
margin of the switched film over the Rozanov limit of the same spectrum,
eta_sw - eta_R, over the centre wavelength, the span, the probe FWHM and the
pump delay: a coarse grid, then a Nelder-Mead refinement. It also reports the
best static film (any fixed drop between 0 and depth) for the same spectrum.
B = 2 pi^2 n0 d / cos(theta), the Rozanov bound of a TM wave on a conducting film.
Films of 5 and 7.5 nm are an outlook: the Drude set of a 40 nm film may not
hold for them. Results go to results/r1_design.json.
"""
import json
import os
import time
import numpy as np
from scipy.optimize import minimize
from bound_scope import gamma_film_on_pec, rozanov_max_absorption
from chirp_scope import drop_profile, t
from ito_model import eps_ito, C0

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
LAM = np.linspace(600e-9, 3600e-9, 3001)
DROPS = np.linspace(0, 0.30, 121)
PUMPS = np.arange(-1000e-15, 201e-15, 25e-15)
N0 = 1.5


def rmap(d, deg):
    th = np.deg2rad(deg)
    return np.array([np.abs(gamma_film_on_pec(N0 ** 2, eps_ito(LAM, "Tirole2022", wp_scale=1 - x),
                                               d, LAM, th, "p")) ** 2 for x in DROPS])


def interp(R, lam, drop):
    i = np.clip((lam - LAM[0]) / (LAM[1] - LAM[0]), 0, len(LAM) - 1.001)
    j = np.clip(drop / (DROPS[1] - DROPS[0]), 0, len(DROPS) - 1.001)
    i0, j0 = i.astype(int), j.astype(int)
    fi, fj = i - i0, j - j0
    return ((1 - fi) * (1 - fj) * R[j0, i0] + fi * (1 - fj) * R[j0, i0 + 1]
            + (1 - fi) * fj * R[j0 + 1, i0] + fi * fj * R[j0 + 1, i0 + 1])


def sweep(lam_c, span):
    """Centre angular frequency and its sweep across the FWHM."""
    w_red, w_blue = 2 * np.pi * C0 / (lam_c + span / 2), 2 * np.pi * C0 / (lam_c - span / 2)
    return 0.5 * (w_red + w_blue), w_blue - w_red


def spectrum_lam(lam_c, span, T):
    """Power spectrum per unit wavelength of the linearly chirped Gaussian."""
    wc, dw = sweep(lam_c, span)
    fw = np.sqrt((4 * np.log(2) / T) ** 2 + dw ** 2)
    w = 2 * np.pi * C0 / LAM
    Sw = np.exp(-4 * np.log(2) * ((w - wc) / fw) ** 2)
    return Sw * w ** 2                    # |dw/dlam| = 2 pi c / lam^2 = w^2 / (2 pi c)


class Design:
    def __init__(self, d, deg, depth, tau_dec=210e-15):
        self.d, self.deg, self.depth, self.tau_dec = d, deg, depth, tau_dec
        self.R = rmap(d, deg)
        self.B = 2 * np.pi ** 2 * N0 * d / np.cos(np.deg2rad(deg))
        self.profiles = np.array([drop_profile(tp, depth, tau_dec=tau_dec) for tp in PUMPS])
        self.cache = {}

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
        """Absorbed fraction for every pump delay on the grid."""
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


SPAN_MAX, LAM_MIN, LAM_MAX = 700e-9, 1000e-9, 2400e-9


def allowed(lam_c, span, T):
    """Search limits: spans up to 700 nm, sweep inside 1000-2400 nm across the
    FWHM, probe FWHM 100-900 fs."""
    return (150e-9 <= span <= SPAN_MAX and lam_c - span / 2 >= LAM_MIN and lam_c + span / 2 <= LAM_MAX
            and 100e-15 <= T <= 900e-15)


def search(d, deg, depth, tau_dec=210e-15):
    D = Design(d, deg, depth, tau_dec)
    best = None
    for lam_c in np.arange(1350e-9, 1901e-9, 50e-9):
        for span in np.arange(150e-9, 701e-9, 50e-9):
            if not allowed(lam_c, span, 300e-15):
                continue
            for T in (200e-15, 300e-15, 450e-15, 600e-15):
                eR = D.eta_R(lam_c, span, T)
                sw = D.switched_all(lam_c, span, T)
                k = int(np.argmax(sw))
                m = sw[k] - eR
                if best is None or m > best[0]:
                    best = (m, lam_c, span, T, PUMPS[k])

    def neg(x):
        lam_c, span, T, tp = x[0] * 1e-9, x[1] * 1e-9, x[2] * 1e-15, x[3] * 1e-15
        if not allowed(lam_c, span, T):
            return 1.0
        return -(D.switched_at(lam_c, span, T, tp) - D.eta_R(lam_c, span, T))
    x0 = np.array([best[1] * 1e9, best[2] * 1e9, best[3] * 1e15, best[4] * 1e15])
    opt = minimize(neg, x0, method="Nelder-Mead",
                   options=dict(xatol=1.0, fatol=1e-5, maxiter=400, initial_simplex=None))
    lam_c, span, T, tp = opt.x[0] * 1e-9, opt.x[1] * 1e-9, opt.x[2] * 1e-15, opt.x[3] * 1e-15
    sw = D.switched_at(lam_c, span, T, tp)
    eR = D.eta_R(lam_c, span, T)
    st, st_drop = D.static(lam_c, span, T)
    return dict(d=d, deg=deg, depth=depth, tau_dec=tau_dec, B=D.B, lam_c=lam_c, span=span, T=T, pump=tp,
                switched=sw, eta_R=eR, static=st, static_drop=st_drop, margin=sw - eR,
                coarse_margin=float(best[0]))


if __name__ == "__main__":
    t0 = time.time()
    rows = []
    for d in (5e-9, 7.5e-9, 10e-9, 15e-9, 20e-9):
        for deg in (60.0, 65.0, 70.0, 75.0, 80.0):
            for depth in (0.10, 0.20, 0.30):
                r = search(d, deg, depth)
                rows.append(r)
                print(f"d {d*1e9:4.1f} nm, {deg:4.0f} deg, depth {depth:.2f}: margin {r['margin']:+.4f} "
                      f"(switched {r['switched']:.4f}, limit {r['eta_R']:.4f}, static {r['static']:.4f}) "
                      f"centre {r['lam_c']*1e9:.0f} nm, span {r['span']*1e9:.0f} nm, FWHM {r['T']*1e15:.0f} fs, "
                      f"pump {r['pump']*1e15:+.0f} fs", flush=True)
    with open(os.path.join(OUT, "r1_design.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    for depth in (0.10, 0.20, 0.30):
        b = max([r for r in rows if r["depth"] == depth], key=lambda r: r["margin"])
        print(f"depth {depth:.2f}: best margin {b['margin']:+.4f} at d {b['d']*1e9:.1f} nm, {b['deg']:.0f} deg "
              f"(switched {b['switched']:.4f}, limit {b['eta_R']:.4f}, static {b['static']:.4f})")
    print(f"saved results/r1_design.json ({time.time() - t0:.0f} s)")
