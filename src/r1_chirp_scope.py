"""r1_chirp_scope.py -- route 1 with a chirped probe: adiabatic first look.

The Rozanov bound depends only on the power spectrum of the pulse. A chirped
pulse keeps the spectrum of a few-cycle pulse but spreads it over hundreds of
femtoseconds, so the slow, measured response of ITO has time to follow it.
This script asks whether a thin ITO film on a perfect conductor, whose ENZ
dip returns to the blue as the electrons cool, can absorb more of such a
pulse than any static non-magnetic absorber of the same thickness may, at a fixed angle in
the glass (Rozanov bound B = 2 pi^2 n0 d / cos(theta) of a TM wave on a film
that conducts at zero frequency).

Probe: Gaussian intensity envelope of FWHM T, wavelength falling linearly in
time by `span` across the FWHM (red first), so the power spectrum is a
Gaussian of FWHM `span` centred on lam_c. Switched film: drop of wp from a
220 fs pump through the kernel of Tirole et al. (30 fs rise, 210 fs decay),
largest drop dmax. Static film: any fixed drop between 0 and dmax. Adiabatic
limit: each instant of the probe sees the static reflectance at its
instantaneous wavelength and the instantaneous drop.
"""
import json
import os
import numpy as np
from bound_scope import gamma_film_on_pec, rozanov_max_absorption
from chirp_scope import drop_profile, t
from ito_model import eps_ito

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
LAM = np.linspace(600e-9, 3200e-9, 2601)
DROPS = np.linspace(0, 0.30, 121)
PROFILES = {}                                   # drop profiles per largest drop, for PUMP_DELAYS
PUMP_DELAYS = np.arange(-1200e-15, 301e-15, 25e-15)


def rmap(d, deg):
    th = np.deg2rad(deg)
    return np.array([np.abs(gamma_film_on_pec(2.25, eps_ito(LAM, "Tirole2022", wp_scale=1 - x),
                                               d, LAM, th, "p")) ** 2 for x in DROPS])


def interp(R, lam, drop):
    i = np.clip((lam - LAM[0]) / (LAM[1] - LAM[0]), 0, len(LAM) - 1.001)
    j = np.clip(drop / (DROPS[1] - DROPS[0]), 0, len(DROPS) - 1.001)
    i0, j0 = i.astype(int), j.astype(int)
    fi, fj = i - i0, j - j0
    return ((1 - fi) * (1 - fj) * R[j0, i0] + fi * (1 - fj) * R[j0, i0 + 1]
            + (1 - fi) * fj * R[j0 + 1, i0] + fi * fj * R[j0 + 1, i0 + 1])


def spectrum(lam_c, span):
    return np.exp(-4 * np.log(2) * ((LAM - lam_c) / span) ** 2)


def eta_R(lam_c, span, B):
    S = spectrum(lam_c, span)
    return rozanov_max_absorption(S, LAM, B)[0]


def static_best(R, lam_c, span, dmax):
    S = spectrum(lam_c, span)
    vals = [(np.trapezoid(S * (1 - R[j]), LAM) / np.trapezoid(S, LAM), DROPS[j])
            for j in range(len(DROPS)) if DROPS[j] <= dmax + 1e-12]
    return max(vals)


def switched_best(R, lam_c, span, dmax, Ts=(150e-15, 300e-15, 450e-15, 600e-15, 900e-15), pumps=PUMP_DELAYS):
    best = (0, None, None)
    key = round(dmax, 6)
    if key not in PROFILES:
        PROFILES[key] = {tp: drop_profile(tp, dmax) for tp in pumps}
    profiles = PROFILES[key]
    for T in Ts:
        I = np.exp(-4 * np.log(2) * (t / T) ** 2)
        lam_t = lam_c - span * t / T
        norm = np.trapezoid(I, t)
        for tp, prof in profiles.items():
            a = np.trapezoid(I * (1 - interp(R, lam_t, prof)), t) / norm
            if a > best[0]:
                best = (float(a), float(T), float(tp))
    return best


if __name__ == "__main__":
    rows = []
    for d in (5e-9, 10e-9, 20e-9):
        for deg in (50.0, 60.0, 70.0):
            R = rmap(d, deg)
            B = 2 * np.pi ** 2 * 1.5 * d / np.cos(np.deg2rad(deg))
            for dmax in (0.10, 0.20, 0.30):
                best_row = None
                for span in (150e-9, 250e-9, 350e-9, 500e-9):
                    for lam_c in np.arange(1300e-9, 1801e-9, 50e-9):
                        eR = eta_R(lam_c, span, B)
                        st, st_drop = static_best(R, lam_c, span, dmax)
                        sw, T, tp = switched_best(R, lam_c, span, dmax)
                        row = dict(d=d, deg=deg, dmax=dmax, span=span, lam_c=lam_c, B=B, eta_R=eR,
                                   static=st, static_drop=st_drop, switched=sw, T=T, pump=tp)
                        rows.append(row)
                        if best_row is None or sw - eR > best_row["switched"] - best_row["eta_R"]:
                            best_row = row
                r = best_row
                print(f"d {d*1e9:4.0f} nm, {deg:4.0f} deg, B {B*1e9:5.0f} nm, depth {dmax:.2f}: largest margin "
                      f"{r['switched'] - r['eta_R']:+.3f} (switched {r['switched']:.3f}, limit {r['eta_R']:.3f}, "
                      f"best static {r['static']:.3f}) at span {r['span']*1e9:.0f} nm, centre "
                      f"{r['lam_c']*1e9:.0f} nm, FWHM {r['T']*1e15:.0f} fs, pump {r['pump']*1e15:+.0f} fs",
                      flush=True)
    with open(os.path.join(OUT, "r1_chirp_scope.json"), "w") as fh:
        json.dump(rows, fh)
    print("saved results/r1_chirp_scope.json")
