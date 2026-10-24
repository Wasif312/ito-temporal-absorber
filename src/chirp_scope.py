"""chirp_scope.py -- adiabatic first look at a resonance that follows a chirp.

The pump red-shifts the ENZ absorption dip of the ITO/Au mirror of Tirole et
al. (2022), and the dip returns as the electrons cool (decay time 210 fs). A
probe pulse whose wavelength falls in time (red first, blue later) can follow
the returning dip. This script estimates, in the adiabatic limit, how much of
such a chirped pulse the switched film absorbs, and compares the result with
the best static film, which may sit at any fixed shift of the plasma frequency
between 0 and the calibrated 10 %.

Adiabatic limit: the pulse is long compared with the optical cycle (4.3 fs)
and with the storage time of the film (about 5 fs), and each instant of a
strongly chirped pulse carries one wavelength. Non-adiabatic effects
(frequency conversion at the time interface) are left to the FDTD stage.
"""
import numpy as np
from ito_model import eps_ito, eps_gold, tmm, delta_wp_from_pump

TH = np.arcsin(np.sin(np.deg2rad(65)) / 1.5)
D = [40e-9, 100e-9]
DMAX = 0.10                                     # calibrated drop of omega_p

lam = np.linspace(1100e-9, 1700e-9, 241)
dl = np.linspace(0.0, DMAX, 41)                 # drop of omega_p, 0 ... 10 %
Rmap = np.array([[tmm([2.25, eps_ito(wl, "Tirole2022", wp_scale=1 - x), eps_gold(wl), 1.0],
                      D, wl, TH, "p")[0] for wl in lam] for x in dl])


def R_of(lam_t, drop_t):
    """Bilinear interpolation of R(lambda, drop)."""
    i = np.clip((lam_t - lam[0]) / (lam[1] - lam[0]), 0, len(lam) - 1.001)
    j = np.clip(drop_t / (dl[1] - dl[0]), 0, len(dl) - 1.001)
    i0, j0 = i.astype(int), j.astype(int)
    fi, fj = i - i0, j - j0
    return ((1 - fi) * (1 - fj) * Rmap[j0, i0] + fi * (1 - fj) * Rmap[j0, i0 + 1]
            + (1 - fi) * fj * Rmap[j0 + 1, i0] + fi * fj * Rmap[j0 + 1, i0 + 1])


t = np.arange(-1500e-15, 2500e-15, 1e-15)


def drop_profile(t_pump, depth, tau_rise=30e-15, tau_dec=210e-15):
    pump = np.exp(-4 * np.log(2) * ((t - t_pump) / 220e-15) ** 2)
    d = np.abs(delta_wp_from_pump(t, pump, tau_rise=tau_rise, tau_dec=tau_dec))
    return d / d.max() * depth


def absorbed(lam_c, span, T, drop_t):
    """Chirped Gaussian probe: intensity FWHM T, wavelength falling linearly
    by `span` across the FWHM, centred on lam_c at t = 0."""
    I = np.exp(-4 * np.log(2) * (t / T) ** 2)
    lam_t = lam_c - span * t / T
    return np.trapezoid(I * (1 - R_of(lam_t, drop_t)), t) / np.trapezoid(I, t)


if __name__ == "__main__":
    # the same chirped probe for both films: the static film may sit at any
    # fixed drop between 0 and 10 %, the switched film at any pump delay
    for tau_dec in (210e-15, 400e-15):
        print(f"\ndecay time {tau_dec*1e15:.0f} fs")
        for span in (100e-9, 150e-9, 200e-9):
            rows = []
            for lam_c in np.arange(1250e-9, 1501e-9, 10e-9):
                a_s = max(absorbed(lam_c, span, 300e-15, np.full_like(t, x)) for x in dl)
                for T in (150e-15, 300e-15, 450e-15, 600e-15):
                    a_v, tp = max((absorbed(lam_c, span, T, drop_profile(tp, DMAX, tau_dec=tau_dec)), tp)
                                  for tp in np.arange(-900e-15, 301e-15, 25e-15))
                    rows.append((a_v - a_s, a_v, a_s, lam_c, T, tp))
            g = max(rows)
            b = max(rows, key=lambda r: r[1])
            print(f"   span {span*1e9:3.0f} nm: largest gain {g[0]:+.3f} "
                  f"(switched {g[1]:.3f} vs static {g[2]:.3f}; centre {g[3]*1e9:.0f} nm, "
                  f"T {g[4]*1e15:.0f} fs, pump at {g[5]*1e15:.0f} fs)")
            print(f"                best switched {b[1]:.3f} vs static {b[2]:.3f} "
                  f"(centre {b[3]*1e9:.0f} nm, T {b[4]*1e15:.0f} fs, pump at {b[5]*1e15:.0f} fs)")
