"""gold_fit.py -- a gold model that the fixed-angle solver can use at large angles.

The fixed-angle scheme of angle1d.py needs a high-frequency permittivity above
(n0 sin theta)^2 = 2.10 at 75 deg in glass in every layer, and the Drude fit of
Olmon et al. (2012) for gold has eps_inf = 1. Interband transitions give gold a
background permittivity well above 1, so we refit a Drude model with eps_inf = 5
to the Olmon permittivity over the probe band, 1.0-2.5 um, and check the
reflectance of glass / ITO / gold against the Olmon model. Also provides the
p-polarised reflection coefficient of a film on a metal half-space.
Results go to results/gold_fit.json.
"""
import json
import os
import numpy as np
from scipy.optimize import least_squares
from ito_model import eps_drude, eps_gold, eps_ito, C0, HBAR_EV

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
EPS_INF_AU = 5.0
BAND = (1.0e-6, 2.5e-6)


def fit(eps_inf=EPS_INF_AU, band=BAND):
    lam = np.linspace(band[0], band[1], 301)
    w = 2 * np.pi * C0 / lam
    target = eps_gold(lam)

    def res(p):
        e = eps_drude(w, eps_inf, p[0] * 1e16, p[1] * 1e13)
        r = (e - target) / np.abs(target)
        return np.concatenate([r.real, r.imag])
    s = least_squares(res, x0=[1.3, 7.0])
    wp, g = s.x[0] * 1e16, s.x[1] * 1e13
    e = eps_drude(w, eps_inf, wp, g)
    return dict(eps_inf=eps_inf, wp=wp, gamma=g, max_rel_err=float(np.max(np.abs(e - target) / np.abs(target))))


_P = None


def params():
    global _P
    if _P is None:
        p = os.path.join(OUT, "gold_fit.json")
        _P = json.load(open(p))["fit"] if os.path.exists(p) else fit()
    return _P


def eps_gold_fit(lam):
    p = params()
    w = 2 * np.pi * C0 / np.asarray(lam, dtype=float)
    return eps_drude(w, p["eps_inf"], p["wp"], p["gamma"])


def gamma_film_on_metal(eps0, eps1, eps2, d, lam, theta0):
    """p-polarised reflection coefficient of medium 0 / film 1 (thickness d) /
    metal half-space 2, angle theta0 in medium 0, time dependence exp(-i w t)."""
    k0 = 2 * np.pi / np.asarray(lam, dtype=float)
    kx = np.sqrt(eps0 + 0j) * np.sin(theta0)

    def kz(e):
        q = np.sqrt(e - kx ** 2 + 0j)
        return np.where(q.imag < 0, -q, q)
    q0, q1, q2 = kz(eps0 + 0 * k0), kz(eps1 + 0 * k0), kz(eps2 + 0 * k0)
    r01 = (eps1 * q0 - eps0 * q1) / (eps1 * q0 + eps0 * q1)
    r12 = (eps2 * q1 - eps1 * q2) / (eps2 * q1 + eps1 * q2)
    ph = np.exp(2j * k0 * q1 * d)
    return (r01 + r12 * ph) / (1 + r01 * r12 * ph)


if __name__ == "__main__":
    from ito_model import tmm
    f = fit()
    print(f"Drude fit with eps_inf = {f['eps_inf']}: hbar wp = {f['wp']*HBAR_EV:.3f} eV, "
          f"1/gamma = {1/f['gamma']*1e15:.2f} fs; largest relative error against Olmon over "
          f"1.0-2.5 um: {f['max_rel_err']:.1e}")
    _P = f
    lam = np.linspace(1000e-9, 2400e-9, 1401)
    checks = []
    for d in (7.5e-9, 10e-9):
        for deg in (70.0, 75.0):
            th = np.deg2rad(deg)
            R_ol = np.abs(gamma_film_on_metal(2.25, eps_ito(lam), eps_gold(lam), d, lam, th)) ** 2
            R_fit = np.abs(gamma_film_on_metal(2.25, eps_ito(lam), eps_gold_fit(lam), d, lam, th)) ** 2
            R_tmm = np.array([tmm([2.25, eps_ito(wl), eps_gold(wl)], [d], wl, th, "p")[0] for wl in lam[::50]])
            e1 = float(np.max(np.abs(R_fit - R_ol)))
            e2 = float(np.max(np.abs(R_ol[::50] - R_tmm)))
            checks.append(dict(d=d, deg=deg, max_dR_fit=e1, max_dR_tmm=e2, A_min=float(1 - R_ol.max()),
                               R_min=float(R_ol.min())))
            print(f"d {d*1e9:.1f} nm, {deg:.0f} deg: |R(fit) - R(Olmon)| <= {e1:.1e}; closed form against "
                  f"transfer matrix {e2:.1e}; R min {R_ol.min():.4f}")
    with open(os.path.join(OUT, "gold_fit.json"), "w") as fh:
        json.dump(dict(fit=f, checks=checks), fh, indent=1)
    print("saved results/gold_fit.json")
