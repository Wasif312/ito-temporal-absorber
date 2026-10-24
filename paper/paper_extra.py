"""Experimental tolerances of the flagship design, from its full-wave spectra.

  sensitivity   Rozanov limit of the incident spectrum when the film thickness
                or the angle assumed for the budget is off by a little
  passband      absorbed fractions and limit inside the band of a long-pass
                filter that blocks the reflected pump (220 fs pump at 1315 nm)
  equivalent    thickness that a static absorber needs to match the switched
                film, and the errors of thickness and angle that erase the margin
  rotated_pump  a pump in a rotated plane of incidence whose front sweeps along
                the plane of the probe at the speed of the probe footprint
  dense         the curves of the figure

Usage: python paper_extra.py [--src DIR] [--results DIR]   writes data/extra.json
"""
import json
import os
from math import erfc

import numpy as np
from scipy.optimize import brentq

from paper_paths import arguments, read_json, write_json

args = arguments(__doc__.splitlines()[0])
from bound_scope import rozanov_max_absorption  # noqa: E402

C0 = 299792458.0
N0 = 1.5
sp = read_json(os.path.join(args.results, "route1_spectra.json"))
lam = np.array(sp["lam"])
Si, Ss, Sw = (np.array(sp[k]) for k in ("S_inc", "S_static", "S_switched"))
des = sp["design"]
d0, deg0 = des["d"], des["deg"]


def budget(d, deg):
    return 2 * np.pi ** 2 * N0 * d / np.cos(np.deg2rad(deg))


def frac(S, mask):
    return np.trapezoid(np.where(mask, S, 0.0), lam)


out = {}
eta0 = rozanov_max_absorption(Si, lam, budget(d0, deg0))[0]
sens = dict(eta_R=float(eta0), thickness=[], angle=[])
for dd in (-1.0, -0.5, -0.25, 0.25, 0.5, 1.0):
    e = rozanov_max_absorption(Si, lam, budget(d0 + dd * 1e-9, deg0))[0]
    sens["thickness"].append(dict(dd_nm=dd, eta_R=float(e), change=float(e - eta0)))
for da in (-1.0, -0.5, -0.2, -0.1, 0.1, 0.2, 0.5, 1.0):
    e = rozanov_max_absorption(Si, lam, budget(d0, deg0 + da))[0]
    sens["angle"].append(dict(dtheta_deg=da, eta_R=float(e), change=float(e - eta0)))
out["sensitivity"] = sens

# reflected pump: transform-limited Gaussian of 220 fs at 1315 nm
tau, lp = 220e-15, 1315e-9
nu0 = C0 / lp
dnu = 0.441 / tau                       # intensity FWHM of the spectrum
sig = dnu / (2 * np.sqrt(2 * np.log(2)))
tot_in = np.trapezoid(Si, lam)
eta_full = 1 - np.trapezoid(Sw, lam) / tot_in
static_full = 1 - np.trapezoid(Ss, lam) / tot_in
pb = []
for cut in (1300e-9, 1330e-9, 1340e-9, 1350e-9, 1360e-9):
    m = lam >= cut
    tin = frac(Si, m)
    e_sw = 1 - frac(Sw, m) / tin
    e_st = 1 - frac(Ss, m) / tin
    e_R = rozanov_max_absorption(np.where(m, Si, 0.0), lam, budget(d0, deg0))[0]
    leak = 0.5 * erfc((nu0 - C0 / cut) / (np.sqrt(2) * sig))
    pb.append(dict(cut_nm=cut * 1e9, probe_blocked=float(1 - tin / tot_in), switched=float(e_sw),
                   static=float(e_st), eta_R=float(e_R), margin=float(e_sw - e_R), pump_leak=float(leak)))
# dense curves for the figure
dense = dict(dd_nm=[], m_d=[], dth=[], m_th=[], cut_nm=[], m_cut=[], blocked=[], leak=[])
for dd in np.linspace(-1.5, 1.5, 61):
    dense["dd_nm"].append(float(dd))
    dense["m_d"].append(float(eta_full - rozanov_max_absorption(Si, lam, budget(d0 + dd * 1e-9, deg0))[0]))
for da in np.linspace(-2.0, 2.0, 41):
    dense["dth"].append(float(da))
    dense["m_th"].append(float(eta_full - rozanov_max_absorption(Si, lam, budget(d0, deg0 + da))[0]))
for cut in [c / 1e9 for c in range(1300, 1401, 5)]:
    m = lam >= cut
    tin = frac(Si, m)
    e_R = rozanov_max_absorption(np.where(m, Si, 0.0), lam, budget(d0, deg0))[0]
    dense["cut_nm"].append(float(cut * 1e9))
    dense["m_cut"].append(float(1 - frac(Sw, m) / tin - e_R))
    dense["blocked"].append(float(1 - tin / tot_in))
    dense["leak"].append(float(0.5 * erfc((nu0 - C0 / cut) / (np.sqrt(2) * sig))))
out["dense"] = dense
# thickness that a static absorber needs to reach the switched absorption
d_eq = brentq(lambda x: rozanov_max_absorption(Si, lam, budget(x, deg0))[0] - eta_full, d0, 3 * d0)
dd0 = brentq(lambda x: rozanov_max_absorption(Si, lam, budget(d0 + x, deg0))[0] - eta_full, 0.0, 5e-9)
da0 = brentq(lambda x: rozanov_max_absorption(Si, lam, budget(d0, deg0 + x))[0] - eta_full, 0.0, 5.0)
out["equivalent"] = dict(d_eq_nm=float(d_eq * 1e9), ratio=float(d_eq / d0), dd_zero_nm=float(dd0 * 1e9),
                         dtheta_zero_deg=float(da0))
# a pump in a rotated plane of incidence with the same trace speed along the probe plane
th0, thp = np.deg2rad(deg0), np.deg2rad(80.0)
phi = np.arccos(np.sin(th0) / np.sin(thp))
grad_fs_um = N0 * np.sin(thp) * np.sin(phi) / C0 * 1e-6 * 1e15
u_probe = np.array([np.sin(th0), 0.0, np.cos(th0)])
u_pump = np.array([np.sin(thp) * np.cos(phi), np.sin(thp) * np.sin(phi), np.cos(thp)])
sep = np.degrees(np.arccos(u_probe @ u_pump))
out["rotated_pump"] = dict(theta_pump_deg=80.0, azimuth_deg=float(np.degrees(phi)), delay_fs_per_um=float(grad_fs_um),
                           separation_deg=float(sep))
out["passband"] = dict(pump_fwhm_nm=float(lp ** 2 * dnu / C0), full=dict(switched=float(eta_full),
                       static=float(static_full), eta_R=float(eta0), margin=float(eta_full - eta0)), cuts=pb)

path = write_json("extra.json", out)
print(json.dumps({k: v for k, v in out.items() if k != "dense"}, indent=1))
print("wrote", path)
