"""Derived quantities for the manuscript, computed with the repository code.

  budgets     Rozanov budget B = 2 pi^2 n0 d / cos(theta) of the four film geometries
  static      reflectance of the 7.5 nm film on a perfect conductor at 75 deg in
              the glass for drops of wp of 0-30 %, and the wavelength, depth and
              band (R < 0.5) of its absorption dip for drops of 0-40 %
  inferred    drop of wp behind two measurements: the saturated reflectance
              0.66 of the mirror of Tirole et al. and the index change 0.72 of
              the film of Alam et al. (fixed damping and rising damping)

Usage: python paper_data.py [--src DIR]   writes data/paper_data.json
"""
import json

import numpy as np
from scipy.optimize import brentq

from paper_paths import arguments, write_json

arguments(__doc__.splitlines()[0], results=False)
import hot_electron as he  # noqa: E402
from bound_scope import gamma_film_on_pec  # noqa: E402
from ito_model import eps_gold, eps_ito, tmm  # noqa: E402

N0 = 1.5
out = {}

out["budgets"] = [dict(d=d, deg=deg, B=2 * np.pi ** 2 * N0 * d / np.cos(np.deg2rad(deg)),
                       ratio=N0 / np.cos(np.deg2rad(deg)))
                  for d, deg in ((5e-9, 80.0), (7.5e-9, 75.0), (10e-9, 70.0), (10e-9, 75.0))]

lam = np.arange(1000e-9, 2400.1e-9, 2e-9)
th = np.deg2rad(75.0)
d = 7.5e-9


def R_film(drop, lam_):
    return np.abs(gamma_film_on_pec(N0 ** 2, eps_ito(lam_, "Tirole2022", wp_scale=1 - drop), d, lam_, th, "p")) ** 2


out["static"] = dict(d=d, deg=75.0, lam=lam.tolist(),
                     R={f"{x:.2f}": R_film(x, lam).tolist() for x in (0.0, 0.1, 0.2, 0.3)})
fine = np.arange(1000e-9, 3000.1e-9, 0.25e-9)
dips = []
for x in np.round(np.arange(0.0, 0.401, 0.05), 3):
    R = R_film(x, fine)
    k = int(np.argmin(R))
    half = fine[R < 0.5]
    dips.append(dict(drop=float(x), lam_dip=float(fine[k]), R_min=float(R[k]),
                     width_R05=float(half.max() - half.min()) if len(half) else 0.0,
                     lo_R05=float(half.min()) if len(half) else float("nan"),
                     hi_R05=float(half.max()) if len(half) else float("nan")))
out["static"]["dips"] = dips

# Tirole et al. mirror: glass / 40 nm ITO / 100 nm Au / air at 1300 nm, 65 deg in air
lam_t = 1300e-9
TH = np.arcsin(np.sin(np.deg2rad(65)) / 1.5)


def R_mirror(x):
    return tmm([2.25, eps_ito(lam_t, "Tirole2022", wp_scale=1 - x), eps_gold(lam_t), 1.0],
               [40e-9, 100e-9], lam_t, TH, "p")[0]


tir = brentq(lambda x: R_mirror(x) - 0.66, 0.0, 0.3)
# Alam et al. film at 1240 nm: index change 0.72
lam_a = 1240e-9
n_cold = np.sqrt(eps_ito(lam_a, "Alam2016"))
alam_fixed = brentq(lambda x: np.sqrt(eps_ito(lam_a, "Alam2016", wp_scale=1 - x)).real - n_cold.real - 0.72, 0, 0.5)
hm = he.DropMap(0.9)
alam_hot = brentq(lambda x: np.sqrt(eps_ito(lam_a, "Alam2016", wp_scale=1 - x,
                                            gamma_scale=hm.gamma_scale(x))).real - n_cold.real - 0.72, 0, 0.5)
model_hot = brentq(lambda x: np.sqrt(eps_ito(lam_a, "Alam2016", wp_scale=1 - x,
                                             gamma_scale=hm.gamma_scale(x))).real - n_cold.real - 0.526, 0, 0.5)
out["inferred"] = dict(tirole_R066=float(tir), tirole_R_cold=float(R_mirror(0.0)),
                       alam_n_cold=[float(n_cold.real), float(n_cold.imag)],
                       alam_dn072_fixed=float(alam_fixed), alam_dn072_hot=float(alam_hot),
                       alam_dn072_hot_gamma=float(hm.gamma_scale(alam_hot)),
                       alam_model_dn0526_hot=float(model_hot))

path = write_json("paper_data.json", out)
print(json.dumps(out["budgets"], indent=0))
print(json.dumps(out["static"]["dips"], indent=0))
print(json.dumps(out["inferred"], indent=0))
print("wrote", path)
