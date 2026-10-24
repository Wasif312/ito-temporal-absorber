"""validate_angle_gold.py -- the fixed-angle solver with a gold back mirror.

V3  static glass / ITO / 200 nm gold / perfect conductor at 75 deg in the glass,
    films of 7.5 and 10 nm: reflectance from a 5 fs pulse against the closed-form
    reflection coefficient of a film on a gold half-space (gold_fit.py), over
    1000-2400 nm, and the energy balance with the dissipation in ITO and gold.
V4  the same stack switched by a 30 % drop of wp with damping that rises with
    the electron temperature (hot_electron.py): energy balance in both forms.
Results go to results/tests_angle_gold.json; the exit status is 1 if any test fails.
"""
import json
import os
import sys
import numpy as np
from angle1d import AngleStack, simulate, energies
from fdtd1d import Medium, C0
from ito_model import ITO_SETS, eps_ito
from gold_fit import params, eps_gold_fit, gamma_film_on_metal
from hot_electron import load_map
from mirror import gaussian_probe, delta_profile
from slab_drude import spectrum

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ITO = ITO_SETS["Tirole2022"]
AU = params()
HOT = load_map(0.9)


def stack_au(d, d_au=200e-9):
    return AngleStack(1.5, [(Medium(ITO["eps_inf"], 0), d), (Medium(AU["eps_inf"], 1), d_au)],
                      [(ITO["wp"], ITO["gamma"]), (AU["wp"], AU["gamma"])])


def main():
    res = {}
    failed = []
    th = np.deg2rad(75.0)
    lam = np.linspace(1000e-9, 2400e-9, 141)
    probe = gaussian_probe(5e-15, 30e-15, w0=2 * np.pi * C0 / 1500e-9)
    v3 = []
    for d in (7.5e-9, 10e-9):
        h = min(1e-9, d / 10)
        ref = simulate(stack_au(d), th, probe, 700e-15, h_fine=h, reference=True)
        run = simulate(stack_au(d), th, probe, 700e-15, h_fine=h)
        dec = max(1, int(0.2e-15 / run["dt"]))
        Ei = spectrum(ref["recE"][0], ref["dt"], C0 / lam, decimate=dec)
        Er = spectrum(run["recE"][0] - ref["recE"][0], run["dt"], C0 / lam, decimate=dec)
        R = np.abs(Er / Ei) ** 2
        Rx = np.abs(gamma_film_on_metal(2.25, eps_ito(lam), eps_gold_fit(lam), d, lam, th)) ** 2
        ui, ur = energies(run, ref)
        bal = (ui - ur - run["diss"].sum() - run["u_end"]) / ui
        err = float(np.max(np.abs(R - Rx)))
        v3.append(dict(d=d, max_err=err, balance=float(bal), A_ito=float(run["diss"][0] / ui),
                       A_au=float(run["diss"][1] / ui)))
        failed += [] if err < 3e-3 and abs(bal) < 1e-3 else [f"V3 {d*1e9:.1f} nm"]
        print(f"V3 d {d*1e9:.1f} nm: max |R - R_exact| over 1000-2400 nm = {err:.1e}; energy balance {bal:+.1e}; "
              f"absorbed in ITO {run['diss'][0]/ui:.4f}, in gold {run['diss'][1]/ui:.4f}  "
              f"{'PASS' if err < 3e-3 and abs(bal) < 1e-3 else 'FAIL'}", flush=True)
    res["V3"] = v3
    # V4 switched with rising damping
    T, lc, span = 300e-15, 1550e-9, 400e-9
    w_red, w_blue = 2 * np.pi * C0 / (lc + span / 2), 2 * np.pi * C0 / (lc - span / 2)
    tc = 3 * T
    probe = gaussian_probe(T, tc, w0=0.5 * (w_red + w_blue), chirp=(w_blue - w_red) / T)
    t_arr = tc + 2.5e-6 * 1.5 * np.cos(th) / C0
    t_end = t_arr + 3 * T + 500e-15
    ref = simulate(stack_au(10e-9), th, probe, t_end, reference=True)
    dfun = delta_profile(t_arr - 300e-15, 0.30)
    v4 = []
    for mom in (False, True):
        run = simulate(stack_au(10e-9), th, probe, t_end, momentum=mom,
                       wp_scale=[lambda t: 1 + dfun(t), None],
                       gamma_scale=[lambda t: HOT.gamma_scale(-dfun(t)), None])
        ui, ur = energies(run, ref)
        W, D = run["wmod"].sum(), run["diss"].sum()
        bal = (ui + W - ur - D - run["u_end"]) / ui
        v4.append(dict(momentum=mom, eta=float(1 - ur / ui), W=float(W / ui), balance=float(bal)))
        failed += [] if abs(bal) < 1e-3 else [f"V4 {'momentum' if mom else 'current'}"]
        print(f"V4 switched, {'momentum' if mom else 'current'} form: absorbed {1 - ur/ui:.4f}, W_mod {W/ui:+.2e}, "
              f"balance {bal:+.1e}  {'PASS' if abs(bal) < 1e-3 else 'FAIL'}", flush=True)
    res["V4"] = v4
    res["failed"] = failed
    with open(os.path.join(OUT, "tests_angle_gold.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print("saved results/tests_angle_gold.json")
    print("all tests passed" if not failed else "FAILED: " + "; ".join(failed))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
