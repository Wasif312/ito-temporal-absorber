"""validate_angle1d.py -- tests of the fixed-angle solver angle1d.py.

V0  incident wave in the glass: Ex / Hy equals the TM wave impedance
    eta0 cos(theta0) / n0 at every instant, as a plane wave at a fixed angle must.
V1  static 10 nm and 40 nm ITO films on a perfect conductor at 37.17 deg and
    70 deg in the glass: reflectance from a 5 fs pulse against the exact
    fixed-angle reflection coefficient over 1000-2000 nm, at 1 nm and 0.5 nm cells.
V2  switched film (pump-induced drop of wp by 20 %, chirped probe), both
    switching forms: energy balance U_inc + W_mod = U_r + U_diss + U_end.
Results go to results/tests_angle.json; the exit status is 1 if any test fails.
"""
import json
import os
import sys
import numpy as np
from angle1d import AngleStack, simulate, energies
from fdtd1d import Medium, C0, ETA0
from ito_model import ITO_SETS, eps_ito
from bound_scope import gamma_film_on_pec
from mirror import gaussian_probe, delta_profile
from slab_drude import spectrum

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ITO = ITO_SETS["Tirole2022"]
M_ITO = Medium(ITO["eps_inf"], 0)
SPECIES = [(ITO["wp"], ITO["gamma"])]


def film(d):
    return AngleStack(1.5, [(M_ITO, d)], SPECIES)


def spectra(run, ref, lam):
    f = C0 / lam
    dec = max(1, int(0.2e-15 / run["dt"]))
    Ei = spectrum(ref["recE"][0], ref["dt"], f, decimate=dec)
    Er = spectrum(run["recE"][0] - ref["recE"][0], run["dt"], f, decimate=dec)
    return Ei, Er


def main():
    res = {}
    failed = []
    lam = np.linspace(1000e-9, 2000e-9, 201)
    w0 = 2 * np.pi * C0 / 1400e-9
    T5 = 5e-15
    probe = gaussian_probe(T5, 6 * T5, w0=w0)
    t_end = 6 * T5 + 600e-15

    # V0 and V1
    v1 = []
    for deg in (37.17, 70.0):
        th = np.deg2rad(deg)
        for h in (1e-9, 0.5e-9):
            for d in (10e-9, 40e-9):
                ref = simulate(film(d), th, probe, t_end, h_fine=h, reference=True)
                run = simulate(film(d), th, probe, t_end, h_fine=h)
                if d == 10e-9 and h == 1e-9:
                    E, H = ref["recE"][0], ref["recH"][0]
                    Hm = np.empty_like(H)              # Hy at the time of Ex (leapfrog half step)
                    Hm[1:] = 0.5 * (H[1:] + H[:-1])
                    Hm[0] = 0.5 * H[0]
                    Z = ETA0 * np.cos(th) / 1.5
                    sel = np.abs(E) > 1e-3 * np.abs(E).max()
                    err0 = float(np.max(np.abs(E[sel] - Z * Hm[sel])) / np.abs(E).max())
                    res[f"V0_{deg}"] = err0
                    failed += [] if err0 < 1e-2 else [f"V0 {deg} deg"]
                    print(f"V0 {deg:5.2f} deg: max |Ex - Z Hy| / max|Ex| = {err0:.1e}  "
                          f"{'PASS' if err0 < 1e-2 else 'FAIL'}")
                Ei, Er = spectra(run, ref, lam)
                R = np.abs(Er / Ei) ** 2
                Rx = np.abs(gamma_film_on_pec(2.25, eps_ito(lam, "Tirole2022"), d, lam, th, "p")) ** 2
                err = float(np.max(np.abs(R - Rx)))
                ui, ur = energies(run, ref)
                v1.append(dict(deg=deg, h=h, d=d, max_err=err, R_min=float(Rx.min()),
                               balance=float((ui - ur - run["diss"].sum() - run["u_end"]) / ui)))
                failed += [] if err < 3e-3 else [f"V1 {deg} deg {d*1e9:.0f} nm {h*1e9:.1f} nm cells"]
                print(f"V1 {deg:5.2f} deg, d {d*1e9:2.0f} nm, cells {h*1e9:.1f} nm: max |R - R_exact| over "
                      f"1000-2000 nm = {err:.1e} (R_exact min {Rx.min():.3f}); energy balance "
                      f"{v1[-1]['balance']:+.1e}  {'PASS' if err < 3e-3 else 'FAIL'}", flush=True)
    res["V1"] = v1

    # V2 energy balance with switching
    th = np.deg2rad(70.0)
    T, span, lc = 300e-15, 300e-9, 1550e-9
    wc = 2 * np.pi * C0 / lc
    dw = 2 * np.pi * C0 * (1 / (lc - span / 2) - 1 / (lc + span / 2))
    tc = 3 * T
    probe = gaussian_probe(T, tc, w0=wc, chirp=dw / T)
    t_end = tc + 3 * T + 300e-15
    v2 = []
    ref = simulate(film(10e-9), th, probe, t_end, reference=True)
    for mom in (False, True):
        dfun = delta_profile(tc - 300e-15, 0.20)
        run = simulate(film(10e-9), th, probe, t_end, wp_scale=[lambda t, d=dfun: 1 + d(t)], momentum=mom)
        ui, ur = energies(run, ref)
        W, D = run["wmod"].sum(), run["diss"].sum()
        bal = (ui + W - ur - D - run["u_end"]) / ui
        v2.append(dict(momentum=mom, eta=1 - ur / ui, W=W / ui, D=D / ui, balance=bal))
        failed += [] if abs(bal) < 1e-3 else [f"V2 {'momentum' if mom else 'current'}"]
        print(f"V2 switched film, {'momentum' if mom else 'current'} form: absorbed {1 - ur/ui:.4f}, "
              f"W_mod {W/ui:+.2e}, dissipated {D/ui:.4f}, balance {bal:+.1e}  "
              f"{'PASS' if abs(bal) < 1e-3 else 'FAIL'}", flush=True)
    res["V2"] = v2
    res["failed"] = failed
    with open(os.path.join(OUT, "tests_angle.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print("saved results/tests_angle.json")
    print("all tests passed" if not failed else "FAILED: " + "; ".join(failed))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
