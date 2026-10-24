"""route3_angles_adiabatic.py -- adiabatic curves that accompany route3_angles.py.

At the in-plane wavevector of the 1370 nm centre wavelength and a probe FWHM of
300 fs (the choice of route3_angles.py at every prism angle): the best static
film and the best switched film against the angle inside the glass, and the
switched film against the pump delay at 50 deg. Results go to
results/route3_angles_adiabatic.json.
"""
import json
import os
import numpy as np
from route3_angles import adiabatic, OUT, LC, WC, DW, DMAX
from mirror import delta_profile
from ito_model import eps_ito, eps_gold, tmm
from fdtd1d import C0
import route3_angles as ra

if __name__ == "__main__":
    ra.T_CHOICES = (300e-15,)
    angles = np.arange(30.0, 60.01, 2.5)
    rows = []
    for deg in angles:
        a = adiabatic(float(deg))
        st, sw = a["static"][300e-15], a["switched"][300e-15]
        rows.append(dict(deg=float(deg), static=st[0], static_drop=st[1], switched=sw[0], pump=sw[1]))
        print(f"{deg:5.1f} deg: static {st[0]:.3f} (drop {st[1]*100:.2f} %), switched {sw[0]:.3f} "
              f"(pump {sw[1]*1e15:+.0f} fs)", flush=True)
    # switched film against pump delay at 50 deg
    deg, T = 50.0, 300e-15
    kx = 2 * np.pi * 1.5 / LC * np.sin(np.deg2rad(deg))
    lam_g = np.linspace(1150e-9, 1650e-9, 201)
    lam_g = lam_g[kx * lam_g / (2 * np.pi * 1.5) < 0.995]
    dl = np.linspace(0, DMAX, 81)
    Rm = np.array([[tmm([2.25, eps_ito(wl, "Tirole2022", wp_scale=1 - x), eps_gold(wl), 1.0],
                        [40e-9, 100e-9], wl, np.arcsin(kx * wl / (2 * np.pi * 1.5)), "p")[0]
                    for wl in lam_g] for x in dl])

    def R_of(lam, drop):
        i = np.clip((lam - lam_g[0]) / (lam_g[1] - lam_g[0]), 0, len(lam_g) - 1.001)
        j = np.clip(drop / (dl[1] - dl[0]), 0, len(dl) - 1.001)
        i0, j0 = i.astype(int), j.astype(int)
        fi, fj = i - i0, j - j0
        return ((1 - fi) * (1 - fj) * Rm[j0, i0] + fi * (1 - fj) * Rm[j0, i0 + 1]
                + (1 - fi) * fj * Rm[j0 + 1, i0] + fi * fj * Rm[j0 + 1, i0 + 1])
    t = np.arange(-4 * T, 4 * T, 1e-15)
    I = np.exp(-4 * np.log(2) * (t / T) ** 2)
    lam_t = 2 * np.pi * C0 / (WC + DW / T * t)
    offs = np.arange(-700e-15, 0.1e-15, 10e-15)
    curve = [float(np.trapezoid(I * (1 - R_of(lam_t, -delta_profile(o, DMAX)(t))), t) / np.trapezoid(I, t))
             for o in offs]
    out = dict(angles=rows, delay_50deg=dict(offsets_fs=(offs * 1e15).tolist(), switched=curve))
    with open(os.path.join(OUT, "route3_angles_adiabatic.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("saved results/route3_angles_adiabatic.json")
