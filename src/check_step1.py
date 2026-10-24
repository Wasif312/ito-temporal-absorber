"""check_step1.py -- the material model against published numbers.

1. Re(eps) = 0 wavelength of each ITO set against the value its paper states.
2. Complex index of the Alam set at 1240 nm against n = 0.42 (Alam 2016) and
   0.42 + 0.42i (Zhou et al. 2020, SI Note 6).
3. Gold permittivity from the Olmon Drude fit.
4. Transfer-matrix sanity checks: a lossless slab conserves energy, and a
   single interface reproduces the Fresnel formulas.
5. Tirole et al. (2022) mirror, glass / 40 nm ITO / 100 nm Au, p-polarised.
   The paper probes at 65 degrees from air, through the glass substrate,
   which is 37.17 degrees inside the glass. Unpumped reflectance against
   angle (published 0.07), then the drop of omega_p that gives the
   saturated pumped reflectance of 0.66 (calibration of the pump).
"""
import numpy as np
from ito_model import (ITO_SETS, enz_wavelength, eps_ito, eps_gold, tmm, C0)


def main():
    print("1. ENZ wavelengths")
    stated = {"Alam2016": 1240e-9, "Bohn2021": C0 / 243e12, "Tirole2022": C0 / 227e12,
              "Galiffi2026": 1196e-9}
    for k, p in ITO_SETS.items():
        lam = enz_wavelength(p["eps_inf"], p["wp"], p["gamma"])
        dev = 100 * (lam - stated[k]) / stated[k]
        print(f"   {k:11s} computed {lam*1e9:7.1f} nm   stated {stated[k]*1e9:7.1f} nm"
              f"   ({dev:+.2f} %)   gamma/wp = {p['gamma']/p['wp']:.3f}")

    print("2. Alam set at 1240 nm")
    e = eps_ito(1240e-9, "Alam2016")
    n = np.sqrt(e)
    print(f"   eps = {e.real:.3f} + {e.imag:.3f}i,  n = {n.real:.3f} + {n.imag:.3f}i")

    print("3. Gold (Olmon Drude)")
    for lam in (1300e-9, 1550e-9):
        eg = eps_gold(lam)
        print(f"   {lam*1e9:.0f} nm: eps = {eg.real:.1f} + {eg.imag:.2f}i")

    print("4. TMM checks")
    lam = 1300e-9
    R, T, A = tmm([1.0, 2.25, 1.0], [400e-9], lam, np.deg2rad(30), "p")
    print(f"   lossless slab, p, 30 deg: R+T-1 = {R+T-1:.1e}")
    R, T, A = tmm([1.0, 2.25, 1.0], [400e-9], lam, np.deg2rad(30), "s")
    print(f"   lossless slab, s, 30 deg: R+T-1 = {R+T-1:.1e}")
    n1, n2, th = 1.0, 1.5, np.deg2rad(40)
    th2 = np.arcsin(n1 * np.sin(th) / n2)
    rp = (n2 * np.cos(th) - n1 * np.cos(th2)) / (n2 * np.cos(th) + n1 * np.cos(th2))
    R, _, _ = tmm([n1**2, n2**2, n2**2], [100e-9], lam, th, "p")
    print(f"   single interface, p, 40 deg: TMM R = {R:.6f}, Fresnel R = {rp**2:.6f}")

    print("5. Tirole 2022 mirror at 1300 nm, p-polarised, light through the glass")
    def eps_stack(lam, s=1.0):
        return [2.25, eps_ito(lam, "Tirole2022", wp_scale=s), eps_gold(lam), 1.0]
    d_stack = [40e-9, 100e-9]
    TH = np.arcsin(np.sin(np.deg2rad(65)) / 1.5)          # 65 deg in air -> glass
    R, T, A = tmm(eps_stack(lam), d_stack, lam, TH, "p")
    print(f"   65 deg in air = {np.rad2deg(TH):.2f} deg in glass: R = {R:.3f}, T = {T:.1e}")
    ang = np.linspace(0, 80, 8001)
    Ra = np.array([tmm(eps_stack(lam), d_stack, lam, np.deg2rad(a), "p")[0] for a in ang])
    cross = ang[1:][np.diff(np.sign(Ra - 0.07)) != 0]
    print(f"   R = 0.07 at {', '.join(f'{a:.1f}' for a in cross)} deg in glass "
          f"(above {np.rad2deg(np.arcsin(1/1.5)):.1f} deg only with a prism)")
    print(f"   taking 65 deg inside the glass instead gives R = "
          f"{tmm(eps_stack(lam), d_stack, lam, np.deg2rad(65), 'p')[0]:.3f}")


    def drop_for(Rtarget, th):
        dd = np.linspace(0, 0.40, 4001)
        Rd = np.array([tmm(eps_stack(lam, 1 - x), d_stack, lam, th, "p")[0] for x in dd])
        return np.interp(Rtarget, Rd, dd)


    print("   pumped state, omega_p -> (1 - drop) omega_p, at 37.17 deg in glass:")
    for d in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.42):
        R, _, _ = tmm(eps_stack(lam, 1 - d), d_stack, lam, TH, "p")
        print(f"     drop {100*d:4.0f} %: R = {R:.3f}")
    print(f"   calibration: R = 0.66 needs a drop of {100*drop_for(0.66, TH):.2f} % "
          f"(at the prism-only solution near 56 deg it would need "
          f"{100*drop_for(0.66, np.deg2rad(56.0)):.1f} %)")
    print("   linear extrapolation of omega_p,2 = -0.42 % per GW cm^-2 to 100 GW cm^-2: "
          f"drop 42 %, R = {tmm(eps_stack(lam, 0.58), d_stack, lam, TH, 'p')[0]:.3f}")
    lam_g = np.linspace(1200e-9, 1600e-9, 4001)
    for s in (1.0, 0.9):
        Rl = np.array([tmm(eps_stack(wl, s), d_stack, wl, TH, "p")[0] for wl in lam_g])
        print(f"   omega_p scale {s}: reflectance dip at {lam_g[Rl.argmin()]*1e9:.1f} nm, R = {Rl.min():.3f}")


if __name__ == "__main__":
    main()
