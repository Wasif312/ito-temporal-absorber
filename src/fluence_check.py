"""fluence_check.py -- pump fluence, absorbed energy and lattice heating of the
route-1 films, set against films that survived published pump-probe
experiments.

For every route-1 film of r1_ttm.py (perfect conductor, fixed and rising
damping, four geometries, largest drops of 20-40 %) the script reruns the
two-temperature model at the pump intensity that gives the drop and records
the incident fluence, the energy absorbed per unit film area and per unit
volume, the peak electron temperature, the lattice temperature at 2.5 ps and
the common temperature that electrons and lattice reach once they equilibrate
(before any heat leaves the film). The same numbers are computed for

  tirole    ITO (40 nm)/Au mirror of Tirole et al., Phys. Rev. Applied 18,
            054067 (2022): 220 fs pump at 1300 nm and 75 deg in air, at their
            low-power map (22 GW/cm^2), at saturation (100 GW/cm^2) and at the
            largest intensity of their setup (708 GW/cm^2)
  alam      310 nm ITO film of Alam et al., Science 352, 795 (2016): 150 fs at
            1240 nm and 60 deg, 150 GW/cm^2, uniform heating

and against the single-shot damage fluence of 0.21-0.40 J/cm^2 that Bian et
al., Opt. Laser Technol. 45, 395 (2013) report for 120-160 nm ITO films
(800 nm, 60-600 fs). Results go to results/fluence_check_<coupling>.json.

Usage: python3 fluence_check.py [--coupling baxter|sarkar]
"""
import argparse
import json
import os
import time
import numpy as np
from scipy.optimize import brentq

import ttm
import r1_ttm as m
from ito_model import eps_gold, tmm
from ttm_validate import ito_eps, table

OUT = m.OUT
GW = 1e13
FLUENCE = np.sqrt(np.pi / (4 * np.log(2)))      # fluence = FLUENCE * I_peak * FWHM
DAMAGE_J_CM2 = (0.21, 0.40)                     # Bian et al. (2013), 60-600 fs, 800 nm


def heating(gas, absorb, I_GWcm2, d, cos_t, fwhm, G=None):
    """Two-temperature run for one pump; energies and temperatures."""
    t = np.arange(-4 * fwhm, 2500e-15, 0.5e-15)
    r = ttm.run(gas, absorb, ttm.gaussian_pulse(I_GWcm2 * GW, fwhm, 0.0), d, cos_t, t, G_const=G)
    q = r["absorbed_J_m2"] / d                                  # J/m^3
    U0 = gas.energy(ttm.T_ROOM)
    T_eq = brentq(lambda T: gas.energy(T) - U0 + ttm.C_L * (T - ttm.T_ROOM) - q, ttm.T_ROOM, 2e4)
    F = FLUENCE * I_GWcm2 * GW * fwhm                           # J/m^2 in the incidence medium
    return dict(I_GWcm2=I_GWcm2, fluence_mJcm2=F * 0.1, fluence_film_mJcm2=F * cos_t * 0.1,
                absorbed_mJcm2=r["absorbed_J_m2"] * 0.1, absorbed_share=r["absorbed_J_m2"] / (F * cos_t),
                energy_density_GJm3=q * 1e-9, Te_max=float(r["Te"].max()), drop_max=float(r["drop"].max()),
                Tl_2500fs=float(r["Tl"][-1]), T_eq=float(T_eq),
                Te_clipped=bool(r["Te"].max() > 0.99 * gas.T[-1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--coupling", choices=["baxter", "sarkar"], default="baxter")
    args = ap.parse_args()
    t0 = time.time()
    G = m.G_SARKAR if args.coupling == "sarkar" else None
    gas = m.gas()
    out = dict(coupling=args.coupling, damage_J_cm2=DAMAGE_J_CM2)

    # Tirole et al. (2022): 40 nm ITO on 100 nm gold, pump through the glass
    th_air = np.deg2rad(75.0)
    th_glass = np.arcsin(np.sin(th_air) / 1.5)
    T_ag = tmm([1.0, 2.25], [], 1300e-9, th_air, "p")[1]
    A_ito = table(lambda T: ttm.layer_absorption([2.25, ito_eps(1300e-9, "Tirole2022", T, gas), eps_gold(1300e-9),
                                                  1.0], [40e-9, 100e-9], 1300e-9, th_glass, "p")[2][0])
    out["tirole"] = [dict(label=lab, **heating(gas, lambda T: A_ito(T) * T_ag, I, 40e-9, np.cos(th_air), 220e-15, G))
                     for lab, I in (("low-power map", 22.0), ("saturation", 100.0), ("largest", 708.0))]

    # Alam et al. (2016): 310 nm film, uniform heating
    th_a = np.deg2rad(60.0)
    A_alam = table(lambda T: ttm.layer_absorption([1.0, ito_eps(1240e-9, "Alam2016", T, gas), 2.25], [310e-9],
                                                  1240e-9, th_a, "p")[2][0])
    out["alam"] = [dict(label="largest", **heating(gas, A_alam, 150.0, 310e-9, np.cos(th_a), 150e-15, G))]

    # route 1
    with open(os.path.join(OUT, f"r1_ttm_{args.coupling}.json")) as fh:
        profs = json.load(fh)["profiles"]
    rows = []
    for p in profs:
        _, absorb, _ = m.pump_setup(p["var"], p["d"], p["deg"])
        row = dict(var=p["var"], d=p["d"], deg=p["deg"], depth=p["depth"],
                   **heating(gas, absorb, p["I_GWcm2"], p["d"], np.cos(np.deg2rad(p["deg"])), m.FWHM_PUMP, G))
        rows.append(row)
    out["route1"] = rows

    ref = max(r["energy_density_GJm3"] for r in out["tirole"])
    for name in ("tirole", "alam"):
        for r in out[name]:
            print(f"{name:6s} {r['label']:13s} {r['I_GWcm2']:6.0f} GW/cm^2, "
                  f"{r['fluence_mJcm2']:6.1f} mJ/cm^2 incident, {r['absorbed_mJcm2']:5.2f} mJ/cm^2 absorbed "
                  f"({r['absorbed_share']*100:4.1f} % of the fluence on the film), "
                  f"{r['energy_density_GJm3']:.2f} GJ/m^3, Te {r['Te_max']:.0f} K, T_eq {r['T_eq']:.0f} K"
                  + (" [Te beyond the table]" if r["Te_clipped"] else ""))
    for r in rows:
        print(f"{r['var']:9s} {r['d']*1e9:4.1f} nm {r['deg']:.0f} deg drop {r['depth']:.2f}: "
              f"{r['I_GWcm2']:6.1f} GW/cm^2, {r['fluence_mJcm2']:6.1f} mJ/cm^2 "
              f"({r['fluence_mJcm2'] / (DAMAGE_J_CM2[0] * 1e3):.2f} of the lower damage fluence), "
              f"{r['absorbed_mJcm2']:5.2f} mJ/cm^2 absorbed ({r['absorbed_share']*100:4.1f} %), "
              f"{r['energy_density_GJm3']:.2f} GJ/m^3 ({r['energy_density_GJm3']/ref:.2f} of Tirole's largest), "
              f"Te {r['Te_max']:.0f} K, Tl(2.5 ps) {r['Tl_2500fs']:.0f} K, T_eq {r['T_eq']:.0f} K")
    with open(os.path.join(OUT, f"fluence_check_{args.coupling}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"saved results/fluence_check_{args.coupling}.json ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
