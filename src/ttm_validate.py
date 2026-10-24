"""ttm_validate.py -- the two-temperature model of ttm.py against measurements,
and the pump that route 1 needs.

  gas       Sommerfeld coefficient against Sarkar, Un and Sivan (12.7 J m^-3 K^-2
            for n_e = 1.5e27 m^-3, E_F = 0.88 eV), electron cooling time C_e/g_ep
  tirole    ITO/Au mirror of Tirole et al. (2022): 220 fs pump at 1300 nm and 75 deg
            in air through the glass, probe at 1300 nm and 65 deg in air; the pump
            intensity that gives the measured reflectance of 0.66, the saturation of
            the reflectance at high intensity, and the decay of the reflectance
  alam      310 nm ITO film of Alam et al. (2016) at 60 deg, TM, 150 fs pulses at
            1240 nm, 150 GW/cm^2: peak index change (measured 0.72) and its decay
            (measured about 360 fs), with uniform heating of the film
  route1    7.5 nm ITO film at 75 deg in the glass on a perfect conductor (route 1):
            peak drop of wp against pump intensity, the intensity and fluence for
            drops of 20, 30 and 40 %, and the drop and damping profiles
Each case runs with the coupling g_ep(Te) of Baxter et al. (A_ep = 5.25e-4 eV^2)
and with the constant G = 3e16 W m^-3 K^-1 of Sarkar et al. Pump polarisation is
p in every case (an assumption for the Tirole mirror, whose pump polarisation we
did not find). Results go to results/ttm_validation.json.
"""
import json
import os
import time
import numpy as np
import ttm
from ito_model import eps_ito, eps_gold, tmm
from bound_scope import gamma_film_on_pec

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
GW = 1e13                                       # 1 GW/cm^2 in W/m^2
T_TAB = np.geomspace(300.0, 59000.0, 220)
G_SARKAR = 3e16
FLUENCE = np.sqrt(np.pi / (4 * np.log(2)))      # fluence = FLUENCE * I_peak * FWHM


def ito_eps(lam, name, T, gas):
    return eps_ito(lam, name, wp_scale=np.sqrt(gas.wp2_ratio(T)), gamma_scale=ttm.gamma_baxter(T))


def table(fun):
    vals = np.array([fun(T) for T in T_TAB])
    return lambda T: np.interp(T, T_TAB, vals)


def scan(gas, absorb, d, cos_t, fwhm, intensities, observe=None, G=None):
    t = np.arange(-4 * fwhm, 2500e-15, 0.25e-15)
    rows = []
    for I0 in intensities:
        r = ttm.run(gas, absorb, ttm.gaussian_pulse(I0 * GW, fwhm, 0.0), d, cos_t, t, G_const=G)
        row = dict(I_GWcm2=I0, fluence_mJcm2=FLUENCE * I0 * GW * fwhm * 0.1, Te_max=float(r["Te"].max()),
                   drop_max=float(r["drop"].max()), absorbed_J_m2=r["absorbed_J_m2"],
                   drop_rise_fs=ttm.rise_time(t, r["drop"]) * 1e15, drop_decay_fs=ttm.decay_time(t, r["drop"]) * 1e15)
        if observe is not None:
            obs = observe(r["Te"])
            row["obs_max"] = float(obs.max()) if obs.max() - obs[0] >= obs[0] - obs.min() else float(obs.min())
            row["obs_decay_fs"] = ttm.decay_time(t, np.abs(obs - obs[0])) * 1e15
        rows.append(row)
    return rows


def main():
    t0 = time.time()
    out = {}
    gas = ttm.load_gas(0.9)
    gas88 = ttm.load_gas(0.88)
    out["gas"] = dict(N=gas.N, gamma_e=gas.gamma_e(), gamma_e_088=gas88.gamma_e(), N_088=gas88.N,
                      g_ep_300=float(gas.g_ep(300.0)),
                      rows=[dict(Te=T, drop=float(gas.drop(T)), Ce=float(gas.ce(T)), g_ep=float(gas.g_ep(T)),
                                 tau_fs=float(gas.ce(T) / gas.g_ep(T) * 1e15),
                                 tau_sarkar_fs=float(gas.ce(T) / G_SARKAR * 1e15))
                            for T in (500.0, 1000.0, 2000.0, 4866.0, 8532.0, 13195.0, 19857.0)])
    print(f"Sommerfeld coefficient {gas88.gamma_e():.2f} J m^-3 K^-2 at E_F = 0.88 eV (Sarkar et al. 12.7); "
          f"g_ep(300 K) = {gas.g_ep(300.0):.3e} W m^-3 K^-1")

    # ---------------------------------------------------------------- Tirole mirror
    th_air = np.deg2rad(75.0)
    th_glass = np.arcsin(np.sin(th_air) / 1.5)
    T_ag = tmm([1.0, 2.25], [], 1300e-9, th_air, "p")[1]
    A_ito = table(lambda T: ttm.layer_absorption([2.25, ito_eps(1300e-9, "Tirole2022", T, gas), eps_gold(1300e-9), 1.0],
                                                 [40e-9, 100e-9], 1300e-9, th_glass, "p")[2][0])
    th_probe = np.arcsin(np.sin(np.deg2rad(65.0)) / 1.5)
    R_probe = table(lambda T: tmm([2.25, ito_eps(1300e-9, "Tirole2022", T, gas), eps_gold(1300e-9), 1.0],
                                  [40e-9, 100e-9], 1300e-9, th_probe, "p")[0])
    Is = [2, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300]
    out["tirole"] = dict(T_air_glass=T_ag, A_ito_cold=float(A_ito(300.0)), R_cold=float(R_probe(300.0)), cases={})
    for label, G in (("g_ep", None), ("G_sarkar", G_SARKAR)):
        rows = scan(gas, lambda T: A_ito(T) * T_ag, 40e-9, np.cos(th_air), 220e-15, Is, observe=R_probe, G=G)
        Rmax = np.array([r["obs_max"] for r in rows])
        I066 = float(np.interp(0.66, Rmax, Is)) if Rmax.max() >= 0.66 else None
        out["tirole"]["cases"][label] = dict(rows=rows, I_for_R066=I066)
        print(f"Tirole mirror, {label}: cold R {R_probe(300.0):.3f}, A_ITO {A_ito(300.0):.3f}; R = 0.66 at "
              f"{I066} GW/cm^2 in air; " + "; ".join(
                  f"{r['I_GWcm2']}: drop {r['drop_max']*100:.1f} %, R {r['obs_max']:.3f}, "
                  f"decay {r['obs_decay_fs']:.0f} fs" for r in rows))

    # ---------------------------------------------------------------- Alam film
    th_a = np.deg2rad(60.0)
    A_alam = table(lambda T: ttm.layer_absorption([1.0, ito_eps(1240e-9, "Alam2016", T, gas), 2.25], [310e-9],
                                                  1240e-9, th_a, "p")[2][0])
    n_alam = table(lambda T: np.sqrt(ito_eps(1240e-9, "Alam2016", T, gas)).real)
    out["alam"] = dict(A_cold=float(A_alam(300.0)), n_cold=float(n_alam(300.0)), cases={})
    for label, G in (("g_ep", None), ("G_sarkar", G_SARKAR)):
        rows = scan(gas, A_alam, 310e-9, np.cos(th_a), 150e-15, [50, 100, 150], observe=n_alam, G=G)
        for r in rows:
            r["dn_max"] = r["obs_max"] - float(n_alam(300.0))
        out["alam"]["cases"][label] = dict(rows=rows)
        print(f"Alam film, {label}: cold A {A_alam(300.0):.3f}, n {n_alam(300.0):.3f}; " + "; ".join(
            f"{r['I_GWcm2']} GW/cm^2: Te {r['Te_max']:.0f} K, dn {r['dn_max']:+.3f}, decay {r['obs_decay_fs']:.0f} fs"
            for r in rows))

    # ---------------------------------------------------------------- route 1 film
    th_r = np.deg2rad(75.0)
    lam_grid = np.linspace(1000e-9, 2200e-9, 241)
    A_cold = 1 - np.abs(gamma_film_on_pec(2.25, eps_ito(lam_grid, "Tirole2022"), 7.5e-9, lam_grid, th_r, "p")) ** 2
    lam_p = float(lam_grid[np.argmax(A_cold)])
    A_r1 = table(lambda T: 1 - abs(gamma_film_on_pec(2.25, ito_eps(np.array([lam_p]), "Tirole2022", T, gas),
                                                     7.5e-9, np.array([lam_p]), th_r, "p")[0]) ** 2)
    Is_r1 = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000]
    out["route1"] = dict(lam_pump=lam_p, A_cold=float(A_r1(300.0)), cases={})
    for label, G in (("g_ep", None), ("G_sarkar", G_SARKAR)):
        rows = scan(gas, A_r1, 7.5e-9, np.cos(th_r), 220e-15, Is_r1, G=G)
        drops = np.array([r["drop_max"] for r in rows])
        need = {}
        for target in (0.10, 0.20, 0.23, 0.30, 0.40):
            if drops.max() >= target:
                I_t = float(np.interp(target, drops, Is_r1))
                need[f"{target:.2f}"] = dict(I_GWcm2=I_t, fluence_mJcm2=FLUENCE * I_t * GW * 220e-15 * 0.1)
        out["route1"]["cases"][label] = dict(rows=rows, needed=need)
        print(f"route 1 film, {label}: pump at {lam_p*1e9:.0f} nm, cold A {A_r1(300.0):.3f}; " + "; ".join(
            f"{r['I_GWcm2']}: drop {r['drop_max']*100:.1f} % (Te {r['Te_max']:.0f} K, "
            f"decay {r['drop_decay_fs']:.0f} fs)" for r in rows))
        print("   needed: " + "; ".join(f"drop {k}: {v['I_GWcm2']:.0f} GW/cm^2, {v['fluence_mJcm2']:.1f} mJ/cm^2"
                                         for k, v in need.items()))
    with open(os.path.join(OUT, "ttm_validation.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"saved results/ttm_validation.json ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
