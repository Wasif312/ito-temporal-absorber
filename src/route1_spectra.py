"""route1_spectra.py -- spectra behind the best full-wave route-1 design.

For the design with a 30 % drop in results/route1_fdtd.json: incident and
reflected energy spectra per unit wavelength for the best static film and the
switched film, and the reflected spectrum of the water-filling optimum, the
best that any static non-magnetic absorber of the same thickness could do at
this angle.
Also the probe wavelength and the dip wavelength of the pumped film against
time, in the adiabatic picture. Results go to results/route1_spectra.json.
"""
import json
import os
import numpy as np
from angle1d import simulate, energies
from fdtd1d import C0
from mirror import delta_profile
from slab_drude import spectrum
from bound_scope import rozanov_max_absorption
from r1_design import Design, sweep, N0, LAM, DROPS
from route1_fdtd import film, h_for, setup, Z_SRC, LAM_S, OUT

if __name__ == "__main__":
    data = json.load(open(os.path.join(OUT, "route1_fdtd.json")))
    tag = "best_0.30"
    des = data["designs"][tag]
    summ = [s for s in data["summary"] if s["tag"] == tag][0]
    s = setup(des, True)
    kw = dict(h_fine=h_for(des["d"]), z_src=Z_SRC)
    ref = simulate(film(des["d"]), s["th"], s["probe"], s["t_end"], reference=True, **kw)
    x = summ["static_drop"]
    st = simulate(film(des["d"]), s["th"], s["probe"], s["t_end"],
                  wp_scale=[lambda t: np.full_like(np.asarray(t, float), 1 - x)], **kw)
    dfun = delta_profile(s["t_arr"] + summ["pump"], des["depth"])
    sw = simulate(film(des["d"]), s["th"], s["probe"], s["t_end"], wp_scale=[lambda t: 1 + dfun(t)], **kw)
    dec = max(1, int(0.25e-15 / ref["dt"]))
    f = C0 / LAM_S
    w = 2 * np.pi * f
    Ei = spectrum(ref["recE"][0], ref["dt"], f, decimate=dec)
    Es = spectrum(st["recE"][0] - ref["recE"][0], ref["dt"], f, decimate=dec)
    Ew = spectrum(sw["recE"][0] - ref["recE"][0], ref["dt"], f, decimate=dec)
    Si = np.abs(Ei) ** 2 * w ** 2
    norm = Si.max()
    Si, Ss, Sw = Si / norm, np.abs(Es) ** 2 * w ** 2 / norm, np.abs(Ew) ** 2 * w ** 2 / norm
    B = 2 * np.pi ** 2 * N0 * des["d"] / np.cos(np.deg2rad(des["deg"]))
    eta, kappa = rozanov_max_absorption(Si, LAM_S, B)
    S_opt = np.minimum(Si, kappa)                     # reflected spectrum of the optimum
    tot = np.trapezoid(Si, LAM_S)
    print(f"absorbed: static {1 - np.trapezoid(Ss, LAM_S)/tot:.4f}, switched {1 - np.trapezoid(Sw, LAM_S)/tot:.4f}, "
          f"water-filling optimum {1 - np.trapezoid(S_opt, LAM_S)/tot:.4f} (eta_R {eta:.4f})")
    ui, ur = energies(sw, ref)
    print(f"switched film from the fluxes: {1 - ur/ui:.4f}")
    # adiabatic tracking picture
    D = Design(des["d"], des["deg"], des["depth"])
    tt = np.arange(-2.5 * des["T"], 2.5 * des["T"], 1e-15)
    wc, dw = sweep(des["lam_c"], des["span"])
    lam_t = 2 * np.pi * C0 / (wc + dw / des["T"] * tt)
    prof = delta_profile(summ["pump"], des["depth"])(tt)
    A = 1 - D.R                                       # absorptance map (drop, wavelength)
    dip = np.array([LAM[np.argmax(A[np.argmin(np.abs(DROPS + p))])] for p in prof])
    I = np.exp(-4 * np.log(2) * (tt / des["T"]) ** 2)
    out = dict(tag=tag, design=des, lam=LAM_S.tolist(), S_inc=Si.tolist(), S_static=Ss.tolist(),
               S_switched=Sw.tolist(), S_optimum=S_opt.tolist(), kappa=float(kappa), eta_R=float(eta),
               t_fs=(tt * 1e15).tolist(), lam_probe=lam_t.tolist(), lam_dip=dip.tolist(), intensity=I.tolist(),
               drop=(-prof).tolist())
    with open(os.path.join(OUT, "route1_spectra.json"), "w") as fh:
        json.dump(out, fh)
    print("saved results/route1_spectra.json")
