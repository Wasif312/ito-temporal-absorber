"""route3_fdtd.py -- the chirp-matched absorber in full-wave FDTD.

Probe: Gaussian intensity envelope of FWHM 450 fs centred at 1370 nm, with a
linear chirp whose instantaneous frequency rises by the amount that takes the
wavelength from 1445 nm to 1295 nm across the FWHM (red first). ITO/Au mirror
of Tirole et al. (2022), p-polarised at the in-plane wavevector of 1370 nm at
65 deg in air (37.17 deg in the glass at the centre wavelength).

Static films: any fixed drop of wp between 0 and 10 %. Switched film: the
pump-induced drop of Tirole et al. (220 fs pump, 30 fs rise, 210 fs decay,
largest drop 10 %), scanned over pump timing; both switching forms at the
best timing. Energy balance in every run, and the adiabatic estimate at the
same in-plane wavevector for comparison. Results go to results/route3.json.
"""
import json
import os
import numpy as np
from fdtd1d import simulate, energies, C0
from mirror import stack, gaussian_probe, arrival_time, delta_profile
from ito_model import eps_ito, eps_gold, tmm
from runopts import Workers, n_workers

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
LC = 1370e-9
WC = 2 * np.pi * C0 / LC
KX = 2 * np.pi / LC * np.sin(np.deg2rad(65))
T_P = 450e-15
DW = 2 * np.pi * C0 * (1 / 1295e-9 - 1 / 1445e-9)        # sweep across the FWHM
CHIRP = DW / T_P                                           # rad s^-2
TC = 3 * T_P
T_END = TC + 3 * T_P + 150e-15
PROBE = gaussian_probe(T_P, TC, w0=WC, chirp=CHIRP)
T_ARR = arrival_time(TC)
DMAX = 0.10
REF = None


def init():
    global REF
    REF = simulate(stack(), KX, PROBE, T_END, reference=True)


def one(args):
    kind, x, mom = args
    if kind == "static":
        scale = [lambda t, x=x: np.full_like(np.asarray(t, float), 1 - x), None]
    else:
        dfun = delta_profile(T_ARR + x, DMAX)
        scale = [lambda t: 1 + dfun(t), None]
    run = simulate(stack(), KX, PROBE, T_END, wp_scale=scale, momentum=mom)
    U_inc, U_r, U_t = energies(run, REF)
    W, D = run["wmod"].sum(), run["diss"].sum()
    return dict(kind=kind, x=x, momentum=mom, eta=1 - U_r / U_inc, Wmod=W / U_inc, diss=D / U_inc,
                T=U_t / U_inc, balance=(U_inc + W - U_r - U_t - D - run["u_end"]) / U_inc)


def adiabatic(pump_offsets, drops_static):
    """Instantaneous-frequency estimate at the same in-plane wavevector."""
    t = np.arange(-4 * T_P, 4 * T_P, 1e-15)
    I = np.exp(-4 * np.log(2) * (t / T_P) ** 2)
    w_t = WC + CHIRP * t
    lam_t = 2 * np.pi * C0 / w_t
    lam_g = np.linspace(1150e-9, 1650e-9, 201)
    dl = np.linspace(0, DMAX, 41)
    def th(lam):
        return np.arcsin(KX * lam / (2 * np.pi * 1.5))
    Rm = np.array([[tmm([2.25, eps_ito(lam, "Tirole2022", wp_scale=1 - x), eps_gold(lam), 1.0],
                        [40e-9, 100e-9], lam, th(lam), "p")[0] for lam in lam_g] for x in dl])

    def R_of(lam, drop):
        i = np.clip((lam - lam_g[0]) / (lam_g[1] - lam_g[0]), 0, len(lam_g) - 1.001)
        j = np.clip(drop / (dl[1] - dl[0]), 0, len(dl) - 1.001)
        i0, j0 = i.astype(int), j.astype(int)
        fi, fj = i - i0, j - j0
        return ((1 - fi) * (1 - fj) * Rm[j0, i0] + fi * (1 - fj) * Rm[j0, i0 + 1]
                + (1 - fi) * fj * Rm[j0 + 1, i0] + fi * fj * Rm[j0 + 1, i0 + 1])
    st = [float(np.trapezoid(I * (1 - R_of(lam_t, np.full_like(t, x))), t) / np.trapezoid(I, t))
          for x in drops_static]
    sw = []
    for off in pump_offsets:
        dfun = delta_profile(off, DMAX)
        sw.append(float(np.trapezoid(I * (1 - R_of(lam_t, -dfun(t))), t) / np.trapezoid(I, t)))
    return st, sw


if __name__ == "__main__":
    workers = n_workers(__doc__)
    drops = np.round(np.arange(0, 0.1001, 0.005), 4)
    offsets = np.arange(-650e-15, -99e-15, 50e-15)
    jobs = [("static", float(x), False) for x in drops] + [("switched", float(o), False) for o in offsets]
    with Workers(workers, initializer=init) as pool:
        res = pool.map(one, jobs)
        best = max([r for r in res if r["kind"] == "switched"], key=lambda r: r["eta"])
        extra = pool.map(one, [("switched", best["x"], True)])
    res += extra
    st_ad, sw_ad = adiabatic(offsets, drops)
    s_best = max([r for r in res if r["kind"] == "static"], key=lambda r: r["eta"])
    print(f"best static film: absorbed {s_best['eta']:.3f} at a fixed drop of {s_best['x']*100:.1f} % "
          f"(adiabatic estimate best {max(st_ad):.3f})")
    for r, a in zip([r for r in res if r["kind"] == "switched" and not r["momentum"]], sw_ad, strict=True):
        print(f"   pump {r['x']*1e15:+5.0f} fs: switched absorbed {r['eta']:.3f} (adiabatic {a:.3f}), "
              f"W_mod {r['Wmod']:+.4f}, balance {r['balance']:+.1e}")
    m = extra[0]
    print(f"best switched {best['eta']:.3f} at pump {best['x']*1e15:+.0f} fs; momentum form {m['eta']:.3f} "
          f"(W_mod {m['Wmod']:+.4f}); gain over best static {best['eta'] - s_best['eta']:+.3f}")
    out = dict(results=res, adiabatic_static=st_ad, adiabatic_switched=sw_ad,
               drops=drops.tolist(), offsets_fs=(offsets * 1e15).tolist())
    with open(os.path.join(OUT, "route3.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("saved results/route3.json")
