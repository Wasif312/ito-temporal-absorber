"""bench_a4.py -- benchmark A4, the time-resolved ITO/Au mirror of Tirole et al. (2022).

Full-wave FDTD of a 220 fs probe at 1300 nm (65 deg in air) reflected by the
mirror while the plasma frequency of ITO follows the model of the paper: a
220 fs pump convolved with a kernel that rises linearly over 30 fs and decays
with 210 fs. The modulation depth (largest drop of wp) sets the pump
intensity; 10 % is the depth that reproduces the saturated reflectance 0.66.

1. Reflectance against pump-probe delay for depths of 2.5, 5 and 10 %
   (current-continuous form) and 10 % (momentum-continuous form), with the
   adiabatic estimate for comparison; 10-90 % rise time and decay time.
2. Spectrum of the reflected probe at a delay of -90 fs, where Tirole et al.
   saw the largest broadening (214-245 THz at 1e-3 of the peak, 708 GW cm^-2):
   full-wave and adiabatic, for rise times of 30, 10, 3 and 1 fs and depths of
   10, 20 and 30 %.
Results go to results/a4.json.
"""
import json
import os
import time
import numpy as np
from fdtd1d import C0, energies
from mirror import KX, gaussian_probe, arrival_time, delta_profile, run_pair
from ito_model import eps_ito, eps_gold, tmm
from runopts import Workers, n_workers

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
T_P = 220e-15
TC = 700e-15
T_END = TC + 750e-15
PROBE = gaussian_probe(T_P, TC)
T_ARR = arrival_time(TC)
REF = None


def r_tmm(lam, drop):
    th = np.arcsin(KX * lam / (2 * np.pi * 1.5))
    return tmm([2.25, eps_ito(lam, "Tirole2022", wp_scale=1 - drop), eps_gold(lam), 1.0],
               [40e-9, 100e-9], lam, th, "p", return_r=True)


def init():
    global REF
    _, REF = run_pair(PROBE, T_END)


def one(args):
    depth, tau, mom, rise, keep = args
    dfun = delta_profile(T_ARR - tau, depth, tau_rise=rise)
    run, _ = run_pair(PROBE, T_END, wp_scale=[lambda t: 1 + dfun(t), None], momentum=mom, ref=REF)
    U_inc, U_r, U_t = energies(run, REF)
    out = dict(depth=depth, tau=tau, momentum=mom, rise=rise, R=U_r / U_inc,
               Wmod=run["wmod"].sum() / U_inc, diss=run["diss"].sum() / U_inc)
    if keep:
        out["Er"] = (run["recE"][0] - REF["recE"][0]).astype(np.float32)
        out["dt"] = run["dt"]
    return out


def spectrum(E, dt, nfft=1 << 22):
    F = np.abs(np.fft.rfft(E, nfft)) ** 2
    f = np.fft.rfftfreq(nfft, dt)
    sel = (f > 150e12) & (f < 320e12)
    return f[sel], F[sel]


def widths(f, S, level=1e-3):
    S = S / S.max()
    above = np.nonzero(S >= level)[0]
    half = np.nonzero(S >= 0.5)[0]
    centroid = np.sum(f * S) / np.sum(S)
    return dict(lo=f[above[0]], hi=f[above[-1]], width=f[above[-1]] - f[above[0]],
                fwhm=f[half[-1]] - f[half[0]], centroid=centroid)


if __name__ == "__main__":
    workers = n_workers(__doc__)
    t0 = time.time()
    init()
    delays = np.concatenate([np.arange(-600, 400, 20), np.arange(400, 1201, 40)]) * 1e-15
    jobs = [(d, tau, False, 30e-15, False) for d in (0.025, 0.05, 0.10) for tau in delays]
    jobs += [(0.10, tau, True, 30e-15, False) for tau in delays]
    spec_cases = [(0.10, r) for r in (30e-15, 10e-15, 3e-15, 1e-15)] + [(0.20, 30e-15), (0.30, 30e-15)]
    jobs += [(d, -90e-15, False, r, True) for d, r in spec_cases]
    jobs += [(0.10, -90e-15, True, r, True) for r in (3e-15, 1e-15)]
    jobs += [(0.0, -90e-15, False, 30e-15, True)]
    print(f"{len(jobs)} runs")
    with Workers(workers, initializer=init) as pool:
        results = pool.map(one, jobs, chunksize=4)
    print(f"done in {(time.time()-t0)/60:.1f} min")

    # adiabatic estimates of the traces: probe-weighted, spectrally averaged
    f0 = C0 / 1300e-9
    fs_ = np.linspace(f0 - 4e12, f0 + 4e12, 41)
    S = np.exp(-4 * np.log(2) * ((fs_ - f0) / (0.441 / T_P)) ** 2)
    drops = np.linspace(0, 0.32, 161)
    Rbar = np.array([np.sum(S * np.abs([r_tmm(C0 / f, x) for f in fs_]) ** 2) / S.sum() for x in drops])
    tt = np.arange(T_ARR - 2e-12, T_ARR + 2.5e-12, 1e-15)
    Ip = np.exp(-4 * np.log(2) * ((tt - T_ARR) / T_P) ** 2)
    out = dict(delays_fs=(delays * 1e15).tolist(), traces={}, adiabatic={}, inst={}, spectra={})
    for d in (0.025, 0.05, 0.10):
        for mom in (False, True):
            rows = [r for r in results if r["depth"] == d and r["momentum"] == mom
                    and r["rise"] == 30e-15 and "Er" not in r]
            if not rows:
                continue
            rows.sort(key=lambda r: r["tau"])
            out["traces"][f"{d:.3f} {'momentum' if mom else 'current'}"] = dict(
                R=[r["R"] for r in rows], Wmod=[r["Wmod"] for r in rows])
        Rad = []
        Rinst = None
        for tau in delays:
            dfun = delta_profile(T_ARR - tau, d)
            Rad.append(float(np.sum(Ip * np.interp(-dfun(tt), drops, Rbar)) / Ip.sum()))
        out["adiabatic"][f"{d:.3f}"] = Rad
        # instantaneous reflectance (no probe averaging), pump peak at t = 0
        dfun = delta_profile(0.0, d)
        ti = np.arange(-600e-15, 1200e-15, 2e-15)
        out["inst"][f"{d:.3f}"] = dict(t_fs=(ti * 1e15).tolist(),
                                      R=np.interp(-dfun(ti), drops, Rbar).tolist())

    # spectra of the reflected probe
    for r in results:
        if "Er" not in r:
            continue
        f, Sx = spectrum(r["Er"].astype(float), r["dt"])
        w = widths(f, Sx)
        key = f"depth {r['depth']:.2f} rise {r['rise']*1e15:.0f} fs {'momentum' if r['momentum'] else 'current'}"
        # adiabatic field: r(t) times the incident envelope, at the carrier
        tt2 = np.arange(T_ARR - 1.2e-12, T_ARR + 1.2e-12, 0.2e-15)
        A = np.exp(-2 * np.log(2) * ((tt2 - T_ARR) / T_P) ** 2)
        dfun = delta_profile(T_ARR - r["tau"], r["depth"], tau_rise=r["rise"])
        dd = -dfun(tt2)
        grid = np.linspace(0, max(dd.max(), 1e-6), 400)
        rg = np.array([r_tmm(1300e-9, x) for x in grid])
        rt = np.interp(dd, grid, rg.real) + 1j * np.interp(dd, grid, rg.imag)
        Fa = np.abs(np.fft.fft(rt * A, 1 << 20)) ** 2
        # exp(-i w t) convention: numpy places a component exp(-i dw t) at -dw
        fa = f0 - np.fft.fftfreq(1 << 20, 0.2e-15)
        order = np.argsort(fa)
        fa, Fa = fa[order], Fa[order]
        wa = widths(fa, Fa)
        out["spectra"][key] = dict(R=r["R"], fdtd={k: float(v) for k, v in w.items()},
                                   adiabatic={k: float(v) for k, v in wa.items()},
                                   f_THz=(f[::40] / 1e12).tolist(), S=(Sx[::40] / Sx.max()).tolist())
        print(f"   {key:36s}: R {r['R']:.3f}  1e-3 width FDTD {w['width']/1e12:5.1f} THz "
              f"({w['lo']/1e12:.1f}-{w['hi']/1e12:.1f}), adiabatic {wa['width']/1e12:5.1f} THz, "
              f"FWHM {w['fwhm']/1e12:.2f} THz, centroid {w['centroid']/1e12:.2f} THz")

    with open(os.path.join(OUT, "a4.json"), "w") as fh:
        json.dump(out, fh)
    print("saved results/a4.json")
