"""spec_a4.py -- spectra of the reflected probe for benchmark A4, saved in full.

Same runs as the spectral part of bench_a4.py (delay -90 fs), with the whole
spectrum between 200 and 260 THz kept for the figure.
"""
import json
import os
import numpy as np
import bench_a4 as b
from ito_model import envelope_tirole2023
from mirror import run_pair
from fdtd1d import energies
from runopts import Workers, n_workers

OUT = b.OUT
_tt = np.arange(-50e-15, 3000e-15, 0.05e-15)
F_MAX = envelope_tirole2023(_tt).max()


def logistic(depth, t_switch):
    """Fast switch of Tirole et al. (2023): rise rate 0.5 fs^-1, decay 400 fs."""
    return lambda t: -depth * envelope_tirole2023(np.asarray(t) - t_switch) / F_MAX


def one_log(args):
    depth, ts_rel = args
    dfun = logistic(depth, b.T_ARR + ts_rel)
    run, _ = run_pair(b.PROBE, b.T_END, wp_scale=[lambda t: 1 + dfun(t), None], ref=b.REF)
    U_inc, U_r, U_t = energies(run, b.REF)
    return dict(depth=depth, ts=ts_rel, R=U_r / U_inc, Wmod=run["wmod"].sum() / U_inc,
                Er=(run["recE"][0] - b.REF["recE"][0]).astype(np.float32), dt=run["dt"])


def adiabatic_spectrum(depth, rise, tau):
    f0 = b.C0 / 1300e-9
    tt = np.arange(b.T_ARR - 1.2e-12, b.T_ARR + 1.2e-12, 0.2e-15)
    A = np.exp(-2 * np.log(2) * ((tt - b.T_ARR) / b.T_P) ** 2)
    dfun = b.delta_profile(b.T_ARR - tau, depth, tau_rise=rise)
    dd = -dfun(tt)
    grid = np.linspace(0, max(dd.max(), 1e-6), 400)
    rg = np.array([b.r_tmm(1300e-9, x) for x in grid])
    rt = np.interp(dd, grid, rg.real) + 1j * np.interp(dd, grid, rg.imag)
    F = np.abs(np.fft.fft(rt * A, 1 << 20)) ** 2
    fa = f0 - np.fft.fftfreq(1 << 20, 0.2e-15)
    o = np.argsort(fa)
    return fa[o], F[o]


if __name__ == "__main__":
    workers = n_workers(__doc__)
    cases = [(0.0, 30e-15, False), (0.10, 30e-15, False), (0.10, 10e-15, False),
             (0.10, 3e-15, False), (0.10, 1e-15, False), (0.20, 30e-15, False),
             (0.30, 30e-15, False), (0.10, 3e-15, True), (0.10, 1e-15, True)]
    jobs = [(d, -90e-15, m, r, True) for d, r, m in cases]
    log_jobs = [(dd, ts) for dd in (0.10, 0.20) for ts in (-150e-15, -90e-15, -30e-15, 0.0, 30e-15, 90e-15)]
    with Workers(workers, initializer=b.init) as pool:
        res = pool.map(b.one, jobs)
        res_log = pool.map(one_log, log_jobs)
    out = {}
    for r in res:
        f, S = b.spectrum(r["Er"].astype(float), r["dt"])
        w = b.widths(f, S)
        fa, Fa = adiabatic_spectrum(r["depth"], r["rise"], r["tau"])
        wa = b.widths(fa[(fa > 150e12) & (fa < 320e12)], Fa[(fa > 150e12) & (fa < 320e12)])
        key = f"depth {r['depth']:.2f} rise {r['rise']*1e15:.0f} fs {'momentum' if r['momentum'] else 'current'}"
        keep = (f > 200e12) & (f < 260e12)
        keepa = (fa > 200e12) & (fa < 260e12)
        out[key] = dict(R=r["R"], Wmod=r["Wmod"],
                        fdtd={k: float(v) for k, v in w.items()},
                        adiabatic={k: float(v) for k, v in wa.items()},
                        f_THz=(f[keep] / 1e12).tolist(), S=(S[keep] / S.max()).tolist(),
                        fa_THz=(fa[keepa][::4] / 1e12).tolist(), Sa=(Fa[keepa][::4] / Fa.max()).tolist())
        print(f"{key:36s}: R {r['R']:.3f}, W_mod {r['Wmod']:+.4f}; 1e-3 extent FDTD "
              f"{w['lo']/1e12:.1f}-{w['hi']/1e12:.1f} THz ({w['width']/1e12:.1f}), adiabatic "
              f"{wa['width']/1e12:.1f} THz; FWHM {w['fwhm']/1e12:.2f} THz; centroid {w['centroid']/1e12:.2f} THz")
    for r in res_log:
        f, S = b.spectrum(r["Er"].astype(float), r["dt"])
        w = b.widths(f, S)
        key = f"logistic depth {r['depth']:.2f} switch {r['ts']*1e15:+.0f} fs"
        keep = (f > 200e12) & (f < 260e12)
        out[key] = dict(R=r["R"], Wmod=r["Wmod"], fdtd={k: float(v) for k, v in w.items()},
                        f_THz=(f[keep] / 1e12).tolist(), S=(S[keep] / S.max()).tolist())
        print(f"{key:36s}: R {r['R']:.3f}, W_mod {r['Wmod']:+.4f}; 1e-3 extent "
              f"{w['lo']/1e12:.1f}-{w['hi']/1e12:.1f} THz ({w['width']/1e12:.1f}); FWHM {w['fwhm']/1e12:.2f} THz; "
              f"centroid {w['centroid']/1e12:.2f} THz")
    with open(os.path.join(OUT, "a4_spectra.json"), "w") as fh:
        json.dump(out, fh)
    print("saved results/a4_spectra.json")
