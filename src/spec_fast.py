"""spec_fast.py -- fast switch of Tirole et al. (2023): FDTD against the adiabatic
spectrum, and the two switching forms, at a depth of 10 %."""
import json
import os
import numpy as np
import bench_a4 as b
from spec_a4 import logistic
from mirror import run_pair
from fdtd1d import energies
from runopts import Workers, n_workers


def one(args):
    depth, ts, mom = args
    dfun = logistic(depth, b.T_ARR + ts)
    run, _ = run_pair(b.PROBE, b.T_END, wp_scale=[lambda t: 1 + dfun(t), None], momentum=mom, ref=b.REF)
    U_inc, U_r, _ = energies(run, b.REF)
    f, S = b.spectrum((run["recE"][0] - b.REF["recE"][0]), run["dt"])
    return dict(depth=depth, ts=ts, mom=mom, R=U_r / U_inc, Wmod=run["wmod"].sum() / U_inc,
                w=b.widths(f, S), f=f, S=S / S.max())


def adiabatic(depth, ts):
    f0 = b.C0 / 1300e-9
    tt = np.arange(b.T_ARR - 1.2e-12, b.T_ARR + 1.2e-12, 0.1e-15)
    A = np.exp(-2 * np.log(2) * ((tt - b.T_ARR) / b.T_P) ** 2)
    dd = -logistic(depth, b.T_ARR + ts)(tt)
    grid = np.linspace(0, dd.max(), 600)
    rg = np.array([b.r_tmm(1300e-9, x) for x in grid])
    rt = np.interp(dd, grid, rg.real) + 1j * np.interp(dd, grid, rg.imag)
    F = np.abs(np.fft.fft(rt * A, 1 << 21)) ** 2
    fa = f0 - np.fft.fftfreq(1 << 21, 0.1e-15)
    o = np.argsort(fa)
    fa, F = fa[o], F[o]
    sel = (fa > 150e12) & (fa < 320e12)
    return b.widths(fa[sel], F[sel]), fa[sel], F[sel] / F[sel].max()


if __name__ == "__main__":
    workers = n_workers(__doc__)
    jobs = [(0.10, -30e-15, False), (0.10, -30e-15, True), (0.10, 0.0, True)]
    with Workers(workers, initializer=b.init) as pool:
        res = pool.map(one, jobs)
    out = {}
    for r in res:
        wa, fa, Sa = adiabatic(r["depth"], r["ts"])
        key = f"depth {r['depth']:.2f} switch {r['ts']*1e15:+.0f} fs {'momentum' if r['mom'] else 'current'}"
        out[key] = dict(R=r["R"], Wmod=r["Wmod"], width_fdtd=r["w"]["width"], width_adiabatic=wa["width"],
                        lo=r["w"]["lo"], hi=r["w"]["hi"], fwhm=r["w"]["fwhm"], centroid=r["w"]["centroid"])
        print(f"{key:36s}: R {r['R']:.3f}, W_mod {r['Wmod']:+.4f}, 1e-3 extent FDTD {r['w']['width']/1e12:.1f} THz "
              f"({r['w']['lo']/1e12:.1f}-{r['w']['hi']/1e12:.1f}), adiabatic {wa['width']/1e12:.1f} THz "
              f"({wa['lo']/1e12:.1f}-{wa['hi']/1e12:.1f})")
    for ts in (-90e-15, 0.0, 30e-15):
        wa, _, _ = adiabatic(0.10, ts)
        print(f"adiabatic only, switch {ts*1e15:+.0f} fs: 1e-3 extent {wa['width']/1e12:.1f} THz")
    json.dump(out, open(os.path.join(b.OUT, "a4_fast.json"), "w"), indent=1)
