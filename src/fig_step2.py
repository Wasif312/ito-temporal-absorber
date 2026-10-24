"""fig_step2.py -- where the Rozanov bound limits an optical absorber, and chirp matching.

(a) Largest fraction of a Gaussian pulse at 1300 nm that any static,
    non-magnetic absorber of thickness d on a perfect conductor can absorb
    (Rozanov bound, normal incidence).
(b) The static ENZ film of Tirole et al. (2022) under the same pulses, with
    the limit set by its own oblique TM bound.
(c) Chirp-matched switching: the probe wavelength falls in time while the
    absorption dip of the pumped film returns to the blue.
(d) Absorbed fraction of the chirped probe against the pump delay, for the
    switched film and for the best static film.
"""
import os
import numpy as np
import figstyle as fs
from bound_scope import eta_R, static_pulse_absorption
import chirp_scope as cs


def main():
    plt = fs.setup()
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

    TH = np.arcsin(np.sin(np.deg2rad(65)) / 1.5)
    FAC = 1.5 / np.cos(TH)                     # oblique TM bound from glass
    T = np.geomspace(2e-15, 300e-15, 60)

    fig, axs = plt.subplots(2, 2, figsize=(fs.FULL, 4.8), constrained_layout=True)

    # (a) the bound against pulse duration
    ax = axs[0, 0]
    for d, col, ls in ((40e-9, fs.BLUE, "-"), (20e-9, fs.TEAL, "--"), (10e-9, fs.ORANGE, ":")):
        e = [eta_R(x, d) for x in T]
        ax.plot(T * 1e15, e, color=col, ls=ls, label=f"d = {d*1e9:.0f} nm")
    ax.legend(loc="lower right", handlelength=2.0)
    ax.set_xscale("log")
    ax.set_xlim(2, 300)
    ax.set_xticks([2, 5, 10, 20, 50, 100, 200])
    ax.set_xticklabels(["2", "5", "10", "20", "50", "100", "200"])
    ax.minorticks_off()
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Pulse duration (fs)")
    ax.set_ylabel("Largest absorbed fraction")
    fs.grid(ax)
    fs.title(ax, "a", "Rozanov limit, normal incidence")

    # (b) the static ENZ film and its own bound
    ax = axs[0, 1]
    a_au = [static_pulse_absorption(x, backing="au") for x in T]
    lim = [eta_R(x, 40e-9, factor=FAC) for x in T]
    ax.plot(T * 1e15, lim, color=fs.GREY, ls="--")
    ax.plot(T * 1e15, a_au, color=fs.BLUE)
    fs.label_end(ax, 40, 1.0, "limit for 40 nm", dx=0, dy=-4, va="top", ha="center", color=fs.GREY)
    fs.label_end(ax, 60, np.interp(60e-15, T, a_au), "static ITO/Au", dx=0, dy=-6, va="top", ha="left")
    ax.set_xscale("log")
    ax.set_xlim(2, 300)
    ax.set_xticks([2, 5, 10, 20, 50, 100, 200])
    ax.set_xticklabels(["2", "5", "10", "20", "50", "100", "200"])
    ax.minorticks_off()
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Pulse duration (fs)")
    ax.set_ylabel("Absorbed fraction")
    fs.grid(ax)
    fs.title(ax, "b", "Static ENZ film, TM, 65°")

    # (c) mechanism, the case with the largest gain for a 150 nm span
    lam_c, span, Tp, tau = 1370e-9, 150e-9, 450e-15, 210e-15
    delays = np.arange(-900e-15, 301e-15, 25e-15)
    a_sw = np.array([cs.absorbed(lam_c, span, Tp, cs.drop_profile(x, cs.DMAX, tau_dec=tau))
                     for x in delays])
    best = delays[np.argmax(a_sw)]
    a_st = max(cs.absorbed(lam_c, span, Tp, np.full_like(cs.t, x)) for x in cs.dl)

    ax = axs[1, 0]
    def dip(row):
        """Dip wavelength with a parabolic refinement around the grid minimum."""
        i = int(np.clip(np.argmin(row), 1, len(row) - 2))
        y0, y1, y2 = row[i - 1], row[i], row[i + 1]
        off = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
        return cs.lam[i] + off * (cs.lam[1] - cs.lam[0])


    lam_dip = np.array([dip(r) for r in cs.Rmap])          # dip wavelength for each drop
    drop = cs.drop_profile(best, cs.DMAX, tau_dec=tau)
    tt = cs.t * 1e15
    I = np.exp(-4 * np.log(2) * (cs.t / Tp) ** 2)
    ax.fill_between(tt, 1200, 1200 + 300 * I, color=fs.BAND, lw=0)
    ax.plot(tt, (lam_c - span * cs.t / Tp) * 1e9, color=fs.ORANGE, ls="--")
    ax.plot(tt, np.interp(drop, cs.dl, lam_dip) * 1e9, color=fs.BLUE)
    fs.label_end(ax, -300, 1476, "probe wavelength", dx=0, dy=0, ha="left", va="bottom", color=fs.DARKGREY)
    fs.label_end(ax, 430, 1314, "absorption dip", dx=0, dy=0, ha="left", va="bottom", color=fs.DARKGREY)
    fs.label_end(ax, -770, 1206, "probe intensity", dx=0, dy=0, ha="left", va="bottom", color=fs.GREY)
    ax.set_xlim(-800, 1000)
    ax.set_ylim(1200, 1500)
    ax.set_xlabel("Time (fs)")
    ax.set_ylabel("Wavelength (nm)")
    fs.grid(ax)
    fs.title(ax, "c", "Dip follows a chirped probe")

    # (d) absorbed fraction against pump delay
    ax = axs[1, 1]
    ax.plot(delays * 1e15, a_sw, color=fs.BLUE)
    ax.axhline(a_st, color=fs.ORANGE, ls="--", lw=fs.LW)
    fs.label_end(ax, 300, a_st, "best static film", dx=-2, dy=-4, ha="right", va="top")
    fs.label_end(ax, best * 1e15, a_sw.max(), "switched", dx=6, dy=2, ha="left", va="bottom")
    ax.set_xlim(-900, 300)
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("Pump time relative to the probe centre (fs)")
    ax.set_ylabel("Absorbed fraction")
    fs.grid(ax)
    fs.title(ax, "d", "Adiabatic estimate, 150 nm chirp")

    fs.save(fig, "fig_step2_scope", OUT)
    print(f"switched {a_sw.max():.3f} at delay {best*1e15:.0f} fs; best static {a_st:.3f}")
    print(f"static film absorbs {np.interp(10e-15, T, a_au):.3f} of a 10 fs pulse; "
          f"limit {np.interp(10e-15, T, lim):.3f}")


if __name__ == "__main__":
    main()
