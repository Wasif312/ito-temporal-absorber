"""fig_step3.py -- the FDTD solver checked, and benchmark A4.

(a) Static ITO/Au mirror: FDTD reflectance against the transfer matrix.
(b) Uniform ITO whose plasma frequency drops by 10 % at 20 fs: FDTD field for
    the two switching models against the exact solution.
(c) Reflectance against pump-probe delay, FDTD and adiabatic, for three
    modulation depths.
(d) Spectrum of the reflected probe at -90 fs delay.
"""
import json
import os
import numpy as np
import figstyle as fs
from bulk import run_bulk, step_scale
from matplotlib.lines import Line2D


def main():
    plt = fs.setup()
    HERE = os.path.dirname(os.path.abspath(__file__))
    OUT = os.path.join(HERE, "results")
    tests = json.load(open(os.path.join(OUT, "tests.json")))
    a4 = json.load(open(os.path.join(OUT, "a4.json")))
    sp = json.load(open(os.path.join(OUT, "a4_spectra.json")))

    fig, axs = plt.subplots(2, 2, figsize=(fs.FULL, 4.9), constrained_layout=True)

    # (a) static spectrum
    ax = axs[0, 0]
    t1 = tests["T1"]["10"]
    lam, Rf, Rt = map(np.array, (t1["lam_nm"], t1["R_fdtd"], t1["R_tmm"]))
    o = np.argsort(lam)
    ax.plot(lam[o], Rt[o], color=fs.BLUE, label="transfer matrix")
    ax.plot(lam[o][::2], Rf[o][::2], ls="none", marker="o", ms=fs.MS - 0.6, mfc="white",
            mec=fs.BLACK, mew=0.8, label="FDTD")
    ax.text(0.97, 0.04, f"largest difference\n{t1['max_err']*1e4:.0f} × 10$^{{-4}}$",
            transform=ax.transAxes, color=fs.DARKGREY, fontsize=8, ha="right", va="bottom", linespacing=1.3)
    ax.set_xlim(1100, 1600)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.legend(loc="upper center")
    fs.grid(ax)
    fs.title(ax, "a", "Static mirror, 65°")

    # (b) uniform medium switched at 20 fs
    ax = axs[0, 1]
    sc = step_scale(20e-15, 0.10)
    for mom, col, ls, lab in ((False, fs.BLUE, "-", "current form"),
                              (True, fs.ORANGE, "--", "momentum form")):
        r = run_bulk(sc, mom, 0.0, courant=0.25, t_end=45e-15)
        t = r["t"] * 1e15
        sel = t > 8
        ax.plot(t[sel], r["E"][sel], color=col, ls=ls, label=lab)
        step = max(1, int(0.4e-15 / r["dt"]))
        ax.plot(t[sel][::step], r["E_exact"][sel][::step], ls="none", marker=".", ms=1.6,
                color=fs.BLACK, label="exact" if mom else None)
    ax.axvline(20, color=fs.GREY, lw=fs.LW_THIN, ls=":")
    ax.set_xlim(8, 45)
    ax.set_ylim(-1.3, 1.95)
    ax.set_xlabel("Time (fs)")
    ax.set_ylabel("Field $E_x$ (normalised)")
    ax.legend(loc="upper center", ncol=3, handlelength=1.6, columnspacing=1.0)
    ax.text(20.5, -1.2, "switch", color=fs.GREY, fontsize=8, va="center")
    fs.grid(ax)
    fs.title(ax, "b", "Uniform ITO, ω$_p$ drops by 10 %")

    # (c) pump-probe traces
    ax = axs[1, 0]
    d = np.array(a4["delays_fs"])
    for key, col, lab in (("0.100", fs.BLUE, "10 %"), ("0.050", fs.TEAL, "5 %"), ("0.025", fs.ORANGE, "2.5 %")):
        R = np.array(a4["traces"][f"{key} current"]["R"])
        ax.plot(d, R, color=col, label=lab)
        ax.plot(d, a4["adiabatic"][key], color=col, ls=(0, (2, 2)), lw=fs.LW_THIN + 0.2)
    Rm = np.array(a4["traces"]["0.100 momentum"]["R"])
    ax.plot(d[::3], Rm[::3], ls="none", marker="o", ms=fs.MS - 1.2, mfc="none", mec=fs.BLACK, mew=0.6)
    ax.axhline(0.66, color=fs.GREY, lw=fs.LW_THIN, ls=":")
    key = [Line2D([], [], color=fs.DARKGREY, lw=fs.LW, label="FDTD"),
           Line2D([], [], color=fs.DARKGREY, lw=fs.LW_THIN + 0.2, ls=(0, (2, 2)), label="adiabatic"),
           Line2D([], [], ls="none", marker="o", ms=fs.MS - 1.2, mfc="none", mec=fs.BLACK, mew=0.6,
                  label="momentum")]
    leg2 = ax.legend(handles=key, loc="upper left", ncol=3, handlelength=1.5, columnspacing=0.8,
                     handletextpad=0.5, fontsize=7.5)
    ax.add_artist(leg2)
    ax.text(1180, 0.675, "measured 0.66", color=fs.GREY, fontsize=8, ha="right", va="bottom")
    ax.set_xlim(-600, 1200)
    ax.set_ylim(0, 0.86)
    ax.set_xlabel("Pump–probe delay (fs)")
    ax.set_ylabel("Reflectance of the probe")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.80), title="depth", title_fontsize=8,
              handlelength=1.6)
    fs.grid(ax)
    fs.title(ax, "c", "Pump–probe trace, FDTD and adiabatic")

    # (d) reflected spectra at -90 fs
    ax = axs[1, 1]
    ax.axvspan(214, 245, color=fs.BAND, lw=0, zorder=0)
    for key, col, ls, lab in (("depth 0.00 rise 30 fs current", fs.GREY, "-", "unpumped"),
                              ("depth 0.10 rise 30 fs current", fs.BLUE, "-", "10 %, pump"),
                              ("depth 0.30 rise 30 fs current", fs.TEAL, ":", "30 %, pump"),
                              ("logistic depth 0.10 switch -30 fs", fs.ORANGE, "--", "10 %, fast")):
        s = sp[key]
        ax.semilogy(s["f_THz"], np.maximum(s["S"], 1e-9), color=col, ls=ls, label=lab)
    ax.axhline(1e-3, color=fs.DARKGREY, lw=fs.LW_THIN, ls=":")
    ax.text(254, 1.4e-3, "10$^{-3}$", color=fs.DARKGREY, fontsize=8, ha="right", va="bottom")
    ax.set_xlim(205, 255)
    ax.set_ylim(1e-6, 100)
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Reflected spectrum (normalised)")
    ax.legend(loc="upper left", handlelength=1.8, fontsize=7.5)
    fs.grid(ax)
    fs.title(ax, "d", "Reflected probe spectrum")

    fs.save(fig, "fig_step3_fdtd_a4", OUT)
    print("saved")


if __name__ == "__main__":
    main()
