"""fig_step6.py -- route 3 at prism angles.

(a) Absorbed fraction of the chirped probe against the angle inside the glass:
    best static film and best switched film, adiabatic estimate (lines) and
    full-wave FDTD (markers).
(b) Switched film at 50 deg against the pump delay, FDTD and adiabatic, with
    the best static film.
"""
import json
import os
import numpy as np
import figstyle as fs


def main():
    plt = fs.setup()
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    ad = json.load(open(os.path.join(OUT, "route3_angles_adiabatic.json")))
    fd = json.load(open(os.path.join(OUT, "route3_angles.json")))

    fig, axs = plt.subplots(1, 2, figsize=(fs.FULL, 2.6), constrained_layout=True)

    # (a) against angle
    ax = axs[0]
    crit = np.rad2deg(np.arcsin(1 / 1.5))
    ax.axvspan(crit, 61, color=fs.BAND, lw=0, zorder=0)
    ax.text(51.5, 0.585, "prism coupling", ha="center", va="bottom", fontsize=7.5, color=fs.GREY)
    A = ad["angles"]
    deg = [r["deg"] for r in A]
    ax.plot(deg, [r["switched"] for r in A], color=fs.BLUE, lw=fs.LW_THIN + 0.3, ls=(0, (2, 1.5)))
    ax.plot(deg, [r["static"] for r in A], color=fs.ORANGE, lw=fs.LW_THIN + 0.3, ls=(0, (2, 1.5)))
    S = fd["summary"]
    ax.plot([s["deg"] for s in S], [s["switched"] for s in S], ls="none", marker="o", ms=fs.MS + 0.4,
            color=fs.BLUE, label="switched film")
    ax.plot([s["deg"] for s in S], [s["static"] for s in S], ls="none", marker="s", ms=fs.MS + 0.4,
            color=fs.ORANGE, label="best static film")
    ax.axvline(np.rad2deg(np.arcsin(np.sin(np.deg2rad(65)) / 1.5)), color=fs.GREY, lw=fs.LW_THIN, ls=":")
    ax.text(36.6, 0.585, "65° in air", ha="right", va="bottom", fontsize=7.5, color=fs.GREY)
    ax.set_xlim(29, 61)
    ax.set_ylim(0.57, 1.0)
    ax.set_xlabel("Angle inside the glass (deg)")
    ax.set_ylabel("Absorbed fraction of the probe")
    ax.legend(loc="upper left")
    ax.text(30.5, A[0]["switched"] + 0.012, "adiabatic", fontsize=7, color=fs.GREY, va="bottom")
    fs.grid(ax)
    fs.title(ax, "a", "Best films against angle")

    # (b) against pump delay at 50 deg
    ax = axs[1]
    rs = sorted([r for r in fd["results"] if r["deg"] == 50.0 and r["kind"] == "switched" and not r["momentum"]],
                key=lambda r: r["x"])
    d50 = ad["delay_50deg"]
    ax.plot(d50["offsets_fs"], d50["switched"], color=fs.BLUE, lw=fs.LW_THIN + 0.3, ls=(0, (2, 1.5)),
            label="switched, adiabatic")
    ax.plot([r["x"] * 1e15 for r in rs], [r["eta"] for r in rs], ls="none", marker="o", ms=fs.MS,
            color=fs.BLUE, label="switched, FDTD")
    s50 = [s for s in S if s["deg"] == 50.0][0]
    ax.axhline(s50["static"], color=fs.ORANGE, ls="--", lw=fs.LW_THIN + 0.3, label="best static film, FDTD")
    ax.set_xlim(-700, 0)
    ax.set_ylim(0.75, 0.98)
    ax.set_xlabel("Pump peak relative to the probe centre (fs)")
    ax.set_ylabel("Absorbed fraction of the probe")
    ax.legend(loc="lower center")
    fs.grid(ax)
    fs.title(ax, "b", "Switched film at 50°")

    fs.save(fig, "fig_step6_route3_angles", OUT)
    print("saved")


if __name__ == "__main__":
    main()
