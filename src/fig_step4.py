"""fig_step4.py -- benchmark A5 and route 3 in full-wave FDTD.

(a) Switched lossy layer of Firestein et al.: absorbed fraction against
    Tp/T0 at sigma2 = 0.18 S/m, with the Rozanov limit for the same pulse.
(b) Margin over the Rozanov limit against sigma2 for three pulse bandwidths.
(c) Chirp-matched ITO/Au film: absorbed fraction against pump timing,
    full-wave and adiabatic, with the best static film.
(d) Where the energy goes: reflected, dissipated and modulation work.
"""
import json
import os
import numpy as np
import figstyle as fs


def main():
    plt = fs.setup()
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    a5 = json.load(open(os.path.join(OUT, "a5.json")))
    r3 = json.load(open(os.path.join(OUT, "route3.json")))
    T = a5["table"]

    fig, axs = plt.subplots(2, 2, figsize=(fs.FULL, 4.9), constrained_layout=True)

    # (a) absorbed fraction against Tp/T0 at 0.18 S/m
    ax = axs[0, 0]
    rows = sorted([t for t in T if t["sigma"] == 0.18], key=lambda t: t["ratio"])
    q = np.array([t["ratio"] for t in rows])
    ax.plot(q, [t["rozanov"] for t in rows], color=fs.BLACK, ls="--", lw=fs.LW_THIN + 0.3,
            label="Rozanov limit")
    ax.plot(q, [t["ltv"] for t in rows], color=fs.BLUE, marker="o", ms=fs.MS - 0.8, label="switched")
    ax.plot(q, [t["sig_only"] for t in rows], ls="none", marker="s", ms=fs.MS + 0.6, mfc="none",
            mec=fs.ORANGE, mew=0.9, label="loss switched only")
    ax.plot(q, [t["lti"] for t in rows], color=fs.GREY, label="static, final material")
    ax.set_xlim(0, 3)
    ax.set_ylim(0.5, 1.02)
    ax.set_xlabel("Pulse bandwidth $T_p/T_0$")
    ax.set_ylabel("Absorbed fraction")
    ax.legend(loc="lower right")
    fs.grid(ax)
    fs.title(ax, "a", "Switched layer, σ$_2$ = 0.18 S/m")

    # (b) margin over the bound against sigma2
    ax = axs[0, 1]
    for qq, col, ls in ((0.992, fs.BLUE, "-"), (1.9, fs.TEAL, "--"), (3.0, fs.ORANGE, ":")):
        rr = sorted([t for t in T if t["ratio"] == qq], key=lambda t: t["sigma"])
        s = np.array([t["sigma"] for t in rr])
        d = np.array([t["ltv"] - t["rozanov"] for t in rr])
        ax.semilogx(s, d, color=col, ls=ls, marker="o", ms=fs.MS - 1.2, label=f"$T_p/T_0$ = {qq:g}")
    ax.axhline(0, color=fs.BLACK, lw=fs.LW_THIN)
    ax.text(0.0065, 0.008, "Rozanov limit", color=fs.DARKGREY, fontsize=8, va="bottom")
    ax.set_xlim(0.005, 1.2)
    ax.set_ylim(-0.4, 0.12)
    ax.set_xlabel("Conductivity after the switch σ$_2$ (S/m)")
    ax.set_ylabel("Absorbed fraction minus limit")
    ax.legend(loc="lower right")
    fs.grid(ax)
    fs.title(ax, "b", "Margin over the Rozanov limit")

    # (c) route 3 full-wave
    ax = axs[1, 0]
    res = r3["results"]
    sw = sorted([r for r in res if r["kind"] == "switched" and not r["momentum"]], key=lambda r: r["x"])
    off = np.array([r["x"] for r in sw]) * 1e15
    ax.plot(off, [r["eta"] for r in sw], color=fs.BLUE, marker="o", ms=fs.MS - 0.8, label="switched, FDTD")
    ax.plot(r3["offsets_fs"], r3["adiabatic_switched"], color=fs.BLUE, ls=(0, (2, 2)), lw=fs.LW_THIN + 0.2,
            label="switched, adiabatic")
    best_static = max(r["eta"] for r in res if r["kind"] == "static")
    ax.axhline(best_static, color=fs.ORANGE, ls="--", label="best static film, FDTD")
    ax.set_xlim(-670, -80)
    ax.set_ylim(0.55, 0.9)
    ax.set_xlabel("Pump peak relative to the probe centre (fs)")
    ax.set_ylabel("Absorbed fraction of the probe")
    ax.legend(loc="lower left")
    fs.grid(ax)
    fs.title(ax, "c", "Chirp-matched ITO/Au film")

    # (d) energy accounting
    ax = axs[1, 1]
    r18 = [t for t in T if t["ratio"] == 0.992 and t["sigma"] == 0.18][0]
    best = max(sw, key=lambda r: r["eta"])
    mom = [r for r in res if r["kind"] == "switched" and r["momentum"]][0]
    cases = [("A5 switch", 1 - r18["ltv"], r18["diss"], r18["wmod"]),
             ("A5 loss only", 1 - r18["sig_only"], r18["sig_only"] + r18["sig_only_wmod"], r18["sig_only_wmod"]),
             ("route 3,\ncurrent", 1 - best["eta"], best["diss"], best["Wmod"]),
             ("route 3,\nmomentum", 1 - mom["eta"], mom["diss"], mom["Wmod"])]
    x = np.arange(len(cases))
    wdt = 0.26
    for i, (lab, col) in enumerate((("reflected", fs.GREY), ("dissipated", fs.BLUE), ("modulation work", fs.ORANGE))):
        vals = [c[1 + i] for c in cases]
        ax.bar(x + (i - 1) * wdt, vals, width=wdt * 0.92, color=col, label=lab)
        for xi, v in zip(x, vals, strict=True):
            if i == 2:
                ax.text(xi + (i - 1) * wdt, max(v, 0) + 0.03, f"{v:+.3f}" if abs(v) < 0.01 else f"{v:+.2f}",
                        ha="center", va="bottom", fontsize=6.5, color=fs.DARKGREY, rotation=90)
    ax.axhline(0, color=fs.AXIS, lw=fs.LW_THIN)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in cases], fontsize=7.5)
    ax.set_ylim(0, 1.75)
    ax.set_ylabel("Energy / incident energy")
    ax.legend(loc="upper right", fontsize=7.5)
    fs.grid(ax, axis="y")
    fs.title(ax, "d", "Where the energy goes")

    fs.save(fig, "fig_step4_a5_route3", OUT)
    print("saved")


if __name__ == "__main__":
    main()
