"""fig_step7.py -- route 1 with a chirped probe.

(a) Probe wavelength and the absorption dip of the pumped film against time
    (best design with a 30 % drop, adiabatic picture).
(b) Incident and reflected energy spectra: best static film, switched film
    (full-wave), and the water-filling optimum, the best any static
    non-magnetic absorber of the same thickness can do at this angle.
(c) Margin of the switched film over the Rozanov limit against the drop of wp.
(d) Absorbed fractions for the same spectrum: static film, switched film with
    a transform-limited pulse, switched film with the chirped pulse.
"""
import json
import os
import numpy as np
import figstyle as fs


def main():
    plt = fs.setup()
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    sp = json.load(open(os.path.join(OUT, "route1_spectra.json")))
    fd = json.load(open(os.path.join(OUT, "route1_fdtd.json")))
    th_ad = json.load(open(os.path.join(OUT, "r1_threshold.json")))
    des_ad = json.load(open(os.path.join(OUT, "r1_design.json")))
    th_fd = json.load(open(os.path.join(OUT, "route1_threshold_fdtd.json")))

    fig, axs = plt.subplots(2, 2, figsize=(fs.FULL, 5.0), constrained_layout=True)

    # (a) tracking
    ax = axs[0, 0]
    t = np.array(sp["t_fs"])
    T = sp["design"]["T"] * 1e15
    ax.axvspan(-T / 2, T / 2, color=fs.BAND, lw=0, zorder=0)
    ax.text(0, 1012, "probe FWHM", ha="center", va="bottom", fontsize=7.5, color=fs.GREY)
    ax.plot(t, np.array(sp["lam_probe"]) * 1e9, color=fs.BLUE, label="probe wavelength")
    ax.plot(t, np.array(sp["lam_dip"]) * 1e9, color=fs.ORANGE, ls="--", label="absorption dip of the film")
    ax.set_xlim(-1.6 * T, 1.6 * T)
    ax.set_ylim(1000, 2300)
    ax.set_xlabel("Time from the probe centre (fs)")
    ax.set_ylabel("Wavelength (nm)")
    ax.legend(loc="upper right")
    fs.grid(ax)
    fs.title(ax, "a", "The dip follows the chirp")

    # (b) spectra
    ax = axs[0, 1]
    lam = np.array(sp["lam"]) * 1e9
    ax.fill_between(lam, sp["S_inc"], color=fs.GRID, lw=0, label="incident")
    ax.plot(lam, sp["S_static"], color=fs.ORANGE, label="reflected, best static film")
    ax.plot(lam, sp["S_optimum"], color=fs.BLACK, ls="--", lw=fs.LW_THIN + 0.3,
            label="reflected, Rozanov optimum")
    ax.plot(lam, sp["S_switched"], color=fs.BLUE, label="reflected, switched film")
    ax.set_xlim(1000, 2600)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Energy per unit wavelength (norm.)")
    ax.legend(loc="upper right", fontsize=7)
    fs.grid(ax)
    fs.title(ax, "b", "Spectra, 30 % drop")

    # (c) margin against depth
    ax = axs[1, 0]
    for (d, deg), col, mk, ls in (((7.5e-9, 75.0), fs.BLUE, "o", "-"), ((10e-9, 75.0), fs.TEAL, "s", "--"),
                                  ((10e-9, 70.0), fs.ORANGE, "^", ":")):
        rows = [r for r in th_ad if abs(r["tau_dec"] - 210e-15) < 1e-18 and abs(r["d"] - d) < 1e-12 and r["deg"] == deg]
        rows += [r for r in des_ad if abs(r["d"] - d) < 1e-12 and r["deg"] == deg and abs(r["depth"] - 0.10) < 1e-9]
        rows = sorted(rows, key=lambda r: r["depth"])
        ax.plot([r["depth"] * 100 for r in rows], [r["margin"] for r in rows], color=col, ls=ls, lw=fs.LW_THIN + 0.4)
        pts = [(s["depth"], s["margin"]) for s in fd["summary"] if abs(s["d"] - d) < 1e-12 and s["deg"] == deg]
        pts += [(s["depth"], s["margin"]) for s in th_fd["summary"] if abs(s["d"] - d) < 1e-12 and s["deg"] == deg]
        pts = sorted(set(pts))
        ax.plot([p[0] * 100 for p in pts], [p[1] for p in pts], ls="none", marker=mk, ms=fs.MS + 0.3, color=col,
                label=f"{d*1e9:g} nm, {deg:g}°")
    ax.axhline(0, color=fs.BLACK, lw=fs.LW_THIN)
    ax.text(10.3, 0.002, "Rozanov limit", fontsize=7.5, color=fs.DARKGREY, va="bottom")
    ax.axvline(19, color=fs.GREY, lw=fs.LW_THIN, ls=":")
    ax.text(18.6, -0.052, "largest\nmeasured\nchange", fontsize=7, color=fs.GREY, ha="right", va="bottom")
    ax.text(30.5, -0.05, "lines: adiabatic\nmarkers: full wave", fontsize=7, color=fs.GREY, ha="right", va="bottom")
    ax.set_xlim(9, 31)
    ax.set_ylim(-0.055, 0.045)
    ax.set_xlabel("Largest drop of $\\omega_p$ (%)")
    ax.set_ylabel("Absorbed fraction minus limit")
    ax.legend(loc="upper left", fontsize=7.5)
    fs.grid(ax)
    fs.title(ax, "c", "Margin over the bound")

    # (d) bars
    ax = axs[1, 1]
    tl = fd["transform_limited"]
    s30 = [s for s in fd["summary"] if s["tag"] == tl["tag"]][0]
    labels = ["best static\nfilm", "switched,\ntransform-limited", "switched,\nchirped"]
    vals = [s30["static"], tl["switched"], s30["switched"]]
    cols = [fs.ORANGE, fs.GREY, fs.BLUE]
    x = np.arange(3)
    ax.bar(x, vals, width=0.55, color=cols)
    for xi, v in zip(x, vals, strict=True):
        ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, color=fs.DARKGREY)
    ax.axhline(s30["eta_R"], color=fs.BLACK, ls="--", lw=fs.LW_THIN + 0.3)
    ax.text(-0.42, s30["eta_R"] + 0.012, f"Rozanov limit {s30['eta_R']:.3f}", fontsize=7.5, color=fs.DARKGREY,
            va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Absorbed fraction of the probe")
    fs.grid(ax, axis="y")
    fs.title(ax, "d", "Same spectrum, three cases")

    fs.save(fig, "fig_step7_route1", OUT)
    print("saved")


if __name__ == "__main__":
    main()
