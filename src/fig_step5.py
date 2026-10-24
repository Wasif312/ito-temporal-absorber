"""fig_step5.py -- benchmark A6: periodic modulation of a lossy Drude slab.

(a) Rozanov integral of the equivalent reflection coefficient against the
    modulation depth, best phase, for the two test pulses (broadband case).
(b) |Gamma_eq| against wavelength: the sine-carrier pulse has almost no energy
    at long wavelengths, where Gamma_eq divides converted light by it.
(c) Absorbed energy of the narrowband pulse against the modulation depth, best
    and mean over the phase, at the paper's modulation frequency and at 0.25.
(d) Absorbed energy minus the water-filling limit against the modulation
    frequency.
"""
import json
import os
import numpy as np
import figstyle as fs


def main():
    plt = fs.setup()
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    d = json.load(open(os.path.join(OUT, "a6.json")))
    B = d["B"]
    F_B = np.array(d["F_B"])
    res = d["results"]
    lim = d["limits"]
    broad = [r for r in res if r["pulse"] in ("gauss", "sine")]
    narrow_sweep = [r for r in res if r["pulse"] == "narrow" and "AE_lam" in r]
    narrow_map = [r for r in res if r["pulse"] == "narrow" and "AE_lam" not in r]
    ext = d["extra"]

    fig, axs = plt.subplots(2, 2, figsize=(fs.FULL, 5.2), constrained_layout=True)

    # (a) Rozanov integral against modulation depth
    ax = axs[0, 0]
    ax.axvspan(0.004, 0.0135, color=fs.BAND, lw=0, zorder=0)
    ax.text(0.0073, 19, "ITO\nat 2$\\omega_0$", ha="center", va="top", fontsize=7, color=fs.GREY)
    for name, col, mk, lab in (("sine", fs.BLUE, "o", "sine-carrier pulse"),
                               ("gauss", fs.ORANGE, "s", "unipolar pulse")):
        fm = sorted({r["f"] for r in broad if r["pulse"] == name and r["f"] > 0})
        ir = [max(r["IR"] for r in broad if r["pulse"] == name and r["f"] == f) / B for f in fm]
        ax.loglog(fm, ir, color=col, marker=mk, ms=fs.MS - 0.8, label=lab)
    static = [r for r in broad if r["pulse"] == "gauss" and r["f"] == 0][0]["IR"] / B
    ax.axhline(1.0, color=fs.BLACK, ls="--", lw=fs.LW_THIN + 0.3)
    ax.text(1.05e-3 * 4.2, 1.07, "Rozanov bound", fontsize=7.5, color=fs.DARKGREY, va="bottom")
    ax.axhline(static, color=fs.GREY, ls=":", lw=fs.LW_THIN + 0.3)
    ax.text(0.03, static * 0.9, "static slab", fontsize=7.5, color=fs.GREY, va="top")
    ax.set_xlim(3.5e-3, 1.2)
    ax.set_ylim(0.2, 25)
    ax.set_xlabel("Modulation depth $f_\\mathrm{mod}$")
    ax.set_ylabel("$I_R$ / 2$\\pi^2 d$")
    ax.legend(loc="upper left", bbox_to_anchor=(0.24, 1.0))
    fs.grid(ax)
    fs.title(ax, "a", "Rozanov integral of $\\Gamma_\\mathrm{eq}$")

    # (b) equivalent reflection coefficient against wavelength
    ax = axs[0, 1]
    lam = 1 / F_B
    ax.axvspan(100, 1000, color=fs.BAND, lw=0, zorder=0)
    ax.text(290, 0.06, "sine pulse:\n4 × 10$^{-11}$\nof its energy", ha="center", va="bottom",
            fontsize=6.5, color=fs.GREY)
    st = [r for r in res if r["pulse"] == "sine" and r["f"] == 0][0]
    ax.semilogx(lam, st["absG"], color=fs.GREY, lw=fs.LW, label="static slab")
    for r in ext:
        if r["pulse"] == "sine" and "absG" in r and r["dzf"] == 1.0 and r["Tf"] == 1.0 and not r["momentum"]:
            ls = "-" if r["f"] > 0.5 else (0, (4, 1.5))
            ax.semilogx(lam, r["absG"], color=fs.BLUE, ls=ls, lw=fs.LW,
                        label=f"sine, $f_\\mathrm{{mod}}$ = {r['f']:g}")
    for r in ext:
        if r["pulse"] == "gauss" and "absG" in r and r["f"] > 0.5:
            ax.semilogx(lam, r["absG"], color=fs.ORANGE, ls=(0, (1, 1.2)), lw=fs.LW + 0.2,
                        label=f"unipolar, $f_\\mathrm{{mod}}$ = {r['f']:g}")
    ax.set_xlim(0.02, 1000)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Wavelength $\\lambda / d$")
    ax.set_ylabel("|$\\Gamma_\\mathrm{eq}$|")
    ax.legend(loc="upper left", fontsize=7)
    fs.grid(ax)
    fs.title(ax, "b", "Equivalent reflection, best phase")

    # (c) absorbed energy of the narrowband pulse against modulation depth
    ax = axs[1, 0]
    L = lim["narrow"]
    for wm, col, mk in ((0.1, fs.TEAL, "o"), (0.25, fs.BLUE, "s")):
        rows = narrow_map if wm != d["narrow"]["wm"] else [r for r in narrow_sweep
                                                           if abs(r["phi"] - d["narrow"]["phi_paper"]) > 1e-9]
        fm = sorted({r["f"] for r in rows if r["wm"] == wm})
        best = [max(r["AE"] for r in rows if r["wm"] == wm and r["f"] == f) for f in fm]
        mean = [np.mean([r["AE"] for r in rows if r["wm"] == wm and r["f"] == f]) for f in fm]
        st0 = [r for r in narrow_sweep if r["f"] == 0][0]["AE"]
        ax.plot([0] + fm, [st0] + best, color=col, marker=mk, ms=fs.MS - 0.8,
                label=f"$\\omega_\\mathrm{{mod}}d/2\\pi c_0$ = {wm:g}, best phase")
        ax.plot([0] + fm, [st0] + mean, color=col, ls=(0, (3, 2)), lw=fs.LW_THIN + 0.4,
                label=f"$\\omega_\\mathrm{{mod}}d/2\\pi c_0$ = {wm:g}, mean over phase")
    ax.axhline(L["eta_R_energy"], color=fs.BLACK, ls="--", lw=fs.LW_THIN + 0.3)
    ax.text(0.02, L["eta_R_energy"] + 0.003, "Rozanov limit for this pulse", fontsize=7.5, color=fs.DARKGREY,
            va="bottom")
    ax.axhline(L["best_static_energy"]["AE"], color=fs.GREY, ls=":", lw=fs.LW_THIN + 0.3)
    ax.text(0.62, L["best_static_energy"]["AE"] - 0.003, "best static slab", fontsize=7.5, color=fs.GREY,
            va="top")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.76, 1.01)
    ax.set_xlabel("Modulation depth $f_\\mathrm{mod}$")
    ax.set_ylabel("Absorbed fraction of the pulse energy")
    ax.legend(loc="lower left", fontsize=7)
    fs.grid(ax)
    fs.title(ax, "c", "Narrowband pulse, absorbed energy")

    # (d) margin over the limit against modulation frequency
    ax = axs[1, 1]
    wms = d["map_wm"]
    for fm, col, mk, ls in ((1.0, fs.BLUE, "o", "-"), (0.8, fs.TEAL, "s", "--"), (0.6, fs.GREY, "^", ":")):
        mg = [max(r["AE"] for r in narrow_map if r["wm"] == wm and r["f"] == fm) - L["eta_R_energy"]
              for wm in wms]
        ax.plot(wms, mg, color=col, ls=ls, marker=mk, ms=fs.MS - 0.6, label=f"$f_\\mathrm{{mod}}$ = {fm:g}")
    ax.axhline(0, color=fs.BLACK, lw=fs.LW_THIN)
    ax.text(0.505, 0.0012, "Rozanov limit", fontsize=7.5, color=fs.DARKGREY, va="bottom", ha="right")
    ax.set_xlim(0.04, 0.51)
    ax.set_ylim(-0.045, 0.02)
    ax.set_xlabel("Modulation frequency $\\omega_\\mathrm{mod}d/2\\pi c_0$")
    ax.set_ylabel("Absorbed fraction minus limit")
    ax.legend(loc="lower center")
    fs.grid(ax)
    fs.title(ax, "d", "Margin over the Rozanov limit")

    fs.save(fig, "fig_step5_a6", OUT)
    print("saved")


if __name__ == "__main__":
    main()
