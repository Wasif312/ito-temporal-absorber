"""fig_ttm.py -- route 1 with the two-temperature model.

(a) Drop of wp after the pump for the 7.5 nm film at 75 deg (damping rising
    with the electron temperature), for largest drops of 20, 30 and 40 %,
    against the kernel of Tirole et al. (2022) scaled to 30 %.
(b) Incident pump fluence that each largest drop needs, damping fixed and
    rising, against the largest pump of Tirole et al. and the single-shot
    damage fluence of ITO films (Bian et al. 2013).
(c) Margin over the Rozanov limit against the damping at the largest drop,
    with the two-temperature time course (solid) and the kernel (dashed),
    and the full-wave checks (open markers).
(d) Common temperature of electrons and lattice once they equilibrate,
    against the largest drop, with the film of Tirole et al. at its three
    published pump intensities.
Reads results/r1_ttm_baxter.json, r1_ttm_damping_baxter.json,
route1_ttm_fdtd_baxter*.json, fluence_check_baxter.json, r1_damping.json,
r1_fixed_wide.json and r1_real_wide.json; writes results/fig_ttm.pdf and .png.
"""
import json
import os

import numpy as np

import figstyle as fs
from mirror import delta_profile

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
GEOM = (7.5e-9, 75.0)


def load(name):
    with open(os.path.join(OUT, name)) as fh:
        return json.load(fh)


def at_geom(rows):
    return [r for r in rows if abs(r["d"] - GEOM[0]) < 1e-12 and r["deg"] == GEOM[1]]


def main():
    plt = fs.setup()
    ttm = load("r1_ttm_baxter.json")
    damp = load("r1_ttm_damping_baxter.json")["designs"]
    flu = load("fluence_check_baxter.json")
    fd = load("route1_ttm_fdtd_baxter.json")["summary"] + load("route1_ttm_fdtd_baxter_damping.json")["summary"]
    kern = load("r1_damping.json") + load("r1_fixed_wide.json") + [
        r for r in load("r1_real_wide.json") if r["var"] == "pec_hot"]

    fig, axs = plt.subplots(2, 2, figsize=(fs.FULL, 11.0 * fs.CM), constrained_layout=True)

    # (a) time course of the drop
    ax = axs[0, 0]
    for depth, ls in ((0.40, "--"), (0.30, "-"), (0.20, ":")):
        p = [q for q in at_geom(ttm["profiles"]) if q["var"] == "pec_hot" and abs(q["depth"] - depth) < 1e-9][0]
        t = np.array(p["t"]) * 1e15
        ax.plot(t, 100 * np.array(p["drop"]), color=fs.BLUE, ls=ls,
                label=f"{depth*100:.0f} %, {p['I_GWcm2']:.0f} GW cm$^{{-2}}$")
    tk = np.arange(-800e-15, 2500e-15, 1e-15)
    ax.plot(tk * 1e15, -100 * delta_profile(0.0, 0.30)(tk), color=fs.GREY, ls="-.", lw=fs.LW_THIN + 0.3,
            label="kernel, 30 %")
    ax.legend(loc="upper right", handlelength=2.6)
    ax.set_xlim(-600, 2000)
    ax.set_ylim(0, 45)
    ax.set_xlabel("time after the pump peak (fs)")
    ax.set_ylabel("drop of $\\omega_p$ (%)")
    fs.grid(ax)
    fs.title(ax, "a", "two-temperature model, 7.5 nm, 75°")

    # (b) fluence against drop
    ax = axs[0, 1]
    rows = at_geom(flu["route1"])
    for var, col, mk, lab in (("pec_fixed", fs.ORANGE, "s", "damping fixed"),
                              ("pec_hot", fs.BLUE, "o", "damping rising")):
        sel = sorted([r for r in rows if r["var"] == var], key=lambda r: r["depth"])
        x = [100 * r["depth"] for r in sel]
        y = [r["fluence_mJcm2"] for r in sel]
        ax.plot(x, y, color=col, marker=mk, ms=fs.MS, mfc="white", mew=0.9, label=lab)
    ax.legend(loc="lower right")
    lo, hi = (1e3 * v for v in flu["damage_J_cm2"])
    ax.axhspan(lo, hi, color=fs.BAND, lw=0)
    ax.text(20.3, np.sqrt(lo * hi), "single-shot damage of ITO", va="center", fontsize=7.5, color=fs.GREY)
    big = [r for r in flu["tirole"] if r["label"] == "largest"][0]
    ax.axhline(big["fluence_mJcm2"], color=fs.BLACK, lw=fs.LW_THIN, ls="--")
    ax.text(20.3, big["fluence_mJcm2"] * 0.86, "largest pump of Tirole et al.", va="top", fontsize=7.5,
            color=fs.DARKGREY)
    ax.set_yscale("log")
    ax.set_xlim(19, 41)
    ax.set_ylim(3, 600)
    ax.set_xlabel("largest drop of $\\omega_p$ (%)")
    ax.set_ylabel("pump fluence (mJ cm$^{-2}$)")
    fs.grid(ax)
    fs.title(ax, "b", "220 fs pump at 1315 nm")

    # (c) margin against damping
    ax = axs[1, 0]
    for depth, col, mk in ((0.30, fs.BLUE, "o"), (0.40, fs.VIOLET, "^")):
        sel = sorted([r for r in at_geom(damp) if abs(r["depth"] - depth) < 1e-9], key=lambda r: r["gamma_scale_max"])
        g = [r["gamma_scale_max"] for r in sel]
        m = [r["margin"] for r in sel]
        ax.plot(g, m, color=col, marker=mk, ms=fs.MS - 0.6)
        if depth > 0.35:
            fs.label_end(ax, g[0], m[0], f"{depth*100:.0f} % drop", dx=4, dy=8, color=col)
        else:
            fs.label_end(ax, g[0], m[0], f"{depth*100:.0f} % drop", dx=-5, dy=-8, ha="right", color=col)
        ks = sorted([r for r in at_geom(kern) if abs(r["depth"] - depth) < 1e-9],
                    key=lambda r: r.get("gamma_scale_max", 1.0 if r["var"] == "pec_fixed" else 2.0))
        gk = [r.get("gamma_scale_max", 1.0 if r["var"] == "pec_fixed" else (1.84 if depth < 0.35 else 2.28))
              for r in ks]
        ax.plot(gk, [r["margin"] for r in ks], color=col, ls="--", lw=fs.LW_THIN + 0.3)
    for s in at_geom(fd):
        g = [r["gamma_scale_max"] for r in damp if r["var"] == ("pec_T0_15k" if s["var"] == "pec_hot" else s["var"])
             and abs(r["d"] - s["d"]) < 1e-12 and r["deg"] == s["deg"] and abs(r["depth"] - s["depth"]) < 1e-9]
        if g:
            ax.plot(g[0], s["margin"], ls="none", marker="D", ms=fs.MS + 0.4, mfc="none", mec=fs.BLACK, mew=0.8)
    ax.axhline(0, color=fs.BLACK, lw=fs.LW_THIN)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=fs.DARKGREY, lw=fs.LW, label="two-temperature model"),
                       Line2D([], [], color=fs.DARKGREY, lw=fs.LW_THIN + 0.3, ls="--", label="kernel of Tirole et al."),
                       Line2D([], [], ls="none", marker="D", ms=fs.MS + 0.4, mfc="none", mec=fs.BLACK, mew=0.8,
                              label="full wave")], loc="lower left", handlelength=2.4, labelspacing=0.3)
    ax.set_xlim(0.0, 2.4)
    ax.set_ylim(-0.095, 0.17)
    ax.set_xlabel("damping at the largest drop, $\\gamma/\\gamma_0$")
    ax.set_ylabel("margin over the Rozanov limit")
    fs.grid(ax)
    fs.title(ax, "c", "margin against the damping, 7.5 nm, 75°")

    # (d) temperature after equilibration
    ax = axs[1, 1]
    rows = sorted([r for r in at_geom(flu["route1"]) if r["var"] == "pec_hot"], key=lambda r: r["depth"])
    ax.plot([100 * r["depth"] for r in rows], [r["T_eq"] for r in rows], color=fs.BLUE, marker="o", ms=fs.MS,
            mfc="white", mew=0.9, label="7.5 nm film, route 1")
    for i, r in enumerate(flu["tirole"]):
        ax.plot(100 * r["drop_max"], r["T_eq"], ls="none", marker="s", ms=fs.MS, color=fs.ORANGE,
                label="40 nm film of Tirole et al." if i == 0 else None)
        fs.label_end(ax, 100 * r["drop_max"], r["T_eq"], f"{r['I_GWcm2']:.0f} GW cm$^{{-2}}$",
                     dx=-6, dy=7, ha="right", color=fs.ORANGE)
    ax.legend(loc="upper left")
    ax.set_xlim(0, 42)
    ax.set_ylim(300, 1400)
    ax.set_xlabel("largest drop of $\\omega_p$ (%)")
    ax.set_ylabel("film temperature after 2.5 ps (K)")
    fs.grid(ax)
    fs.title(ax, "d", "heat left in the film")

    fs.save(fig, "fig_ttm", OUT)
    print("saved results/fig_ttm.pdf and .png")


if __name__ == "__main__":
    main()
