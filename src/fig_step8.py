"""fig_step8.py -- route 1 with realistic material choices.

(a) Damping scale and electron temperature against the drop of wp, from the
    nonparabolic band and the damping law gamma0 (1 + Te/T0).
(b) Best margin over the Rozanov limit against the drop of wp, for the perfect
    conductor and gold, with fixed and rising damping (lines adiabatic,
    markers full wave).
(c) Margin against the rise of the damping at the largest drop, for drops of
    30 % and 40 % (7.5 nm film at 75 deg).
(d) Absorbed fractions with rising damping, full wave: best static film,
    switched film and the Rozanov limit.
"""
import json
import os
import numpy as np
import figstyle as fs


def main():
    plt = fs.setup()
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


    def load(name):
        return json.load(open(os.path.join(OUT, name)))


    he = load("hot_electron.json")
    des = load("r1_design.json")
    thr = load("r1_threshold.json")
    fxw = load("r1_fixed_wide.json")
    real = load("r1_real.json")
    wide = load("r1_real_wide.json")
    damp = load("r1_damping.json")
    fd1 = load("route1_fdtd.json")["summary"]
    fd2 = load("route1_threshold_fdtd.json")["summary"]
    fdr = load("route1_real_fdtd.json")["summary"]

    fig, axs = plt.subplots(2, 2, figsize=(fs.FULL, 5.1), constrained_layout=True)

    # (a) damping against drop
    ax = axs[0, 0]
    for EF, ls, lab in (("0.88", ":", None), ("0.9", "-", "damping, $E_F$ = 0.9 eV"), ("1.0", ":", None)):
        d = np.array(he[EF]["drop"]) * 100
        te = np.array(he[EF]["te"])
        g = (1 + te / 15000) / (1 + 300 / 15000)
        sel = d <= 41
        ax.plot(d[sel], g[sel], color=fs.ORANGE, ls=ls, lw=fs.LW if ls == "-" else fs.LW_THIN, label=lab)
    for row in he["0.9"]["table"]:
        if row["drop"] in (0.10, 0.20, 0.30, 0.40):
            ax.plot(row["drop"] * 100, row["gamma_scale"], "o", color=fs.ORANGE, ms=fs.MS)
            ax.annotate(f"{row['Te']/1000:.1f} kK", (row["drop"] * 100, row["gamma_scale"]), xytext=(-4, 6),
                        textcoords="offset points", ha="right", fontsize=7, color=fs.DARKGREY)
    ax.text(2, 2.18, "dotted: $E_F$ = 0.88 and 1.0 eV", fontsize=7, color=fs.GREY, va="top")
    ax.set_xlim(0, 42)
    ax.set_ylim(0.95, 2.4)
    ax.set_xlabel("Drop of $\\omega_p$ (%)")
    ax.set_ylabel("Damping $\\gamma / \\gamma_0$")
    ax.legend(loc="lower right")
    fs.grid(ax)
    fs.title(ax, "a", "Hot electrons raise the damping")

    # (b) margin against drop
    ax = axs[0, 1]


    def best_by_depth(rows, var=None):
        out = {}
        for r in rows:
            if var is not None and r.get("var") != var:
                continue
            k = round(r["depth"], 3)
            out[k] = max(out.get(k, -9), r["margin"])
        ks = sorted(out)
        return np.array(ks) * 100, np.array([out[k] for k in ks])


    series = [
        ("perfect conductor, damping fixed",
         [r for r in des + thr + fxw if abs(r.get("tau_dec", 210e-15) - 210e-15) < 1e-18], None, fs.BLUE, "o", "-"),
        ("gold, damping fixed", real, "au_fixed", fs.VIOLET, "s", "--"),
        ("perfect conductor, damping rising", real + wide, "pec_hot", fs.ORANGE, "^", "-."),
        ("gold, damping rising", real, "au_hot", fs.GREY, "D", ":"),
    ]
    for lab, rows, var, col, _mk, ls in series:
        x, y = best_by_depth([r for r in rows if r["depth"] >= 0.099], var)
        ax.plot(x, y, color=col, ls=ls, lw=fs.LW_THIN + 0.4, label=lab)
    fdpts = {"perfect conductor, damping fixed": [(s["depth"], s["margin"]) for s in fd1 + fd2],
             "gold, damping fixed": [(s["depth"], s["margin"]) for s in fdr if s["var"] == "au_fixed"],
             "perfect conductor, damping rising": [(s["depth"], s["margin"]) for s in fdr if s["var"] == "pec_hot"],
             "gold, damping rising": []}
    for lab, _rows, _var, col, mk, _ls in series:
        pts = {}
        for dpt, m in fdpts[lab]:
            pts[round(dpt, 3)] = max(pts.get(round(dpt, 3), -9), m)
        ks = sorted(pts)
        ax.plot(np.array(ks) * 100, [pts[k] for k in ks], ls="none", marker=mk, ms=fs.MS, color=col)
    ax.axhline(0, color=fs.BLACK, lw=fs.LW_THIN)
    ax.text(10.5, 0.004, "Rozanov limit", fontsize=7.5, color=fs.DARKGREY, va="bottom")
    ax.set_xlim(9, 41)
    ax.set_ylim(-0.16, 0.11)
    ax.set_xlabel("Largest drop of $\\omega_p$ (%)")
    ax.set_ylabel("Absorbed fraction minus limit")
    ax.legend(loc="lower left", fontsize=6.8)
    fs.grid(ax)
    fs.title(ax, "b", "Best margin over the bound")

    # (c) margin against damping rise
    ax = axs[1, 0]
    T0s = {"pec_fixed": np.inf, "pec_hot_T120k": 120000.0, "pec_hot_T60k": 60000.0, "pec_hot_T30k": 30000.0,
           "pec_hot": 15000.0}
    pool = fxw + damp + wide
    for depth, col, mk, ls in ((0.30, fs.BLUE, "o", "-"), (0.40, fs.ORANGE, "s", "--")):
        te = float(np.interp(depth, he["0.9"]["drop"], he["0.9"]["te"]))
        xs, ys = [], []
        for var, T0 in T0s.items():
            r = [q for q in pool if q["var"] == var and abs(q["d"] - 7.5e-9) < 1e-12 and q["deg"] == 75.0
                 and abs(q["depth"] - depth) < 1e-9]
            if not r:
                continue
            rise = 0.0 if np.isinf(T0) else ((1 + te / T0) / (1 + 300 / T0) - 1) * 100
            xs.append(rise)
            ys.append(r[0]["margin"])
        o = np.argsort(xs)
        ax.plot(np.array(xs)[o], np.array(ys)[o], color=col, ls=ls, marker=mk, ms=fs.MS - 0.5,
                label=f"drop {depth*100:.0f} %")
        model = ((1 + te / 15000) / (1 + 300 / 15000) - 1) * 100
        ax.axvline(model, color=col, lw=fs.LW_THIN, ls=":")
        ax.text(model - 2, 0.095 if depth == 0.30 else 0.08, f"model at {depth*100:.0f} %", fontsize=7, color=col,
                va="bottom", ha="right")
    ax.axhline(0, color=fs.BLACK, lw=fs.LW_THIN)
    ax.set_xlim(-5, 140)
    ax.set_ylim(-0.04, 0.11)
    ax.set_xlabel("Rise of the damping at the largest drop (%)")
    ax.set_ylabel("Absorbed fraction minus limit")
    ax.legend(loc="lower left")
    fs.grid(ax)
    fs.title(ax, "c", "How much damping the bound tolerates")

    # (d) absorbed fractions with rising damping
    ax = axs[1, 1]
    cases = [("au_hot_7.5nm_75_30", "gold\n7.5 nm, 30 %"), ("pec_hot_7.5nm_75_40", "ideal mirror\n7.5 nm, 40 %"),
             ("pec_hot_5nm_80_40", "ideal mirror\n5 nm, 40 %")]
    x = np.arange(len(cases))
    w = 0.34
    for i, (tag, _lab) in enumerate(cases):
        s = [q for q in fdr if q["tag"] == tag][0]
        ax.bar(i - w / 2, s["static"], width=w * 0.92, color=fs.ORANGE, label="best static film" if i == 0 else None)
        ax.bar(i + w / 2, s["switched"], width=w * 0.92, color=fs.BLUE, label="switched, chirped" if i == 0 else None)
        ax.plot([i - w, i + w], [s["eta_R"]] * 2, color=fs.BLACK, ls="--", lw=fs.LW_THIN + 0.3,
                label="Rozanov limit" if i == 0 else None)
        ax.text(i + w / 2, s["switched"] - 0.015, f"{s['switched']:.3f}", ha="center", va="top", fontsize=6.8,
                color="white")
        ax.text(i - w / 2, s["static"] - 0.015, f"{s['static']:.3f}", ha="center", va="top", fontsize=6.8,
                color="white")
    ax.set_xticks(x)
    ax.set_xticklabels([c[1] for c in cases], fontsize=7)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Absorbed fraction of the probe")
    ax.legend(loc="upper right", fontsize=7)
    fs.grid(ax, axis="y")
    fs.title(ax, "d", "Rising damping, full wave")

    fs.save(fig, "fig_step8_route1_real", OUT)
    print("saved")


if __name__ == "__main__":
    main()
