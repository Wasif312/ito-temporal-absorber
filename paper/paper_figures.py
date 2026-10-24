"""Figures of the manuscript, drawn from the result files of the repository.

The colours are those of the Okabe-Ito palette, which colour-blind readers
can tell apart: blue for the switched film, vermillion for the best static
film, green and orange for further series, and shades of blue for increasing
drops of the plasma frequency. The Rozanov limit is a black dashed line, and
line styles or markers separate the curves as well. Figures go to
figures/<name>.png at 300 dpi, 16 cm wide.

Usage: python paper_figures.py [--results DIR] [--show] [figure names]
"""
import os

import matplotlib

from paper_paths import DATA, FIGURES, arguments, read_json

NAMES = ("fig_concept", "fig_gas", "fig_validation", "fig_benchmarks", "fig_mirror", "fig_route1", "fig_design",
         "fig_damping", "fig_experiments", "fig_ttm", "fig_requirements")


def _options(ap):
    ap.add_argument("names", nargs="*", metavar="name", help="figures to draw (default: all): " + ", ".join(NAMES))
    ap.add_argument("--show", action="store_true", help="also show each figure in a window")


ARGS = arguments(__doc__.splitlines()[0], extra=_options)
if not ARGS.show:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle  # noqa: E402

RES = ARGS.results
OUT = FIGURES
os.makedirs(OUT, exist_ok=True)
PD = read_json(os.path.join(DATA, "paper_data.json"))

BLUE, VERM, GREEN, ORANGE, GREY, INK = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#999999", "#222222"
MUTED = "#666666"
W = 16 / 2.54
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.4, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300, "figure.dpi": 100,
})


def load(n):
    return read_json(os.path.join(RES, n))


def label(ax, letter, x=-0.16, y=1.04):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", pad_inches=0.03)
    if not ARGS.show:
        plt.close(fig)
    print("saved", name)


def blues(n):
    return [plt.cm.Blues(x) for x in np.linspace(0.45, 1.0, n)]


def minus(txt):
    return txt.replace("-", "−")


# ----------------------------------------------------------------- concept
def fig_concept():
    fig = plt.figure(figsize=(W, 2.6))
    ax = fig.add_axes([0.0, 0.0, 0.38, 1.0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Polygon([[0.2, 2.2], [9.8, 2.2], [8.6, 5.3], [1.4, 5.3]], closed=True, fc="#E8EEF4", ec="#9AA8B6",
                         lw=0.6))
    ax.text(5.0, 4.85, "glass prism, $n_0$ = 1.5", ha="center", fontsize=7.5, color=INK)
    ax.add_patch(Rectangle((0.2, 2.0), 9.6, 0.2, fc=ORANGE, ec="none"))
    ax.add_patch(Rectangle((0.2, 1.35), 9.6, 0.65, fc="#D9D9D9", ec="#8C8C8C", hatch="////", lw=0.0))
    ax.text(5.0, 1.0, "ITO film, 5–10 nm, on a mirror", ha="center", fontsize=7, color=INK)
    th = np.deg2rad(75)
    x0, y0 = 6.1, 2.2
    L = 4.2
    dx, dy = -L * np.sin(th), L * np.cos(th)
    ax.add_patch(FancyArrowPatch((x0 - 1.0 + dx, y0 + dy), (x0 - 1.0, y0), arrowstyle="-|>", mutation_scale=8,
                                 lw=1.3, color=VERM))
    ax.add_patch(FancyArrowPatch((x0 + dx, y0 + dy), (x0, y0), arrowstyle="-|>", mutation_scale=8, lw=1.3,
                                 color=BLUE))
    ax.add_patch(FancyArrowPatch((x0, y0), (x0 - dx * 0.62, y0 + dy * 0.62), arrowstyle="-|>", mutation_scale=8,
                                 lw=1.0, color=BLUE, ls="--"))
    ax.text(x0 - 1.0 + dx - 0.05, y0 + dy + 0.5, "pump, 220 fs", color=VERM, fontsize=7, ha="left")
    ax.text(x0 + dx + 0.05, y0 + dy + 0.12, "chirped probe", color=BLUE, fontsize=7, ha="left")
    ax.text(x0 - dx * 0.62, y0 + dy * 0.62 + 0.12, "reflected", color=BLUE, fontsize=7, ha="right")
    ax.plot([x0, x0], [y0, y0 + 1.25], color=MUTED, lw=0.6, ls=":")
    ax.text(x0 - 0.5, y0 + 0.75, "$\\theta_0$", fontsize=8, color=INK)
    ax.add_patch(FancyArrowPatch((0.4, 0.42), (1.9, 0.42), arrowstyle="-|>", mutation_scale=7, lw=0.8, color=INK))
    ax.text(2.05, 0.42, "pump front and probe footprint\nmove at $c_0/(n_0\\sin\\theta_0)$", fontsize=6.6, color=INK,
            va="center")
    fig.text(0.0, 0.985, "(a)", fontsize=10, fontweight="bold", va="top")
    ax2 = fig.add_axes([0.445, 0.17, 0.245, 0.76])
    s = PD["static"]
    lam = np.array(s["lam"]) * 1e9
    for (k, lab), c in zip((("0.00", "0 %"), ("0.10", "10 %"), ("0.20", "20 %"), ("0.30", "30 %")), blues(4),
                           strict=True):
        ax2.plot(lam, s["R"][k], color=c, label=lab)
    ax2.set_xlim(1000, 2400)
    ax2.set_ylim(0, 1.0)
    ax2.set_xticks([1000, 1400, 1800, 2200])
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Reflectance of the film")
    ax2.legend(title="drop", title_fontsize=7, fontsize=6.8, loc="lower right", handlelength=1.2)
    label(ax2, "b", x=-0.3)
    ax3 = fig.add_axes([0.765, 0.17, 0.23, 0.76])
    dips = s["dips"]
    x = [r["drop"] * 100 for r in dips]
    ax3.fill_between(x, [r["lo_R05"] * 1e9 for r in dips], [r["hi_R05"] * 1e9 for r in dips], color="#DCE6F0", lw=0,
                     label="$R$ < 0.5")
    ax3.plot(x, [r["lam_dip"] * 1e9 for r in dips], "-o", ms=3, color=BLUE, label="dip")
    for r in dips[::2]:
        v = r["R_min"]
        txt = f"{v:.3f}".lstrip("0") if v >= 0.001 else "3e−4"
        ax3.text(r["drop"] * 100 + 1.0, r["lam_dip"] * 1e9 - 30, txt, fontsize=6, color=MUTED, va="top")
    ax3.set_xlim(-1, 42)
    ax3.set_ylim(1100, 2450)
    ax3.set_xlabel("Drop of $\\omega_\\mathrm{p}$ (%)")
    ax3.set_ylabel("Wavelength (nm)")
    ax3.legend(loc="upper left", fontsize=6.8, handlelength=1.2)
    label(ax3, "c", x=-0.36)
    save(fig, "fig_concept")


# ----------------------------------------------------------------- electron gas
def fig_gas():
    import ttm  # noqa: E402  (src/ is on the import path)
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.35))
    fig.subplots_adjust(wspace=0.48, left=0.08, right=0.97, bottom=0.2, top=0.9)
    h = load("hot_electron.json")
    for ef, c in zip(("0.88", "0.9", "1.0"), blues(3), strict=True):
        axs[0].plot(np.array(h[ef]["te"]) / 1e3, np.array(h[ef]["drop"]) * 100, color=c,
                    label=f"$E_\\mathrm{{F}}$ = {ef} eV")
    axs[0].set_xlim(0, 22)
    axs[0].set_ylim(0, 45)
    axs[0].set_xlabel("$T_\\mathrm{e}$ (10$^3$ K)")
    axs[0].set_ylabel("Drop of $\\omega_\\mathrm{p}$ (%)")
    axs[0].legend(loc="upper left", fontsize=6.8)
    label(axs[0], "a", x=-0.32)
    te = np.linspace(300, 20000, 300)
    laws = ((15000, VERM, 1.7), (30000, plt.cm.Oranges(0.65), 1.0), (60000, plt.cm.Oranges(0.5), 1.0),
            (120000, plt.cm.Oranges(0.35), 1.0), (-120000, plt.cm.Blues(0.45), 1.0),
            (-60000, plt.cm.Blues(0.65), 1.0), (-40000, BLUE, 1.0))
    for T0, c, lw in laws:
        g = (T0 + te) / (T0 + 300.0)
        axs[1].plot(te / 1e3, g, color=c, lw=lw)
        axs[1].text(20.5, g[-1], minus(f"{T0 / 1e3:+.0f}k"), fontsize=6, va="center", color=c)
    axs[1].axhline(1.0, color="black", ls="--", lw=0.8)
    axs[1].text(0.8, 2.12, "$T_0$ = 15k: Baxter et al.", fontsize=6.3, color=VERM)
    axs[1].set_xlim(0, 24.5)
    axs[1].set_xticks([0, 5, 10, 15, 20])
    axs[1].set_xlabel("$T_\\mathrm{e}$ (10$^3$ K)")
    axs[1].set_ylabel("Damping, $\\gamma/\\gamma_0$")
    label(axs[1], "b", x=-0.32)
    gas = ttm.load_gas(0.9)
    T = np.linspace(300, 30000, 400)
    axs[2].plot(T / 1e3, gas.ce(T) / gas.g_ep(T) * 1e15, color=BLUE,
                label="$C_\\mathrm{e}/g_\\mathrm{ep}(T_\\mathrm{e})$")
    axs[2].plot(T / 1e3, gas.ce(T) / 3e16 * 1e15, color=VERM, ls="-.", label="$C_\\mathrm{e}/G$, constant $G$")
    axs[2].set_yscale("log")
    axs[2].set_xlim(0, 30)
    axs[2].set_ylim(30, 4000)
    axs[2].set_xlabel("$T_\\mathrm{e}$ (10$^3$ K)")
    axs[2].set_ylabel("Cooling time (fs)")
    axs[2].legend(loc="lower right", fontsize=6.8)
    label(axs[2], "c", x=-0.32)
    save(fig, "fig_gas")


# ----------------------------------------------------------------- solver validation
def fig_validation():
    tests, ta, tg = load("tests.json"), load("tests_angle.json"), load("tests_angle_gold.json")
    fig = plt.figure(figsize=(W, 2.8))
    ax0 = fig.add_axes([0.065, 0.18, 0.22, 0.72])
    t1 = tests["T1"]["5"]
    ax0.plot(t1["lam_nm"], t1["R_tmm"], color="black", lw=1.0, label="transfer matrix")
    ax0.plot(t1["lam_nm"][::2], t1["R_fdtd"][::2], "o", ms=3, mfc="white", mec=BLUE, mew=0.9, label="FDTD")
    ax0.set_xlabel("Wavelength (nm)")
    ax0.set_ylabel("Reflectance, ITO–gold mirror")
    ax0.legend(loc="upper center", fontsize=6.8)
    label(ax0, "a", x=-0.36)
    dev = [("T1 mirror, transfer matrix", [tests["T1"][k]["max_err"] for k in ("10", "5")]),
           ("T2 switched medium, exact", [v["err"] for v in tests["T2"].values() if "err" in v]),
           ("T4 slow modulation, adiabatic", [abs(v[m] - v["adiabatic"]) for v in tests["T4"].values()
                                              for m in ("current", "momentum")]),
           ("V0 incident wave, TM impedance", [ta["V0_37.17"], ta["V0_70.0"]]),
           ("V1 films on conductor, exact", [x["max_err"] for x in ta["V1"]]),
           ("V3 films on gold, transfer matrix", [x["max_err"] for x in tg["V3"]])]
    pec, au = [], []
    for f in ("route1_fdtd.json", "route1_threshold_fdtd.json", "route1_real_fdtd.json", "route1_ttm_fdtd_baxter.json",
              "route1_ttm_fdtd_baxter_damping.json"):
        for x in load(f)["summary"]:
            if "balance" in x:
                (au if str(x.get("var", x.get("tag", ""))).startswith("au") else pec).append(abs(x["balance"]))
    mir = [abs(x["balance"]) for x in load("route3.json")["results"]] + \
          [abs(x["balance"]) for x in load("route3_angles.json")["summary"]]
    cv = load("convergence.json")["summary"]
    base = {x["case"]: x for x in cv if x["level"] == "base"}
    ch = []
    for x in cv:
        if x["level"] == "base":
            continue
        for k in ("static", "switched", "eta_R", "margin"):
            if x.get(k) is not None and base[x["case"]].get(k) is not None:
                ch.append(max(abs(x[k] - base[x["case"]][k]), 1.2e-7))
    res = [("T2 switched medium", [abs(v["balance"]) for v in tests["T2"].values() if "balance" in v]),
           ("T3 switched mirror", [abs(v["balance"]) for v in tests["T3"].values()]),
           ("V2, V4 switched films", [abs(x["balance"]) for x in ta["V2"]] + [abs(x["balance"]) for x in tg["V4"]]),
           ("absorbers on a conductor", pec), ("absorbers on gold", au), ("ITO–gold mirror, chirped", mir),
           ("change under refinement", ch)]
    for pos, rows, lab, title, xmin in (([0.535, 0.18, 0.16, 0.72], dev, "b", "Deviation from reference", 5e-6),
                                        ([0.835, 0.18, 0.16, 0.72], res, "c", "Residual or change", 1e-7)):
        ax = fig.add_axes(pos)
        for i, (name, vals) in enumerate(rows):
            y = len(rows) - 1 - i
            ax.plot(vals, [y] * len(vals), "o", ms=3.2, mfc="none", mec=BLUE, mew=0.8)
            ax.text(-0.05, y, name, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=6.2)
        ax.axvspan(0.031, 0.047, color="#F6E0D3", lw=0)
        ax.text(0.038, len(rows) - 0.5, "margins", ha="center", fontsize=6.2, color=VERM)
        ax.set_xscale("log")
        ax.set_xlim(xmin, 0.1)
        ax.set_ylim(-0.6, len(rows) - 0.1)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel(title, fontsize=8)
        ax.text(-1.3, 1.04, f"({lab})", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
    save(fig, "fig_validation")


# ----------------------------------------------------------------- benchmarks
def fig_benchmarks():
    fig, axs = plt.subplots(2, 2, figsize=(W, 4.7))
    fig.subplots_adjust(wspace=0.3, hspace=0.45, left=0.09, right=0.99, bottom=0.1, top=0.96)
    axs = axs.ravel()
    a = load("a4.json")
    t = np.array(a["delays_fs"])
    for (k, lab), c in zip((("0.025", "2.5 %"), ("0.050", "5 %"), ("0.100", "10 %")), blues(3), strict=True):
        axs[0].plot(t, a["traces"][f"{k} current"]["R"], color=c, label=lab)
        axs[0].plot(t[::4], np.array(a["adiabatic"][k])[::4], "o", ms=3, mfc="white", mec=c, mew=0.8)
    axs[0].plot([], [], "o", ms=3, mfc="white", mec=MUTED, label="adiabatic")
    axs[0].set_xlabel("Pump delay (fs)")
    axs[0].set_ylabel("Reflectance")
    axs[0].set_xlim(-600, 1200)
    axs[0].legend(loc="upper right", handlelength=1.4)
    label(axs[0], "a", x=-0.17)
    d = load("a5.json")
    rows = sorted([r for r in d["table"] if r["sigma"] == 0.18], key=lambda r: r["ratio"])
    x = [r["ratio"] for r in rows]
    axs[1].plot(x, [r["ltv"] for r in rows], "-o", ms=3, color=BLUE, label="switched")
    axs[1].plot(x, [r["lti"] for r in rows], "-s", ms=3, color=VERM, label="static, final layer")
    axs[1].plot(x, [r["rozanov"] for r in rows], "--", color="black", lw=1.0, label="Rozanov limit")
    axs[1].set_xlabel("$T_\\mathrm{p}/T_0$")
    axs[1].set_ylabel("Absorbed fraction")
    axs[1].set_ylim(0.5, 1.02)
    axs[1].legend(loc="lower right")
    label(axs[1], "b", x=-0.17)
    a6 = load("a6.json")
    lim = a6["limits"]["narrow"]
    best = {}
    for r in a6["results"] + a6["extra"]:
        if r["pulse"] != "narrow" or r["dzf"] != 1.0 or r["Tf"] != 1.0 or r["momentum"]:
            continue
        k = (round(r["wm"], 3), r["f"])
        if k not in best or r["AE"] > best[k]["AE"]:
            best[k] = r
    for wm, c, st, mk, lab in ((0.25, BLUE, "-", "o", "$\\omega_\\mathrm{m}d/2\\pi c_0$ = 0.25"),
                               (0.1, ORANGE, "--", "^", "$\\omega_\\mathrm{m}d/2\\pi c_0$ = 0.1")):
        f = sorted(k[1] for k in best if k[0] == wm and k[1] > 0)
        axs[2].plot(f, [best[(wm, v)]["AE"] for v in f], ls=st, marker=mk, ms=3, color=c, label=lab)
    axs[2].axhline(lim["best_static_energy"]["AE"], color=VERM, ls="-.", lw=1.0, label="best static slab")
    axs[2].axhline(lim["eta_R_energy"], color="black", ls="--", lw=1.0, label="water-filling limit")
    axs[2].set_xlabel("Modulation depth $f$")
    axs[2].set_ylabel("Absorbed fraction")
    axs[2].set_ylim(0.93, 1.0)
    axs[2].legend(loc="lower right", bbox_to_anchor=(1.0, 0.07), fontsize=6.6)
    label(axs[2], "c", x=-0.17)
    B = a6["B"]
    for pulse, c, mk, lab in (("gauss", BLUE, "o", "unipolar Gaussian"), ("sine", ORANGE, "^", "sine carrier")):
        by = {}
        for r in a6["results"]:
            if r["pulse"] == pulse and r["dzf"] == 1.0 and r["Tf"] == 1.0 and not r["momentum"]:
                by.setdefault(r["f"], []).append(r["IR"] / B)
        f = sorted(by)
        axs[3].plot(f, [max(by[v]) for v in f], "-" + mk, ms=3, color=c, label=f"{lab}, best phase")
        axs[3].plot(f, [min(by[v]) for v in f], "--", color=c, lw=1.0, label=f"{lab}, worst phase")
    axs[3].axhline(1.0, color="black", ls="--", lw=1.0)
    axs[3].text(0.97, 1.7, "budget", fontsize=6.5, color=MUTED, ha="right")
    axs[3].axhline(0.0, color="black", lw=0.5)
    axs[3].set_xlabel("Modulation depth $f$")
    axs[3].set_ylabel("Rozanov integral / budget")
    axs[3].legend(loc="upper left", fontsize=6.5)
    label(axs[3], "d", x=-0.17)
    save(fig, "fig_benchmarks")


# ----------------------------------------------------------------- ITO-gold mirror
def fig_mirror():
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.45))
    fig.subplots_adjust(wspace=0.3, left=0.08, right=0.99, bottom=0.19, top=0.92)
    r = load("route3.json")
    sw = sorted([x for x in r["results"] if x["kind"] == "switched" and not x["momentum"]], key=lambda x: x["x"])
    st = max([x for x in r["results"] if x["kind"] == "static"], key=lambda x: x["eta"])
    off = np.array(r["offsets_fs"])
    axs[0].plot(off, r["adiabatic_switched"], color=BLUE, label="switched, adiabatic")
    axs[0].plot([x["x"] * 1e15 for x in sw], [x["eta"] for x in sw], "o", ms=3.5, color=BLUE, mfc="white",
                label="switched, full wave")
    axs[0].axhline(st["eta"], color=VERM, ls="-.", lw=1.0, label="best static mirror")
    axs[0].set_xlabel("Pump delay (fs)")
    axs[0].set_ylabel("Absorbed fraction")
    axs[0].set_ylim(0.55, 0.9)
    axs[0].legend(loc="lower center", ncol=1)
    label(axs[0], "a", x=-0.13)
    ad = load("route3_angles_adiabatic.json")["angles"]
    ang = [x["deg"] for x in ad]
    axs[1].plot(ang, [x["switched"] for x in ad], color=BLUE, label="switched, adiabatic")
    axs[1].plot(ang, [x["static"] for x in ad], color=VERM, ls="-.", label="best static, adiabatic")
    fw = load("route3_angles.json")["summary"]
    angs = [37.17] + [x["deg"] for x in fw]
    axs[1].plot(angs, [max(x["eta"] for x in sw)] + [x["switched"] for x in fw], "o", ms=3.5, color=BLUE, mfc="white",
                label="switched, full wave")
    axs[1].plot(angs, [st["eta"]] + [x["static"] for x in fw], "s", ms=3.5, color=VERM, mfc="white",
                label="best static, full wave")
    axs[1].set_xlabel("Angle in the glass (°)")
    axs[1].set_ylabel("Absorbed fraction")
    axs[1].set_ylim(0.55, 1.0)
    axs[1].legend(loc="lower right")
    label(axs[1], "b", x=-0.13)
    save(fig, "fig_mirror")


# ----------------------------------------------------------------- 7.5 nm film, kernel model
def fig_route1():
    fig, axs = plt.subplots(2, 2, figsize=(W, 4.9))
    fig.subplots_adjust(wspace=0.3, hspace=0.42, left=0.09, right=0.99, bottom=0.1, top=0.96)
    s = load("route1_spectra.json")
    t = np.array(s["t_fs"])
    ax = axs[0, 0]
    T = s["design"]["T"] * 1e15
    ax.axvspan(-T / 2, T / 2, color="#EEEEEE", lw=0)
    ax.plot(t, np.array(s["lam_probe"]) * 1e9, color=BLUE, label="probe wavelength")
    ax.plot(t, np.array(s["lam_dip"]) * 1e9, color=VERM, ls="--", label="absorption dip")
    ax.text(0, 1030, "probe FWHM", ha="center", fontsize=7, color=MUTED)
    ax.set_xlim(-520, 520)
    ax.set_ylim(1000, 2300)
    ax.set_xlabel("Time from the probe peak (fs)")
    ax.set_ylabel("Wavelength (nm)")
    ax.legend(loc="upper right")
    label(ax, "a", x=-0.17)
    ax = axs[0, 1]
    lam = np.array(s["lam"]) * 1e9
    ax.fill_between(lam, 0, s["S_inc"], color="#E3E3E3", lw=0, label="incident")
    ax.plot(lam, s["S_static"], color=VERM, ls="-.", label="reflected, best static film")
    ax.plot(lam, s["S_optimum"], color="black", ls="--", lw=1.0, label="reflected, Rozanov optimum")
    ax.plot(lam, s["S_switched"], color=BLUE, label="reflected, switched film")
    ax.set_xlim(1000, 2600)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Energy per unit wavelength")
    ax.legend(loc="upper right")
    label(ax, "b", x=-0.15)
    ax = axs[1, 0]
    fw = load("route1_threshold_fdtd.json")["summary"] + load("route1_fdtd.json")["summary"]
    ad = load("r1_threshold.json") + load("r1_design.json")
    for (d, deg), c, mk, ls in (((7.5e-9, 75.0), BLUE, "o", "-"), ((10e-9, 75.0), GREEN, "s", "--"),
                                ((10e-9, 70.0), ORANGE, "^", ":")):
        pa = {}
        for r in ad:
            if abs(r["d"] - d) < 1e-12 and r["deg"] == deg and abs(r.get("tau_dec", 210e-15) - 210e-15) < 1e-18:
                pa[r["depth"]] = r["margin"]
        xs = sorted(pa)
        ax.plot([x * 100 for x in xs], [pa[x] for x in xs], color=c, ls=ls, lw=1.1)
        pf = sorted({(r["depth"], r["margin"]) for r in fw if abs(r["d"] - d) < 1e-12 and r["deg"] == deg})
        ax.plot([p[0] * 100 for p in pf], [p[1] for p in pf], mk, ms=4, color=c, mfc="white", mew=1.0,
                label=f"{d * 1e9:g} nm, {deg:g}°")
    ax.axhline(0, color="black", lw=0.7)
    ax.axvline(18.7, color=MUTED, lw=0.7, ls=":")
    ax.text(18.3, -0.032, "largest drop\nbehind a\nmeasurement", ha="right", fontsize=6.5, color=MUTED)
    ax.text(30.5, -0.034, "lines: adiabatic\nmarkers: full wave", ha="right", fontsize=6.5, color=MUTED)
    ax.set_xlim(9, 31)
    ax.set_ylim(-0.037, 0.042)
    ax.set_xlabel("Largest drop of $\\omega_\\mathrm{p}$ (%)")
    ax.set_ylabel("Margin over the Rozanov limit")
    ax.legend(loc="upper left")
    label(ax, "c", x=-0.17)
    ax = axs[1, 1]
    f = load("route1_fdtd.json")
    best = [x for x in f["summary"] if x["tag"] == "best_0.30"][0]
    tl = f["transform_limited"]
    vals = [best["static"], tl["switched"], best["switched"], best["momentum"]]
    labs = ["best static\nfilm", "switched,\nunchirped", "switched,\nchirped", "switched,\nchirped,\nmomentum law"]
    cols = [VERM, GREY, BLUE, BLUE]
    bars = ax.bar(range(4), vals, color=cols, width=0.62)
    bars[3].set_hatch("////")
    bars[3].set_facecolor("white")
    bars[3].set_edgecolor(BLUE)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=7.5)
    ax.axhline(best["eta_R"], color="black", ls="--", lw=1.0)
    ax.text(-0.45, best["eta_R"] + 0.015, f"Rozanov limit {best['eta_R']:.3f}", fontsize=7.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labs, fontsize=7)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Absorbed fraction of the probe")
    label(ax, "d", x=-0.15)
    save(fig, "fig_route1")


# ----------------------------------------------------------------- design map
def fig_design():
    a = load("r1_design.json")
    ds = sorted({r["d"] for r in a})
    degs = sorted({r["deg"] for r in a})
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.35), sharey=True)
    fig.subplots_adjust(wspace=0.08, left=0.08, right=0.9, bottom=0.2, top=0.88)
    vmax = 0.06
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", [VERM, "#F2D3C2", "#F0F0F0", "#C6DBEF", BLUE])
    for ax, depth in zip(axs, (0.10, 0.20, 0.30), strict=True):
        M = np.full((len(ds), len(degs)), np.nan)
        for r in a:
            if abs(r["depth"] - depth) < 1e-9:
                M[ds.index(r["d"]), degs.index(r["deg"])] = r["margin"]
        im = ax.imshow(np.clip(M, -vmax, vmax), cmap=cmap, vmin=-vmax, vmax=vmax, origin="lower", aspect="auto")
        for i in range(len(ds)):
            for j in range(len(degs)):
                v = M[i, j]
                ax.text(j, i, minus(f"{v * 100:+.1f}"), ha="center", va="center", fontsize=6.5,
                        color="white" if abs(v) > 0.8 * vmax else INK, fontweight="bold" if v > 0 else "normal")
        ax.set_xticks(range(len(degs)))
        ax.set_xticklabels([f"{x:g}" for x in degs])
        ax.set_yticks(range(len(ds)))
        ax.set_yticklabels([f"{x * 1e9:g}" for x in ds])
        ax.set_xlabel("Angle in the glass (°)")
        ax.set_title(f"largest drop {depth * 100:.0f} %", fontsize=8.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    axs[0].set_ylabel("Film thickness (nm)")
    cax = fig.add_axes([0.915, 0.2, 0.015, 0.68])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Margin (10$^{-2}$)", fontsize=8)
    cb.set_ticks([-0.06, -0.03, 0, 0.03, 0.06])
    cb.set_ticklabels(["−6", "−3", "0", "+3", "+6"])
    cb.outline.set_visible(False)
    for ax, lt in zip(axs, "abc", strict=True):
        label(ax, lt, x=-0.05 if lt != "a" else -0.28, y=1.02)
    save(fig, "fig_design")


# ----------------------------------------------------------------- damping and gold
def fig_damping():
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.75))
    fig.subplots_adjust(wspace=0.3, left=0.08, right=0.99, bottom=0.17, top=0.93)
    ax = axs[0]
    t = load("r1_ttm_damping_baxter.json")["designs"]
    nb = read_json(os.path.join(DATA, "numbers.json"))
    for depth, c, mk in ((0.30, BLUE, "o"), (0.40, ORANGE, "^")):
        p = sorted((r["gamma_scale_max"], r["margin"]) for r in t
                   if abs(r["d"] - 7.5e-9) < 1e-12 and r["deg"] == 75.0 and abs(r["depth"] - depth) < 1e-9)
        ax.plot([x for x, _ in p], [y for _, y in p], "-" + mk, ms=3.2, color=c,
                label=f"{depth * 100:.0f} %, two-temperature")
        k = nb[f"damping_zero_kernel_7.5nm_75_{depth:.2f}"]["points"]
        ax.plot([x for x, _ in k], [y for _, y in k], "--", color=c, lw=1.0, label=f"{depth * 100:.0f} %, kernel")
    fw = load("route1_ttm_fdtd_baxter.json")["summary"] + load("route1_ttm_fdtd_baxter_damping.json")["summary"]
    gmap = {"pec_fixed_7.5nm_75_30": 1.0, "pec_fixed_7.5nm_75_25": None, "pec_hot_7.5nm_75_40": 2.278,
            "pec_T0_30k_7.5nm_75_30": 1.426, "pec_T0_-60k_7.5nm_75_30": 0.784}
    first = True
    for s in fw:
        g = gmap.get(s["tag"])
        if g is None:
            continue
        ax.plot(g, s["margin"], "D", ms=5, mfc="none", mec="black", mew=0.9, label="full wave" if first else None)
        first = False
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel("Damping at the largest drop, $\\gamma/\\gamma_0$")
    ax.set_ylabel("Margin over the Rozanov limit")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_xlim(0.4, 2.4)
    label(ax, "a", x=-0.15)
    ax = axs[1]
    rr = load("r1_real.json")
    base = {r["depth"]: r["margin"] for r in load("r1_threshold.json") + load("r1_design.json")
            if abs(r["d"] - 7.5e-9) < 1e-12 and r["deg"] == 75.0 and abs(r.get("tau_dec", 210e-15) - 210e-15) < 1e-18}
    xs = sorted(x for x in base if x >= 0.2)
    ax.plot([x * 100 for x in xs], [base[x] for x in xs], "-o", ms=3, color=BLUE, label="conductor, fixed damping")
    for var, c, ls, mk, lab in (("pec_hot", VERM, "-", "s", "conductor, rising damping"),
                                ("au_fixed", GREEN, "--", "^", "gold, fixed damping"),
                                ("au_hot", ORANGE, "--", "v", "gold, rising damping")):
        p = sorted((r["depth"], r["margin"]) for r in rr if r["var"] == var and abs(r["d"] - 7.5e-9) < 1e-12
                   and r["deg"] == 75.0)
        ax.plot([x * 100 for x, _ in p], [y for _, y in p], ls=ls, marker=mk, ms=3, color=c, label=lab)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel("Largest drop of $\\omega_\\mathrm{p}$ (%)")
    ax.set_ylabel("Margin, 7.5 nm film at 75°")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.14), fontsize=7)
    label(ax, "b", x=-0.15)
    save(fig, "fig_damping")


# ----------------------------------------------------------------- two-temperature model against experiments
def fig_experiments():
    v = load("ttm_validation.json")
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.35))
    fig.subplots_adjust(wspace=0.45, left=0.07, right=0.99, bottom=0.2, top=0.9)
    for coup, c, mk, lab in (("g_ep", BLUE, "o", "$g_\\mathrm{ep}(T_\\mathrm{e})$"),
                             ("G_sarkar", VERM, "s", "constant $G$")):
        rows = v["tirole"]["cases"][coup]["rows"]
        I = [r["I_GWcm2"] for r in rows]
        axs[0].plot(I, [r["obs_max"] for r in rows], "-" + mk, ms=3, color=c, label=lab)
        dec = [(r["I_GWcm2"], r["obs_decay_fs"]) for r in rows
               if r["obs_decay_fs"] == r["obs_decay_fs"] and r["obs_decay_fs"]]
        axs[1].plot([x for x, _ in dec], [y for _, y in dec], "-" + mk, ms=3, color=c, label=lab)
        rows = v["alam"]["cases"][coup]["rows"]
        axs[2].plot([r["I_GWcm2"] for r in rows], [r["dn_max"] for r in rows], "-" + mk, ms=3, color=c, label=lab)
    axs[0].axhline(0.66, color="black", ls="--", lw=0.8)
    axs[0].text(2.0, 0.68, "measured saturation", ha="left", fontsize=6.5, color=MUTED)
    axs[0].set_xscale("log")
    axs[0].set_xlabel("Pump intensity (GW cm$^{-2}$)")
    axs[0].set_ylabel("Peak reflectance")
    axs[0].legend(loc="lower right", fontsize=6.5)
    label(axs[0], "a", x=-0.32)
    a4 = load("a4.json")
    kd = []
    for k in ("0.025", "0.050", "0.100"):
        tt = np.array(a4["inst"][k]["t_fs"])
        R = np.array(a4["inst"][k]["R"]) - a4["inst"][k]["R"][0]
        i = int(np.argmax(R))
        kd.append(tt[i + int(np.where(R[i:] <= R[i] / np.e)[0][0])] - tt[i])
    axs[1].axhspan(min(kd), max(kd), color="#E6E6E6", lw=0)
    axs[1].text(1.6, max(kd) * 1.08, "kernel fitted to\nthe measurement", ha="left", fontsize=6.5, color=MUTED)
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    axs[1].set_yticks([150, 300, 500, 1000, 2000])
    axs[1].set_yticklabels(["150", "300", "500", "1000", "2000"])
    axs[1].minorticks_off()
    axs[1].set_xlabel("Pump intensity (GW cm$^{-2}$)")
    axs[1].set_ylabel("Decay of the reflectance (fs)")
    label(axs[1], "b", x=-0.32)
    axs[2].plot([150], [0.72], "D", ms=5, color="black", mfc="none", label="measured")
    axs[2].set_xlabel("Pump intensity (GW cm$^{-2}$)")
    axs[2].set_ylabel("Index change at 1240 nm")
    axs[2].set_xlim(30, 170)
    axs[2].legend(loc="upper left", fontsize=6.5)
    label(axs[2], "c", x=-0.32)
    save(fig, "fig_experiments")


# ----------------------------------------------------------------- two-temperature model of the 7.5 nm film
def fig_ttm():
    fig, axs = plt.subplots(2, 2, figsize=(W, 4.8))
    fig.subplots_adjust(wspace=0.32, hspace=0.42, left=0.09, right=0.99, bottom=0.1, top=0.96)
    axs = axs.ravel()
    t = load("r1_ttm_baxter.json")
    ax = axs[0]
    for depth, c in zip((0.20, 0.30, 0.40), blues(3), strict=True):
        p = [x for x in t["profiles"] if x["var"] == "pec_fixed" and abs(x["d"] - 7.5e-9) < 1e-12
             and x["deg"] == 75.0 and abs(x["depth"] - depth) < 1e-9][0]
        ax.plot(np.array(p["t"]) * 1e15, np.array(p["drop"]) * 100, color=c,
                label=f"{depth * 100:.0f} %, {p['I_GWcm2']:.0f} GW cm$^{{-2}}$")
    s = load("route1_spectra.json")
    tt = np.array(s["t_fs"]) - s["design"]["pump"] * 1e15
    ax.plot(tt, np.array(s["drop"]) * 100, color=GREY, ls="--", lw=1.0, label="kernel, 30 %")
    ax.set_xlim(-500, 2000)
    ax.set_xlabel("Time after the pump peak (fs)")
    ax.set_ylabel("Drop of $\\omega_\\mathrm{p}$ (%)")
    ax.legend(loc="upper right", fontsize=6.8)
    label(ax, "a", x=-0.17)
    ax = axs[1]
    for var, c, mk, lab in (("pec_fixed", BLUE, "o", "fixed damping"), ("pec_hot", VERM, "s", "rising damping")):
        p = sorted((x["depth"], x["rise_fs"], x["decay_fs"]) for x in t["profiles"] if x["var"] == var
                   and abs(x["d"] - 7.5e-9) < 1e-12 and x["deg"] == 75.0)
        ax.plot([q[0] * 100 for q in p], [q[2] for q in p], "-" + mk, ms=3, color=c, label=f"decay, {lab}")
        ax.plot([q[0] * 100 for q in p], [q[1] for q in p], "--" + mk, ms=3, color=c, mfc="white", lw=1.0,
                label=f"rise, {lab}")
    ax.axhline(275, color=GREY, lw=0.9)
    ax.axhline(188, color=GREY, lw=0.9, ls="--")
    ax.text(40.4, 268, "kernel decay", fontsize=6.5, color=MUTED, ha="right", va="top")
    ax.text(40.4, 181, "kernel rise", fontsize=6.5, color=MUTED, ha="right", va="top")
    ax.set_ylim(140, 470)
    ax.set_xlabel("Largest drop of $\\omega_\\mathrm{p}$ (%)")
    ax.set_ylabel("Rise or decay time (fs)")
    ax.legend(loc="upper left", fontsize=6.3, ncol=2, columnspacing=0.8, handlelength=1.8)
    label(ax, "b", x=-0.17)
    h = load("fluence_check_baxter.json")
    ax = axs[2]
    for var, c, mk, lab in (("pec_fixed", BLUE, "o", "fixed damping"), ("pec_hot", VERM, "s", "rising damping")):
        p = sorted((r["depth"], r["fluence_mJcm2"]) for r in h["route1"] if r["var"] == var
                   and abs(r["d"] - 7.5e-9) < 1e-12 and r["deg"] == 75.0)
        ax.plot([x * 100 for x, _ in p], [y for _, y in p], "-" + mk, ms=3, color=c, label=lab)
    lo, hi = [x * 1e3 for x in h["damage_J_cm2"]]
    ax.axhspan(lo, hi, color="#E6E6E6", lw=0)
    ax.text(20, lo * 1.08, "single-shot damage", fontsize=6.5, color=MUTED)
    big = [r for r in h["tirole"] if r["label"] == "largest"][0]
    ax.axhline(big["fluence_mJcm2"], color="black", ls="--", lw=0.8)
    ax.text(19.8, big["fluence_mJcm2"] * 0.82, "largest pump on the\nITO–gold mirror", fontsize=6.5, color=MUTED,
            va="top")
    ax.set_yscale("log")
    ax.set_ylim(3, 600)
    ax.set_xlabel("Largest drop of $\\omega_\\mathrm{p}$ (%)")
    ax.set_ylabel("Pump fluence (mJ cm$^{-2}$)")
    ax.legend(loc="lower right", fontsize=6.8)
    label(ax, "c", x=-0.17)
    ax = axs[3]
    for var, c, mk, lab in (("pec_fixed", BLUE, "o", "7.5 nm, fixed"), ("pec_hot", VERM, "s", "7.5 nm, rising")):
        p = sorted((r["depth"], r["T_eq"]) for r in h["route1"] if r["var"] == var
                   and abs(r["d"] - 7.5e-9) < 1e-12 and r["deg"] == 75.0)
        ax.plot([x * 100 for x, _ in p], [y for _, y in p], "-" + mk, ms=3, color=c, label=lab)
    ex = [(r["drop_max"], r["T_eq"]) for r in h["tirole"]] + [(r["drop_max"], r["T_eq"]) for r in h["alam"]]
    ax.plot([x * 100 for x, _ in ex[:3]], [y for _, y in ex[:3]], "D", ms=4, color=GREEN, label="ITO–gold mirror")
    ax.plot([ex[3][0] * 100], [ex[3][1]], "P", ms=5, color=ORANGE, label="thick film")
    ax.set_xlabel("Largest drop of $\\omega_\\mathrm{p}$ (%)")
    ax.set_ylabel("$T_\\mathrm{eq}$ (K)")
    ax.legend(loc="upper left", fontsize=6.8)
    label(ax, "d", x=-0.17)
    save(fig, "fig_ttm")


# ----------------------------------------------------------------- what an experiment must control
def fig_requirements():
    e = read_json(os.path.join(DATA, "extra.json"))
    dn = e["dense"]
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.3))
    fig.subplots_adjust(wspace=0.42, left=0.08, right=0.99, bottom=0.21, top=0.9)
    for ax, x, y, xl, lt, zero in ((axs[0], dn["dd_nm"], dn["m_d"], "Error of the assumed thickness (nm)", "a",
                                    e["equivalent"]["dd_zero_nm"]),
                                   (axs[1], dn["dth"], dn["m_th"], "Error of the assumed angle (°)", "b",
                                    e["equivalent"]["dtheta_zero_deg"])):
        ax.plot(x, y, color=BLUE)
        ax.axhline(0, color="black", lw=0.7)
        ax.axvline(0, color=GREY, lw=0.6, ls=":")
        ax.plot([zero], [0], "o", ms=4, mfc="white", mec=VERM, mew=1.0)
        ax.text(zero, 0.005, minus(f"{zero:+.2f}"), ha="center", fontsize=6.8, color=VERM)
        ax.set_xlabel(xl, fontsize=8)
        ax.set_ylabel("Inferred margin")
        label(ax, lt, x=-0.32)
    ax = axs[2]
    ax.plot(dn["cut_nm"], dn["m_cut"], color=BLUE, label="long-pass filter")
    ax.axhline(e["passband"]["full"]["margin"], color="black", ls="--", lw=0.9, label="full band")
    for c in (1330.0, 1340.0, 1350.0):
        i = int(np.argmin(np.abs(np.array(dn["cut_nm"]) - c)))
        ax.plot(c, dn["m_cut"][i], "o", ms=3.5, color=VERM)
        exp = int(np.floor(np.log10(dn["leak"][i])))
        ax.text(c + 2.5, dn["m_cut"][i] + 0.0012, f"pump 10$^{{{exp}}}$", fontsize=6.2, color=VERM)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel("Filter edge (nm)", fontsize=8)
    ax.set_ylabel("Margin in the passband")
    ax.set_ylim(-0.006, 0.036)
    ax.legend(loc="lower left", fontsize=6.8)
    label(ax, "c", x=-0.32)
    save(fig, "fig_requirements")


if __name__ == "__main__":
    unknown = [n for n in ARGS.names if n not in NAMES]
    if unknown:
        raise SystemExit(f"unknown figure {', '.join(unknown)}; choose from {', '.join(NAMES)}")
    for name in NAMES:
        if not ARGS.names or name in ARGS.names:
            globals()[name]()
    print(f"figures in {OUT}")
    if ARGS.show:
        plt.show()
