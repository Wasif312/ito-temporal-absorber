"""Tables of the manuscript, written straight from the result files.

Every number in a table comes from a JSON file in src/results, or from
data/paper_data.json and data/extra.json, which paper_data.py and
paper_extra.py compute with the repository code, so no result is typed by
hand. Each table goes to tables/<name>.md as a pandoc pipe table with its
caption. The two tables of input parameters, tables/materials.md and
tables/numerics.md, are written by hand and left untouched.

Usage: python paper_tables.py [--results DIR] [--print]
"""
import math
import os
import re
import sys

from paper_paths import DATA, TABLES, arguments, read_json

ARGS = arguments(__doc__.splitlines()[0], extra=lambda ap: ap.add_argument(
    "--print", action="store_true", help="also print every table to the console"))
RES = ARGS.results
OUTD = TABLES
os.makedirs(OUTD, exist_ok=True)
PD = read_json(os.path.join(DATA, "paper_data.json"))
SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
MINUS = "−"


def load(name):
    return read_json(os.path.join(RES, name))


def num(x, nd=4, sign=False):
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    if sign and float(s) == 0:
        s = f"{0:.{nd}f}"
    return s.replace("-", MINUS)


def sci(x, nd=1):
    """$2.1\\times10^{-5}$ as inline math, which never breaks across lines."""
    if x == 0:
        return "0"
    m, e = f"{x:.{nd}e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


MATH = [
    ("ΔR", r"$\Delta R$"),
    ("I_R/B", r"$I_\mathrm{R}/B$"),
    ("depth f", r"depth $f$"),
    ("E_x from Z H_y", r"$E_x$ from $ZH_y$"),
    ("ω_m d/2πc₀", r"$\omega_\mathrm{m}d/2\pi c_0$"),
    ("g_ep(T_e)", r"$g_\mathrm{ep}(T_\mathrm{e})$"),
    ("C_e/g_ep", r"$C_\mathrm{e}/g_\mathrm{ep}$"),
    ("C_e/G", r"$C_\mathrm{e}/G$"),
    ("F/F_damage", r"$F/F_\mathrm{damage}$"),
    ("T_p/T_0", r"$T_\mathrm{p}/T_0$"),
    ("ħω_p", r"$\hbar\omega_\mathrm{p}$"),
    ("T_eq", r"$T_\mathrm{eq}$"),
    ("T_e", r"$T_\mathrm{e}$"),
    ("E_F", r"$E_\mathrm{F}$"),
    ("g_ep", r"$g_\mathrm{ep}$"),
    ("η_R", r"$\eta_\mathrm{R}$"),
    ("W_m", r"$W_\mathrm{m}$"),
    ("U_d", r"$U_\mathrm{d}$"),
    ("Δn", r"$\Delta n$"),
    ("ε∞", r"$\varepsilon_\infty$"),
    ("γ/γ₀", r"$\gamma/\gamma_0$"),
    ("n₀/cos θ₀", r"$n_0/\cos\theta_0$"),
]


RULES = [(r"(?<![\w\\_{])η(?![\w_])", r"$\eta$"), (r"(?<![\w\\_{])R(?=[ ,]|$)", r"$R$"),
         (r"(?<![\w\\_{])F(?= \()", r"$F$"), (r"(?<![\w\\_{])B(?= \()", r"$B$"),
         (r"(?<![\w\\_{])d(?= \(nm\))", r"$d$"), (r"(?<=constant )G(?![\w])", r"$G$"),
         (r"(?<![\w\\_{])C_e(?![\w])", r"$C_\mathrm{e}$")]


def _sub_outside_math(text, rx, tex):
    """Replace the matches of rx by tex in the parts of text outside $...$."""
    parts = text.split("$")
    for k in range(0, len(parts), 2):
        parts[k] = rx.sub(lambda m: tex, parts[k])
    return "$".join(parts)


def mathify(text):
    """Write the symbols of headers, captions and cells as inline math."""
    for plain, tex in MATH:
        rx = re.compile(r"(?<![\w\\])" + re.escape(plain) + r"(?![\w])")
        text = _sub_outside_math(text, rx, tex)
    for pat, tex in RULES:
        text = _sub_outside_math(text, re.compile(pat), tex)
    return text


def show(text):
    """Print a table; characters the console cannot encode are escaped."""
    enc = sys.stdout.encoding or "utf-8"
    print(text.encode(enc, errors="backslashreplace").decode(enc) + "\n")


def table(name, caption, header, rows):
    caption = mathify(caption)
    header = [mathify(h) for h in header]
    rows = [[mathify(str(c)) for c in r] for r in rows]
    lines = [f"Table: TAB:{name} {caption}", ""]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join([":-:"] * len(header)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    with open(os.path.join(OUTD, name + ".md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    if ARGS.print:
        show("\n".join(lines))
    return name, len(rows)


def nm(x):
    return f"{x * 1e9:.0f}"


def dnm(x):
    return f"{x * 1e9:g}"


def fs(x):
    return f"{x * 1e15:.0f}".replace("-", MINUS)


def pct(x):
    return f"{x * 100:.1f}".rstrip("0").rstrip(".")


def sweep_nm(r):
    return f"{nm(r['lam_c'] + r['span'] / 2)}→{nm(r['lam_c'] - r['span'] / 2)}"


TWO_PI = 2 * math.pi

# ------------------------------------------------------------------ validation


def t_tests_kx():
    t = load("tests.json")
    rows = []
    for k in ("10", "5"):
        rows.append(["T1", f"static mirror, cells of {k} nm", "largest ΔR against the transfer matrix",
                     sci(t["T1"][k]["max_err"])])
    for key, v in t["T2"].items():
        if "gamma" not in key:
            continue
        form, g, c = key.split()[0], float(key.split()[1].split("=")[1]), key.split()[2].split("=")[1]
        gtxt = "lossless" if g == 0 else "lossy"
        rows.append(["T2", f"uniform medium, {form} form, {gtxt}, Courant number {c}",
                     "field error, balance, W_m", f"{sci(v['err'])}, {sci(v['balance'])}, {num(v['wmod'], 3, True)}"])
    for key in ("time reflection current", "time reflection momentum"):
        v = t["T2"][key]
        rows.append(["T2", f"sudden lossless switch, {key.split()[-1]} form",
                     "forward and backward amplitude, frequency (THz)",
                     f"{v['E_forward']:.3f}, {v['E_backward']:.3f}, "
                     f"{v['w_before'] / TWO_PI / 1e12:.1f} → {v['w_after'] / TWO_PI / 1e12:.1f}"])
    for key, v in t["T3"].items():
        delay = key.split()[1]
        rows.append(["T3", f"switched mirror, pump at {delay.replace('-', MINUS)} fs, {key.split()[-1]} form",
                     "R, W_m, U_d, balance",
                     f"{v['R']:.4f}, {num(v['Wmod'], 4, True)}, {v['diss']:.4f}, {sci(v['balance'])}"])
    for key, v in t["T4"].items():
        rows.append(["T4", f"slow modulation, pump at {key.split()[1].replace('-', MINUS)} fs",
                     "R current, momentum, adiabatic",
                     f"{v['current']:.4f}, {v['momentum']:.4f}, {v['adiabatic']:.4f}"])
    return table("tests_kx", "Tests of the solver at a fixed in-plane wavevector.",
                 ["Test", "Case", "Quantities", "Values"], rows)


def t_tests_angle():
    a = load("tests_angle.json")
    g = load("tests_angle_gold.json")
    rows = [["V0", "incident wave at 37.17°", "largest relative deviation of E_x from Z H_y", sci(a["V0_37.17"])],
            ["V0", "incident wave at 70°", "largest relative deviation of E_x from Z H_y", sci(a["V0_70.0"])]]
    for v in a["V1"]:
        rows.append(["V1", f"{v['d'] * 1e9:.0f} nm film, {v['deg']:g}°, cells of {v['h'] * 1e9:g} nm",
                     "largest ΔR against the exact value, balance", f"{sci(v['max_err'])}, {sci(v['balance'])}"])
    for v in a["V2"]:
        rows.append(["V2", f"switched film, {'momentum' if v['momentum'] else 'current'} form", "η, W_m, U_d, balance",
                     f"{v['eta']:.4f}, {num(v['W'], 4, True)}, {v['D']:.4f}, {sci(v['balance'])}"])
    for v in g["V3"]:
        rows.append(["V3", f"{v['d'] * 1e9:g} nm film on gold", "largest ΔR against the exact value, balance",
                     f"{sci(v['max_err'])}, {sci(v['balance'])}"])
    for v in g["V4"]:
        rows.append(["V4", f"switched film on gold, {'momentum' if v['momentum'] else 'current'} form",
                     "η, W_m, balance", f"{v['eta']:.4f}, {num(v['W'], 4, True)}, {sci(v['balance'])}"])
    return table("tests_angle", "Tests of the solver at a fixed angle of incidence.",
                 ["Test", "Case", "Quantities", "Values"], rows)


def t_trace():
    d = load("fig1b_trace.json")
    rows = []
    for i, t in enumerate(d["t_fs"]):
        rows.append([f"{t:.1f}", num(d["current"]["E"][i], 4), num(d["current"]["E_exact"][i], 4),
                     num(d["momentum"]["E"][i], 4), num(d["momentum"]["E_exact"][i], 4)])
    return table("trace", "Field of a uniform medium after a 10 % drop of the plasma frequency, sampled at the "
                 "field maxima.", ["t (fs)", "E current", "E exact", "E momentum", "E exact"], rows)


# ------------------------------------------------------------------ benchmarks


def t_a4():
    a = load("a4.json")
    rows = []
    for key, v in a["spectra"].items():
        parts = key.split()
        depth, rise, form = float(parts[1]), parts[3], parts[5]
        f = v["fdtd"]
        rows.append([pct(depth), rise, form, f"{v['R']:.3f}", f"{f['width'] / 1e12:.1f}",
                     f"{v['adiabatic']['width'] / 1e12:.1f}", f"{f['fwhm'] / 1e12:.2f}", f"{f['centroid'] / 1e12:.2f}"])
    return table("a4", "Reflected spectrum of the ITO–gold mirror under a 220 fs probe at 1300 nm.",
                 ["Drop (%)", "Rise (fs)", "Form", "R", "Width, full wave (THz)", "Width, adiabatic (THz)",
                  "FWHM (THz)", "Centroid (THz)"], rows)


def t_a4_trace():
    a = load("a4.json")
    rows = []
    idx = list(range(0, len(a["delays_fs"]), 5))
    for i in idx:
        rows.append([f"{a['delays_fs'][i]:.0f}".replace("-", MINUS)]
                    + [f"{a['traces'][k]['R'][i]:.3f}" for k in ("0.025 current", "0.050 current", "0.100 current",
                                                                  "0.100 momentum")]
                    + [f"{a['adiabatic'][k][i]:.3f}" for k in ("0.025", "0.050", "0.100")])
    return table("a4_trace", "Reflectance of the ITO–gold mirror against the pump delay.",
                 ["Delay (fs)", "2.5 %", "5 %", "10 %", "10 %, momentum", "2.5 %, adiabatic", "5 %, adiabatic",
                  "10 %, adiabatic"], rows)


def t_a5():
    d = load("a5.json")
    rows = []
    for r in sorted([r for r in d["table"] if r["sigma"] == 0.18], key=lambda r: r["ratio"]):
        rows.append([f"{r['ratio']:g}", f"{r['ltv']:.4f}", f"{r['lti']:.4f}", f"{r['rozanov']:.4f}",
                     num(r["ltv"] - r["rozanov"], 4, True), num(r["wmod"], 3, True)])
    return table("a5", "Switched lossy layer with a final conductivity of 0.18 S/m.",
                 ["T_p/T_0", "η switched", "η static", "η_R", "Margin", "W_m"], rows)


def t_a5_all():
    d = load("a5.json")
    rows = []
    for r in sorted(d["table"], key=lambda r: (r["sigma"], r["ratio"])):
        rows.append([f"{r['sigma']:g}", f"{r['ratio']:g}", f"{r['ltv']:.4f}", f"{r['lti']:.4f}", f"{r['rozanov']:.4f}",
                     f"{r['sig_only']:.4f}", num(r["wmod"], 3, True)])
    return table("a5_all", "Switched lossy layer for every final conductivity and pulse duration.",
                 ["σ₂ (S/m)", "T_p/T_0", "η switched", "η static", "η_R", "η, conductivity only", "W_m"], rows)


def t_a6():
    a = load("a6.json")
    lim = a["limits"]["narrow"]
    best = {}
    for r in a["results"] + a["extra"]:
        if r["pulse"] != "narrow" or r["dzf"] != 1.0 or r["Tf"] != 1.0 or r["momentum"]:
            continue
        k = (round(r["wm"], 3), r["f"])
        if k not in best or r["AE"] > best[k]["AE"]:
            best[k] = r
    bs = lim["best_static_energy"]
    eR = lim["eta_R_energy"]
    rows = [["static slab, best plasma frequency", f"{bs['AE']:.4f}", num(bs["AE"] - eR, 4, True), "0", "", "", ""],
            ["water-filling limit", f"{eR:.4f}", "0", "", "", "", ""]]
    for f_ in sorted(k[1] for k in best if k[0] == 0.25 and k[1] > 0):
        r = best[(0.25, f_)]
        q = best.get((0.1, f_))
        rows.append([f"modulation depth {f_:g}", f"{r['AE']:.4f}", num(r["AE"] - eR, 4, True), num(r["W"], 5, True),
                     sci(r["balance"]), f"{q['AE']:.4f}" if q else "", num(q["AE"] - eR, 4, True) if q else ""])
    return table("a6", "Periodically modulated slab under its narrow-band pulse at the best modulation phase, for "
                 "modulation frequencies ω_m d/2πc₀ of 0.25 and 0.1.",
                 ["Case", "Absorbed, 0.25", "Margin, 0.25", "W_m, 0.25", "Balance, 0.25", "Absorbed, 0.1",
                  "Margin, 0.1"], rows)


def t_a6_broad():
    a = load("a6.json")
    B = a["B"]
    agg = {}
    for r in a["results"]:
        if r["pulse"] not in ("gauss", "sine") or r["dzf"] != 1.0 or r["Tf"] != 1.0 or r["momentum"]:
            continue
        b = agg.setdefault((r["pulse"], r["f"]), dict(AE=-1.0, hi=-1e9, lo=1e9))
        b["AE"] = max(b["AE"], r["AE"])
        b["hi"] = max(b["hi"], r["IR"] / B)
        b["lo"] = min(b["lo"], r["IR"] / B)
    fs_ = sorted({k[1] for k in agg})
    rows = []
    for f_ in fs_:
        g, sn = agg[("gauss", f_)], agg[("sine", f_)]
        rows.append([f"{f_:g}", num(g["hi"], 3), num(g["lo"], 3), f"{g['AE']:.4f}", num(sn["hi"], 3), num(sn["lo"], 3),
                     f"{sn['AE']:.4f}"])
    lg, ls = a["limits"]["gauss"]["eta_R_energy"], a["limits"]["sine"]["eta_R_energy"]
    rows.append(["water-filling limit", "", "", f"{lg:.4f}", "", "", f"{ls:.4f}"])
    return table("a6_broad", "Periodically modulated slab under two broadband pulses, with the largest and smallest "
                 "Rozanov integral of the equivalent reflection coefficient over the modulation phase, relative to the "
                 "budget, and the largest absorbed fraction.",
                 ["Modulation depth f", "I_R/B largest, Gaussian", "I_R/B smallest, Gaussian", "Absorbed, Gaussian",
                  "I_R/B largest, sine carrier", "I_R/B smallest, sine carrier", "Absorbed, sine carrier"], rows)


def t_t1():
    t = load("tests.json")["T1"]
    lam = t["5"]["lam_nm"]
    order = sorted(range(len(lam)), key=lambda i: lam[i])
    rows = []
    for i in order[::3]:
        rows.append([f"{lam[i]:.0f}", f"{t['5']['R_tmm'][i]:.4f}", f"{t['10']['R_fdtd'][i]:.4f}",
                     f"{t['5']['R_fdtd'][i]:.4f}"])
    return table("t1_spectrum", "Reflectance of the static ITO–gold mirror from the transfer matrix and from the "
                 "solver at a fixed in-plane wavevector on two grids.",
                 ["Wavelength (nm)", "Transfer matrix", "FDTD, cells of 10 nm", "FDTD, cells of 5 nm"], rows)


# ------------------------------------------------------------------ route 3


def t_route3():
    r = load("route3.json")
    sw = sorted([x for x in r["results"] if x["kind"] == "switched" and not x["momentum"]], key=lambda x: x["x"])
    ad = dict(zip([round(o) for o in r["offsets_fs"]], r["adiabatic_switched"], strict=False))
    rows = []
    for x in sw:
        o = round(x["x"] * 1e15)
        rows.append([str(o).replace("-", MINUS), f"{x['eta']:.3f}", f"{ad[o]:.3f}", num(x["Wmod"], 4, True),
                     sci(x["balance"])])
    st = [x for x in r["results"] if x["kind"] == "static"]
    bs = max(st, key=lambda x: x["eta"])
    mom = [x for x in r["results"] if x["kind"] == "switched" and x["momentum"]][0]
    rows.append(["static, drop " + pct(bs["x"]) + " %", f"{bs['eta']:.3f}", f"{max(r['adiabatic_static']):.3f}", "0",
                 sci(bs["balance"])])
    rows.append([f"momentum form, {fs(mom['x'])} fs", f"{mom['eta']:.3f}", "", num(mom["Wmod"], 4, True),
                 sci(mom["balance"])])
    return table("route3", "Absorbed fraction of the chirped probe on the ITO–gold mirror at 37.17° in the glass "
                 "against the pump delay.",
                 ["Pump delay (fs)", "η full wave", "η adiabatic", "W_m", "Balance"], rows)


def t_route3_angles():
    a = load("route3_angles.json")
    ad = load("route3_angles_adiabatic.json")
    rows = []
    for s in a["summary"]:
        rows.append([f"{s['deg']:g}", f"{s['static']:.3f}", pct(s["static_drop"]), f"{s['switched']:.3f}",
                     f"{s['momentum']:.3f}", fs(s["pump"]), num(s["gain"], 3, True), num(s["Wmod"], 4, True),
                     sci(s["balance"])])
    t1 = table("route3_angles", "Chirp-matched switching of the ITO–gold mirror at prism angles.",
               ["Angle (°)", "Best static", "Its drop (%)", "Switched", "Momentum form", "Pump (fs)", "Gain",
                "W_m", "Balance"], rows)
    rows = [[f"{x['deg']:g}", f"{x['static']:.3f}", pct(x["static_drop"]), f"{x['switched']:.3f}", fs(x["pump"]),
             num(x["switched"] - x["static"], 3, True)] for x in ad["angles"]]
    t2 = table("route3_angles_ad", "Adiabatic estimate for the ITO–gold mirror against the angle in the glass.",
               ["Angle (°)", "Best static", "Its drop (%)", "Switched", "Pump (fs)", "Gain"], rows)
    return t1, t2


# ------------------------------------------------------------------ route 1, kernel


def t_static():
    s = PD["static"]
    lam = s["lam"]
    rows = []
    for L in range(1100, 2401, 100):
        i = min(range(len(lam)), key=lambda k: abs(lam[k] - L * 1e-9))
        rows.append([str(L)] + [f"{s['R'][k][i]:.3f}" for k in ("0.00", "0.10", "0.20", "0.30")])
    t1 = table("static_R", "Reflectance of the 7.5 nm film on a perfect conductor at 75° in the glass.",
               ["Wavelength (nm)", "No drop", "Drop 10 %", "Drop 20 %", "Drop 30 %"], rows)
    rows = [[pct(x["drop"]), f"{x['lam_dip'] * 1e9:.0f}", f"{x['R_min']:.4f}",
             f"{x['lo_R05'] * 1e9:.0f}–{x['hi_R05'] * 1e9:.0f}", f"{x['width_R05'] * 1e9:.0f}"]
            for x in s["dips"]]
    t2 = table("static_dip", "Absorption dip of the 7.5 nm film at 75° against the drop of the plasma frequency.",
               ["Drop (%)", "Dip wavelength (nm)", "Smallest R", "Band with R below 0.5 (nm)", "Width (nm)"], rows)
    rows = [[dnm(b["d"]), f"{b['deg']:g}", f"{b['ratio']:.2f}", f"{b['B'] * 1e9:.0f}"] for b in PD["budgets"]]
    t3 = table("budgets", "Rozanov budget of the film geometries.",
               ["d (nm)", "θ₀ (°)", "n₀/cos θ₀", "B (nm)"], rows)
    return t1, t2, t3


def t_route1_fw():
    f = load("route1_fdtd.json")
    rows = []
    for s in f["summary"]:
        rows.append([dnm(s["d"]), f"{s['deg']:g}", pct(s["depth"]), sweep_nm(s), fs(s["T"]), f"{s['eta_R']:.4f}",
                     f"{s['static']:.4f}", f"{s['switched']:.4f}", f"{s['momentum']:.4f}", num(s["margin"], 4, True),
                     num(s["Wmod"], 4, True)])
    tl = f["transform_limited"]
    t1 = table("route1_fw", "Full-wave results for thin ITO films on a perfect conductor with the kernel model.",
               ["d (nm)", "θ₀ (°)", "Drop (%)", "Sweep (nm)", "FWHM (fs)", "η_R", "Best static", "Switched",
                "Momentum form", "Margin", "W_m"], rows)
    return t1, tl


def t_threshold():
    f = load("route1_threshold_fdtd.json")
    rows = [[dnm(s["d"]), f"{s['deg']:g}", pct(s["depth"]), f"{s['eta_R']:.4f}", f"{s['static']:.4f}",
             f"{s['switched']:.4f}", num(s["margin"], 4, True), num(s["margin_adiabatic"], 4, True),
             num(s["Wmod"], 4, True), sci(s["balance"])] for s in f["summary"]]
    t1 = table("threshold_fw", "Full-wave margins near the threshold drop with the kernel model.",
               ["d (nm)", "θ₀ (°)", "Drop (%)", "η_R", "Best static", "Switched", "Margin", "Margin, adiabatic",
                "W_m", "Balance"], rows)
    a = load("r1_threshold.json")
    rows = [[f"{r['tau_dec'] * 1e15:.0f}", dnm(r["d"]), f"{r['deg']:g}", pct(r["depth"]), sweep_nm(r), fs(r["T"]),
             fs(r["pump"]), f"{r['eta_R']:.4f}", f"{r['static']:.4f}", f"{r['switched']:.4f}",
             num(r["margin"], 4, True)] for r in a]
    t2 = table("threshold_ad", "Adiabatic margins near the threshold for decay times of 210 fs and 360 fs.",
               ["Decay (fs)", "d (nm)", "θ₀ (°)", "Drop (%)", "Sweep (nm)", "FWHM (fs)", "Pump (fs)", "η_R",
                "Best static", "Switched", "Margin"], rows)
    return t1, t2


def t_design():
    a = load("r1_design.json")
    rows = [[dnm(r["d"]), f"{r['deg']:g}", pct(r["depth"]), f"{r['B'] * 1e9:.0f}", sweep_nm(r), fs(r["T"]),
             fs(r["pump"]), f"{r['eta_R']:.4f}", f"{r['static']:.4f}", f"{r['switched']:.4f}",
             num(r["margin"], 4, True)] for r in a]
    t1 = table("design", "Adiabatic design search for thin ITO films on a perfect conductor with the kernel model.",
               ["d (nm)", "θ₀ (°)", "Drop (%)", "B (nm)", "Sweep (nm)", "FWHM (fs)", "Pump (fs)", "η_R",
                "Best static", "Switched", "Margin"], rows)
    c = load("r1_chirp_scope.json")
    best = {}
    for r in c:
        k = (r["d"], r["deg"], r["dmax"])
        m = r["switched"] - r["eta_R"]
        if k not in best or m > best[k][0]:
            best[k] = (m, r)
    rows = []
    for k in sorted(best):
        m, r = best[k]
        rows.append([dnm(r["d"]), f"{r['deg']:g}", pct(r["dmax"]), f"{r['B'] * 1e9:.0f}", sweep_nm(r), fs(r["T"]),
                     fs(r["pump"]), f"{r['eta_R']:.3f}", f"{r['static']:.3f}", f"{r['switched']:.3f}", num(m, 3, True)])
    t2 = table("scope", "Coarse survey of thin ITO films on a perfect conductor over thickness, angle and drop on "
               "a fixed grid of probes.",
               ["d (nm)", "θ₀ (°)", "Drop (%)", "B (nm)", "Sweep (nm)", "FWHM (fs)", "Pump (fs)", "η_R",
                "Best static", "Switched", "Margin"], rows)
    return t1, t2


def t_spectra():
    s = load("route1_spectra.json")
    rows = []
    lam = s["lam"]
    for L in range(1100, 2601, 100):
        i = min(range(len(lam)), key=lambda k: abs(lam[k] - L * 1e-9))
        rows.append([str(L), f"{s['S_inc'][i]:.3f}", f"{s['S_static'][i]:.3f}", f"{s['S_optimum'][i]:.3f}",
                     f"{s['S_switched'][i]:.3f}"])
    t1 = table("spectra", "Incident and reflected energy spectra of the 7.5 nm film at 75° with a 30 % drop, "
               "normalised to the peak of the incident spectrum.",
               ["Wavelength (nm)", "Incident", "Best static", "Rozanov optimum", "Switched"], rows)
    rows = []
    t = s["t_fs"]
    for T in range(-500, 501, 50):
        i = min(range(len(t)), key=lambda k: abs(t[k] - T))
        rows.append([str(T).replace("-", MINUS), f"{s['lam_probe'][i] * 1e9:.0f}", f"{s['lam_dip'][i] * 1e9:.0f}",
                     f"{s['intensity'][i]:.3f}", pct(s["drop"][i])])
    t2 = table("dip_track", "Wavelength of the probe and of the absorption dip of the film against time from the "
               "probe peak.", ["Time (fs)", "Probe (nm)", "Dip (nm)", "Probe intensity", "Drop (%)"], rows)
    return t1, t2


# ------------------------------------------------------------------ damping and gold


def t_variants():
    f = load("route1_real_fdtd.json")
    names = {"pec_hot": "conductor, rising damping", "au_fixed": "gold, fixed damping",
             "au_hot": "gold, rising damping", "pec_hot_T30k": "conductor, half the rise"}
    rows = [[names.get(s["var"], s["var"]), dnm(s["d"]), f"{s['deg']:g}", pct(s["depth"]), f"{s['eta_R']:.4f}",
             f"{s['static']:.4f}", f"{s['switched']:.4f}", f"{s['momentum']:.4f}", num(s["margin"], 4, True),
             num(s["Wmod"], 4, True), f"{s['diss_au']:.4f}"] for s in f["summary"]]
    t1 = table("variants_fw", "Full-wave results with rising damping and with a gold mirror, kernel model.",
               ["Back mirror, damping", "d (nm)", "θ₀ (°)", "Drop (%)", "η_R", "Best static", "Switched",
                "Momentum form", "Margin", "W_m", "Absorbed in gold"], rows)
    a = load("r1_real.json")
    labels = {"pec_hot": "conductor, rising", "au_hot": "gold, rising", "au_fixed": "gold, fixed",
              "pec_hot_g15": "conductor, rising, 1.5 γ₀", "pec_hot_bohn": "conductor, rising, second set"}
    rows = [[labels[r["var"]], dnm(r["d"]), f"{r['deg']:g}", pct(r["depth"]), f"{r['gamma_scale_max']:.2f}",
             sweep_nm(r), fs(r["T"]), f"{r['eta_R']:.4f}", f"{r['static']:.4f}", f"{r['switched']:.4f}",
             num(r["margin"], 4, True)] for r in a]
    t2 = table("variants", "Adiabatic search with rising damping, a gold mirror, extra damping and a second Drude set.",
               ["Variant", "d (nm)", "θ₀ (°)", "Drop (%)", "γ/γ₀", "Sweep (nm)", "FWHM (fs)", "η_R", "Best static",
                "Switched", "Margin"], rows)
    w = load("r1_real_wide.json")
    rows = [[dnm(r["d"]), f"{r['deg']:g}", pct(r["depth"]), f"{r['gamma_scale_max']:.2f}", sweep_nm(r), fs(r["T"]),
             f"{r['eta_R']:.4f}", f"{r['static']:.4f}", f"{r['switched']:.4f}", num(r["margin"], 4, True)] for r in w]
    t3 = table("variants_wide", "Adiabatic search with rising damping over a wider range of thickness and angle.",
               ["d (nm)", "θ₀ (°)", "Drop (%)", "γ/γ₀", "Sweep (nm)", "FWHM (fs)", "η_R", "Best static",
                "Switched", "Margin"], rows)
    k = load("r1_damping.json")
    rows = [[f"{float(r['var'].split('T')[1].rstrip('k')):g}", dnm(r["d"]), f"{r['deg']:g}", pct(r["depth"]),
             f"{r['gamma_scale_max']:.2f}", sweep_nm(r), f"{r['eta_R']:.4f}", f"{r['static']:.4f}",
             f"{r['switched']:.4f}", num(r["margin"], 4, True)] for r in k]
    t4 = table("damping_kernel", "Adiabatic margins with weaker rises of the damping, kernel model.",
               ["T₀ (10³ K)", "d (nm)", "θ₀ (°)", "Drop (%)", "γ/γ₀", "Sweep (nm)", "η_R", "Best static", "Switched",
                "Margin"], rows)
    return t1, t2, t3, t4


# ------------------------------------------------------------------ hot electrons and the two-temperature model


def t_hot():
    h = load("hot_electron.json")
    rows = []
    for i, r in enumerate(h["0.9"]["table"]):
        rows.append([pct(r["drop"]), f"{h['0.88']['table'][i]['Te']:.0f}", f"{r['Te']:.0f}",
                     f"{h['1.0']['table'][i]['Te']:.0f}", f"{r['gamma_scale']:.2f}"])
    return table("hot", "Electron temperature that gives each drop of the plasma frequency, and the rising damping "
                 "at that temperature.",
                 ["Drop (%)", "T_e, E_F = 0.88 eV (K)", "T_e, E_F = 0.9 eV (K)", "T_e, E_F = 1.0 eV (K)",
                  "γ/γ₀ (E_F = 0.9 eV)"], rows)


def t_gas():
    v = load("ttm_validation.json")
    rows = [[f"{r['Te']:.0f}", pct(r["drop"]), f"{r['Ce'] / 1e4:.3f}", f"{r['g_ep'] / 1e17:.3f}", f"{r['tau_fs']:.0f}",
             f"{r['tau_sarkar_fs']:.0f}"] for r in v["gas"]["rows"]]
    return table("gas", "Heat capacity, electron–phonon coupling and cooling time C_e/g_ep of the electron gas with "
                 "E_F = 0.9 eV, compared with a constant coupling.",
                 ["T_e (K)", "Drop (%)", "C_e (10⁴ J m⁻³ K⁻¹)", "g_ep (10¹⁷ W m⁻³ K⁻¹)", "C_e/g_ep (fs)",
                  "C_e/G, constant G (fs)"], rows)


def t_ttm_exp():
    v = load("ttm_validation.json")

    def dec(x):
        return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.0f}"
    a, b = v["tirole"]["cases"]["g_ep"]["rows"], v["tirole"]["cases"]["G_sarkar"]["rows"]
    rows = [[f"{x['I_GWcm2']:g}", pct(x["drop_max"]), f"{x['obs_max']:.3f}", dec(x["obs_decay_fs"]),
             pct(y["drop_max"]), f"{y['obs_max']:.3f}", dec(y["obs_decay_fs"])] for x, y in zip(a, b, strict=False)]
    t1 = table("ttm_tirole", "The ITO–gold mirror in the two-temperature model with the coupling g_ep(T_e) and with a "
               "constant coupling.",
               ["I₀ in air (GW cm⁻²)", "Drop, g_ep (%)", "R, g_ep", "Decay, g_ep (fs)", "Drop, constant G (%)",
                "R, constant G", "Decay, constant G (fs)"], rows)
    a, b = v["alam"]["cases"]["g_ep"]["rows"], v["alam"]["cases"]["G_sarkar"]["rows"]
    rows = [[f"{x['I_GWcm2']:g}", f"{x['Te_max']:.0f}", pct(x["drop_max"]), num(x["dn_max"], 3, True),
             dec(x["obs_decay_fs"]), f"{y['Te_max']:.0f}", pct(y["drop_max"]), num(y["dn_max"], 3, True),
             dec(y["obs_decay_fs"])] for x, y in zip(a, b, strict=False)]
    t2 = table("ttm_alam", "The thick ITO film in the two-temperature model.",
               ["I₀ (GW cm⁻²)", "T_e, g_ep (K)", "Drop, g_ep (%)", "Δn, g_ep", "Decay, g_ep (fs)",
                "T_e, constant G (K)", "Drop, constant G (%)", "Δn, constant G", "Decay, constant G (fs)"], rows)
    a, b = v["route1"]["cases"]["g_ep"]["rows"], v["route1"]["cases"]["G_sarkar"]["rows"]
    rows = [[f"{x['I_GWcm2']:g}", f"{x['Te_max']:.0f}", pct(x["drop_max"]), dec(x["drop_decay_fs"]),
             f"{y['Te_max']:.0f}", pct(y["drop_max"]), dec(y["drop_decay_fs"])] for x, y in zip(a, b, strict=False)]
    t3 = table("ttm_route1", "The 7.5 nm film at 75° in the two-temperature model with rising damping against the "
               "pump intensity.",
               ["I₀ (GW cm⁻²)", "T_e, g_ep (K)", "Drop, g_ep (%)", "Decay, g_ep (fs)", "T_e, constant G (K)",
                "Drop, constant G (%)", "Decay, constant G (fs)"], rows)
    inf = PD["inferred"]
    rows = [["ITO–gold mirror, saturated reflectance 0.66 at 1300 nm", "fixed", pct(inf["tirole_R066"])],
            ["thick ITO film, index change 0.72 at 1240 nm", "fixed", pct(inf["alam_dn072_fixed"])],
            ["thick ITO film, index change 0.72 at 1240 nm",
             f"rising (γ/γ₀ = {inf['alam_dn072_hot_gamma']:.2f})", pct(inf["alam_dn072_hot"])],
            ["model of the thick film at 150 GW cm⁻², index change 0.526", "rising",
             pct(inf["alam_model_dn0526_hot"])]]
    t4 = table("inferred", "Drop of the plasma frequency behind two measured optical changes in the Drude model of "
               "each film.", ["Measured change", "Damping", "Drop (%)"], rows)
    return t1, t2, t3, t4


def t_ttm_profiles():
    t = load("r1_ttm_baxter.json")
    rows = [["fixed" if p["var"] == "pec_fixed" else "rising", dnm(p["d"]), f"{p['deg']:g}", pct(p["drop_max"]),
             f"{p['lam_pump'] * 1e9:.0f}", f"{p['A_cold']:.3f}", f"{p['I_GWcm2']:.1f}", f"{p['fluence_mJcm2']:.1f}",
             f"{p['Te_max']:.0f}", f"{p['rise_fs']:.0f}", f"{p['decay_fs']:.0f}"] for p in t["profiles"]]
    t1 = table("ttm_profiles", "Pump that gives each largest drop in the two-temperature model, with the rise and "
               "decay of the drop.", ["Damping", "d (nm)", "θ₀ (°)", "Drop (%)", "Pump (nm)", "Cold absorption",
                                     "I₀ (GW cm⁻²)", "F (mJ cm⁻²)", "Peak T_e (K)", "Rise (fs)", "Decay (fs)"], rows)
    rows = [["fixed" if r["var"] == "pec_fixed" else "rising", dnm(r["d"]), f"{r['deg']:g}", pct(r["depth"]),
             sweep_nm(r), fs(r["T"]), fs(r["pump"]), f"{r['eta_R']:.4f}", f"{r['static']:.4f}", f"{r['switched']:.4f}",
             num(r["margin"], 4, True)] for r in t["designs"]]
    t2 = table("ttm_designs", "Adiabatic design search with the time course of the two-temperature model.",
               ["Damping", "d (nm)", "θ₀ (°)", "Drop (%)", "Sweep (nm)", "FWHM (fs)", "Pump (fs)", "η_R",
                "Best static", "Switched", "Margin"], rows)
    return t1, t2


def t_ttm_drop():
    t = load("r1_ttm_baxter.json")
    s = load("route1_spectra.json")
    prof = {}
    for depth in (0.20, 0.30, 0.40):
        p = [x for x in t["profiles"] if x["var"] == "pec_fixed" and abs(x["d"] - 7.5e-9) < 1e-12
             and x["deg"] == 75.0 and abs(x["depth"] - depth) < 1e-9][0]
        prof[depth] = p
    tk = [x - s["design"]["pump"] * 1e15 for x in s["t_fs"]]
    rows = []
    for T in list(range(-400, 801, 100)) + [1000, 1500, 2000]:
        r = [str(T).replace("-", MINUS)]
        for depth in (0.20, 0.30, 0.40):
            tt = prof[depth]["t"]
            i = min(range(len(tt)), key=lambda k: abs(tt[k] * 1e15 - T))
            r.append(f"{prof[depth]['drop'][i] * 100:.1f}")
        if tk[0] <= T <= tk[-1]:
            i = min(range(len(tk)), key=lambda k: abs(tk[k] - T))
            r.append(f"{s['drop'][i] * 100:.1f}")
        else:
            r.append("")
        rows.append(r)
    return table("ttm_drop", "Drop of the plasma frequency of the 7.5 nm film at 75° against time in the "
                 "two-temperature model with fixed damping, and in the kernel model.",
                 ["Time after the pump peak (fs)", "Largest drop 20 % (%)", "30 % (%)", "40 % (%)", "Kernel, 30 % (%)"],
                 rows)


def t_ttm_fw():
    f = load("route1_ttm_fdtd_baxter.json")
    g = load("route1_ttm_fdtd_baxter_damping.json")
    lab = {"pec_fixed": "fixed", "pec_hot": "rising, T₀ = 15\u202f000\u00a0K"}
    rows = []
    for s in f["summary"]:
        rows.append([lab[s["var"]], dnm(s["d"]), f"{s['deg']:g}", pct(s["depth"]), f"{s['I_GWcm2']:.0f}",
                     f"{s['eta_R']:.4f}", f"{s['static']:.4f}", f"{s['switched']:.4f}", f"{s['momentum']:.4f}",
                     num(s["margin"], 4, True), num(s["Wmod"], 4, True), sci(s["balance"])])
    for s in g["summary"]:
        T0 = s["tag"].split("_")[2].replace("k", "")
        rows.append([f"T₀ = {int(T0) * 1000:,}\u00a0K".replace(",", "\u202f").replace("-", MINUS), dnm(s["d"]),
                     f"{s['deg']:g}", pct(s["depth"]), f"{s['I_GWcm2']:.0f}", f"{s['eta_R']:.4f}", f"{s['static']:.4f}",
                     f"{s['switched']:.4f}", f"{s['momentum']:.4f}", num(s["margin"], 4, True),
                     num(s["Wmod"], 4, True), sci(s["balance"])])
    return table("ttm_fw", "Full-wave results with the time course of the two-temperature model.",
                 ["Damping", "d (nm)", "θ₀ (°)", "Drop (%)", "I₀ (GW cm⁻²)", "η_R", "Best static", "Switched",
                  "Momentum form", "Margin", "W_m", "Balance"], rows)


def t_ttm_damping():
    t = load("r1_ttm_damping_baxter.json")
    rows = []
    for r in t["designs"]:
        T0 = r["var"].replace("pec_", "")
        T0 = "fixed" if T0 == "fixed" else f"{float(T0.split('_')[1].rstrip('k')):g}".replace("-", MINUS)
        rows.append([T0, dnm(r["d"]), f"{r['deg']:g}", pct(r["depth"]), f"{r['gamma_scale_max']:.2f}",
                     f"{r['I_GWcm2']:.0f}", f"{r['decay_fs']:.0f}", f"{r['eta_R']:.4f}", f"{r['static']:.4f}",
                     f"{r['switched']:.4f}", num(r["margin"], 4, True)])
    return table("ttm_damping", "Adiabatic margins with the two-temperature model as the damping law changes from "
                 "falling (T₀ < 0) to rising (T₀ > 0).",
                 ["T₀ (10³ K)", "d (nm)", "θ₀ (°)", "Drop (%)", "γ/γ₀", "I₀ (GW cm⁻²)", "Decay (fs)", "η_R",
                  "Best static", "Switched", "Margin"], rows)


def t_heat():
    h = load("fluence_check_baxter.json")
    lab = {"low-power map": "ITO–gold mirror, low-power map", "saturation": "ITO–gold mirror, saturation",
           "largest": "ITO–gold mirror, largest pump"}
    rows = []
    for r in h["tirole"]:
        rows.append([lab[r["label"]], f"{r['I_GWcm2']:.0f}", f"{r['fluence_mJcm2']:.1f}", f"{r['absorbed_mJcm2']:.2f}",
                     f"{r['energy_density_GJm3']:.2f}", f"{r['Te_max']:.0f}", f"{r['T_eq']:.0f}"])
    for r in h["alam"]:
        rows.append(["thick ITO film, largest pump", f"{r['I_GWcm2']:.0f}", f"{r['fluence_mJcm2']:.1f}",
                     f"{r['absorbed_mJcm2']:.2f}", f"{r['energy_density_GJm3']:.2f}", f"{r['Te_max']:.0f}",
                     f"{r['T_eq']:.0f}"])
    t1 = table("heat_exp", "Heating of the films of two experiments in the two-temperature model.",
               ["Film and pump", "I₀ (GW cm⁻²)", "F (mJ cm⁻²)", "Absorbed (mJ cm⁻²)", "Energy density (GJ m⁻³)",
                "Peak T_e (K)", "T_eq (K)"], rows)
    dmg = h["damage_J_cm2"][0] * 1e3
    rows = [["fixed" if r["var"] == "pec_fixed" else "rising", dnm(r["d"]), f"{r['deg']:g}", pct(r["depth"]),
             f"{r['I_GWcm2']:.1f}", f"{r['fluence_mJcm2']:.1f}", f"{r['fluence_mJcm2'] / dmg:.2f}",
             f"{r['absorbed_mJcm2']:.2f}", f"{r['energy_density_GJm3']:.2f}", f"{r['Te_max']:.0f}", f"{r['T_eq']:.0f}"]
            for r in h["route1"]]
    t2 = table("heat_r1", "Pump fluence, absorbed energy and heating of the thin films on a perfect conductor in the "
               "two-temperature model.",
               ["Damping", "d (nm)", "θ₀ (°)", "Drop (%)", "I₀ (GW cm⁻²)", "F (mJ cm⁻²)", "F/F_damage",
                "Absorbed (mJ cm⁻²)", "Energy density (GJ m⁻³)", "Peak T_e (K)", "T_eq (K)"], rows)
    return t1, t2


def t_convergence():
    c = load("convergence.json")
    lab = {"route1": "7.5 nm, 75°, 30 %, kernel", "realistic": "5 nm, 80°, 40 %, rising damping",
           "route3": "ITO–gold mirror"}
    lev = {"base": "base grid", "cells/2": "film cells halved", "cells/4": "film cells quartered",
           "dt/2": "time step halved", "glass/2": "glass cells halved"}
    rows = []
    for s in c["summary"]:
        rows.append([lab[s["case"]], lev[s["level"]], f"{s['static']:.5f}", f"{s['switched']:.5f}",
                     f"{s['eta_R']:.5f}" if s.get("eta_R") is not None else "",
                     num(s["margin"], 5, True) if s.get("margin") is not None else "", sci(abs(s["balance"]))])
    return table("convergence", "Convergence of the key full-wave results.",
                 ["Case", "Grid", "Best static", "Switched", "η_R", "Margin", "Largest balance"], rows)


def t_gold():
    g = load("gold_fit.json")
    f = g["fit"]
    rows = [[dnm(c["d"]), f"{c['deg']:g}", sci(c["max_dR_fit"]), sci(c["max_dR_tmm"]), f"{c['R_min']:.4f}"]
            for c in g["checks"]]
    hb = 6.582119569e-16
    return table("gold", f"Reflectance of ITO films on gold with the Drude refit of gold (ε∞ = 5, "
                 f"ħω_p = {hb * f['wp']:.3f} eV, 1/γ = {1e15 / f['gamma']:.2f} fs) against the reference model.",
                 ["d (nm)", "θ₀ (°)", "largest ΔR against the reference model", "Closed form against transfer matrix",
                  "Smallest R"], rows)


# ------------------------------------------------------------------ residuals and experimental requirements
def t_residuals():
    tests, ta, tg = load("tests.json"), load("tests_angle.json"), load("tests_angle_gold.json")
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
    ch = [abs(x[k] - base[x["case"]][k]) for x in cv if x["level"] != "base"
          for k in ("static", "switched", "eta_R", "margin")
          if x.get(k) is not None and base[x["case"]].get(k) is not None]
    groups = [("tests T2, switched uniform medium",
               [abs(v["balance"]) for v in tests["T2"].values() if "balance" in v]),
              ("tests T3, switched ITO–gold mirror", [abs(v["balance"]) for v in tests["T3"].values()]),
              ("tests V1, static films on a conductor", [abs(x["balance"]) for x in ta["V1"]]),
              ("tests V2 and V4, switched films",
               [abs(x["balance"]) for x in ta["V2"]] + [abs(x["balance"]) for x in tg["V4"]]),
              ("absorbers on a perfect conductor", pec), ("absorbers on gold", au),
              ("ITO–gold mirror under a chirped probe", mir)]
    rows = [[g, str(len(v)), sci(max(v))] for g, v in groups]
    rows.append(["change of any result under refinement", str(len(ch)), sci(max(ch))])
    return table("residuals", "Largest residual of the energy balance in each group of runs, and the largest change "
                 "of an absorbed fraction, limit or margin when the grid or the time step is refined.",
                 ["Runs", "Number", "Largest value"], rows)


def t_requirements():
    e = read_json(os.path.join(DATA, "extra.json"))
    s = e["sensitivity"]
    eta = e["passband"]["full"]["switched"]
    rows = [[minus_sign(f"{r['dd_nm']:+g}"), "0", f"{r['eta_R']:.4f}", num(eta - r["eta_R"], 4, True)]
            for r in s["thickness"]]
    rows += [["0", minus_sign(f"{r['dtheta_deg']:+g}"), f"{r['eta_R']:.4f}", num(eta - r["eta_R"], 4, True)]
             for r in s["angle"]]
    rows.insert(3, ["0", "0", f"{s['eta_R']:.4f}", num(eta - s["eta_R"], 4, True)])
    t1 = table("sensitivity", "Rozanov limit of the incident spectrum of the 7.5 nm film at 75° when the thickness "
               "or the angle assumed for the budget is in error, and the margin that the full-wave absorbed fraction "
               "of 0.8784 then appears to have.",
               ["Thickness error (nm)", "Angle error (°)", "η_R", "Inferred margin"], rows)
    rows = [["none", "0", f"{eta:.4f}", f"{e['passband']['full']['static']:.4f}",
             f"{e['passband']['full']['eta_R']:.4f}", num(e["passband"]["full"]["margin"], 4, True), "1"]]
    for r in e["passband"]["cuts"]:
        leak = r["pump_leak"]
        rows.append([f"{r['cut_nm']:.0f}", f"{r['probe_blocked']:.3f}", f"{r['switched']:.4f}", f"{r['static']:.4f}",
                     f"{r['eta_R']:.4f}", num(r["margin"], 4, True), sci(leak) if leak < 0.01 else f"{leak:.3f}"])
    t2 = table("passband", "Absorbed fractions and Rozanov limit of the 7.5 nm film at 75° inside the band that a "
               "long-pass filter transmits, and the fraction of a 220 fs pump at 1315 nm that the filter passes.",
               ["Filter edge (nm)", "Probe energy blocked", "Switched", "Static film", "η_R", "Margin",
                "Pump passed"], rows)
    return t1, t2


def minus_sign(t):
    return t.replace("-", MINUS)


if __name__ == "__main__":
    made = []
    for f in (t_tests_kx, t_tests_angle, t_trace, t_a4, t_a4_trace, t_a5, t_a5_all, t_a6, t_route3, t_route3_angles,
              t_static, t_threshold, t_design, t_spectra, t_variants, t_hot, t_gas, t_ttm_profiles, t_ttm_fw,
              t_ttm_damping, t_heat, t_convergence, t_gold, t_ttm_exp, t_ttm_drop, t_a6_broad, t_residuals,
              t_requirements, t_t1):
        r = f()
        if isinstance(r, tuple) and r and isinstance(r[0], tuple):
            made.extend(r)
        elif isinstance(r, tuple) and isinstance(r[0], str):
            made.append(r)
    r1 = t_route1_fw()
    made.append(r1[0])
    print("transform-limited control:", {k: v for k, v in r1[1].items() if k != "rows"})
    for n, k in made:
        print(f"{n}: {k} rows")
    print(f"wrote {len(made)} tables to {OUTD}")
