"""bench_a6.py -- benchmark A6: periodic modulation of a lossy Drude slab
(Hayran and Monticone, Phys. Rev. Applied 21, 044007 (2024)).

Slab of thickness d on a perfect conductor, eps_inf = 1, Drude current with
wp^2(t) = wp0^2 [1 + f sin(w_m (t - t_a) + phi)] in the current form of their
Eqs. (3)-(4); t_a is the time at which the pulse centre reaches the slab.
Lengths in units of d (d = 1 um in the code), frequencies in c0/d, so that
wp0 d/2 pi c0 = 6 reads wp = 6 below.

Part 1, the broadband case of their Fig. 3 (wp = 6, gamma = 330, w_m = 0.133).
The Rozanov integral I_R of the equivalent reflection coefficient
Gamma_eq = E_r(w)/E_i(w) over lambda/d = 0.02-1000, against the bound 2 pi^2 d,
and the absorbed energy fraction against the water-filling limit, for two test
pulses with the same short-wavelength content: a unipolar Gaussian, whose
spectrum stays flat down to zero frequency, and a Gaussian times a sine
carrier, whose spectrum falls in proportion to f at low frequency.
Part 2, the narrowband case of their Fig. 4 (wp = 5.5, gamma = 100, w_m = 0.1,
their phase phi/pi = 0.45, pulse band lambda/d = 1.66-200). Their absorption
efficiency (Eq. 19, weighted by wavelength) against their bound AE_R
(Eqs. 20-21), and the absorbed energy fraction against the water-filling limit.
Part 3. The absorbed energy fraction of the narrowband pulse against w_m and f,
best and mean over the modulation phase.
Checks: static FDTD against the exact formula, energy balance, grid and time
convergence, momentum form. Results go to results/a6.json.
"""
import json
import os
import time
import numpy as np
from slab_drude import run, flux_parts, spectrum, gamma_static, C0
from bound_scope import rozanov_max_absorption
from runopts import Workers, n_workers

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
D = 1e-6
U = C0 / D
B = 2 * np.pi ** 2                    # Rozanov bound of a non-magnetic slab, in units of d
LV = 1.2 * D
LAY = [(D, 1.0, True)]
T_AFTER = 80.0                        # simulated time after the pulse reaches the slab, d/c0

PULSES = {"gauss": dict(tau=0.0137, fc=0.0, dz=D / 1000, dec=10),
          "sine": dict(tau=0.0106, fc=25.0, dz=D / 1000, dec=10),
          "narrow": dict(tau=0.888, fc=0.3035, dz=D / 200, dec=20)}
BROAD = dict(wp=6.0, g=330.0, wm=0.133)
NARROW = dict(wp=5.5, g=100.0, wm=0.1, phi_paper=0.45 * np.pi)
F_B = np.exp(np.linspace(np.log(1e-3), np.log(50.0), 3000))       # c0/d
F_N = np.linspace(1e-3, 1.5, 3000)
BAND_N = (1.66, 200.0)
PHI12 = np.arange(12) * np.pi / 6
PHI16 = np.arange(16) * np.pi / 8


def pulse_fn(name):
    p = PULSES[name]
    tau, fc = p["tau"] / U, p["fc"] * U
    t0 = 6 * tau

    def f(t):
        x = np.asarray(t) - t0
        env = np.exp(-(x / tau) ** 2)
        return env if fc == 0 else env * np.sin(2 * np.pi * fc * x)
    return f, t0 + 0.75 * LV / C0


REF = {}


def get_ref(name, dz, T):
    key = (name, dz, T)
    if key not in REF:
        src, ta = pulse_fn(name)
        r = run(LAY, 1.0, 1.0, src, ta + T / U, dz, LV, reference=True)
        F = F_N if name == "narrow" else F_B
        dec = PULSES[name]["dec"] * int(round(PULSES[name]["dz"] / dz))
        r["Ei"] = spectrum(r["recE"][0], r["dt"], F * U, decimate=dec)
        r["dec"] = dec
        REF[key] = r
    return REF[key]


def one(job):
    name, wp, g, wm, fm, ph, dzf, Tf, mom, spec = job
    dz = PULSES[name]["dz"] * dzf
    T = T_AFTER * Tf
    src, ta = pulse_fn(name)
    ref = get_ref(name, dz, T)
    W_M = 2 * np.pi * wm * U
    scale = (lambda t: 1 + fm * np.sin(W_M * (np.asarray(t) - ta) + ph)) if fm > 0 else None
    r = run(LAY, 2 * np.pi * wp * U, 2 * np.pi * g * U, src, ta + T / U, dz, LV,
            wp2_scale=scale, momentum=mom)
    Ui, Ur = flux_parts(r, ref)
    out = dict(pulse=name, wp=wp, g=g, wm=wm, f=fm, phi=ph, dzf=dzf, Tf=Tf, momentum=mom,
               AE=1 - Ur / Ui, W=r["wmod"] / Ui, diss=r["diss"] / Ui,
               balance=(Ui - Ur + r["wmod"] - r["diss"] - r["u_end"]) / Ui)
    if spec:
        F = F_N if name == "narrow" else F_B
        Er = spectrum(r["recE"][0] - ref["recE"][0], r["dt"], F * U, decimate=ref["dec"])
        Ei = ref["Ei"]
        lnG = np.log(np.abs(Er / Ei))
        if name == "narrow":
            w = 1 / F ** 2                                   # d(lambda) = df / f^2
            sel = (1 / F >= BAND_N[0]) & (1 / F <= BAND_N[1])
            out["IR"] = float(-np.trapezoid(lnG[sel] * w[sel], F[sel]))
            out["AE_lam"] = float(1 - np.trapezoid(np.abs(Er) ** 2 * w, F)
                                  / np.trapezoid(np.abs(Ei) ** 2 * w, F))
            out["AE_spec"] = float(1 - np.trapezoid(np.abs(Er) ** 2, F) / np.trapezoid(np.abs(Ei) ** 2, F))
        else:
            lnf = np.log(F)
            out["IR"] = float(-np.trapezoid(lnG / F, lnf))
            lam = 1 / F
            parts = []
            for a, b in ((0.02, 1), (1, 15), (15, 100), (100, 1000)):
                sel = (lam >= a) & (lam <= b)
                parts.append(float(-np.trapezoid(lnG[sel] / F[sel], lnf[sel])))
            out["IR_parts"] = parts
        if spec == 2:
            out["absG"] = np.abs(Er / Ei).tolist()
            out["Er2"] = (np.abs(Er) ** 2).tolist()
    return out


def limits(name):
    """Water-filling limits and the best static slab for the incident
    spectrum of the FDTD reference run."""
    ref = get_ref(name, PULSES[name]["dz"], T_AFTER)
    F = F_N if name == "narrow" else F_B
    Sf = np.abs(ref["Ei"]) ** 2
    Sf = Sf / Sf.max()
    # energy weighting: S per unit lambda = S_f f^2 (c0 = d = 1)
    o = np.argsort(1 / F)
    lam = (1 / F)[o]
    Se = Sf * F ** 2
    out = dict(eta_R_energy=float(rozanov_max_absorption((Se / Se.max())[o], lam, B)[0]))
    if name == "narrow":
        out["eta_R_lam"] = float(rozanov_max_absorption(Sf[o], lam, B)[0])
        w = 1 / F ** 2
        sel = (1 / F >= BAND_N[0]) & (1 / F <= BAND_N[1])
        G0 = np.exp(-B / (BAND_N[1] - BAND_N[0]))
        refl = np.trapezoid(np.where(sel, G0 ** 2, 1.0) * Sf * w, F) / np.trapezoid(Sf * w, F)
        out["AE_R_paper"] = float(1 - refl)
        out["Gamma0"] = float(G0)
    # best static slab of the same thickness and damping, plasma frequency free
    case = NARROW if name == "narrow" else BROAD
    best_e, best_l = (0, -1), (0, -1)
    for wp in np.arange(1.0, 40.01, 0.25):
        G2 = np.abs(gamma_static(F * U, LAY, 2 * np.pi * wp * U, 2 * np.pi * case["g"] * U)) ** 2
        ae = 1 - np.trapezoid(Sf * G2, F) / np.trapezoid(Sf, F)
        best_e = max(best_e, (ae, wp))
        if name == "narrow":
            ael = 1 - np.trapezoid(Sf * G2 / F ** 2, F) / np.trapezoid(Sf / F ** 2, F)
            best_l = max(best_l, (ael, wp))
    out["best_static_energy"] = dict(AE=float(best_e[0]), wp=float(best_e[1]))
    if name == "narrow":
        out["best_static_lam"] = dict(AE=float(best_l[0]), wp=float(best_l[1]))
    # energy of the pulse beyond lambda = 100 d
    sel = F < 0.01
    out["energy_frac_beyond_100d"] = float(np.trapezoid(Sf[sel], F[sel]) / np.trapezoid(Sf, F)) \
        if name != "narrow" else None
    return out


def static_check():
    """Static slab of the broadband case: FDTD against the exact formula."""
    r = one(("gauss", BROAD["wp"], BROAD["g"], BROAD["wm"], 0.0, 0.0, 1.0, 1.0, False, 2))
    G_x = np.abs(gamma_static(F_B * U, LAY, 2 * np.pi * BROAD["wp"] * U, 2 * np.pi * BROAD["g"] * U))
    dl = np.abs(np.log(np.array(r["absG"])) - np.log(G_x))
    IR_x = float(-np.trapezoid(np.log(G_x) / F_B, np.log(F_B)))
    lam = 1 / F_B
    return dict(IR_fdtd=r["IR"], IR_exact=IR_x, max_dln_lam_ge_1=float(dl[lam >= 1].max()),
                max_dln_all=float(dl.max()), balance=r["balance"],
                AE_fdtd=r["AE"])


if __name__ == "__main__":
    workers = n_workers(__doc__)
    t_start = time.time()
    jobs = []
    for name in ("gauss", "sine"):
        jobs.append((name, BROAD["wp"], BROAD["g"], BROAD["wm"], 0.0, 0.0, 1.0, 1.0, False, 2))
        for fm in (0.004, 0.0135, 0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.96):
            for ph in PHI12:
                jobs.append((name, BROAD["wp"], BROAD["g"], BROAD["wm"], fm, ph, 1.0, 1.0, False, 1))
    nf = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.96, 1.0]
    jobs.append(("narrow", NARROW["wp"], NARROW["g"], NARROW["wm"], 0.0, 0.0, 1.0, 1.0, False, 2))
    for fm in nf:
        for ph in list(PHI16) + [NARROW["phi_paper"]]:
            jobs.append(("narrow", NARROW["wp"], NARROW["g"], NARROW["wm"], fm, ph, 1.0, 1.0, False, 1))
    map_wm = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    map_f = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for wm in map_wm:
        for fm in map_f:
            for ph in PHI16:
                jobs.append(("narrow", NARROW["wp"], NARROW["g"], wm, fm, ph, 1.0, 1.0, False, 0))
    with Workers(workers) as pool:
        res = pool.map(one, jobs, chunksize=4)
        print(f"{len(jobs)} runs in {time.time() - t_start:.0f} s")
        res_b = [r for r in res if r["pulse"] != "narrow"]
        res_n = [r for r in res if r["pulse"] == "narrow" and r["wm"] == NARROW["wm"] and
                 ("AE_lam" in r)]
        res_m = [r for r in res if r["pulse"] == "narrow" and "AE_lam" not in r]

        def best(rows, key="AE"):
            return max(rows, key=lambda r: r[key])
        # spectra of the best cases and the checks
        extra = []
        for name in ("gauss", "sine"):
            for fm in (0.05, 0.96):
                b = best([r for r in res_b if r["pulse"] == name and r["f"] == fm], "IR")
                extra.append((name, BROAD["wp"], BROAD["g"], BROAD["wm"], fm, b["phi"], 1.0, 1.0, False, 2))
        bn = best([r for r in res_n if r["f"] == 1.0])
        extra.append(("narrow", NARROW["wp"], NARROW["g"], NARROW["wm"], 1.0, bn["phi"], 1.0, 1.0, False, 2))
        bm = best(res_m)
        extra.append(("narrow", NARROW["wp"], NARROW["g"], bm["wm"], bm["f"], bm["phi"], 1.0, 1.0, False, 2))
        # convergence and momentum form
        extra += [("narrow", NARROW["wp"], NARROW["g"], bm["wm"], bm["f"], bm["phi"], 0.5, 1.0, False, 0),
                  ("narrow", NARROW["wp"], NARROW["g"], bm["wm"], bm["f"], bm["phi"], 1.0, 2.0, False, 0),
                  ("narrow", NARROW["wp"], NARROW["g"], bm["wm"], bm["f"], bm["phi"], 1.0, 1.0, True, 0)]
        bs = best([r for r in res_b if r["pulse"] == "sine" and r["f"] == 0.05], "IR")
        extra += [("sine", BROAD["wp"], BROAD["g"], BROAD["wm"], 0.05, bs["phi"], 0.5, 1.0, False, 1),
                  ("sine", BROAD["wp"], BROAD["g"], BROAD["wm"], 0.05, bs["phi"], 1.0, 2.0, False, 1),
                  ("sine", BROAD["wp"], BROAD["g"], BROAD["wm"], 0.05, bs["phi"], 1.0, 1.0, True, 1)]
        res_x = pool.map(one, extra)
    lim = {name: limits(name) for name in PULSES}
    chk = static_check()

    print("\nstatic check, broadband slab: I_R FDTD %.4f d, exact %.4f d (bound %.4f d); "
          "max |d ln|Gamma|| %.1e for lambda >= d, %.1e overall" %
          (chk["IR_fdtd"], chk["IR_exact"], B, chk["max_dln_lam_ge_1"], chk["max_dln_all"]))
    print("largest |energy balance| over all runs: %.1e" % max(abs(r["balance"]) for r in res + res_x))
    print("\nPart 1. Broadband case of Fig. 3 (best phase for each f)")
    for name in ("gauss", "sine"):
        L = lim[name]
        print(f"  {name}: water-filling limit {L['eta_R_energy']:.4f}, best static slab "
              f"{L['best_static_energy']['AE']:.4f} at wp = {L['best_static_energy']['wp']}; "
              f"energy beyond 100 d {L['energy_frac_beyond_100d']:.1e}")
        for fm in [0.0, 0.004, 0.0135, 0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.96]:
            rows = [r for r in res_b if r["pulse"] == name and r["f"] == fm]
            bi = best(rows, "IR")
            ba = best(rows, "AE")
            print(f"    f = {fm:6.4f}: I_R/B best {bi['IR']/B:6.3f} (phi/pi {bi['phi']/np.pi:.2f}, parts/B "
                  + " ".join(f"{p/B:+.3f}" for p in bi["IR_parts"])
                  + f"), worst {min(r['IR'] for r in rows)/B:6.3f}; absorbed best {ba['AE']:.4f}, "
                  f"mean over phase {np.mean([r['AE'] for r in rows]):.4f}, W {ba['W']:+.1e}")
    L = lim["narrow"]
    print("\nPart 2. Narrowband case of Fig. 4: AE_R (paper) %.4f with Gamma0 %.4f; water-filling limit "
          "%.4f (energy), %.4f (wavelength weighting); best static %.4f (energy, wp %.2f), %.4f "
          "(wavelength weighting, wp %.2f)" % (L["AE_R_paper"], L["Gamma0"], L["eta_R_energy"],
                                              L["eta_R_lam"], L["best_static_energy"]["AE"],
                                              L["best_static_energy"]["wp"], L["best_static_lam"]["AE"],
                                              L["best_static_lam"]["wp"]))
    for fm in [0.0] + nf:
        rows_all = [r for r in res_n if r["f"] == fm]
        # best and mean over the 16 equally spaced phases; the paper's phase separately
        rows = [r for r in rows_all if abs(r["phi"] - NARROW["phi_paper"]) > 1e-9]
        b = best(rows)
        bl = best(rows, "AE_lam")
        pp = [r for r in rows_all if abs(r["phi"] - NARROW["phi_paper"]) < 1e-9]
        txt = f", paper phase: AE {pp[0]['AE']:.4f}, AE_lam {pp[0]['AE_lam']:.4f}" if pp else ""
        print(f"    f = {fm:4.2f}: absorbed energy best {b['AE']:.4f} (phi/pi {b['phi']/np.pi:.3f}), mean "
              f"{np.mean([r['AE'] for r in rows]):.4f}; AE_lam best {bl['AE_lam']:.4f}; I_R/B best "
              f"{max(r['IR'] for r in rows)/B:.3f}{txt}")
    print("\nPart 3. Narrowband pulse, best and mean absorbed energy over the phase")
    for wm in map_wm:
        line = []
        for fm in map_f:
            rows = [r for r in res_m if r["wm"] == wm and r["f"] == fm]
            line.append(f"{best(rows)['AE']:.3f}")
        print(f"    w_m = {wm:4.2f}: " + " ".join(line))
    print(f"    best {bm['AE']:.4f} at w_m {bm['wm']}, f {bm['f']}, phi/pi {bm['phi']/np.pi:.3f}, W {bm['W']:+.1e}")
    for r in res_x:
        if r["dzf"] != 1.0 or r["Tf"] != 1.0 or r["momentum"]:
            print(f"    check {r['pulse']}: dz x{r['dzf']}, time x{r['Tf']}, momentum {r['momentum']}: "
                  f"AE {r['AE']:.5f}" + (f", I_R/B {r['IR']/B:.4f}" if "IR" in r else "")
                  + f", W {r['W']:+.1e}, balance {r['balance']:+.1e}")
    out = dict(B=B, broad=BROAD, narrow=NARROW, pulses=PULSES, F_B=F_B.tolist(), F_N=F_N.tolist(),
               limits=lim, static_check=chk, results=res, extra=res_x, map_wm=map_wm, map_f=map_f)
    with open(os.path.join(OUT, "a6.json"), "w") as fh:
        json.dump(out, fh)
    print(f"saved results/a6.json ({time.time() - t_start:.0f} s)")
