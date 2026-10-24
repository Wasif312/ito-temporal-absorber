"""validate_fdtd.py -- four tests of the FDTD solver at a fixed in-plane wavevector.

T1  static ITO/Au mirror: FDTD reflectance spectrum against the transfer
    matrix at the same in-plane wavevector, 1100-1600 nm, target 1e-3.
T2  uniform Drude medium switched by a 10 % drop of wp: FDTD against the exact
    single-mode solution, both forms, target 1e-3; energy change equals the
    modulation work minus the dissipation.
T3  the mirror switched while a 220 fs probe is on it: energy balance
    U_inc + W_mod = U_r + U_t + U_diss + U_end, both forms, target 1e-3.
T4  slow modulation (rise 300 fs, decay 3 ps): both forms agree and match
    the adiabatic estimate.
Results go to results/tests.json; the exit status is 1 if any test fails.
"""
import json
import os
import sys
import numpy as np
from fdtd1d import C0, energies
from mirror import KX, gaussian_probe, arrival_time, delta_profile, run_pair
from bulk import run_bulk, step_scale, forward_mode, ode_matrix
from ito_model import eps_ito, eps_gold, tmm

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def tmm_R(lam, scale=1.0):
    th = np.arcsin(KX * lam / (2 * np.pi * 1.5))
    return tmm([2.25, eps_ito(lam, "Tirole2022", wp_scale=scale), eps_gold(lam), 1.0],
               [40e-9, 100e-9], lam, th, "p")[0]


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {}
    failed = []

    # ----------------------------------------------------------------- T1
    print("T1  static mirror, FDTD against transfer matrix")
    T, tc = 6e-15, 30e-15
    src = gaussian_probe(T, tc)
    t1 = {}
    for hc in (10e-9, 5e-9):
        run, ref = run_pair(src, 250e-15, h_coarse=hc)
        Ei = ref["recE"][0]
        Er = run["recE"][0] - Ei
        nfft = 1 << 22
        f = np.fft.rfftfreq(nfft, run["dt"])[1:]
        R = (np.abs(np.fft.rfft(Er, nfft)) ** 2 / np.abs(np.fft.rfft(Ei, nfft)) ** 2)[1:]
        lam = C0 / f
        sel = np.nonzero((lam > 1100e-9) & (lam < 1600e-9))[0]
        idx = sel[np.linspace(0, len(sel) - 1, 60).astype(int)]
        Rt = np.array([tmm_R(wl) for wl in lam[idx]])
        err = np.abs(R[idx] - Rt)
        t1[f"{hc*1e9:.0f}"] = dict(max_err=float(err.max()), worst_nm=float(lam[idx][err.argmax()] * 1e9),
                                   lam_nm=(lam[idx] * 1e9).tolist(), R_fdtd=R[idx].tolist(), R_tmm=Rt.tolist())
        ok = err.max() < 1e-3
        failed += [] if ok else [f"T1 coarse cells {hc*1e9:.0f} nm"]
        print(f"    coarse cells {hc*1e9:.0f} nm: max |R_FDTD - R_TMM| = {err.max():.1e} "
              f"(worst at {lam[idx][err.argmax()]*1e9:.0f} nm)  {'PASS' if ok else 'FAIL'}")
    res["T1"] = t1

    # ----------------------------------------------------------------- T2
    print("T2  uniform Drude medium, 10 % drop of wp at t = 20 fs, FDTD against the exact solution")
    t2 = {}
    sc = step_scale(20e-15, 0.10)
    for gamma in (0.0, 1.45e14):
        for mom in (False, True):
            for cour in (0.5, 0.25):
                r = run_bulk(sc, mom, gamma, courant=cour)
                err = float(np.max(np.abs(r["E"] - r["E_exact"])) / np.max(np.abs(r["E_exact"])))
                bal = float((r["W1"] - r["W0"] - r["wmod"] + r["diss"]) / r["W0"])
                key = f"{'momentum' if mom else 'current'} gamma={gamma:.2e} courant={cour}"
                t2[key] = dict(err=err, balance=bal, wmod=float(r["wmod"] / r["W0"]),
                               diss=float(r["diss"] / r["W0"]), W1=float(r["W1"] / r["W0"]))
                ok = err < 1e-3 and abs(bal) < 1e-3
                failed += [] if ok else [f"T2 {key}"]
                print(f"    {key:42s}: field error {err:.1e}, energy balance {bal:+.1e}, "
                      f"W_mod/W0 {r['wmod']/r['W0']:+.3f}  {'PASS' if ok else 'FAIL'}")
    # time reflection after a sudden lossless switch, exact single-mode solution
    L, N = 2.0e-6, 400
    dz = L / N
    kt = 2 / dz * np.sin(2 * np.pi / L * dz / 2)
    eps_inf, wp0 = 4.08, 2.88e15
    for mom in (False, True):
        lam1, v1 = forward_mode(kt, eps_inf, wp0 ** 2, 0.0, mom)
        A2 = ode_matrix(kt, eps_inf, (0.9 * wp0) ** 2, 0.0, mom)
        ev, V = np.linalg.eig(A2)
        a = np.linalg.solve(V, v1)                 # the state just before the switch is continuous
        # forward: Im(lambda) < 0 large; backward: Im(lambda) > 0 large; static: |lambda| ~ 0
        i_f, i_b = int(np.argmin(ev.imag)), int(np.argmax(ev.imag))
        Ef, Eb = abs(a[i_f] * V[0, i_f]), abs(a[i_b] * V[0, i_b])
        w1, w2 = abs(lam1.imag), abs(ev[i_f].imag)
        t2[f"time reflection {'momentum' if mom else 'current'}"] = dict(
            E_forward=float(Ef), E_backward=float(Eb), w_before=float(w1), w_after=float(w2))
        print(f"    sudden lossless switch, {'momentum' if mom else 'current '} form: forward E {Ef:.3f}, "
              f"backward E {Eb:.3f}, frequency {w1/2/np.pi/1e12:.1f} -> {w2/2/np.pi/1e12:.1f} THz")
    res["T2"] = t2

    # ----------------------------------------------------------------- T3 and T4 share the probe
    T_p = 220e-15
    tc = 700e-15
    t_end = tc + 750e-15
    probe = gaussian_probe(T_p, tc)
    t_arr = arrival_time(tc)
    _, ref = run_pair(probe, t_end, wp_scale=None)

    print("T3  mirror switched under a 220 fs probe, energy balance")
    t3 = {}
    for tau in (-90e-15, 100e-15):
        dfun = delta_profile(t_arr - tau, 0.10)
        for mom in (False, True):
            run, _ = run_pair(probe, t_end, wp_scale=[lambda t, d=dfun: 1 + d(t), None],
                              momentum=mom, ref=ref)
            U_inc, U_r, U_t = energies(run, ref)
            W, D = run["wmod"].sum(), run["diss"].sum()
            bal = (U_inc + W - U_r - U_t - D - run["u_end"]) / U_inc
            key = f"delay {tau*1e15:+.0f} fs {'momentum' if mom else 'current'}"
            t3[key] = dict(R=U_r / U_inc, T=U_t / U_inc, Wmod=W / U_inc, diss=D / U_inc,
                           end=run["u_end"] / U_inc, balance=bal)
            ok = abs(bal) < 1e-3
            failed += [] if ok else [f"T3 {key}"]
            print(f"    {key:28s}: R {U_r/U_inc:.4f}, W_mod {W/U_inc:+.4f}, diss {D/U_inc:.4f}, "
                  f"balance {bal:+.1e}  {'PASS' if ok else 'FAIL'}")
    res["T3"] = t3

    print("T4  slow modulation: both forms against the adiabatic estimate")
    # adiabatic estimate: probe-weighted, spectrally averaged static reflectance
    f0 = C0 / 1300e-9
    fs_ = np.linspace(f0 - 4e12, f0 + 4e12, 41)
    S = np.exp(-4 * np.log(2) * ((fs_ - f0) / (0.441 / T_p)) ** 2)
    drops = np.linspace(0, 0.12, 49)
    Rbar = np.array([np.sum(S * np.array([tmm_R(C0 / f, 1 - x) for f in fs_])) / S.sum() for x in drops])
    t4 = {}
    tt = np.arange(t_arr - 1.5e-12, t_arr + 1.5e-12, 1e-15)
    Ip = np.exp(-4 * np.log(2) * ((tt - t_arr) / T_p) ** 2)
    for tau in (-200e-15, 0.0, 500e-15):
        dfun = delta_profile(t_arr - tau, 0.10, tau_rise=300e-15, tau_dec=3000e-15)
        Rad = np.sum(Ip * np.interp(-dfun(tt), drops, Rbar)) / Ip.sum()
        out = {}
        for mom in (False, True):
            run, _ = run_pair(probe, t_end, wp_scale=[lambda t, d=dfun: 1 + d(t), None],
                              momentum=mom, ref=ref)
            U_inc, U_r, U_t = energies(run, ref)
            out["momentum" if mom else "current"] = U_r / U_inc
        t4[f"delay {tau*1e15:+.0f} fs"] = dict(current=out["current"], momentum=out["momentum"],
                                               adiabatic=float(Rad))
        ok = abs(out["current"] - out["momentum"]) < 2e-3 and abs(out["current"] - Rad) < 5e-3
        failed += [] if ok else [f"T4 delay {tau*1e15:+.0f} fs"]
        print(f"    delay {tau*1e15:+5.0f} fs: R current {out['current']:.4f}, momentum {out['momentum']:.4f}, "
              f"adiabatic {Rad:.4f}  {'PASS' if ok else 'FAIL'}")
    res["T4"] = t4

    res["failed"] = failed
    with open(os.path.join(OUT, "tests.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print("saved results/tests.json")
    print("all tests passed" if not failed else "FAILED: " + "; ".join(failed))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
