"""bench_a5.py -- benchmark A5: the switched lossy layer of Firestein et al. (2022).

Layer of thickness d = 0.4 m on a perfect conductor, eps1 = 1.5 and lossless
while a modulated Blackman-type pulse of duration Tp = 1.6 ns enters; when the
pulse sits at the centre of the layer the layer switches to eps2 = 0.75 and a
conductivity sigma2. D and B stay continuous at the switch.

Checks
  S1  lossless static layer: all energy returns (|Gamma| = 1).
  S2  static lossy layer (eps2, sigma2): FDTD reflectance against the exact
      PEC-backed slab formula across the pulse band.
  S3  switched layer: energy balance with the work of the switch.
Reproduction
  absorbed fraction 1 - U_r/U_inc of the switched layer (LTV), of the static
  layer made of the final material (LTI), and the largest absorbed fraction
  that any time-invariant absorber of the same thickness can reach for the
  same pulse (Rozanov bound, water-filling optimum), against Tp/T0 and sigma2.
Results go to results/a5.json.
"""
import json
import os
import numpy as np
from slab1d import run, flux_parts, blackman_pulse, step, C0, EPS0
from bound_scope import rozanov_max_absorption

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
D_LAY = 0.4
EPS1, EPS2 = 1.5, 0.75
TP = 1.6e-9
T0_START = 0.2e-9
LEN_VAC = 1.2
Z_SRC = 0.3
T_END = 120e-9
B_ROZ = 2 * np.pi ** 2 * 1.0 * D_LAY


def t_switch():
    """Pulse centre at the middle of the layer."""
    t_front = T0_START + TP / 2 + (LEN_VAC - Z_SRC) / C0
    return t_front + (D_LAY / 2) * np.sqrt(EPS1) / C0


def pulse(ratio):
    f0 = ratio / TP
    return blackman_pulse(TP, f0, T0_START)


def gamma_exact(f, eps, sig, d=D_LAY):
    w = 2 * np.pi * f
    ec = eps + 1j * sig / (EPS0 * w)
    n = np.sqrt(ec)
    n = np.where(n.imag < 0, -n, n)
    eta = 1 / n                                   # in units of eta0
    zin = -1j * eta * np.tan(w / C0 * n * d)
    return (zin - 1) / (zin + 1)


def spectrum_lambda(E, dt, nfft=1 << 20):
    F = np.abs(np.fft.rfft(E, nfft)) ** 2
    f = np.fft.rfftfreq(nfft, dt)[1:]
    Sf = F[1:]
    lam = C0 / f
    Sl = Sf * f ** 2 / C0
    o = np.argsort(lam)
    return lam[o], Sl[o], f, Sf


def absorbed(ratio, sig2, ref=None, switched=True, eps_after=EPS2):
    src = pulse(ratio)
    if ref is None:
        ref = run(D_LAY, lambda t: np.full_like(t, 1.0), lambda t: np.zeros_like(t), src, T_END,
                  len_vac=LEN_VAC, z_src=Z_SRC, reference=True)
    if switched:
        ts = t_switch()
        eps_fn, sig_fn = step(ts, EPS1, eps_after), step(ts, 0.0, sig2)
    else:
        def eps_fn(t):
            return np.full_like(np.asarray(t, dtype=float), EPS2)

        def sig_fn(t, sig2=sig2):
            return np.full_like(np.asarray(t, dtype=float), sig2)
    r = run(D_LAY, eps_fn, sig_fn, src, T_END, len_vac=LEN_VAC, z_src=Z_SRC)
    U_inc, U_r = flux_parts(r, ref)
    return dict(eta=1 - U_r / U_inc, U_inc=U_inc, U_r=U_r, diss=r["diss"], wmod=r["wmod"],
                end=r["u_end"]), ref, r


if __name__ == "__main__":
    res = {"checks": {}}
    # S1 lossless static layer
    src = pulse(0.992)
    ref = run(D_LAY, lambda t: np.full_like(t, 1.0), lambda t: np.zeros_like(t), src, T_END,
              len_vac=LEN_VAC, z_src=Z_SRC, reference=True)
    r1 = run(D_LAY, lambda t: np.full_like(np.asarray(t, float), EPS1),
             lambda t: np.zeros_like(np.asarray(t, float)), src, T_END, len_vac=LEN_VAC, z_src=Z_SRC)
    Ui, Ur = flux_parts(r1, ref)
    res["checks"]["S1"] = 1 - Ur / Ui
    print(f"S1  lossless layer: 1 - U_r/U_inc = {1 - Ur/Ui:+.1e}  {'PASS' if abs(1 - Ur/Ui) < 1e-4 else 'FAIL'}")

    # S2 static lossy layer against the exact formula
    out, _, r2 = absorbed(0.992, 0.18, ref=ref, switched=False)
    lam, Sl, f, Sf_inc = spectrum_lambda(ref["recE"][0], ref["dt"])
    Er = r2["recE"][0] - ref["recE"][0]
    _, _, _, Sf_r = spectrum_lambda(Er, r2["dt"])
    band = (f > 0.05e9) & (f < 2.5e9) & (Sf_inc > 1e-6 * Sf_inc.max())
    Rf = Sf_r[band] / Sf_inc[band]
    Rx = np.abs(gamma_exact(f[band], EPS2, 0.18)) ** 2
    err = np.abs(Rf - Rx).max()
    eta_exact = 1 - np.sum(Sf_inc * np.abs(gamma_exact(f, EPS2, 0.18)) ** 2) / np.sum(Sf_inc)
    res["checks"]["S2"] = dict(max_err=float(err), eta_fdtd=out["eta"], eta_exact=float(eta_exact))
    print(f"S2  static lossy layer: max |R_FDTD - R_exact| over 0.05-2.5 GHz = {err:.1e}; absorbed "
          f"{out['eta']:.4f} (FDTD) against {eta_exact:.4f} (exact)  {'PASS' if err < 2e-3 else 'FAIL'}")

    # S3 energy balance of the switched layer
    out, _, _ = absorbed(0.992, 0.18, ref=ref, switched=True)
    bal = (out["U_inc"] + out["wmod"] - out["U_r"] - out["diss"] - out["end"]) / out["U_inc"]
    res["checks"]["S3"] = dict(balance=float(bal), wmod=out["wmod"] / out["U_inc"])
    print(f"S3  switched layer: W_mod/U_inc {out['wmod']/out['U_inc']:+.3f}, balance {bal:+.1e}  "
          f"{'PASS' if abs(bal) < 1e-3 else 'FAIL'}")

    # reproduction
    ratios = [0.0, 0.25, 0.5, 0.75, 0.992, 1.25, 1.5, 1.9, 2.5, 3.0]
    sigmas = [0.006, 0.012, 0.024, 0.048, 0.1, 0.18, 0.3, 0.5, 1.0]
    table = []
    for q in ratios:
        src = pulse(q)
        ref = run(D_LAY, lambda t: np.full_like(t, 1.0), lambda t: np.zeros_like(t), src, T_END,
                  len_vac=LEN_VAC, z_src=Z_SRC, reference=True)
        lam, Sl, f, Sf = spectrum_lambda(ref["recE"][0], ref["dt"])
        etaR, _ = rozanov_max_absorption(Sl, lam, B_ROZ)
        for s in sigmas:
            ltv, _, _ = absorbed(q, s, ref=ref, switched=True)
            lti, _, _ = absorbed(q, s, ref=ref, switched=False)
            sig_only, _, _ = absorbed(q, s, ref=ref, switched=True, eps_after=EPS1)
            table.append(dict(ratio=q, sigma=s, ltv=ltv["eta"], lti=lti["eta"], rozanov=float(etaR),
                              wmod=ltv["wmod"] / ltv["U_inc"], diss=ltv["diss"] / ltv["U_inc"],
                              sig_only=sig_only["eta"], sig_only_wmod=sig_only["wmod"] / sig_only["U_inc"]))
        row = [t for t in table if t["ratio"] == q]
        best = max(row, key=lambda t: t["ltv"])
        r18 = [t for t in row if t["sigma"] == 0.18][0]
        bs = max(row, key=lambda t: t["sig_only"])
        print(f"Tp/T0 {q:5.3f}: Rozanov limit {etaR:.3f}; switched at 0.18 S/m {r18['ltv']:.3f} "
              f"(W_mod {r18['wmod']:+.3f}); best switched {best['ltv']:.3f} at {best['sigma']} S/m; "
              f"static {r18['lti']:.3f}; conductivity-only switch best {bs['sig_only']:.3f} at {bs['sigma']} S/m")
    res["table"] = table
    with open(os.path.join(OUT, "a5.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print("saved results/a5.json")
