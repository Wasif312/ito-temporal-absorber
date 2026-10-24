"""slab_drude.py -- normal incidence on Drude layers backed by a perfect conductor,
with a plasma frequency that varies in time.

Benchmark A6 (Hayran and Monticone, Phys. Rev. Applied 21, 044007 (2024)) and
the ITO version of their periodic route. Geometry: vacuum | layers | perfect
conductor. Each layer has a background permittivity eps_inf, and one Drude
species may occupy any of the layers:

    eps0 eps_inf dE/dt = -dH/dz - J,        mu0 dH/dt = -dE/dz,
    dJ/dt + gamma J = eps0 wp(t)^2 E                current-continuous form,
    dQ/dt + gamma Q = eps0 E,  J = wp(t)^2 Q        momentum-continuous form.

Hayran and Monticone use the current form (their Eq. 3) with
wp(t)^2 = wp0^2 [1 + f_mod sin(w_mod t + phi_mod)] (their Eq. 4).

Uniform grid with E on the nodes and H between them; the last node is the
perfect conductor. Graded PML without frequency shift, which absorbs down to
zero frequency as unipolar pulses need. A soft current sheet launches the
pulse; a reference run with vacuum and a PML in place of the layers and the
conductor gives the incident field on identical nodes.

Energy per unit area: flux E H, dissipation gamma J^2/(eps0 wp^2), work of the
modulation (J^2/2 eps0) d(1/wp^2)/dt (current form) or (Q^2/2 eps0) d(wp^2)/dt
(momentum form), kinetic energy J^2/(2 eps0 wp^2).
"""
import numpy as np
from numba import njit

C0 = 299_792_458.0
EPS0 = 8.8541878128e-12
MU0 = 1.25663706212e-6
ETA0 = np.sqrt(MU0 / EPS0)


@njit(cache=True)
def _core(nsteps, dt, dz, E, H, J, eps_n, w_n, bE, aE, bH, aH, psiE, psiH,
          wp0sq, g, s_int, s_half, momentum, k_lo, k_hi, src_k, src_w, mon_k,
          recE, recH, acc, pec):
    N = E.shape[0]
    a1 = (1.0 - 0.5 * g * dt) / (1.0 + 0.5 * g * dt)
    for n in range(nsteps):
        # Drude current from E^n to J^{n+1/2}
        wp2n = wp0sq * s_int[n]
        wp2h = wp0sq * s_half[n]
        wp2x = wp0sq * s_int[n + 1]
        if momentum:
            a2 = EPS0 * dt / (1.0 + 0.5 * g * dt)
            fac = wp2h
        else:
            a2 = EPS0 * wp2n * dt / (1.0 + 0.5 * g * dt)
            fac = 1.0
        sumJ2 = 0.0
        for k in range(k_lo, k_hi):
            J[k] = a1 * J[k] + a2 * E[k]
            jj = fac * J[k]
            sumJ2 += dz * w_n[k] * jj * jj
        if k_hi > k_lo and min(wp2n, wp2h, wp2x) > 1e-9 * wp0sq:
            acc[0] += dt * g * sumJ2 / (EPS0 * wp2h)
            if momentum:
                acc[1] += sumJ2 / (wp2h * wp2h) / (2.0 * EPS0) * (wp2x - wp2n)
            else:
                acc[1] += sumJ2 / (2.0 * EPS0) * (1.0 / wp2x - 1.0 / wp2n)
        # E^{n+1}
        for k in range(1, N - 1):
            dH = (H[k] - H[k - 1]) / dz
            psiE[k] = bE[k] * psiE[k] + aE[k] * dH
            dH = dH + psiE[k]
            E[k] += dt / (EPS0 * eps_n[k]) * (-dH - w_n[k] * fac * J[k])
        E[src_k] -= dt / (EPS0 * eps_n[src_k]) * src_w[n] / dz
        if pec:
            E[N - 1] = 0.0
        # H^{n+3/2}
        for c in range(N - 1):
            dE = (E[c + 1] - E[c]) / dz
            psiH[c] = bH[c] * psiH[c] + aH[c] * dE
            dE = dE + psiH[c]
            H[c] -= dt / MU0 * dE
        for m in range(mon_k.shape[0]):
            recE[m, n] = E[mon_k[m]]
            recH[m, n] = 0.5 * (H[mon_k[m] - 1] + H[mon_k[m]])


def pml(N, n_left, n_right, dz, dt, m=3):
    """Graded PML in vacuum at both ends, no frequency shift."""
    smax = 0.8 * (m + 1) / (ETA0 * dz)

    def prof(pos):
        d = np.zeros_like(pos)
        if n_left:
            L = n_left * dz
            sel = pos < L
            d[sel] = (L - pos[sel]) / L
        if n_right:
            L = n_right * dz
            z0 = (N - 1) * dz - L
            sel = pos > z0
            d[sel] = (pos[sel] - z0) / L
        sig = smax * d ** m
        b = np.exp(-sig * dt / EPS0)
        a = np.where(sig > 0, b - 1, 0.0)
        b = np.where(sig > 0, b, 0.0)
        return b, a
    zn = np.arange(N) * dz
    zc = zn[:-1] + 0.5 * dz
    bE, aE = prof(zn)
    bH, aH = prof(zc)
    return bE, aE, bH, aH


def run(layers, wp0, gamma, src_fn, t_end, dz, len_vac, wp2_scale=None, momentum=False,
        n_pml=40, z_src=None, z_mon=None, courant=0.5, reference=False):
    """layers: list of (thickness, eps_inf, drude) from the front face to the
    conductor; drude is True for the layer(s) that carry the Drude species.
    wp2_scale: callable of time giving wp(t)^2 / wp0^2 (None for a static
    film). The front face sits at z = len_vac; the source and the monitor
    default to 0.25 and 0.5 of len_vac. With reference=True every cell is
    vacuum and a PML replaces the conductor."""
    n_vac = int(round(len_vac / dz))
    n_lay = [max(1, int(round(t / dz))) for t, _, _ in layers]
    n_tot = sum(n_lay)
    n_right = n_pml if reference else 0
    N = n_vac + n_tot + 1 + n_right
    dt = courant * dz / C0
    nsteps = int(np.ceil(t_end / dt))
    t_int = np.arange(nsteps + 1) * dt
    t_half = t_int[:-1] + 0.5 * dt

    eps_c = np.ones(N - 1)
    drude_c = np.zeros(N - 1)
    if not reference:
        c = n_vac
        for (_t, e, dr), nl in zip(layers, n_lay, strict=True):
            eps_c[c:c + nl] = e
            drude_c[c:c + nl] = 1.0 if dr else 0.0
            c += nl
    eps_n = np.ones(N)
    w_n = np.zeros(N)
    eps_n[1:-1] = 0.5 * (eps_c[1:] + eps_c[:-1])
    w_n[1:-1] = 0.5 * (drude_c[1:] + drude_c[:-1])
    eps_n[0] = eps_c[0]
    eps_n[-1] = eps_c[-1]
    if not reference:
        w_n[-1] = 0.0                       # the conductor node
    idx = np.nonzero(w_n)[0]
    k_lo, k_hi = (int(idx[0]), int(idx[-1]) + 1) if len(idx) else (0, 0)

    bE, aE, bH, aH = pml(N, n_pml, n_right, dz, dt)
    src_k = int(round((0.25 * len_vac if z_src is None else z_src) / dz))
    mon_k = np.array([int(round((0.5 * len_vac if z_mon is None else z_mon) / dz))], dtype=np.int64)
    src_w = src_fn(t_half)
    if wp2_scale is None:
        s_int = np.ones(nsteps + 1)
        s_half = np.ones(nsteps)
    else:
        s_int = np.asarray(wp2_scale(t_int), dtype=float)
        s_half = np.asarray(wp2_scale(t_half), dtype=float)
    E = np.zeros(N)
    H = np.zeros(N - 1)
    J = np.zeros(N)
    psiE = np.zeros(N)
    psiH = np.zeros(N - 1)
    recE = np.zeros((1, nsteps))
    recH = np.zeros((1, nsteps))
    acc = np.zeros(2)
    _core(nsteps, dt, dz, E, H, J, eps_n, w_n, bE, aE, bH, aH, psiE, psiH,
          float(wp0) ** 2, float(gamma), s_int, s_half, bool(momentum), k_lo, k_hi,
          src_k, src_w, mon_k, recE, recH, acc, not reference)
    a = mon_k[0]
    wp2_end = wp0 ** 2 * s_half[-1]
    fac = wp2_end if momentum else 1.0
    u_end = (0.5 * EPS0 * np.sum(eps_n[a:] * E[a:] ** 2) + 0.5 * MU0 * np.sum(H[a:] ** 2)) * dz
    if k_hi > k_lo:
        u_end += np.sum(w_n * (fac * J) ** 2) * dz / (2 * EPS0 * wp2_end)
    return dict(dt=dt, t=t_int[1:], recE=recE, recH=recH, diss=acc[0], wmod=acc[1],
                u_end=u_end, z_front=n_vac * dz, z_src=src_k * dz, z_mon=mon_k[0] * dz,
                nsteps=nsteps)


def flux_parts(run_, ref):
    """Incident and reflected energies at the monitor (reference subtraction)."""
    dt = run_["dt"]

    def hmid(H):
        Hm = np.empty_like(H)
        Hm[1:] = 0.5 * (H[1:] + H[:-1])
        Hm[0] = 0.5 * H[0]
        return Hm
    Ei, Hi = ref["recE"][0], hmid(ref["recH"][0])
    Er = run_["recE"][0] - ref["recE"][0]
    Hr = hmid(run_["recH"][0] - ref["recH"][0])
    return np.sum(Ei * Hi) * dt, -np.sum(Er * Hr) * dt


# ------------------------------------------------------------------ spectra
@njit(cache=True)
def _dtft(x, dt, t0, freqs):
    out = np.zeros(freqs.shape[0], dtype=np.complex128)
    for i in range(freqs.shape[0]):
        w = 2.0 * np.pi * freqs[i]
        re = 0.0
        im = 0.0
        for n in range(x.shape[0]):
            ph = w * (t0 + n * dt)
            re += x[n] * np.cos(ph)
            im += x[n] * np.sin(ph)
        out[i] = complex(re * dt, im * dt)
    return out


def spectrum(x, dt, freqs, decimate=1):
    """Fourier transform int x(t) exp(+i 2 pi f t) dt at the given
    frequencies (time dependence exp(-i omega t)), with x sampled at
    t_n = (n + 1) dt. decimate > 1 keeps every decimate-th sample, which is
    safe when x has no content above 1 / (2 decimate dt)."""
    xs = np.ascontiguousarray(x[decimate - 1::decimate])
    return _dtft(xs, dt * decimate, dt * decimate, np.ascontiguousarray(freqs, dtype=float))


def gamma_static(f, layers, wp0, gamma):
    """Exact reflection coefficient of the static stack on a perfect
    conductor (exp(-i omega t)), layers as in run()."""
    w = 2 * np.pi * np.asarray(f, dtype=float)
    zin = np.zeros_like(w, dtype=complex)            # at the conductor
    for t, e, dr in layers[::-1]:
        eps = e - (wp0 ** 2 / (w ** 2 + 1j * gamma * w) if dr else 0.0)
        n = np.sqrt(eps + 0j)
        n = np.where(n.imag < 0, -n, n)
        z = 1 / n
        tn = np.tan(w / C0 * n * t)
        zin = z * (zin - 1j * z * tn) / (z - 1j * zin * tn)
    return (zin - 1) / (zin + 1)


# ------------------------------------------------------------------ pulses
def gauss_pulse(tau, t0):
    """Unipolar Gaussian exp(-((t - t0)/tau)^2): flat spectrum at low
    frequency."""
    return lambda t: np.exp(-((np.asarray(t) - t0) / tau) ** 2)


def sine_pulse(tau, t0, fc):
    """Gaussian times sin(2 pi fc (t - t0)): no zero-frequency content, the
    spectrum rises in proportion to f at low frequency."""
    return lambda t: (np.exp(-((np.asarray(t) - t0) / tau) ** 2)
                      * np.sin(2 * np.pi * fc * (np.asarray(t) - t0)))
