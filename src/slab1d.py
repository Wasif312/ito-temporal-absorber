"""slab1d.py -- normal incidence on a switched layer backed by a perfect conductor.

Benchmark A5 (Firestein, Shlivinski and Hadad, Phys. Rev. Applied 17, 014017
(2022)): vacuum | layer of thickness d | PEC. The layer has a non-dispersive
permittivity eps(t) and a conductivity sigma(t) that switch while a pulse is
inside it. At a switch the charge and the magnetic flux stay continuous, i.e.
D and B, so the update uses D as the state variable:

    dD/dt = -dH/dz - sigma(t) E,     E = D / (eps0 eps(t)),
    mu0 dH/dt = -dE/dz.

Energy bookkeeping (per unit area): flux S = E H, dissipation sigma E^2, and
the work of the switch P_mod = (D^2 / 2 eps0) d(1/eps)/dt.
"""
import numpy as np
from numba import njit

C0 = 299_792_458.0
EPS0 = 8.8541878128e-12
MU0 = 1.25663706212e-6
ETA0 = np.sqrt(MU0 / EPS0)


@njit(cache=True)
def _core(nsteps, dt, dz, D, E, H, eps_int, sig_half, w_node, eps_bg, bE, aE, bH, aH,
          psiE, psiH, src_k, src_w, mon_k, recE, recH, acc, pec):
    N = E.shape[0]
    for n in range(nsteps):
        # D and E at n+1
        for k in range(1, N - 1):
            dH = (H[k] - H[k - 1]) / dz
            psiE[k] = bE[k] * psiE[k] + aE[k] * dH
            dH = dH + psiE[k]
            w = w_node[k]
            eps_new = eps_bg[k] * (1.0 - w) + eps_int[n + 1] * w
            eps_old = eps_bg[k] * (1.0 - w) + eps_int[n] * w
            sig = sig_half[n] * w
            Dn = D[k]
            # a change of eps acts at the start of the step, with D held fixed;
            # its work on the field is D^2/(2 eps0) (1/eps_new - 1/eps_old)
            if w > 0.0 and eps_new != eps_old:
                acc[1] += dz * Dn * Dn / (2.0 * EPS0) * (1.0 / eps_new - 1.0 / eps_old)
            En = Dn / (EPS0 * eps_new)
            Dnew = (Dn + dt * (-dH) - 0.5 * dt * sig * En) / (1.0 + 0.5 * dt * sig / (EPS0 * eps_new))
            if k == src_k:
                Dnew -= dt * src_w[n] / dz
            Enew = Dnew / (EPS0 * eps_new)
            Em = 0.5 * (En + Enew)
            acc[0] += dt * dz * sig * Em * Em
            D[k] = Dnew
            E[k] = Enew
        if pec:
            E[N - 1] = 0.0
            D[N - 1] = 0.0
        # H at n+3/2
        for c in range(N - 1):
            dE = (E[c + 1] - E[c]) / dz
            psiH[c] = bH[c] * psiH[c] + aH[c] * dE
            dE = dE + psiH[c]
            H[c] += dt / MU0 * (-dE)
        for m in range(mon_k.shape[0]):
            recE[m, n] = E[mon_k[m]]
            recH[m, n] = 0.5 * (H[mon_k[m] - 1] + H[mon_k[m]])


def pml(N, n_left, n_right, dz, dt, m=3):
    """Standard graded PML (no frequency shift, so it also absorbs the
    near-zero frequencies of unipolar pulses) in vacuum at both ends."""
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


def run(d, eps_fn, sig_fn, src_fn, t_end, dz=1e-3, len_vac=1.2, n_pml=40,
        z_src=0.3, z_mon=0.6, courant=0.5, reference=False):
    """Vacuum of length len_vac (PML inside it at the left end), then the
    layer of thickness d from z = len_vac to len_vac + d, then a PEC. With
    reference=True the layer is vacuum and a PML replaces the PEC."""
    n_vac = int(round(len_vac / dz))
    n_lay = int(round(d / dz))
    n_right = n_pml if reference else 0
    N = n_vac + n_lay + 1 + n_right
    dt = courant * dz / C0
    nsteps = int(np.ceil(t_end / dt))
    t_int = np.arange(nsteps + 1) * dt
    k0 = n_vac
    w_node = np.zeros(N)
    eps_bg = np.ones(N)
    if not reference:
        w_node[k0 + 1:k0 + n_lay] = 1.0
        w_node[k0] = 0.5
        w_node[k0 + n_lay] = 1.0          # last node is the PEC wall
    eps_int = eps_fn(t_int)
    sig_half = sig_fn(t_int[:-1] + 0.5 * dt)
    bE, aE, bH, aH = pml(N, n_pml, n_right, dz, dt)
    src_k = int(round(z_src / dz))
    mon_k = np.array([int(round(z_mon / dz))], dtype=np.int64)
    src_w = src_fn(t_int[:-1] + 0.5 * dt)
    D = np.zeros(N)
    E = np.zeros(N)
    H = np.zeros(N - 1)
    psiE = np.zeros(N)
    psiH = np.zeros(N - 1)
    recE = np.zeros((1, nsteps))
    recH = np.zeros((1, nsteps))
    acc = np.zeros(2)
    _core(nsteps, dt, dz, D, E, H, eps_int, sig_half, w_node, eps_bg, bE, aE, bH, aH,
          psiE, psiH, src_k, src_w, mon_k, recE, recH, acc, not reference)
    a, b = mon_k[0], N - 1
    eps_end = eps_bg * (1 - w_node) + eps_int[-1] * w_node
    u_end = (np.sum(0.5 * D[a:b] ** 2 / (EPS0 * eps_end[a:b])) + np.sum(0.5 * MU0 * H[a:b] ** 2)) * dz
    return dict(dt=dt, t=t_int[1:], recE=recE, recH=recH, diss=acc[0], wmod=acc[1],
                u_end=u_end, z_front=k0 * dz)


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


# --------------------------------------------------------------- pulses
NUTTALL = (0.355768, 0.487396, 0.144232, 0.012604)   # four-term window (Nuttall 1981)


def blackman_pulse(Tp, f0, t_start, coeffs=NUTTALL):
    """Four-term Blackman-type window of duration Tp starting at t_start,
    modulated by a carrier f0 centred on the window."""
    a0, a1, a2, a3 = coeffs

    def f(t):
        x = (np.asarray(t) - t_start) / Tp
        win = a0 - a1 * np.cos(2 * np.pi * x) + a2 * np.cos(4 * np.pi * x) - a3 * np.cos(6 * np.pi * x)
        win = np.where((x >= 0) & (x <= 1), win, 0.0)
        return win * np.cos(2 * np.pi * f0 * (np.asarray(t) - t_start - Tp / 2))
    return f


def step(t_s, before, after, t_ramp=0.0):
    """Value switching from `before` to `after` at t_s; with t_ramp > 0 a
    linear ramp of that duration centred on t_s."""
    def f(t):
        t = np.asarray(t, dtype=float)
        if t_ramp <= 0:
            return np.where(t < t_s, before, after)
        x = np.clip((t - (t_s - t_ramp / 2)) / t_ramp, 0, 1)
        return before + (after - before) * x
    return f
