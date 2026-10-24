"""fdtd1d.py -- one-dimensional FDTD for planar stacks of time-varying Drude media.

p-polarised light at a fixed in-plane wavevector kx. A film that is uniform in
the plane conserves kx even while it changes in time, so each kx is an exact,
independent one-dimensional problem. With the fields written as
F(z, t) exp(i kx x) and Ez = -i Ez', every field is real:

    eps0 epsinf dEx/dt  = -dHy/dz - Jx
    eps0 epsinf dEz'/dt = -kx Hy  - Jz'
    mu0 dHy/dt          = -dEx/dz + kx Ez'

Each Drude species s (here ITO and Au) carries a current that obeys either the
current-continuous form (new carriers appear at rest)

    dJ/dt + gamma(t) J = eps0 wp(t)^2 E

or the momentum-continuous form (the effective mass changes)

    dQ/dt + gamma(t) Q = eps0 E,     J = wp(t)^2 Q.

Leapfrog in time: E at integer steps, H and J (or Q) at half steps. Non-uniform
grid, interfaces on E nodes with length-weighted averaging of the two cells,
convolutional PML at both ends. A soft current sheet in the incidence medium
launches the probe; a reference run without the stack gives the incident
field, so the reflected field is the difference of the two runs.

Energy bookkeeping (all quantities per unit area, averaged over x, with the
common factor 1/2 of the x-average dropped):
    flux        S = Ex Hy
    dissipation p = gamma J^2 / (eps0 wp^2)
    modulation  P = (J^2 / 2 eps0) d(1/wp^2)/dt        current-continuous
                P = (Q^2 / 2 eps0) d(wp^2)/dt            momentum-continuous
"""
import numpy as np
from numba import njit

C0 = 299_792_458.0
EPS0 = 8.8541878128e-12
MU0 = 1.25663706212e-6
ETA0 = np.sqrt(MU0 / EPS0)


# ----------------------------------------------------------------- grid
def build_grid(d_layers, h_fine, h_coarse, len_left, len_right, ratio=1.08):
    """Node positions: uniform cells of about h_fine inside the stack (the
    stack starts at z = 0), geometric growth up to h_coarse outside it."""
    z = [0.0]
    for d in d_layers:
        n = max(1, int(round(d / h_fine)))
        for _ in range(n):
            z.append(z[-1] + d / n)
    z_end = z[-1]
    right, h, pos = [], h_fine, z_end
    while pos - z_end < len_right:
        h = min(h * ratio, h_coarse)
        pos += h
        right.append(pos)
    left, h, pos = [], h_fine, 0.0
    while -pos < len_left:
        h = min(h * ratio, h_coarse)
        pos -= h
        left.append(pos)
    return np.concatenate([np.array(left[::-1]), np.array(z), np.array(right)])


class Medium:
    """eps_inf plus an optional Drude species index."""
    def __init__(self, eps_inf, species=-1):
        self.eps_inf = float(eps_inf)
        self.species = int(species)


def assign(nodes, d_layers, media_layers, medium_in, medium_out):
    """Cell media from the cell midpoints; node values by length weighting."""
    zc = 0.5 * (nodes[1:] + nodes[:-1])
    edges = np.concatenate([[0.0], np.cumsum(d_layers)])
    cell_med = []
    for z in zc:
        if z < 0:
            cell_med.append(medium_in)
        elif z > edges[-1]:
            cell_med.append(medium_out)
        else:
            i = min(np.searchsorted(edges, z) - 1, len(d_layers) - 1)
            cell_med.append(media_layers[i])
    return cell_med


# ----------------------------------------------------------------- CPML
def cpml_profiles(nodes, n_pml, eps_left, eps_right, dt, m=3, kappa_max=1.0,
                  alpha_max=2 * np.pi * EPS0 * 20e12):
    N = len(nodes)
    zc = 0.5 * (nodes[1:] + nodes[:-1])
    zl, zr = nodes[n_pml], nodes[N - 1 - n_pml]
    Ll, Lr = zl - nodes[0], nodes[-1] - zr
    hl, hr = nodes[1] - nodes[0], nodes[-1] - nodes[-2]
    smax_l = 0.8 * (m + 1) / (ETA0 * hl * np.sqrt(eps_left))
    smax_r = 0.8 * (m + 1) / (ETA0 * hr * np.sqrt(eps_right))

    def coeffs(z):
        d = np.zeros_like(z)
        smax = np.zeros_like(z)
        left = z < zl
        right = z > zr
        d[left] = (zl - z[left]) / Ll
        d[right] = (z[right] - zr) / Lr
        smax[left], smax[right] = smax_l, smax_r
        sig = smax * d ** m
        kap = 1 + (kappa_max - 1) * d ** m
        alp = np.where(left | right, alpha_max * (1 - d), 0.0)
        b = np.exp(-(sig / kap + alp) * dt / EPS0)
        with np.errstate(invalid="ignore", divide="ignore"):
            a = np.where(sig > 0, sig / (sig * kap + kap ** 2 * alp) * (b - 1), 0.0)
        b = np.where(left | right, b, 0.0)
        return b, a, kap

    bE, aE, kE = coeffs(nodes)
    bH, aH, kH = coeffs(zc)
    return bE, aE, kE, bH, aH, kH


# ----------------------------------------------------------------- core loop
@njit(cache=True)
def _core(nsteps, dt, kx, Ex, Ez, Hy, dzn, dzh, cEn, cEc,
          bE, aE, kE, bH, aH, kH, psiE, psiH,
          wn, wc, n0, n1, c0, c1, Jx, Jz, wp0sq, g0,
          wp2_int, wp2_half, g_int, momentum,
          src_k, src_w, mon_k, recE, recH, diss, wmod):
    N = Ex.shape[0]
    ns = Jx.shape[0]
    fac = np.zeros(ns)
    for n in range(nsteps):
        # Drude currents from E^n to J^{n+1/2}
        for s in range(ns):
            g = g0[s] * g_int[s, n]
            wp2n = wp0sq[s] * wp2_int[s, n]
            a1 = (1.0 - 0.5 * g * dt) / (1.0 + 0.5 * g * dt)
            if momentum:
                a2 = EPS0 * dt / (1.0 + 0.5 * g * dt)
            else:
                a2 = EPS0 * wp2n * dt / (1.0 + 0.5 * g * dt)
            for k in range(n0[s], n1[s]):
                Jx[s, k] = a1 * Jx[s, k] + a2 * Ex[k]
            for c in range(c0[s], c1[s]):
                Jz[s, c] = a1 * Jz[s, c] + a2 * Ez[c]
            wp2h = wp0sq[s] * wp2_half[s, n]
            if momentum:
                fac[s] = wp2h
            else:
                fac[s] = 1.0
            sumJ2 = 0.0
            for k in range(n0[s], n1[s]):
                jj = fac[s] * Jx[s, k]
                sumJ2 += dzn[k] * wn[s, k] * jj * jj
            for c in range(c0[s], c1[s]):
                jj = fac[s] * Jz[s, c]
                sumJ2 += dzh[c] * wc[s, c] * jj * jj
            diss[s] += dt * g * sumJ2 / (EPS0 * wp2h)
            wp2next = wp0sq[s] * wp2_int[s, n + 1]
            if momentum:
                wmod[s] += sumJ2 / (wp2h * wp2h) / (2.0 * EPS0) * (wp2next - wp2n)
            else:
                wmod[s] += sumJ2 / (2.0 * EPS0) * (1.0 / wp2next - 1.0 / wp2n)
        # E^{n+1}
        for k in range(1, N - 1):
            dH = (Hy[k] - Hy[k - 1]) / dzn[k]
            psiE[k] = bE[k] * psiE[k] + aE[k] * dH
            dH = dH / kE[k] + psiE[k]
            J = 0.0
            for s in range(ns):
                J += wn[s, k] * fac[s] * Jx[s, k]
            Ex[k] += cEn[k] * (-dH - J)
        Ex[src_k] -= cEn[src_k] * src_w[n] / dzn[src_k]
        for c in range(N - 1):
            J = 0.0
            for s in range(ns):
                J += wc[s, c] * fac[s] * Jz[s, c]
            Ez[c] += cEc[c] * (-kx * Hy[c] - J)
        # H^{n+3/2}
        for c in range(N - 1):
            dE = (Ex[c + 1] - Ex[c]) / dzh[c]
            psiH[c] = bH[c] * psiH[c] + aH[c] * dE
            dE = dE / kH[c] + psiH[c]
            Hy[c] += dt / MU0 * (-dE + kx * Ez[c])
        for m in range(mon_k.shape[0]):
            recE[m, n] = Ex[mon_k[m]]
            recH[m, n] = 0.5 * (Hy[mon_k[m] - 1] + Hy[mon_k[m]])


# ----------------------------------------------------------------- driver
class Stack:
    """Planar stack: incidence medium | layers | exit medium, with Drude
    species given as (wp0, gamma0) and optional time scalings."""
    def __init__(self, medium_in, layers, medium_out, species):
        self.medium_in = medium_in
        self.layers = layers            # list of (Medium, thickness)
        self.medium_out = medium_out
        self.species = species          # list of (wp0, gamma0)


def simulate(stack, kx, src_wave_fn, t_end, h_fine=1e-9, h_coarse=10e-9,
             len_left=3.0e-6, len_right=1.2e-6, n_pml=24, courant=0.5,
             z_src=-2.0e-6, z_mon_in=-1.0e-6, z_mon_out=0.8e-6,
             wp_scale=None, gamma_scale=None, momentum=False, reference=False):
    """Run one simulation. wp_scale[s] and gamma_scale[s] are callables of
    time (s) giving wp(t)/wp0 and gamma(t)/gamma0 for species s. With
    reference=True every cell holds the incidence medium."""
    d_layers = [d for _, d in stack.layers]
    nodes = build_grid(d_layers, h_fine, h_coarse, len_left, len_right)
    if reference:
        med_layers = [stack.medium_in] * len(d_layers)
        med_out = stack.medium_in
    else:
        med_layers = [m for m, _ in stack.layers]
        med_out = stack.medium_out
    cell_med = assign(nodes, d_layers, med_layers, stack.medium_in, med_out)
    N = len(nodes)
    dzh = np.diff(nodes)
    dzn = np.empty(N)
    dzn[1:-1] = 0.5 * (nodes[2:] - nodes[:-2])
    dzn[0], dzn[-1] = dzh[0], dzh[-1]
    dt = courant * dzh.min() / C0
    nsteps = int(np.ceil(t_end / dt))

    eps_c = np.array([m.eps_inf for m in cell_med])
    ns = len(stack.species)
    wc = np.zeros((ns, N - 1))
    for c, m in enumerate(cell_med):
        if m.species >= 0:
            wc[m.species, c] = 1.0
    eps_n = np.empty(N)
    wn = np.zeros((ns, N))
    eps_n[0], eps_n[-1] = eps_c[0], eps_c[-1]
    for k in range(1, N - 1):
        hl, hr = dzh[k - 1], dzh[k]
        eps_n[k] = (hl * eps_c[k - 1] + hr * eps_c[k]) / (hl + hr)
        wn[:, k] = (hl * wc[:, k - 1] + hr * wc[:, k]) / (hl + hr)
    n0 = np.zeros(ns, dtype=np.int64)
    n1 = np.zeros(ns, dtype=np.int64)
    c0 = np.zeros(ns, dtype=np.int64)
    c1 = np.zeros(ns, dtype=np.int64)
    for s in range(ns):
        idx = np.nonzero(wn[s])[0]
        if len(idx):
            n0[s], n1[s] = idx[0], idx[-1] + 1
        idx = np.nonzero(wc[s])[0]
        if len(idx):
            c0[s], c1[s] = idx[0], idx[-1] + 1

    cEn = dt / (EPS0 * eps_n)
    cEc = dt / (EPS0 * eps_c)
    bE, aE, kE, bH, aH, kH = cpml_profiles(nodes, n_pml, cell_med[0].eps_inf,
                                           cell_med[-1].eps_inf, dt)
    t_int = np.arange(nsteps + 1) * dt
    t_half = t_int[:-1] + 0.5 * dt
    wp0sq = np.array([w ** 2 for w, _ in stack.species])
    g0 = np.array([g for _, g in stack.species])
    wp2_int = np.ones((ns, nsteps + 1))
    wp2_half = np.ones((ns, nsteps))
    g_int = np.ones((ns, nsteps + 1))
    for s in range(ns):
        if wp_scale is not None and wp_scale[s] is not None:
            wp2_int[s] = wp_scale[s](t_int) ** 2
            wp2_half[s] = wp_scale[s](t_half) ** 2
        if gamma_scale is not None and gamma_scale[s] is not None:
            g_int[s] = gamma_scale[s](t_int)
    src_k = int(np.argmin(np.abs(nodes - z_src)))
    z_end = float(sum(d_layers))
    mon_k = np.array([np.argmin(np.abs(nodes - z_mon_in)),
                      np.argmin(np.abs(nodes - (z_end + z_mon_out)))], dtype=np.int64)
    src_w = src_wave_fn(t_half)

    Ex = np.zeros(N)
    Hy = np.zeros(N - 1)
    Ez = np.zeros(N - 1)
    psiE = np.zeros(N)
    psiH = np.zeros(N - 1)
    Jx = np.zeros((ns, N))
    Jz = np.zeros((ns, N - 1))
    recE = np.zeros((2, nsteps))
    recH = np.zeros((2, nsteps))
    diss = np.zeros(ns)
    wmod = np.zeros(ns)
    _core(nsteps, dt, kx, Ex, Ez, Hy, dzn, dzh, cEn, cEc, bE, aE, kE, bH, aH, kH,
          psiE, psiH, wn, wc, n0, n1, c0, c1, Jx, Jz, wp0sq, g0,
          wp2_int, wp2_half, g_int, bool(momentum), src_k, src_w, mon_k,
          recE, recH, diss, wmod)

    # energy stored at the end between the two monitors
    a, b = mon_k
    fac = np.array([wp0sq[s] * wp2_half[s, -1] if momentum else 1.0 for s in range(ns)])
    u_em = (0.5 * EPS0 * np.sum(eps_n[a:b + 1] * Ex[a:b + 1] ** 2 * dzn[a:b + 1])
            + 0.5 * EPS0 * np.sum(eps_c[a:b] * Ez[a:b] ** 2 * dzh[a:b])
            + 0.5 * MU0 * np.sum(Hy[a:b] ** 2 * dzh[a:b]))
    u_kin = 0.0
    for s in range(ns):
        wp2 = wp0sq[s] * wp2_half[s, -1]
        u_kin += np.sum(wn[s] * (fac[s] * Jx[s]) ** 2 * dzn) / (2 * EPS0 * wp2)
        u_kin += np.sum(wc[s] * (fac[s] * Jz[s]) ** 2 * dzh) / (2 * EPS0 * wp2)
    return dict(dt=dt, t=t_int[1:], nodes=nodes, mon_k=mon_k, src_k=src_k,
                recE=recE, recH=recH, diss=diss, wmod=wmod, u_end=u_em + u_kin,
                nsteps=nsteps)


def flux(rec, m):
    """Poynting flux S = Ex Hy at monitor m, at the integer times of recE."""
    E = rec["recE"][m]
    H = rec["recH"][m]
    Hm = np.empty_like(H)
    Hm[1:] = 0.5 * (H[1:] + H[:-1])
    Hm[0] = 0.5 * H[0]
    return E * Hm


def energies(run, ref):
    """Incident, reflected and transmitted energies, from a stack run and the
    matching reference run (same grid, same source)."""
    dt = run["dt"]
    S_inc = flux(ref, 0)
    Er = run["recE"][0] - ref["recE"][0]
    Hr = run["recH"][0] - ref["recH"][0]
    Hm = np.empty_like(Hr)
    Hm[1:] = 0.5 * (Hr[1:] + Hr[:-1])
    Hm[0] = 0.5 * Hr[0]
    U_inc = np.sum(S_inc) * dt
    U_r = -np.sum(Er * Hm) * dt
    U_t = np.sum(flux(run, 1)) * dt
    return U_inc, U_r, U_t
