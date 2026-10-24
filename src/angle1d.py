"""angle1d.py -- one-dimensional FDTD at a fixed angle of incidence, for planar
stacks of time-varying Drude media on a perfect conductor.

p-polarised light enters from a medium of index n0 at the angle theta0. Every
field depends on x and t only through tau = t - s x / c0, with s = n0 sin(theta0),
so every frequency keeps the same angle. A pump that sweeps along the film
together with the probe footprint changes the film as a function of tau alone,
which keeps this form, and the problem reduces exactly to one dimension:

    eps0 epsinf dEx/dtau = -dHy/dz - Jx
    mu_eff dHy/dtau      = -dEx/dz + (s/c0) Jz / (eps0 epsinf),   mu_eff = mu0 (1 - s^2/epsinf)
    eps0 epsinf Ez       = -(s/c0) Hy - Pz,     dPz/dtau = Jz
    dJ/dtau + gamma J    = eps0 wp^2 E          (current-continuous form)
    dQ/dtau + gamma Q    = eps0 E, J = wp^2 Q   (momentum-continuous form)

At a fixed angle the Rozanov bound B = 2 pi^2 n0 d / cos(theta0) holds for every static,
non-magnetic stack, so a time-varying result can be tested against it. The explicit scheme
needs epsinf > s^2 in every cell; ITO (4.08) and the glass (2.25) satisfy it,
gold with epsinf = 1 does not above 41.8 deg, so the bound test uses a perfect
conductor behind the film.

Leapfrog in tau: Ex on the nodes at whole steps; Hy, Ez and Pz on the cells at
half steps; Jx on the nodes at half steps; Jz on the cells at whole steps.
Graded PML (stretched coordinate, no frequency shift) at the left end, and at
the right end of the reference run. Energy bookkeeping as in fdtd1d.py:
flux Ex Hy, dissipation gamma J^2/(eps0 wp^2), modulation work
(J^2/2 eps0) d(1/wp^2)/dtau (current form) or (Q^2/2 eps0) d(wp^2)/dtau
(momentum form). Time integrals over tau at fixed x equal those over t.
"""
import numpy as np
from numba import njit
from fdtd1d import build_grid, Medium, C0, EPS0, MU0, ETA0

CHUNK = 1 << 20          # time steps per pass of the kernel (memory of the modulation arrays)


@njit(cache=True)
def _core(nsteps, dt, sc, Ex, Hy, Pz, Jx, Jz, dzn, dzh, cEn, cHc, eps_c,
          bE, aE, bH, aH, psiE, psiH, wn, wc, n0s, n1s, c0s, c1s, wp0sq, g0,
          wp2_int, wp2_half, g_int, g_half, momentum, src_k, src_w, mon_k,
          recE, recH, diss, wmod, pec):
    N = Ex.shape[0]
    ns = Jx.shape[0]
    facx = np.ones(ns)
    facz = np.ones(ns)
    for n in range(nsteps):
        # Jz^{n+1} from Ez^{n+1/2} (cells), centred at tau_{n+1/2}
        for s in range(ns):
            g = g0[s] * g_half[s, n]
            wp2c = wp0sq[s] * wp2_half[s, n]
            a1 = (1.0 - 0.5 * g * dt) / (1.0 + 0.5 * g * dt)
            if momentum:
                a2 = EPS0 * dt / (1.0 + 0.5 * g * dt)
                facz[s] = wp0sq[s] * wp2_int[s, n + 1]
            else:
                a2 = EPS0 * wp2c * dt / (1.0 + 0.5 * g * dt)
                facz[s] = 1.0
            sumJ2 = 0.0
            for c in range(c0s[s], c1s[s]):
                ez = -(sc * Hy[c] + Pz[c]) / (EPS0 * eps_c[c])
                Jz[s, c] = a1 * Jz[s, c] + a2 * ez
                jj = facz[s] * Jz[s, c]
                sumJ2 += dzh[c] * wc[s, c] * jj * jj
            # energy over [tau_{n+1/2}, tau_{n+3/2}] with J at tau_{n+1}
            wp2j = wp0sq[s] * wp2_int[s, n + 1]
            gj = g0[s] * g_int[s, n + 1]
            wa = wp0sq[s] * wp2_half[s, n]
            wb = wp0sq[s] * wp2_half[s, n + 1]
            if c1s[s] > c0s[s] and wa > 0.0 and wb > 0.0 and wp2j > 0.0:
                diss[s] += dt * gj * sumJ2 / (EPS0 * wp2j)
                if momentum:
                    wmod[s] += sumJ2 / (wp2j * wp2j) / (2.0 * EPS0) * (wb - wa)
                else:
                    wmod[s] += sumJ2 / (2.0 * EPS0) * (1.0 / wb - 1.0 / wa)
        # Jx^{n+1/2} from Ex^n (nodes), centred at tau_n
        for s in range(ns):
            g = g0[s] * g_int[s, n]
            wp2n = wp0sq[s] * wp2_int[s, n]
            a1 = (1.0 - 0.5 * g * dt) / (1.0 + 0.5 * g * dt)
            if momentum:
                a2 = EPS0 * dt / (1.0 + 0.5 * g * dt)
                facx[s] = wp0sq[s] * wp2_half[s, n]
            else:
                a2 = EPS0 * wp2n * dt / (1.0 + 0.5 * g * dt)
                facx[s] = 1.0
            sumJ2 = 0.0
            for k in range(n0s[s], n1s[s]):
                Jx[s, k] = a1 * Jx[s, k] + a2 * Ex[k]
                jj = facx[s] * Jx[s, k]
                sumJ2 += dzn[k] * wn[s, k] * jj * jj
            wp2h = wp0sq[s] * wp2_half[s, n]
            gh = g0[s] * g_half[s, n]
            wp2x = wp0sq[s] * wp2_int[s, n + 1]
            if n1s[s] > n0s[s] and wp2n > 0.0 and wp2h > 0.0 and wp2x > 0.0:
                diss[s] += dt * gh * sumJ2 / (EPS0 * wp2h)
                if momentum:
                    wmod[s] += sumJ2 / (wp2h * wp2h) / (2.0 * EPS0) * (wp2x - wp2n)
                else:
                    wmod[s] += sumJ2 / (2.0 * EPS0) * (1.0 / wp2x - 1.0 / wp2n)
        # Ex^{n+1}
        for k in range(1, N - 1):
            dH = (Hy[k] - Hy[k - 1]) / dzn[k]
            psiE[k] = bE[k] * psiE[k] + aE[k] * dH
            dH = dH + psiE[k]
            J = 0.0
            for s in range(ns):
                J += wn[s, k] * facx[s] * Jx[s, k]
            Ex[k] += cEn[k] * (-dH - J)
        Ex[src_k] -= cEn[src_k] * src_w[n] / dzn[src_k]
        if pec:
            Ex[N - 1] = 0.0
        # Hy^{n+3/2} and Pz^{n+3/2}
        for c in range(N - 1):
            dE = (Ex[c + 1] - Ex[c]) / dzh[c]
            psiH[c] = bH[c] * psiH[c] + aH[c] * dE
            dE = dE + psiH[c]
            J = 0.0
            for s in range(ns):
                J += wc[s, c] * facz[s] * Jz[s, c]
            Hy[c] += cHc[c] * (-dE + sc * J / (EPS0 * eps_c[c]))
            Pz[c] += dt * J
        for m in range(mon_k.shape[0]):
            recE[m, n] = Ex[mon_k[m]]
            recH[m, n] = 0.5 * (Hy[mon_k[m] - 1] + Hy[mon_k[m]])


def pml_profiles(nodes, n_left, n_right, dt, m=3):
    """Stretched-coordinate PML without frequency shift, graded as d^m."""
    N = len(nodes)
    zc = 0.5 * (nodes[1:] + nodes[:-1])
    zl = nodes[n_left] if n_left else -np.inf
    zr = nodes[N - 1 - n_right] if n_right else np.inf
    hl = nodes[1] - nodes[0]
    hr = nodes[-1] - nodes[-2]
    smax_l = 0.8 * (m + 1) / (ETA0 * hl)
    smax_r = 0.8 * (m + 1) / (ETA0 * hr)

    def prof(z):
        d = np.zeros_like(z)
        smax = np.zeros_like(z)
        if n_left:
            sel = z < zl
            d[sel] = (zl - z[sel]) / (zl - nodes[0])
            smax[sel] = smax_l
        if n_right:
            sel = z > zr
            d[sel] = (z[sel] - zr) / (nodes[-1] - zr)
            smax[sel] = smax_r
        sig = smax * d ** m
        b = np.exp(-sig * dt / EPS0)
        a = np.where(sig > 0, b - 1, 0.0)
        b = np.where(sig > 0, b, 0.0)
        return b, a
    bE, aE = prof(nodes)
    bH, aH = prof(zc)
    return bE, aE, bH, aH


class AngleStack:
    """Incidence medium (index n0) | layers on a perfect conductor.
    layers: list of (Medium, thickness); species: list of (wp0, gamma0)."""
    def __init__(self, n0, layers, species):
        self.n0 = float(n0)
        self.layers = layers
        self.species = species


def simulate(stack, theta0, src_fn, t_end, h_fine=1e-9, h_coarse=20e-9, len_left=4.0e-6,
             len_right_ref=3.0e-6, n_pml=30, courant=0.5, z_src=-2.5e-6, z_mon=-1.2e-6,
             wp_scale=None, gamma_scale=None, momentum=False, reference=False, chunk=CHUNK):
    """Run one simulation at the angle theta0 (rad) inside the incidence medium.
    wp_scale[s], gamma_scale[s]: callables of tau giving wp/wp0 and gamma/gamma0.
    With reference=True every cell holds the incidence medium and the grid
    continues to a PML on the right, on nodes identical up to the film.
    The modulation and the source are evaluated `chunk` steps at a time, which
    bounds the memory of long runs and leaves the result unchanged."""
    s_ang = stack.n0 * np.sin(theta0)
    sc = s_ang / C0
    d_layers = [d for _, d in stack.layers]
    nodes = build_grid(d_layers, h_fine, h_coarse, len_left,
                       len_right_ref if reference else 0.0)
    N = len(nodes)
    dzh = np.diff(nodes)
    dzn = np.empty(N)
    dzn[1:-1] = 0.5 * (nodes[2:] - nodes[:-2])
    dzn[0], dzn[-1] = dzh[0], dzh[-1]
    zc = 0.5 * (nodes[1:] + nodes[:-1])
    edges = np.concatenate([[0.0], np.cumsum(d_layers)])
    inc = Medium(stack.n0 ** 2)
    cell_med = []
    for z in zc:
        if reference or z < 0 or z > edges[-1]:
            cell_med.append(inc)
        else:
            i = min(np.searchsorted(edges, z) - 1, len(d_layers) - 1)
            cell_med.append(stack.layers[i][0])
    eps_c = np.array([m.eps_inf for m in cell_med])
    if np.any(eps_c <= s_ang ** 2):
        raise ValueError("eps_inf must exceed (n0 sin theta0)^2 in every cell")
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
    if not reference:
        wn[:, -1] = 0.0                       # the conductor node
    n0s = np.zeros(ns, dtype=np.int64)
    n1s = np.zeros(ns, dtype=np.int64)
    c0s = np.zeros(ns, dtype=np.int64)
    c1s = np.zeros(ns, dtype=np.int64)
    for s in range(ns):
        idx = np.nonzero(wn[s])[0]
        if len(idx):
            n0s[s], n1s[s] = idx[0], idx[-1] + 1
        idx = np.nonzero(wc[s])[0]
        if len(idx):
            c0s[s], c1s[s] = idx[0], idx[-1] + 1
    mu_eff = MU0 * (1 - s_ang ** 2 / eps_c)
    # one time step for the stack run and its reference run: the smallest cell
    # with the fastest z-speed of any medium in the stack, c0 / sqrt(eps_inf - s^2)
    eps_all = np.array([stack.n0 ** 2] + [m.eps_inf for m, _ in stack.layers])
    cz_max = C0 / np.sqrt(np.min(eps_all - s_ang ** 2))
    dt = courant * np.min(dzh) / cz_max
    nsteps = int(np.ceil(t_end / dt))
    cEn = dt / (EPS0 * eps_n)
    cHc = dt / mu_eff
    bE, aE, bH, aH = pml_profiles(nodes, n_pml, n_pml if reference else 0, dt)
    wp0sq = np.array([w ** 2 for w, _ in stack.species])
    g0 = np.array([g for _, g in stack.species])
    src_k = int(np.argmin(np.abs(nodes - z_src)))
    mon_k = np.array([int(np.argmin(np.abs(nodes - z_mon)))], dtype=np.int64)
    Ex = np.zeros(N)
    Hy = np.zeros(N - 1)
    Pz = np.zeros(N - 1)
    psiE = np.zeros(N)
    psiH = np.zeros(N - 1)
    Jx = np.zeros((ns, N))
    Jz = np.zeros((ns, N - 1))
    recE = np.zeros((1, nsteps))
    recH = np.zeros((1, nsteps))
    diss = np.zeros(ns)
    wmod = np.zeros(ns)
    for n_a in range(0, nsteps, chunk):
        m = min(chunk, nsteps - n_a)
        # tau at whole and half steps n_a ... n_a + m, as in a single pass
        t_int = np.arange(n_a, n_a + m + 1) * dt
        t_half = t_int + 0.5 * dt
        wp2_int = np.ones((ns, m + 1))
        wp2_half = np.ones((ns, m + 1))
        g_int = np.ones((ns, m + 1))
        g_half = np.ones((ns, m + 1))
        for s in range(ns):
            if wp_scale is not None and wp_scale[s] is not None:
                wp2_int[s] = wp_scale[s](t_int) ** 2
                wp2_half[s] = wp_scale[s](t_half) ** 2
            if gamma_scale is not None and gamma_scale[s] is not None:
                g_int[s] = gamma_scale[s](t_int)
                g_half[s] = gamma_scale[s](t_half)
        src_w = src_fn(t_half[:-1])
        rE = np.zeros((1, m))
        rH = np.zeros((1, m))
        _core(m, dt, sc, Ex, Hy, Pz, Jx, Jz, dzn, dzh, cEn, cHc, eps_c,
              bE, aE, bH, aH, psiE, psiH, wn, wc, n0s, n1s, c0s, c1s, wp0sq, g0,
              wp2_int, wp2_half, g_int, g_half, bool(momentum), src_k, src_w, mon_k,
              rE, rH, diss, wmod, not reference)
        recE[:, n_a:n_a + m] = rE
        recH[:, n_a:n_a + m] = rH
    a = mon_k[0]
    Ez = -(sc * Hy + Pz) / (EPS0 * eps_c)
    u_end = (0.5 * EPS0 * np.sum(eps_n[a:] * Ex[a:] ** 2 * dzn[a:])
             + 0.5 * EPS0 * np.sum(eps_c[a:] * Ez[a:] ** 2 * dzh[a:])
             + 0.5 * MU0 * np.sum(Hy[a:] ** 2 * dzh[a:]))
    return dict(dt=dt, t=np.arange(1, nsteps + 1) * dt, nodes=nodes, mon_k=mon_k, src_k=src_k, recE=recE,
                recH=recH, diss=diss, wmod=wmod, u_end=u_end, nsteps=nsteps, s=s_ang)


def _flux_integral(E, H, dt):
    """Time integral of E Hy at the monitor, with Hy averaged to the times of E
    (H[-1] = 0): sum_n E[n] (H[n] + H[n-1]) / 2 dt."""
    return 0.5 * (np.dot(E, H) + np.dot(E[1:], H[:-1])) * dt


def energies(run, ref):
    """Incident and reflected energies per unit area at the monitor."""
    dt = run["dt"]
    U_inc = _flux_integral(ref["recE"][0], ref["recH"][0], dt)
    U_r = -_flux_integral(run["recE"][0] - ref["recE"][0], run["recH"][0] - ref["recH"][0], dt)
    return U_inc, U_r
