"""bulk.py -- a uniform time-varying Drude medium, FDTD against the exact solution.

Periodic domain, normal incidence (Ex, Hy), one Fourier mode exp(i k z). On the
staggered grid the spatial derivative of that mode is exactly i k~ with
k~ = (2/dz) sin(k dz/2), so the FDTD must follow the three-variable ODE

    eps0 epsinf dE/dt = -i k~ H - J,   mu0 dH/dt = -i k~ E,
    dJ/dt = -gamma J + eps0 wp(t)^2 E              (current-continuous)
    dQ/dt = -gamma Q + eps0 E,  J = wp(t)^2 Q       (momentum-continuous)

up to the second-order time-stepping error. The ODE is integrated to 1e-12.
"""
import numpy as np
from numba import njit
from scipy.integrate import solve_ivp
from fdtd1d import C0, EPS0, MU0


@njit(cache=True)
def _bulk(nsteps, dt, dz, eps_inf, wp0sq, gamma, wp2_int, wp2_half, momentum,
          E, H, J, recE, out):
    N = E.shape[0]
    a1 = (1.0 - 0.5 * gamma * dt) / (1.0 + 0.5 * gamma * dt)
    for n in range(nsteps):
        wp2n = wp0sq * wp2_int[n]
        if momentum:
            a2 = EPS0 * dt / (1.0 + 0.5 * gamma * dt)
            fac = wp0sq * wp2_half[n]
        else:
            a2 = EPS0 * wp2n * dt / (1.0 + 0.5 * gamma * dt)
            fac = 1.0
        s = 0.0
        for k in range(N):
            J[k] = a1 * J[k] + a2 * E[k]
            jj = fac * J[k]
            s += jj * jj * dz
        wp2h = wp0sq * wp2_half[n]
        out[0] += dt * gamma * s / (EPS0 * wp2h)
        wp2next = wp0sq * wp2_int[n + 1]
        if momentum:
            out[1] += s / (wp2h * wp2h) / (2.0 * EPS0) * (wp2next - wp2n)
        else:
            out[1] += s / (2.0 * EPS0) * (1.0 / wp2next - 1.0 / wp2n)
        for k in range(N):
            dH = (H[k] - H[(k - 1 + N) % N]) / dz
            E[k] += dt / (EPS0 * eps_inf) * (-dH - fac * J[k])
        for k in range(N):
            dE = (E[(k + 1) % N] - E[k]) / dz
            H[k] += dt / MU0 * (-dE)
        recE[n] = E[0]


def ode_matrix(kt, eps_inf, wp2, gamma, momentum):
    """d/dt [E, H, J or Q] = A [E, H, J or Q]."""
    if momentum:
        return np.array([[0, -1j * kt / (EPS0 * eps_inf), -wp2 / (EPS0 * eps_inf)],
                         [-1j * kt / MU0, 0, 0],
                         [EPS0, 0, -gamma]], dtype=complex)
    return np.array([[0, -1j * kt / (EPS0 * eps_inf), -1 / (EPS0 * eps_inf)],
                     [-1j * kt / MU0, 0, 0],
                     [EPS0 * wp2, 0, -gamma]], dtype=complex)


def forward_mode(kt, eps_inf, wp2, gamma, momentum):
    """Eigenvalue and eigenvector of the forward optical mode, exp(-i w t)."""
    lam, vec = np.linalg.eig(ode_matrix(kt, eps_inf, wp2, gamma, momentum))
    i = int(np.argmin(lam.imag))          # most negative Im(lambda) = largest Re(w)
    v = vec[:, i] / vec[0, i]             # E amplitude 1
    return lam[i], v


def run_bulk(scale_fn, momentum, gamma, eps_inf=4.08, wp0=2.88e15, L=2.0e-6, N=400,
             t_end=100e-15, courant=0.5):
    dz = L / N
    k = 2 * np.pi / L
    kt = 2 / dz * np.sin(k * dz / 2)
    dt = courant * dz / C0
    nsteps = int(np.ceil(t_end / dt))
    t_int = np.arange(nsteps + 1) * dt
    wp2_int = scale_fn(t_int) ** 2
    wp2_half = scale_fn(t_int[:-1] + 0.5 * dt) ** 2
    wp0sq = wp0 ** 2
    lam, v = forward_mode(kt, eps_inf, wp0sq * scale_fn(np.array([0.0]))[0] ** 2, gamma, momentum)
    z = np.arange(N) * dz
    def amp(t):                                         # valid before the change
        return v * np.exp(lam * t)
    E = np.real(amp(0.0)[0] * np.exp(1j * k * z))
    H = np.real(amp(0.5 * dt)[1] * np.exp(1j * k * (z + 0.5 * dz)))
    J = np.real(amp(-0.5 * dt)[2] * np.exp(1j * k * z))
    W0 = field_energy(E, H, J, dz, eps_inf, wp0sq * wp2_int[0], momentum)
    recE = np.zeros(nsteps)
    out = np.zeros(2)
    _bulk(nsteps, dt, dz, eps_inf, wp0sq, gamma, wp2_int, wp2_half, bool(momentum),
          E, H, J, recE, out)
    W1 = field_energy(E, H, J, dz, eps_inf, wp0sq * wp2_half[-1], momentum)

    # exact solution of the same mode
    def rhs(t, y):
        s = scale_fn(np.array([t]))[0] ** 2
        A = ode_matrix(kt, eps_inf, wp0sq * s, gamma, momentum)
        yc = y[:3] + 1j * y[3:]
        d = A @ yc
        return np.concatenate([d.real, d.imag])
    y0 = np.concatenate([v.real, v.imag])
    sol = solve_ivp(rhs, (0, t_int[-1]), y0, method="DOP853", rtol=1e-12, atol=1e-12 * np.abs(v).max(),
                    t_eval=t_int[1:], max_step=0.5e-15)
    E_exact = sol.y[0]
    return dict(t=t_int[1:], E=recE, E_exact=E_exact, W0=W0, W1=W1, diss=out[0],
                wmod=out[1], kt=kt, dt=dt)


def field_energy(E, H, J, dz, eps_inf, wp2, momentum):
    Jp = wp2 * J if momentum else J
    return dz * (0.5 * EPS0 * eps_inf * np.sum(E ** 2) + 0.5 * MU0 * np.sum(H ** 2)
                 + np.sum(Jp ** 2) / (2 * EPS0 * wp2))


def step_scale(t_s, drop, width=1e-15):
    """wp(t)/wp0 falling smoothly from 1 to 1 - drop around t_s."""
    def f(t):
        return 1 - drop * 0.5 * (1 + np.tanh((np.asarray(t) - t_s) / width))
    return f
