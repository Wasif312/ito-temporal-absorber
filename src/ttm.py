"""ttm.py -- time-dependent hot-electron model of ITO (two-temperature model).

The conduction electrons of ITO absorb the pump and heat up; their temperature
Te sets the plasma frequency (nonparabolic band) and the damping, and the
electrons cool by passing energy to the lattice. Following Baxter et al.
(Adv. Photonics Res. 4, 2200280 (2023), Eqs. 5a-b and SI Eqs. 33-34) and
Sarkar, Un and Sivan (Phys. Rev. Applied 19, 014005 (2023), Eqs. 21-22):

    C_e(Te) dTe/dt = P(t) - g_ep(Te) (Te - Tl)
    C_l     dTl/dt =        g_ep(Te) (Te - Tl)

with
    band          E (1 + C E) = hbar^2 k^2 / (2 m*),  C = 0.4191 eV^-1,
                  m* = 0.3964 me (Liu et al. 2014, as read in Un et al.)
    density of states
                  D(E) = (1 / 2 pi^2) (2 m* / hbar^2)^(3/2) sqrt(E (1 + C E)) (1 + 2 C E)
    chemical potential mu(Te) at fixed electron density N = int D f dE
    electron energy U(Te) = int E D(E) f(E; mu, Te) dE,  C_e = dU/dTe
    electron-phonon coupling (Lin, Zhigilei and Celli 2008 form, used by Baxter et al.)
                  g_ep(Te) = (pi kB A_ep / hbar) (1 / D(E_F)) int D(E)^2 (-df/dE) dE,
                  A_ep = lambda <(hbar omega)^2> = 5.25e-4 eV^2 (Alam et al. 2018, as
                  used by Baxter et al.); the zero-temperature value is pi kB A_ep D(E_F) / hbar
    lattice heat capacity C_l = 2.54e6 J m^-3 K^-1 (Sarkar et al., Baxter et al.)
    plasma frequency
                  wp^2(Te) / wp^2(300 K) from int D v^2 (-df/dE) dE (Baxter et al. Eq. 3)
    damping       gamma(Te) = gamma0 (T0 + Te) / (T0 + 300 K), T0 = 15 000 K
                  (Baxter et al. Eq. 10, scaled to 300 K); other laws can be passed in
    pump          P(t) = A_ITO(Te) I(t) cos(theta) / d for a film of thickness d much
                  thinner than the wavelength, where A_ITO is the fraction of the
                  incident pump power that the ITO layer absorbs at the instantaneous
                  permittivity (self-consistent, adiabatic in the pump envelope) and
                  I(t) the beam intensity in the incidence medium.

The integrals run over the Kane band without an upper cut-off, as in Eq. 3 of
Baxter et al.; electrons at Te of a few 10^4 K reach energies where the
two-band form no longer holds, which limits the model at the largest drops.
"""
import numpy as np

HBAR = 1.054571817e-34      # J s
KB = 1.380649e-23           # J/K
QE = 1.602176634e-19        # C
ME = 9.1093837015e-31       # kg
EPS0 = 8.8541878128e-12
C_L = 2.54e6                # J m^-3 K^-1
A_EP = 5.25e-4 * QE ** 2    # J^2
T_ROOM = 300.0


class ElectronGas:
    """Thermodynamics of the conduction electrons of ITO on a temperature grid."""

    def __init__(self, E_F_eV=0.9, m_rel=0.3964, C_eV=0.4191, te_max=60000.0, n_t=600, n_e=24001):
        self.E_F = E_F_eV * QE
        self.m = m_rel * ME
        self.C = C_eV / QE
        self.a = (2 * self.m / HBAR ** 2) ** 1.5 / (2 * np.pi ** 2)
        kF2 = 2 * self.m / HBAR ** 2 * self.E_F * (1 + self.C * self.E_F)
        self.N = kF2 ** 1.5 / (3 * np.pi ** 2)
        self.D_F = self.dos(self.E_F)
        self.n_e = n_e
        self.T = np.unique(np.concatenate([np.linspace(1.0, T_ROOM, 30),
                                           np.geomspace(T_ROOM, te_max, n_t)]))
        rows = [self._state(T) for T in self.T]
        self.mu = np.array([r[0] for r in rows])
        self.U = np.array([r[1] for r in rows])
        self.D2 = np.array([r[2] for r in rows])
        self.W = np.array([r[3] for r in rows])          # Drude weight up to a constant
        self.Ce = np.gradient(self.U, self.T)
        i300 = np.argmin(np.abs(self.T - T_ROOM))
        self.W300 = self.W[i300]
        self.U300 = self.U[i300]

    def dos(self, E):
        E = np.asarray(E, dtype=float)
        return self.a * np.sqrt(np.clip(E * (1 + self.C * E), 0, None)) * (1 + 2 * self.C * E)

    def _grid(self, mu, T):
        kT = KB * T
        return np.linspace(0.0, max(self.E_F, mu) + 60 * kT, self.n_e)

    def _fermi(self, E, mu, T):
        x = np.clip((E - mu) / (KB * T), -700, 700)
        f = 1.0 / (np.exp(x) + 1.0)
        return f, f * (1 - f) / (KB * T)

    def _density(self, mu, T):
        E = self._grid(mu, T)
        f, _ = self._fermi(E, mu, T)
        return np.trapezoid(self.dos(E) * f, E)

    def _state(self, T):
        lo, hi = self.E_F - 60 * KB * T - 5 * QE, self.E_F + 0.5 * QE
        for _ in range(90):
            mid = 0.5 * (lo + hi)
            if self._density(mid, T) > self.N:
                hi = mid
            else:
                lo = mid
        mu = 0.5 * (lo + hi)
        E = self._grid(mu, T)
        f, mdf = self._fermi(E, mu, T)
        D = self.dos(E)
        U = np.trapezoid(E * D * f, E)
        D2 = np.trapezoid(D ** 2 * mdf, E)
        k = np.sqrt(np.clip(E * (1 + self.C * E), 0, None))
        W = np.trapezoid(k ** 3 / (1 + 2 * self.C * E) * mdf, E)    # g v^2 up to a constant
        return mu, U, D2, W

    # interpolated properties -------------------------------------------------
    def wp2_ratio(self, T):
        return np.interp(T, self.T, self.W) / self.W300

    def drop(self, T):
        return 1.0 - np.sqrt(self.wp2_ratio(T))

    def ce(self, T):
        return np.interp(T, self.T, self.Ce)

    def g_ep(self, T, A_ep=A_EP):
        return np.pi * KB * A_ep / HBAR * np.interp(T, self.T, self.D2) / self.D_F

    def energy(self, T):
        return np.interp(T, self.T, self.U)

    def temperature(self, U):
        return np.interp(U, self.U, self.T)

    def gamma_e(self):
        """Low-temperature Sommerfeld coefficient C_e / Te (J m^-3 K^-2)."""
        sel = (self.T > 50) & (self.T < 300)
        return float(np.mean(self.Ce[sel] / self.T[sel]))


def load_gas(E_F_eV=0.9, m_rel=0.3964, C_eV=0.4191):
    """ElectronGas from results/ttm_gas_<E_F>.json when it exists (building the
    tables takes about 40 s), else built and saved there."""
    import json
    import os
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                     f"ttm_gas_{E_F_eV:g}_{m_rel:g}_{C_eV:g}.json")
    if os.path.exists(p):
        with open(p) as fh:
            d = json.load(fh)
        g = ElectronGas.__new__(ElectronGas)
        for k, v in d.items():
            setattr(g, k, np.array(v) if isinstance(v, list) else v)
        g.n_e = int(g.n_e)
        return g
    g = ElectronGas(E_F_eV, m_rel, C_eV)
    keys = ["E_F", "m", "C", "a", "N", "D_F", "n_e", "T", "mu", "U", "D2", "W", "Ce", "W300", "U300"]
    with open(p, "w") as fh:
        json.dump({k: (getattr(g, k).tolist() if isinstance(getattr(g, k), np.ndarray) else float(getattr(g, k)))
                   for k in keys}, fh)
    return g


def gamma_baxter(T, T0=15000.0):
    """Damping scale gamma(Te)/gamma(300 K), Baxter et al. Eq. 10."""
    return (T0 + np.asarray(T, dtype=float)) / (T0 + T_ROOM)


# ------------------------------------------------------------------ optics
def layer_absorption(eps_list, d_list, lam, theta0, pol):
    """Fraction of the incident power absorbed in each inner layer of a planar
    stack (coherent transfer matrix with forward and backward amplitudes in
    every layer, as in S. J. Byrnes, arXiv:1603.02720). Returns
    (R, T, [A_1, ..., A_n])."""
    n = [np.sqrt(complex(e)) for e in eps_list]
    n = [ni if ni.imag >= 0 else -ni for ni in n]
    k0 = 2 * np.pi / lam
    kx = n[0] * np.sin(theta0)
    cos_t = []
    for ni in n:
        c = np.sqrt(1 - (kx / ni) ** 2 + 0j)
        kz = ni * c
        if kz.imag < 0 or (abs(kz.imag) < 1e-14 and kz.real < 0):
            c = -c
        cos_t.append(c)

    def fresnel(i, j):
        ni, nj, ci, cj = n[i], n[j], cos_t[i], cos_t[j]
        if pol == "s":
            return (ni * ci - nj * cj) / (ni * ci + nj * cj), 2 * ni * ci / (ni * ci + nj * cj)
        return (nj * ci - ni * cj) / (nj * ci + ni * cj), 2 * ni * ci / (nj * ci + ni * cj)

    N = len(n)
    M_list = [None] * N
    for j in range(1, N - 1):
        delta = k0 * n[j] * cos_t[j] * d_list[j - 1]
        r, t = fresnel(j, j + 1)
        M_list[j] = (np.array([[np.exp(-1j * delta), 0], [0, np.exp(1j * delta)]])
                     @ np.array([[1, r], [r, 1]], dtype=complex) / t)
    r01, t01 = fresnel(0, 1)
    Mt = np.array([[1, r01], [r01, 1]], dtype=complex) / t01
    for j in range(1, N - 1):
        Mt = Mt @ M_list[j]
    r = Mt[1, 0] / Mt[0, 0]
    t = 1 / Mt[0, 0]
    vw = [None] * N
    vw[N - 1] = np.array([t, 0j])
    for j in range(N - 2, 0, -1):
        vw[j] = M_list[j] @ vw[j + 1]

    def flux(j, v, w):
        if pol == "s":
            return ((n[j] * cos_t[j] * np.conj(v + w) * (v - w)).real
                    / (n[0] * cos_t[0]).real)
        return ((n[j] * np.conj(cos_t[j]) * (v + w) * np.conj(v - w)).real
                / (n[0] * np.conj(cos_t[0])).real)

    S = [flux(j, vw[j][0], vw[j][1]) for j in range(1, N - 1)] + [flux(N - 1, t, 0j)]
    A = [S[i] - S[i + 1] for i in range(N - 2)]
    R = abs(r) ** 2
    return R, S[-1], A


# ------------------------------------------------------------------ dynamics
def gaussian_pulse(peak_W_m2, fwhm, t_center):
    def intensity(t):
        return peak_W_m2 * np.exp(-4 * np.log(2) * ((np.asarray(t) - t_center) / fwhm) ** 2)
    return intensity


def run(gas, absorb, intensity, d, cos_theta, t, A_ep=A_EP, G_const=None, C_l=C_L):
    """Integrate the two-temperature model on the time grid t (s).

    absorb(Te)   fraction of the incident pump power absorbed in the ITO film
    intensity(t) beam intensity in the incidence medium (W m^-2)
    d            film thickness (m); cos_theta: cosine of the pump angle in the
                 incidence medium (the power per unit film area is I cos theta)
    G_const      if given, a constant electron-phonon coupling (W m^-3 K^-1)
                 in place of g_ep(Te)
    Fourth-order Runge-Kutta in the electron energy density and the lattice
    temperature."""
    def coupling(Te):
        return G_const if G_const is not None else gas.g_ep(Te, A_ep)

    def rhs(tt, y):
        U, Tl = y
        Te = gas.temperature(U)
        P = absorb(Te) * intensity(tt) * cos_theta / d
        q = coupling(Te) * (Te - Tl)
        return np.array([P - q, q / C_l])

    y = np.array([gas.energy(T_ROOM), T_ROOM])
    out = np.empty((len(t), 2))
    out[0] = y
    for i in range(len(t) - 1):
        h = t[i + 1] - t[i]
        k1 = rhs(t[i], y)
        k2 = rhs(t[i] + h / 2, y + h / 2 * k1)
        k3 = rhs(t[i] + h / 2, y + h / 2 * k2)
        k4 = rhs(t[i] + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[i + 1] = y
    Te = gas.temperature(out[:, 0])
    absorbed = np.trapezoid(np.array([absorb(T) for T in Te]) * intensity(t) * cos_theta, t)
    return dict(t=t, Te=Te, Tl=out[:, 1], drop=gas.drop(Te), U=out[:, 0], absorbed_J_m2=float(absorbed))


def decay_time(t, x, start_frac=0.9, end_frac=1 / np.e):
    """Time for x(t) - x(end) to fall from its peak to end_frac of the peak,
    measured from the peak (x relative to its initial value)."""
    x = np.asarray(x) - x[0]
    i = int(np.argmax(x))
    after = np.where(x[i:] <= end_frac * x[i])[0]
    if not len(after):
        return np.nan
    return float(t[i + after[0]] - t[i])


def rise_time(t, x, lo=0.1, hi=0.9):
    x = np.asarray(x) - x[0]
    i = int(np.argmax(x))
    a = np.where(x[:i + 1] >= lo * x[i])[0][0]
    b = np.where(x[:i + 1] >= hi * x[i])[0][0]
    return float(t[b] - t[a])
