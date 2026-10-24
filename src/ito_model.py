"""ito_model.py -- material models for the temporal-absorber study.

Time convention exp(-i*omega*t), so absorbing media have Im(eps) > 0.

Static Drude model of a transparent conducting oxide
    eps(omega) = eps_inf - omega_p**2 / (omega**2 + 1j*gamma*omega)
with the parameter sets published for indium tin oxide (ITO). Each set
records the sample it was fitted to and its source, so that every number in
the study can be traced back to a paper.

Time-varying models follow the phenomenological forms used in the
time-varying ENZ literature: the pump changes the plasma frequency,
    omega_p(t) = omega_p0 * (1 + delta(t)),
with delta(t) the pump intensity convolved with a response kernel
(Tirole et al., Phys. Rev. Applied 18, 054067 (2022)), or a logistic
rise-and-decay envelope (Tirole et al., Nat. Phys. 19, 999 (2023)).
"""
import numpy as np

C0 = 299_792_458.0          # m/s
HBAR_EV = 6.582_119_569e-16 # eV s

# ----------------------------------------------------------------- ITO sets
# eps_inf, omega_p (rad/s), gamma (rad/s)
ITO_SETS = {
    "Alam2016": dict(
        eps_inf=3.8055, wp=2.9719e15, gamma=0.0468 * 2.9719e15,
        sample="310 nm ITO on glass, ENZ at 1240 nm",
        source="Alam, De Leon, Boyd, Science 352, 795 (2016), supplement; "
               "same set in Zhou et al., Nat. Commun. 11, 2180 (2020), SI Note 2"),
    "Bohn2021": dict(
        eps_inf=3.43, wp=2.86e15, gamma=2.24e14,
        sample="60 nm ITO on a coverslip, ENZ at 243 THz",
        source="Bohn et al., Nat. Commun. 12, 1017 (2021), SI S1"),
    "Tirole2022": dict(
        eps_inf=4.08, wp=2.88e15, gamma=0.145e15,
        sample="40 nm ITO under 100 nm Au on glass, ENZ near 227 THz",
        source="Tirole et al., Phys. Rev. Applied 18, 054067 (2022)"),
    # plasma wavelength 597 nm and gamma = 0.13 fs^-1 as given in the paper
    "Galiffi2026": dict(
        eps_inf=4.08, wp=2 * np.pi * 299_792_458.0 / 597e-9, gamma=0.13e15,
        sample="310 nm ITO on 1.1 mm glass, ENZ at 1196 nm",
        source="Galiffi et al., Nat. Photon. 20, 163 (2026), arXiv:2410.16426"),
}


def eps_drude(omega, eps_inf, wp, gamma):
    """Drude permittivity at angular frequency omega (rad/s)."""
    omega = np.asarray(omega, dtype=float)
    return eps_inf - wp**2 / (omega**2 + 1j * gamma * omega)


def eps_ito(lam_m, name="Tirole2022", wp_scale=1.0, gamma_scale=1.0):
    """ITO permittivity at vacuum wavelength lam_m (m) for a published set.

    wp_scale and gamma_scale multiply the plasma frequency and the damping,
    which is how the pumped state is described: omega_p -> (1 + delta) omega_p.
    """
    p = ITO_SETS[name]
    omega = 2 * np.pi * C0 / np.asarray(lam_m, dtype=float)
    return eps_drude(omega, p["eps_inf"], p["wp"] * wp_scale, p["gamma"] * gamma_scale)


def enz_wavelength(eps_inf, wp, gamma):
    """Wavelength (m) where Re(eps) = 0 for the Drude model."""
    w0 = np.sqrt(wp**2 / eps_inf - gamma**2)
    return 2 * np.pi * C0 / w0


# ----------------------------------------------------------------- gold
# Olmon et al., Phys. Rev. B 86, 235147 (2012): Drude fit over 0.5-1.0 eV,
# mean of the three samples, hbar*omega_p = 8.45 eV and tau_D = 14 fs,
# with eps = 1 - omega_p^2 / (omega^2 + i omega Gamma), Gamma = 1/tau_D.
AU_OLMON = dict(eps_inf=1.0, wp=8.45 / HBAR_EV, gamma=1.0 / 14e-15,
                source="Olmon et al., Phys. Rev. B 86, 235147 (2012), Drude fit")


def eps_gold(lam_m):
    omega = 2 * np.pi * C0 / np.asarray(lam_m, dtype=float)
    return eps_drude(omega, AU_OLMON["eps_inf"], AU_OLMON["wp"], AU_OLMON["gamma"])


# ----------------------------------------------------------------- silica
def eps_silica(lam_m):
    """Fused silica, Sellmeier fit of Malitson, JOSA 55, 1205 (1965)."""
    x = (np.asarray(lam_m, dtype=float) * 1e6) ** 2
    n2 = (1 + 0.6961663 * x / (x - 0.0684043**2)
            + 0.4079426 * x / (x - 0.1162414**2)
            + 0.8974794 * x / (x - 9.896161**2))
    return n2 + 0j


# ----------------------------------------------------------------- multilayer
def tmm(eps_list, d_list, lam_m, theta0_rad, pol, return_r=False):
    """Reflectance, transmittance and absorptance of a planar stack.

    eps_list: permittivities [incidence medium, layer 1, ..., exit medium]
    d_list:   thicknesses (m) of the inner layers
    theta0:   angle of incidence in the incidence medium
    pol:      's' (TE) or 'p' (TM)
    return_r: if True, return the complex amplitude reflection coefficient
    Characteristic-matrix formulation with complex cosines, as in
    S. J. Byrnes, arXiv:1603.02720 (tmm package).
    """
    n = [np.sqrt(complex(e)) for e in eps_list]
    n = [ni if ni.imag >= 0 else -ni for ni in n]
    k0 = 2 * np.pi / lam_m
    kx = n[0] * np.sin(theta0_rad)
    cos_t = []
    for ni in n:
        c = np.sqrt(1 - (kx / ni) ** 2 + 0j)
        # forward-going wave: decaying or carrying power in +z
        kz = ni * c
        if kz.imag < 0 or (abs(kz.imag) < 1e-14 and kz.real < 0):
            c = -c
        cos_t.append(c)

    def fresnel(i, j):
        ni, nj, ci, cj = n[i], n[j], cos_t[i], cos_t[j]
        if pol == "s":
            r = (ni * ci - nj * cj) / (ni * ci + nj * cj)
            t = 2 * ni * ci / (ni * ci + nj * cj)
        else:
            r = (nj * ci - ni * cj) / (nj * ci + ni * cj)
            t = 2 * ni * ci / (nj * ci + ni * cj)
        return r, t

    r01, t01 = fresnel(0, 1)
    M = np.array([[1, r01], [r01, 1]], dtype=complex) / t01
    for j in range(1, len(n) - 1):
        delta = k0 * n[j] * cos_t[j] * d_list[j - 1]
        P = np.array([[np.exp(-1j * delta), 0], [0, np.exp(1j * delta)]])
        r, t = fresnel(j, j + 1)
        L = np.array([[1, r], [r, 1]], dtype=complex) / t
        M = M @ P @ L
    r = M[1, 0] / M[0, 0]
    if return_r:
        return r
    t = 1 / M[0, 0]
    R = abs(r) ** 2
    if pol == "s":
        T = abs(t) ** 2 * (n[-1] * cos_t[-1]).real / (n[0] * cos_t[0]).real
    else:
        T = abs(t) ** 2 * (n[-1] * np.conj(cos_t[-1])).real / (n[0] * np.conj(cos_t[0])).real
    return R, T, 1 - R - T


# ----------------------------------------------------------------- time dependence
def kernel_tirole2022(t_s, tau_rise=30e-15, tau_dec=210e-15):
    """Response kernel xi(t) of Tirole et al. (2022): a linear rise over
    tau_rise followed by an exponential decay with tau_dec."""
    t = np.asarray(t_s, dtype=float)
    out = np.zeros_like(t)
    rise = (t > 0) & (t < tau_rise)
    out[rise] = t[rise] / tau_rise
    after = t >= tau_rise
    out[after] = np.exp(-(t[after] - tau_rise) / tau_dec)
    return out


def envelope_tirole2023(t_s, alpha=0.5e15, beta=1 / 400e-15):
    """Logistic rise-and-decay envelope f(t) = 1/[(1+e^{-alpha t})(1+e^{beta t})]
    of Tirole et al. (2023); alpha = 1/2 fs^-1 and beta = 1/400 fs^-1."""
    t = np.asarray(t_s, dtype=float)
    return 1.0 / ((1 + np.exp(-alpha * t)) * (1 + np.exp(beta * t)))


def delta_wp_from_pump(t_s, intensity_GWcm2, wp2_per_GWcm2=-0.42e-2,
                       tau_rise=30e-15, tau_dec=210e-15):
    """delta(t) = (pump intensity convolved with the kernel), scaled so that
    max delta = wp2 * max(I), as in Tirole et al. (2022)."""
    t = np.asarray(t_s, dtype=float)
    dt = t[1] - t[0]
    ker = kernel_tirole2022(np.arange(0, 8 * tau_dec, dt), tau_rise, tau_dec)
    conv = np.convolve(intensity_GWcm2, ker)[: len(t)] * dt
    peak = np.max(np.abs(conv))
    if peak == 0:
        return np.zeros_like(t)
    return conv / peak * wp2_per_GWcm2 * np.max(intensity_GWcm2)
