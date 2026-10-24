"""bound_scope.py -- where does the Rozanov bound limit an optical absorber?

1. For a pulse with spectrum S(lambda), the largest fraction of its energy
   that ANY passive, linear, time-invariant absorber of total thickness d on a
   perfect conductor can absorb follows from the Rozanov bound
       integral_0^inf -ln|Gamma(lambda)| dlambda <= B,   B = 2 pi^2 mu d,
   with mu the static relative permeability (mu = 1 for the non-magnetic
   absorbers studied here; a magnetic absorber has a larger budget).
   Maximising  integral S (1 - g) dlambda  with g = |Gamma|^2 under that
   constraint gives  g = min(1, kappa / S)  (water filling), with kappa set
   so that the constraint is met with equality.

2. The static ENZ absorber of Tirole et al. (2022), with the gold replaced by
   a perfect conductor so that the integral converges: how much of its own
   (oblique, TM) Rozanov budget does it use, and where?
"""
import numpy as np
from ito_model import eps_ito, C0

LAM0 = 1300e-9


# ----------------------------------------------------------------- pulses
def gaussian_S(lam, lam0, T):
    """Energy per unit wavelength of a transform-limited Gaussian pulse of
    intensity FWHM T (s) at carrier wavelength lam0, |E(w)|^2 * dw/dlam."""
    w = 2 * np.pi * C0 / lam
    w0 = 2 * np.pi * C0 / lam0
    Sw = np.exp(-(w - w0) ** 2 * T ** 2 / (4 * np.log(2)))
    return Sw * 2 * np.pi * C0 / lam ** 2


def rozanov_max_absorption(S, lam, B):
    """Largest absorbed fraction of the energy spectrum S(lam) for a budget B
    (m) of the Rozanov integral. Returns (eta, kappa)."""
    tot = np.trapezoid(S, lam)

    def budget(k):
        r = np.where(S > k, S / k, 1.0)
        return np.trapezoid(0.5 * np.log(r), lam)

    lo, hi = S.max() * 1e-300, S.max()
    for _ in range(400):
        mid = np.sqrt(lo * hi)
        if budget(mid) > B:
            lo = mid
        else:
            hi = mid
        if hi / lo < 1 + 1e-12:
            break
    k = np.sqrt(lo * hi)
    eta = np.trapezoid(np.clip(S - k, 0, None), lam) / tot
    return eta, k


def lam_grid(lam0, T):
    """Wavelength grid wide enough for the pulse spectrum (12 spectral FWHM)."""
    w0 = 2 * np.pi * C0 / lam0
    dw = 4 * np.log(2) / T
    wmin = max(w0 - 6 * dw, 0.02 * w0)
    wmax = w0 + 6 * dw
    w = np.linspace(wmin, wmax, 40001)
    return np.sort(2 * np.pi * C0 / w)


def eta_R(T, d, lam0=LAM0, mu=1.0, factor=1.0):
    lam = lam_grid(lam0, T)
    S = gaussian_S(lam, lam0, T)
    return rozanov_max_absorption(S, lam, factor * 2 * np.pi ** 2 * mu * d)[0]


def threshold_T(d, target, lam0=LAM0, factor=1.0):
    """Pulse duration below which even the optimal static absorber of
    thickness d absorbs less than `target` of the pulse energy."""
    lo, hi = 0.5e-15, 2000e-15
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        if eta_R(mid, d, lam0, factor=factor) < target:
            lo = mid
        else:
            hi = mid
    return np.sqrt(lo * hi)


# ----------------------------------------------------------------- PEC-backed film
def gamma_film_on_pec(eps0, eps1, d, lam, theta0, pol):
    """Reflection coefficient of one layer (eps1, thickness d) on a perfect
    conductor, light incident from a medium eps0 at angle theta0."""
    n0 = np.sqrt(eps0 + 0j)
    n1 = np.sqrt(eps1 + 0j)
    n1 = np.where(n1.imag < 0, -n1, n1)
    kx = n0 * np.sin(theta0)
    c0 = np.sqrt(1 - (kx / n0) ** 2 + 0j)
    c1 = np.sqrt(1 - (kx / n1) ** 2 + 0j)
    kz = n1 * c1
    c1 = np.where((kz.imag < 0) | ((np.abs(kz.imag) < 1e-14) & (kz.real < 0)), -c1, c1)
    if pol == "s":
        r01 = (n0 * c0 - n1 * c1) / (n0 * c0 + n1 * c1)
        r12 = -1.0
    else:
        r01 = (n1 * c0 - n0 * c1) / (n1 * c0 + n0 * c1)
        r12 = 1.0
    ph = np.exp(2j * 2 * np.pi / lam * n1 * c1 * d)
    return (r01 + r12 * ph) / (1 + r01 * r12 * ph)


# ----------------------------------------------------------------- static ENZ film under pulses
def static_pulse_absorption(T, lam0=LAM0, backing="pec", d=40e-9):
    """Absorbed fraction of a Gaussian pulse for the static glass / ITO / back
    structure of Tirole et al. (2022), TM, 65 deg in air (37.17 deg in glass)."""
    from ito_model import tmm, eps_gold
    th = np.arcsin(np.sin(np.deg2rad(65)) / 1.5)
    lam = lam_grid(lam0, T)
    S = gaussian_S(lam, lam0, T)
    if backing == "pec":
        R = np.abs(gamma_film_on_pec(2.25, eps_ito(lam, "Tirole2022"), d, lam, th, "p")) ** 2
    else:
        R = np.array([tmm([2.25, eps_ito(wl, "Tirole2022"), eps_gold(wl), 1.0],
                          [d, 100e-9], wl, th, "p")[0] for wl in lam[::20]])
        R = np.interp(lam, lam[::20], R)
    return np.trapezoid(S * (1 - R), lam) / np.trapezoid(S, lam)


def main():
    # check 1: flat-top spectrum, analytic eta = 1 - exp(-2B/W)
    lam = np.linspace(1000e-9, 1600e-9, 60001)
    S = np.where((lam > 1100e-9) & (lam < 1500e-9), 1.0, 1e-30)
    W = 400e-9
    for B in (50e-9, 200e-9, 800e-9):
        eta, _ = rozanov_max_absorption(S, lam, B)
        print(f"flat top W = 400 nm, B = {B*1e9:4.0f} nm: eta = {eta:.5f}, "
              f"analytic {1 - np.exp(-2 * B / W):.5f}")

    # check 2: a lossless film on a perfect conductor reflects everything
    g = gamma_film_on_pec(2.25, 2.0, 100e-9, 1300e-9, np.deg2rad(30), "p")
    print(f"lossless film on PEC: |Gamma| = {abs(g):.12f}")

    print("\nlargest absorbed fraction of a Gaussian pulse at 1300 nm, normal incidence")
    for d in (10e-9, 20e-9, 40e-9, 80e-9):
        row = [f"{eta_R(T, d):.4f}" for T in (5e-15, 10e-15, 20e-15, 50e-15, 220e-15)]
        print(f"   d = {d*1e9:3.0f} nm (B = {2*np.pi**2*d*1e9:6.1f} nm): "
              f"T = 5, 10, 20, 50, 220 fs -> {', '.join(row)}")
    for d in (10e-9, 20e-9, 40e-9, 80e-9):
        print(f"   d = {d*1e9:3.0f} nm: eta_R < 0.99 below {threshold_T(d, 0.99)*1e15:5.1f} fs, "
              f"< 0.90 below {threshold_T(d, 0.90)*1e15:5.1f} fs")

    print("\nTirole structure with the gold replaced by a perfect conductor")
    th = np.arcsin(np.sin(np.deg2rad(65)) / 1.5)
    d = 40e-9
    lam = np.geomspace(150e-9, 2e-3, 400001)
    G = gamma_film_on_pec(2.25, eps_ito(lam, "Tirole2022"), d, lam, th, "p")
    f = -np.log(np.abs(G))
    I = np.trapezoid(f, lam)
    B_tm = 2 * np.pi ** 2 * 1.5 * d / np.cos(th)
    print(f"   theta in glass {np.rad2deg(th):.2f} deg, TM bound B = {B_tm*1e9:.0f} nm")
    print(f"   Rozanov integral I_R = {I*1e9:.1f} nm  ({100*I/B_tm:.1f} % of the bound)")
    for a, b in ((150e-9, 1000e-9), (1000e-9, 1700e-9), (1700e-9, 5e-6), (5e-6, 2e-3)):
        m = (lam >= a) & (lam <= b)
        print(f"      share {a*1e9:7.0f}-{b*1e9:8.0f} nm: {np.trapezoid(f[m], lam[m])*1e9:7.1f} nm")
    R1300 = abs(gamma_film_on_pec(2.25, eps_ito(1300e-9, "Tirole2022"), d, 1300e-9, th, "p"))**2
    print(f"   R(1300 nm) with the perfect conductor = {R1300:.3f}")

    th = np.arcsin(np.sin(np.deg2rad(65)) / 1.5)
    B_tm = 2 * np.pi ** 2 * 1.5 * 40e-9 / np.cos(th)
    print("\nstatic ENZ film (40 nm ITO, TM, 37.17 deg in glass) against its bound")
    for T in (3e-15, 5e-15, 10e-15, 20e-15, 50e-15, 100e-15, 220e-15):
        a_pec = static_pulse_absorption(T, backing="pec")
        a_au = static_pulse_absorption(T, backing="au")
        e = eta_R(T, 40e-9, factor=1.5 / np.cos(th))
        print(f"   T = {T*1e15:5.0f} fs: absorbed {a_pec:.3f} (PEC), {a_au:.3f} (Au); "
              f"bound allows {e:.4f}")
    print(f"   bound-limited threshold for B = {B_tm*1e9:.0f} nm: eta_R < 0.99 below "
          f"{threshold_T(40e-9, 0.99, factor=1.5/np.cos(th))*1e15:.1f} fs, < 0.90 below "
          f"{threshold_T(40e-9, 0.90, factor=1.5/np.cos(th))*1e15:.1f} fs")


if __name__ == "__main__":
    main()
