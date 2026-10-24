"""Material models and the transfer matrix against analytic results and the
numbers the source papers state."""
import numpy as np
import pytest

from ito_model import ITO_SETS, C0, enz_wavelength, eps_gold, eps_ito, tmm
from bound_scope import gamma_film_on_pec
from gold_fit import eps_gold_fit, gamma_film_on_metal, params
from ttm import layer_absorption

TH_GLASS = np.arcsin(np.sin(np.deg2rad(65.0)) / 1.5)      # 65 deg in air = 37.17 deg in the glass


@pytest.mark.parametrize("name, ours_nm, stated_nm", [
    ("Alam2016", 1241.6, 1240.0),
    ("Bohn2021", 1232.8, C0 / 243e12 * 1e9),
    ("Tirole2022", 1328.0, C0 / 227e12 * 1e9),
    ("Galiffi2026", 1210.1, 1196.0),
])
def test_enz_wavelengths(name, ours_nm, stated_nm):
    p = ITO_SETS[name]
    lam = enz_wavelength(p["eps_inf"], p["wp"], p["gamma"]) * 1e9
    assert lam == pytest.approx(ours_nm, abs=0.05)
    assert abs(lam - stated_nm) / stated_nm < 0.012          # within 1.2 % of the paper


def test_enz_is_zero_of_real_permittivity():
    p = ITO_SETS["Tirole2022"]
    lam = enz_wavelength(p["eps_inf"], p["wp"], p["gamma"])
    assert abs(eps_ito(lam, "Tirole2022").real) < 1e-12


def test_alam_index_at_1240nm():
    n = np.sqrt(eps_ito(1240e-9, "Alam2016"))
    assert n.real == pytest.approx(0.423, abs=1e-3)
    assert n.imag == pytest.approx(0.411, abs=1e-3)


def test_time_convention_absorbing():
    lam = np.linspace(800e-9, 2500e-9, 50)
    assert np.all(eps_ito(lam).imag > 0)
    assert np.all(eps_gold(lam).imag > 0)


@pytest.mark.parametrize("pol", ["s", "p"])
def test_tmm_single_interface_is_fresnel(pol):
    n1, n2, th = 1.0, 1.5, np.deg2rad(40.0)
    ct = np.sqrt(1 - (n1 / n2 * np.sin(th)) ** 2)
    if pol == "s":
        r = (n1 * np.cos(th) - n2 * ct) / (n1 * np.cos(th) + n2 * ct)
    else:
        r = (n2 * np.cos(th) - n1 * ct) / (n2 * np.cos(th) + n1 * ct)
    R, T, A = tmm([n1 ** 2, n2 ** 2], [], 1300e-9, th, pol)
    assert R == pytest.approx(abs(r) ** 2, abs=1e-14)
    assert R + T == pytest.approx(1.0, abs=1e-14)


@pytest.mark.parametrize("pol", ["s", "p"])
def test_tmm_lossless_slab_conserves_energy(pol):
    R, T, A = tmm([1.0, 2.25, 4.0, 1.0], [400e-9, 170e-9], 1300e-9, np.deg2rad(30.0), pol)
    assert R + T == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("pol", ["s", "p"])
def test_layer_absorption_sums_to_one(pol):
    eps = [2.25, eps_ito(1300e-9), eps_gold(1300e-9), 1.0]
    R, T, A = layer_absorption(eps, [40e-9, 100e-9], 1300e-9, TH_GLASS, pol)
    assert R + T + sum(A) == pytest.approx(1.0, abs=1e-12)
    R2 = tmm(eps, [40e-9, 100e-9], 1300e-9, TH_GLASS, pol)[0]
    assert R == pytest.approx(R2, abs=1e-12)
    assert all(a > 0 for a in A)


def test_tirole_mirror_calibration():
    """Glass / 40 nm ITO / 100 nm Au / air at 65 deg in air: R = 0.074 unpumped
    (0.07 measured) and R = 0.66 at a 9.98 % drop of wp (README, reference numbers)."""
    def R(scale):
        return tmm([2.25, eps_ito(1300e-9, "Tirole2022", wp_scale=scale), eps_gold(1300e-9), 1.0],
                   [40e-9, 100e-9], 1300e-9, TH_GLASS, "p")[0]
    assert R(1.0) == pytest.approx(0.0743, abs=5e-4)
    assert R(1 - 0.0998) == pytest.approx(0.66, abs=2e-3)


def test_lossless_film_on_pec_reflects_all():
    g = gamma_film_on_pec(2.25, 2.0, 100e-9, 1300e-9, np.deg2rad(30.0), "p")
    assert abs(g) == pytest.approx(1.0, abs=1e-12)


def test_film_on_pec_is_limit_of_good_conductor():
    """A film on a metal half-space of very large |eps| approaches the film on a
    perfect conductor."""
    lam = 1500e-9
    e_film = eps_ito(lam)
    g = gamma_film_on_pec(2.25, e_film, 10e-9, lam, np.deg2rad(75.0), "p")
    R_tmm = tmm([2.25, e_film, -1e8 + 1e6j], [10e-9], lam, np.deg2rad(75.0), "p")[0]
    assert abs(g) ** 2 == pytest.approx(R_tmm, abs=1e-3)


def test_gold_refit():
    """Drude refit of gold with eps_inf = 5 (README, reference numbers): hbar wp = 8.595 eV,
    1/gamma = 14.26 fs, and the reflectance of ITO on gold within 2.1e-3 of
    the Olmon model at 70 and 75 deg."""
    p = params()
    assert p["eps_inf"] == 5.0
    assert p["wp"] * 6.582119569e-16 == pytest.approx(8.595, abs=5e-3)
    assert 1 / p["gamma"] * 1e15 == pytest.approx(14.26, abs=0.02)
    lam = np.linspace(1000e-9, 2400e-9, 281)
    for d in (7.5e-9, 10e-9):
        for deg in (70.0, 75.0):
            th = np.deg2rad(deg)
            R_ol = np.abs(gamma_film_on_metal(2.25, eps_ito(lam), eps_gold(lam), d, lam, th)) ** 2
            R_fit = np.abs(gamma_film_on_metal(2.25, eps_ito(lam), eps_gold_fit(lam), d, lam, th)) ** 2
            assert np.max(np.abs(R_fit - R_ol)) < 2.1e-3


def test_film_on_metal_against_tmm():
    lam, d, th = 1500e-9, 10e-9, np.deg2rad(60.0)
    g = gamma_film_on_metal(2.25, eps_ito(lam), eps_gold_fit(lam), d, lam, th)
    R_tmm = tmm([2.25, eps_ito(lam), eps_gold_fit(lam)], [d], lam, th, "p")[0]
    assert abs(g) ** 2 == pytest.approx(R_tmm, abs=1e-12)
