"""The Rozanov bound: the water-filling limit against its analytic form, and
static films that must respect the bound."""
import numpy as np
import pytest

from bound_scope import eta_R, gamma_film_on_pec, rozanov_max_absorption
from ito_model import eps_ito


@pytest.mark.parametrize("B", [50e-9, 200e-9, 800e-9])
def test_water_filling_flat_top(B):
    """For a flat spectrum of width W the optimum is |Gamma|^2 = const and the
    absorbed fraction is 1 - exp(-2B/W)."""
    lam = np.linspace(1000e-9, 1600e-9, 60001)
    S = np.where((lam > 1100e-9) & (lam < 1500e-9), 1.0, 1e-30)
    eta, _ = rozanov_max_absorption(S, lam, B)
    assert eta == pytest.approx(1 - np.exp(-2 * B / 400e-9), abs=2e-4)


def test_limit_rises_with_budget_and_thickness():
    etas = [eta_R(20e-15, d) for d in (5e-9, 10e-9, 20e-9, 40e-9)]
    assert all(b > a for a, b in zip(etas, etas[1:], strict=False))
    assert 0 < etas[0] < etas[-1] < 1


def test_limit_falls_for_shorter_pulses():
    etas = [eta_R(T, 20e-9) for T in (200e-15, 50e-15, 10e-15, 3e-15)]
    assert all(b < a for a, b in zip(etas, etas[1:], strict=False))


@pytest.mark.parametrize("d, deg", [(7.5e-9, 75.0), (10e-9, 70.0), (40e-9, 37.17)])
def test_static_film_on_pec_respects_bound(d, deg):
    """Rozanov integral of a static ITO film on a perfect conductor, TM at a
    fixed angle in the glass, against B = 2 pi^2 n0 d / cos(theta)."""
    th = np.deg2rad(deg)
    lam = np.geomspace(150e-9, 2e-3, 200001)
    G = gamma_film_on_pec(2.25, eps_ito(lam, "Tirole2022"), d, lam, th, "p")
    integral = np.trapezoid(-np.log(np.abs(G)), lam)
    B = 2 * np.pi ** 2 * 1.5 * d / np.cos(th)
    assert 0 < integral < B


def test_tirole_film_uses_quarter_of_its_bound():
    """The static ITO film of the mirror of Tirole et al., on a perfect conductor in place of
    the gold, spends 342 nm of its Rozanov budget of 1486 nm, 23 %."""
    th = np.arcsin(np.sin(np.deg2rad(65.0)) / 1.5)
    lam = np.geomspace(150e-9, 2e-3, 400001)
    G = gamma_film_on_pec(2.25, eps_ito(lam, "Tirole2022"), 40e-9, lam, th, "p")
    integral = np.trapezoid(-np.log(np.abs(G)), lam)
    B = 2 * np.pi ** 2 * 1.5 * 40e-9 / np.cos(th)
    assert B * 1e9 == pytest.approx(1486, abs=1)
    assert integral * 1e9 == pytest.approx(342, abs=1)


def test_static_absorption_never_beats_limit():
    """Any static film absorbs at most the water-filling limit of its own bound."""
    from r1_design import spectrum_lam, LAM
    d, th = 7.5e-9, np.deg2rad(75.0)
    B = 2 * np.pi ** 2 * 1.5 * d / np.cos(th)
    S = spectrum_lam(1600e-9, 500e-9, 500e-15)
    lim = rozanov_max_absorption(S / S.max(), LAM, B)[0]
    for scale in (1.0, 0.9, 0.8, 0.7):
        R = np.abs(gamma_film_on_pec(2.25, eps_ito(LAM, "Tirole2022", wp_scale=scale), d, LAM, th, "p")) ** 2
        eta = np.trapezoid(S * (1 - R), LAM) / np.trapezoid(S, LAM)
        assert eta < lim
