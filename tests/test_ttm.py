"""Hot-electron model and two-temperature model against published numbers,
limits and conservation laws."""
import numpy as np
import pytest

import ttm
from hot_electron import load_map, drop_of_te


@pytest.fixture(scope="module")
def gas():
    return ttm.load_gas(0.9)


def test_sommerfeld_coefficient_matches_sarkar():
    """Sarkar, Un and Sivan: 12.7 J m^-3 K^-2 for n_e = 1.5e27 m^-3 and E_F = 0.88 eV."""
    g88 = ttm.load_gas(0.88)
    assert g88.N == pytest.approx(1.5e27, rel=0.01)
    assert g88.gamma_e() == pytest.approx(12.7, abs=0.1)


def test_electron_phonon_coupling_at_room_temperature(gas):
    assert gas.g_ep(300.0) == pytest.approx(1.15e17, rel=0.01)


def test_drop_rises_with_temperature(gas):
    T = np.geomspace(300.0, 50000.0, 200)
    d = gas.drop(T)
    assert d[0] == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.diff(d) > 0)


@pytest.mark.parametrize("drop, te_kK", [(0.10, 4.9), (0.20, 8.5), (0.30, 13.2), (0.40, 19.9)])
def test_temperature_for_each_drop(drop, te_kK, gas):
    """README, models and conventions: drops of 10-40 % need 4.9, 8.5, 13.2 and 19.9 kK; the
    tables of hot_electron.py and ttm.py agree."""
    hmap = load_map(0.9)
    te = float(hmap.te_of_drop(drop))
    assert te / 1e3 == pytest.approx(te_kK, abs=0.05)
    assert float(gas.drop(te)) == pytest.approx(drop, abs=2e-3)
    assert float(drop_of_te(te)) == pytest.approx(drop, abs=1e-3)


def test_heat_capacity_limits(gas):
    """C_e = gamma_e T well below the Fermi temperature (10 400 K). Far above
    it the Kane band becomes linear in k (above 1/(C kB) = 27 700 K) and C_e
    tends to 3 n kB, twice the value of a parabolic band."""
    assert gas.ce(600.0) / 600.0 == pytest.approx(gas.gamma_e(), rel=0.02)
    T = np.geomspace(300.0, 59000.0, 50)
    assert np.all(np.diff(gas.ce(T)) > 0)
    ratio = gas.ce(59000.0) / (1.5 * gas.N * ttm.KB)
    assert 1.9 < ratio < 2.0


def test_ttm_conserves_energy(gas):
    """Energy absorbed from the pump = rise of electron energy + lattice heat."""
    d, cos_t = 7.5e-9, np.cos(np.deg2rad(75.0))
    t = np.arange(-900e-15, 2500e-15, 0.5e-15)
    r = ttm.run(gas, lambda T: 0.5 + 0 * T, ttm.gaussian_pulse(1e15, 220e-15, 0.0), d, cos_t, t)
    stored = (r["U"][-1] - r["U"][0]) + ttm.C_L * (r["Tl"][-1] - r["Tl"][0])
    assert stored * d == pytest.approx(r["absorbed_J_m2"], rel=1e-5)


def test_ttm_relaxes_to_common_temperature(gas):
    t = np.arange(-900e-15, 4000e-15, 0.5e-15)
    r = ttm.run(gas, lambda T: 0.3 + 0 * T, ttm.gaussian_pulse(5e14, 220e-15, 0.0), 10e-9, 1.0, t)
    assert r["Te"].max() > 3000
    assert r["Te"][-1] == pytest.approx(r["Tl"][-1], abs=0.5)
    assert r["Tl"][-1] > 300


def test_constant_coupling_option(gas):
    """The constant coupling of Sarkar et al. cools the electrons more slowly."""
    t = np.arange(-900e-15, 6000e-15, 0.5e-15)
    pump = ttm.gaussian_pulse(5e14, 220e-15, 0.0)
    a = ttm.run(gas, lambda T: 0.3 + 0 * T, pump, 10e-9, 1.0, t)
    b = ttm.run(gas, lambda T: 0.3 + 0 * T, pump, 10e-9, 1.0, t, G_const=3e16)
    assert ttm.decay_time(t, b["drop"]) > 2 * ttm.decay_time(t, a["drop"])


def test_baxter_damping_law():
    assert float(ttm.gamma_baxter(300.0)) == pytest.approx(1.0)
    assert float(ttm.gamma_baxter(15600.0)) == pytest.approx(2.0, rel=1e-12)      # (T0 + Te)/(T0 + 300 K)
    assert float(ttm.gamma_baxter(15600.0, T0=-60000.0)) < 1.0                  # falling damping
