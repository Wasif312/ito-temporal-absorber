"""Time-domain solvers against exact solutions, closed-form reflection
coefficients and the energy balance. Each test runs in seconds; the full
validation suites are validate_fdtd.py, validate_angle1d.py and
validate_angle_gold.py."""
import numpy as np
import pytest

from angle1d import AngleStack, CHUNK, energies, simulate
from bound_scope import gamma_film_on_pec
from bulk import run_bulk, step_scale
from fdtd1d import C0, ETA0, Medium
from ito_model import ITO_SETS, eps_ito
from mirror import delta_profile, gaussian_probe
from slab_drude import spectrum

ITO = ITO_SETS["Tirole2022"]


def film(d):
    return AngleStack(1.5, [(Medium(ITO["eps_inf"], 0), d)], [(ITO["wp"], ITO["gamma"])])


@pytest.mark.parametrize("momentum, gain", [(False, 0.099), (True, -0.090)])
@pytest.mark.parametrize("gamma", [0.0, 1.45e14])
def test_uniform_medium_switch_exact(momentum, gain, gamma):
    """A uniform Drude medium whose wp drops by 10 % at 20 fs: the field
    follows the exact single-mode solution, and the energy change equals the
    modulation work minus the dissipation. Without loss the switch adds 9.9 %
    of the energy in the current form and removes 9.0 % in the momentum form."""
    r = run_bulk(step_scale(20e-15, 0.10), momentum, gamma, courant=0.5)
    err = np.max(np.abs(r["E"] - r["E_exact"])) / np.max(np.abs(r["E_exact"]))
    balance = (r["W1"] - r["W0"] - r["wmod"] + r["diss"]) / r["W0"]
    assert err < 1e-3
    assert abs(balance) < 1e-3
    if gamma == 0.0:
        assert r["wmod"] / r["W0"] == pytest.approx(gain, abs=1.5e-3)


@pytest.fixture(scope="module")
def static_pair():
    th = np.deg2rad(70.0)
    probe = gaussian_probe(5e-15, 30e-15, w0=2 * np.pi * C0 / 1400e-9)
    ref = simulate(film(10e-9), th, probe, 300e-15, reference=True)
    run = simulate(film(10e-9), th, probe, 300e-15)
    return th, ref, run


def test_incident_wave_has_tm_impedance(static_pair):
    """At a fixed angle the incident wave in the glass has Ex / Hy = eta0 cos(theta) / n0."""
    th, ref, _ = static_pair
    E, H = ref["recE"][0], ref["recH"][0]
    Hm = np.empty_like(H)
    Hm[1:] = 0.5 * (H[1:] + H[:-1])
    Hm[0] = 0.5 * H[0]
    sel = np.abs(E) > 1e-3 * np.abs(E).max()
    err = np.max(np.abs(E[sel] - ETA0 * np.cos(th) / 1.5 * Hm[sel])) / np.abs(E).max()
    assert err < 1e-2


def test_static_film_on_pec_reflectance(static_pair):
    """10 nm ITO film on a perfect conductor at 70 deg in the glass: FDTD
    reflectance against the exact fixed-angle value over 1000-2000 nm."""
    th, ref, run = static_pair
    lam = np.linspace(1000e-9, 2000e-9, 201)
    dec = max(1, int(0.2e-15 / run["dt"]))
    Ei = spectrum(ref["recE"][0], ref["dt"], C0 / lam, decimate=dec)
    Er = spectrum(run["recE"][0] - ref["recE"][0], run["dt"], C0 / lam, decimate=dec)
    R = np.abs(Er / Ei) ** 2
    Rx = np.abs(gamma_film_on_pec(2.25, eps_ito(lam, "Tirole2022"), 10e-9, lam, th, "p")) ** 2
    assert np.max(np.abs(R - Rx)) < 3e-3
    ui, ur = energies(run, ref)
    assert abs((ui - ur - run["diss"].sum() - run["u_end"]) / ui) < 1e-4


@pytest.mark.parametrize("momentum", [False, True])
def test_switched_film_energy_balance(momentum):
    """Film switched by a 20 % drop of wp under a chirped probe:
    U_inc + W_mod = U_r + U_diss + U_end."""
    th = np.deg2rad(70.0)
    T, span, lc = 150e-15, 300e-9, 1550e-9
    wc = 2 * np.pi * C0 / lc
    dw = 2 * np.pi * C0 * (1 / (lc - span / 2) - 1 / (lc + span / 2))
    probe = gaussian_probe(T, 3 * T, w0=wc, chirp=dw / T)
    t_end = 6 * T + 200e-15
    dfun = delta_profile(3 * T - 150e-15, 0.20)
    ref = simulate(film(10e-9), th, probe, t_end, reference=True)
    run = simulate(film(10e-9), th, probe, t_end, wp_scale=[lambda t: 1 + dfun(t)], momentum=momentum)
    ui, ur = energies(run, ref)
    W, D = run["wmod"].sum(), run["diss"].sum()
    assert abs((ui + W - ur - D - run["u_end"]) / ui) < 1e-3
    assert 0 < 1 - ur / ui < 1


def test_chunked_time_stepping_is_exact():
    """The modulation is evaluated in chunks of time steps; the fields do not
    depend on the chunk length."""
    th = np.deg2rad(75.0)
    probe = gaussian_probe(60e-15, 180e-15, w0=2 * np.pi * C0 / 1500e-9)
    dfun = delta_profile(150e-15, 0.30)
    kw = dict(wp_scale=[lambda t: 1 + dfun(t)], gamma_scale=[lambda t: 1 - 0.5 * dfun(t)], momentum=True)
    a = simulate(film(7.5e-9), th, probe, 400e-15, chunk=CHUNK, **kw)
    b = simulate(film(7.5e-9), th, probe, 400e-15, chunk=7919, **kw)
    assert a["nsteps"] > 7919
    assert np.array_equal(a["recE"], b["recE"]) and np.array_equal(a["recH"], b["recH"])
    assert np.array_equal(a["diss"], b["diss"]) and np.array_equal(a["wmod"], b["wmod"])
