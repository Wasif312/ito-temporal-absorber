"""The stored results against the numbers quoted in the README and the paper,
and against the invariants every run must satisfy (energy balance, bounds,
monotonic trends). These tests read results/ only and run in a second."""
import json
import os

import numpy as np
import pytest

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "results")


def load(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} not computed yet")
    with open(path) as fh:
        return json.load(fh)


def summary(name, tag):
    return [s for s in load(name)["summary"] if s["tag"] == tag][0]


def test_validation_suites_meet_the_quoted_bounds():
    """The largest deviations quoted in the README for tests T1-T4 and V0-V4,
    each below the upper end of its rounding interval."""
    t = load("tests.json")
    assert t["T1"]["10"]["max_err"] < 6.15e-4 and t["T1"]["5"]["max_err"] < 4.45e-4
    t2 = [v for v in t["T2"].values() if "err" in v]
    assert max(v["err"] for v in t2) < 7.75e-4
    assert max(abs(v["balance"]) for v in t2) < 2.25e-4
    assert max(abs(v["balance"]) for v in t["T3"].values()) < 1.65e-4
    for v in t["T4"].values():
        assert abs(v["current"] - v["momentum"]) < 2.25e-4
        assert max(abs(v["current"] - v["adiabatic"]), abs(v["momentum"] - v["adiabatic"])) < 6.15e-4
    a = load("tests_angle.json")
    assert max(a["V0_37.17"], a["V0_70.0"]) < 2.15e-3
    assert max(v["max_err"] for v in a["V1"]) < 2.15e-3
    assert max(v["max_err"] for v in a["V1"] if abs(v["h"] - 0.5e-9) < 1e-15) < 1.95e-3
    assert max(abs(v["balance"]) for v in a["V2"]) < 7.45e-5
    g = load("tests_angle_gold.json")
    assert max(v["max_err"] for v in g["V3"]) < 4.35e-4
    assert max(abs(v["balance"]) for v in g["V4"]) < 7.45e-5


def test_route1_flagship_design():
    """7.5 nm film at 75 deg, 30 % drop (README, reference numbers)."""
    s = summary("route1_fdtd.json", "best_0.30")
    assert s["eta_R"] == pytest.approx(0.848, abs=5e-4)
    assert s["static"] == pytest.approx(0.572, abs=5e-4)
    assert s["switched"] == pytest.approx(0.878, abs=5e-4)
    assert s["momentum"] == pytest.approx(0.880, abs=5e-4)
    assert s["switched"] > s["eta_R"] > s["static"]
    assert s["Wmod"] < 0                       # the film returns energy to the pump: no hidden gain


@pytest.mark.parametrize("name, bound", [("route1_fdtd.json", 1e-4), ("route1_real_fdtd.json", 1.2e-4),
                                         ("route1_threshold_fdtd.json", 1e-4), ("route1_ttm_fdtd_baxter.json", 4.3e-5),
                                         ("route1_ttm_fdtd_baxter_damping.json", 4.3e-5)])
def test_energy_balance_of_every_fixed_angle_run(name, bound):
    d = load(name)
    worst = max(abs(r["balance"]) for r in d["results"] if "balance" in r)
    assert worst < bound


@pytest.mark.parametrize("name", ["route1_fdtd.json", "route1_real_fdtd.json", "route1_ttm_fdtd_baxter.json",
                                  "route1_ttm_fdtd_baxter_damping.json"])
def test_static_films_never_beat_the_bound(name):
    for s in load(name)["summary"]:
        assert s["static"] < s["eta_R"]


@pytest.mark.parametrize("tag, margin", [("pec_hot_5nm_80_40", -0.005), ("pec_hot_7.5nm_75_40", -0.016),
                                         ("au_fixed_10nm_70_40", 0.025), ("au_fixed_7.5nm_75_40", 0.014),
                                         ("pec_hot_T30k_7.5nm_75_40", 0.025), ("pec_hot_T30k_5nm_80_30", 0.006)])
def test_realistic_margins(tag, margin):
    """README, reference numbers: thin films with the kernel model, full wave."""
    assert summary("route1_real_fdtd.json", tag)["margin"] == pytest.approx(margin, abs=6e-4)


@pytest.mark.parametrize("tag, margin", [("pec_fixed_7.5nm_75_25", 0.0219), ("pec_fixed_7.5nm_75_30", 0.0472),
                                         ("pec_fixed_10nm_70_30", 0.0366), ("pec_hot_5nm_80_40", 0.0058),
                                         ("pec_hot_7.5nm_75_40", -0.0052)])
def test_ttm_margins(tag, margin):
    """Full wave with the drop profiles of the two-temperature model."""
    s = summary("route1_ttm_fdtd_baxter.json", tag)
    assert s["margin"] == pytest.approx(margin, abs=1e-4)
    assert abs(s["margin"] - s["margin_adiabatic"]) < 5e-3


def test_margin_falls_as_damping_rises():
    """For every film and drop, the TTM margin falls monotonically as the
    damping at the largest drop rises (falling, fixed and rising damping)."""
    rows = load("r1_ttm_damping_baxter.json")["designs"]
    keys = {(r["d"], r["deg"], r["depth"]) for r in rows}
    for key in keys:
        sel = sorted([r for r in rows if (r["d"], r["deg"], r["depth"]) == key], key=lambda r: r["gamma_scale_max"])
        m = [r["margin"] for r in sel]
        assert all(b < a for a, b in zip(m, m[1:], strict=False)), key


def zero_crossing(points):
    """Damping scale at which the margin crosses zero, by linear interpolation."""
    pts = sorted(points)
    for (g1, m1), (g2, m2) in zip(pts, pts[1:], strict=False):
        if m1 > 0 >= m2:
            return g1 + (g2 - g1) * m1 / (m1 - m2)
    return None


@pytest.mark.parametrize("d, deg, rise_ttm, rise_kernel", [(7.5e-9, 75.0, 0.66, 0.46), (5e-9, 80.0, 0.80, 0.59)])
def test_damping_rise_at_zero_margin(d, deg, rise_ttm, rise_kernel):
    """README: at a 30 % drop the margin reaches zero at a damping rise of 66 %
    (7.5 nm, 75 deg) and 80 % (5 nm, 80 deg) with the two-temperature model,
    against 46 % and 59 % with the kernel."""
    def sel(r):
        return abs(r["d"] - d) < 1e-12 and r["deg"] == deg and abs(r["depth"] - 0.30) < 1e-9
    ttm = [(r["gamma_scale_max"], r["margin"]) for r in load("r1_ttm_damping_baxter.json")["designs"] if sel(r)]
    kern = [(r.get("gamma_scale_max", 1.0), r["margin"]) for r in load("r1_damping.json") + load("r1_fixed_wide.json")
            if sel(r)]
    kern += [(1.84, r["margin"]) for r in load("r1_real_wide.json") if r["var"] == "pec_hot" and sel(r)]
    assert zero_crossing(ttm) - 1 == pytest.approx(rise_ttm, abs=0.006)
    assert zero_crossing(kern) - 1 == pytest.approx(rise_kernel, abs=0.006)


def test_pump_for_a_30_percent_drop():
    """README: 87-150 GW/cm^2, 20-35 mJ/cm^2, 1.3 GJ/m^3 and about 820 K, the
    same energy density as the film of Tirole et al. at 708 GW/cm^2."""
    flu = load("fluence_check_baxter.json")
    rows = [r for r in flu["route1"] if abs(r["depth"] - 0.30) < 1e-9]
    intensity = [r["I_GWcm2"] for r in rows]
    fluence = [r["fluence_mJcm2"] for r in rows]
    assert 86.5 < min(intensity) and max(intensity) < 150.5
    assert 20.0 < min(fluence) and max(fluence) < 35.5
    assert 0.09 < min(fluence) / 210 and max(fluence) / 210 < 0.17          # damage fluence 0.21 J/cm^2
    q = [r["energy_density_GJm3"] for r in rows]
    T = [r["T_eq"] for r in rows]
    assert 1.29 < min(q) and max(q) < 1.36 and 805 < min(T) and max(T) < 835
    big = [r for r in flu["tirole"] if r["label"] == "largest"][0]
    assert big["I_GWcm2"] == 708.0
    assert big["energy_density_GJm3"] == pytest.approx(1.32, abs=0.01)
    assert big["T_eq"] == pytest.approx(819, abs=2)


def test_pump_energy_depends_on_drop_only():
    """The energy that a drop needs per unit volume does not depend on the
    film, the angle or the damping law: it is set by the electron temperature."""
    rows = load("fluence_check_baxter.json")["route1"]
    for depth in {r["depth"] for r in rows}:
        q = [r["energy_density_GJm3"] for r in rows if r["depth"] == depth]
        assert max(q) / min(q) < 1.05
        T = [r["Te_max"] for r in rows if r["depth"] == depth]
        assert max(T) - min(T) < 5.0


def test_convergence():
    """Halving the film cells twice, the time step once and the cells outside
    the film once moves every absorbed fraction, limit and margin by less than
    2e-5."""
    rows = load("convergence.json")["summary"]
    for case in {r["case"] for r in rows}:
        sel = [r for r in rows if r["case"] == case]
        base = [r for r in sel if r["level"] == "base"][0]
        tol = 2e-5
        for r in sel:
            assert abs(r["switched"] - base["switched"]) < tol
            assert abs(r["static"] - base["static"]) < tol
            if "margin" in r:
                assert abs(r["margin"] - base["margin"]) < tol
            assert abs(r["balance"]) < 1e-4
    finest = [r for r in rows if r["refine"] == max(q["refine"] for q in rows) and "margin" in r]
    assert all(r["margin"] > 0 for r in finest if r["case"] == "route1")
    assert np.isfinite([r["switched"] for r in rows]).all()
