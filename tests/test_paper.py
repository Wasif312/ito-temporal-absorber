"""The paper folder: the data, tables and figures behind the manuscript are
complete, well formed and consistent with the result files and the code."""
import json
import os

import numpy as np
import pytest
from scipy.optimize import brentq

from bound_scope import rozanov_max_absorption

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")
RESULTS = os.path.join(ROOT, "src", "results")
FIGURES = ("fig_concept", "fig_gas", "fig_validation", "fig_benchmarks", "fig_mirror", "fig_route1", "fig_design",
           "fig_damping", "fig_experiments", "fig_ttm", "fig_requirements")
TABLES = sorted(n[:-3] for n in os.listdir(os.path.join(PAPER, "tables")) if n.endswith(".md"))
N0 = 1.5


def read(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as fh:
        return json.load(fh)


def budget(d, deg):
    return 2 * np.pi ** 2 * N0 * d / np.cos(np.deg2rad(deg))


@pytest.fixture(scope="module")
def flagship():
    """Spectra of the 7.5 nm film at 75 deg with a 30 % drop, and the data derived from them."""
    sp = read(RESULTS, "route1_spectra.json")
    lam = np.array(sp["lam"])
    S_inc, S_sw = np.array(sp["S_inc"]), np.array(sp["S_switched"])
    return dict(lam=lam, S=S_inc, S_sw=S_sw, eta=1 - np.trapezoid(S_sw, lam) / np.trapezoid(S_inc, lam),
                d=sp["design"]["d"], deg=sp["design"]["deg"], extra=read(PAPER, "data", "extra.json"))


def test_limit_and_absorption_of_the_flagship_design(flagship):
    f = flagship
    eta_R = rozanov_max_absorption(f["S"], f["lam"], budget(f["d"], f["deg"]))[0]
    assert eta_R == pytest.approx(0.8478, abs=1e-4)
    assert f["eta"] == pytest.approx(0.8784, abs=1e-4)
    assert f["extra"]["passband"]["full"]["eta_R"] == pytest.approx(eta_R, rel=1e-9)
    # the energy spectra and the time-domain energies of the full-wave run agree
    run = [s for s in read(RESULTS, "route1_fdtd.json")["summary"]
           if abs(s["d"] - f["d"]) < 1e-12 and s["deg"] == f["deg"] and abs(s["depth"] - 0.3) < 1e-9]
    assert len(run) == 1
    assert run[0]["switched"] == pytest.approx(f["eta"], abs=1e-4)
    assert run[0]["eta_R"] == pytest.approx(eta_R, abs=1e-6)


def test_equivalent_static_thickness(flagship):
    f = flagship
    d_eq = brentq(lambda x: rozanov_max_absorption(f["S"], f["lam"], budget(x, f["deg"]))[0] - f["eta"],
                  f["d"], 3 * f["d"])
    assert d_eq * 1e9 == pytest.approx(8.68, abs=0.01)
    assert f["extra"]["equivalent"]["d_eq_nm"] == pytest.approx(d_eq * 1e9, rel=1e-6)


def test_margin_inside_the_passband_of_a_pump_filter(flagship):
    f = flagship
    cut = [c for c in f["extra"]["passband"]["cuts"] if c["cut_nm"] == 1350.0][0]
    m = f["lam"] >= 1350e-9
    eta_R = rozanov_max_absorption(np.where(m, f["S"], 0.0), f["lam"], budget(f["d"], f["deg"]))[0]
    switched = 1 - np.trapezoid(np.where(m, f["S_sw"], 0.0), f["lam"]) / np.trapezoid(np.where(m, f["S"], 0.0),
                                                                                    f["lam"])
    assert cut["eta_R"] == pytest.approx(eta_R, rel=1e-9)
    assert cut["switched"] == pytest.approx(switched, rel=1e-9)
    assert cut["margin"] == pytest.approx(0.0105, abs=1e-4)
    assert cut["pump_leak"] < 1e-11


def test_threshold_and_damping_tolerance():
    n = read(PAPER, "data", "numbers.json")
    assert n["threshold_kernel_7.5nm_75"]["full_wave"] == pytest.approx(0.229, abs=0.002)
    assert n["damping_zero_ttm_7.5nm_75_0.30"]["zero"] == pytest.approx(1.66, abs=0.01)
    assert n["damping_zero_kernel_7.5nm_75_0.30"]["zero"] == pytest.approx(1.46, abs=0.01)


def test_outputs_are_complete():
    for name in FIGURES:
        assert os.path.getsize(os.path.join(PAPER, "figures", name + ".png")) > 20_000
    for name in ("paper_data.json", "numbers.json", "extra.json"):
        assert os.path.isfile(os.path.join(PAPER, "data", name))
    assert len(TABLES) == 47


@pytest.mark.parametrize("name", TABLES)
def test_tables_are_well_formed(name):
    with open(os.path.join(PAPER, "tables", name + ".md"), encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    assert lines[0].startswith(f"Table: TAB:{name} ")
    rows = [ln for ln in lines[1:] if ln.strip()]
    assert len(rows) >= 3                       # header, alignment and at least one row
    assert all(r.startswith("|") and r.endswith("|") for r in rows)
    assert len({r.count("|") for r in rows}) == 1
