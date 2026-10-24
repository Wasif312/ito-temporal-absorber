"""Derived numbers quoted in the text: zero crossings of the margin against the
drop and against the damping, read from the result files by linear
interpolation.

Usage: python paper_numbers.py [--results DIR]   writes data/numbers.json
"""
import os

from paper_paths import arguments, read_json, write_json

RES = arguments(__doc__.splitlines()[0]).results
out = {}


def load(name):
    return read_json(os.path.join(RES, name))


def zero(points):
    """First zero crossing of y(x) by linear interpolation; points sorted by x."""
    pts = sorted(points)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:], strict=False):
        if y0 == 0:
            return x0
        if (y0 < 0) != (y1 < 0):
            return x0 + (0 - y0) * (x1 - x0) / (y1 - y0)
    return None


def short(v):
    """A result with its points rounded, for the console."""
    if not isinstance(v, dict):
        return v
    return {k: [(round(a, 3), round(b, 4)) for a, b in x] if k in ("points", "fw_points") else x
            for k, x in v.items()}


# threshold drop, kernel model, full wave and adiabatic
fw = load("route1_threshold_fdtd.json")["summary"] + load("route1_fdtd.json")["summary"]
ad = load("r1_threshold.json") + load("r1_design.json")
for d, deg in ((7.5e-9, 75.0), (10e-9, 75.0), (10e-9, 70.0)):
    pf = [(s["depth"], s["margin"]) for s in fw if abs(s["d"] - d) < 1e-12 and s["deg"] == deg]
    pa = [(s["depth"], s["margin"]) for s in ad if abs(s["d"] - d) < 1e-12 and s["deg"] == deg
          and abs(s.get("tau_dec", 210e-15) - 210e-15) < 1e-18]
    pa = list(dict(pa).items())
    out[f"threshold_kernel_{d*1e9:g}nm_{deg:g}"] = dict(full_wave=zero(pf), adiabatic=zero(pa), fw_points=sorted(pf))
pa = [(s["depth"], s["margin"]) for s in load("r1_threshold.json") if abs(s["tau_dec"] - 360e-15) < 1e-18
      and abs(s["d"] - 7.5e-9) < 1e-12 and s["deg"] == 75.0]
out["threshold_kernel_decay360_7.5nm_75"] = zero(pa)
# threshold with the two-temperature model, fixed damping (adiabatic)
for d, deg in ((5e-9, 80.0), (7.5e-9, 75.0), (10e-9, 70.0), (10e-9, 75.0)):
    for var in ("pec_fixed", "pec_hot"):
        p = [(r["depth"], r["margin"]) for r in load("r1_ttm_baxter.json")["designs"]
             if r["var"] == var and abs(r["d"] - d) < 1e-12 and r["deg"] == deg]
        out[f"ttm_{var}_{d*1e9:g}nm_{deg:g}"] = dict(points=sorted(p), zero=zero(p))
# damping tolerance at a 30 % and 40 % drop, two-temperature model
for d, deg in ((7.5e-9, 75.0), (5e-9, 80.0)):
    for depth in (0.30, 0.40):
        p = [(r["gamma_scale_max"], r["margin"]) for r in load("r1_ttm_damping_baxter.json")["designs"]
             if abs(r["d"] - d) < 1e-12 and r["deg"] == deg and abs(r["depth"] - depth) < 1e-9]
        out[f"damping_zero_ttm_{d*1e9:g}nm_{deg:g}_{depth:.2f}"] = dict(points=sorted(p), zero=zero(p))
# damping tolerance, kernel model: fixed damping (r1_design or r1_threshold), r1_damping, r1_real pec_hot
for d, deg in ((7.5e-9, 75.0), (5e-9, 80.0), (10e-9, 70.0)):
    for depth in (0.30, 0.40):
        p = [(r["gamma_scale_max"], r["margin"]) for r in load("r1_damping.json")
             if abs(r["d"] - d) < 1e-12 and r["deg"] == deg and abs(r["depth"] - depth) < 1e-9]
        p += [(r["gamma_scale_max"], r["margin"]) for r in load("r1_real.json") + load("r1_real_wide.json")
              if r["var"] == "pec_hot" and abs(r["d"] - d) < 1e-12 and r["deg"] == deg
              and abs(r["depth"] - depth) < 1e-9]
        p += [(1.0, r["margin"]) for r in load("r1_design.json")
              if abs(r["d"] - d) < 1e-12 and r["deg"] == deg and abs(r["depth"] - depth) < 1e-9]
        p = sorted(dict(p).items())
        out[f"damping_zero_kernel_{d*1e9:g}nm_{deg:g}_{depth:.2f}"] = dict(points=p, zero=zero(p))

path = write_json("numbers.json", out)
for k, v in out.items():
    print(k, short(v))
print("wrote", path)
