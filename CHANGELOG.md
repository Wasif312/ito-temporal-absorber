# Changelog

## 1.1.0 (2026-10-24)

No result file changed. Every number of 1.0.0 stands.

Added
- `paper/`, with the scripts that derive the eleven figures and 47 tables of
  the manuscript from the result files, their outputs (`data/`, `tables/`,
  `figures/`) and a README that maps each figure to its result files and to
  the tables that hold the same data.
- A table row for a pump filter with its edge at 1300 nm, which blocks 4 % of
  the probe and passes almost all of the pump.
- `tests/test_paper.py`, which checks that every paper output exists and that
  the Rozanov limit of the flagship design, the equivalent static thickness,
  the margin inside a pump filter and the zero crossings of the margin agree
  with a fresh computation.
- A `paper` stage in `src/reproduce.py`; `--figures` now also redraws the
  figures and tables of the paper.
- A CI job that regenerates the paper outputs from the stored results and
  fails if a table changes.
- `CHANGELOG.md`, and the repository links in `pyproject.toml` and
  `CITATION.cff`.

Changed
- README rewritten around the result, the numerical methods, a validation
  table and the names that the code and the paper use for the same things.
- The Rozanov bound is stated for non-magnetic absorbers in the README and the
  docstrings, since a static permeability above one enlarges the budget.
- Reference numbers in the README: the decay of the reflectance in the
  two-temperature model (281 fs) is now compared with the measured kernel
  defined the same way, 1/e of the peak change (256 fs); decays with a constant
  coupling span 0.36–2.1 ps over 2–200 GW cm⁻²; deviations of the tests are
  quoted as rounded largest values, and the test suite checks each against the
  upper end of its rounding interval.
- `src/reproduce.py` names the log of a script that lives in another folder
  after the script alone.
- Python 3.11 or newer is required; the tests and the paper outputs are
  checked on Python 3.11, 3.12 and 3.13, which the CI matrix now covers.
- The colours of the study figures carry descriptive names in `figstyle.py`;
  every figure is unchanged.

## 1.0.0 (2026-10-21)

First release: material models, the Rozanov limit, the FDTD solvers, the
hot-electron and two-temperature models, every study script with its result
files, the reproduction pipeline and the test suite.
