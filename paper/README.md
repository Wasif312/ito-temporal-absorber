# Figures and tables of the paper

This folder turns the result files in `src/results/` into the eleven figures
and 47 tables of the manuscript. No number is typed by hand: the scripts read
the JSON files that the studies wrote, and three of them recompute a few
derived quantities with the code in `src/`. Two tables of input parameters,
`tables/materials.md` and `tables/numerics.md`, are the only hand-written files.

## Running the scripts

From the repository folder, in this order (about a minute in all):

```
python paper/paper_data.py       # data/paper_data.json: budgets, static dips, drops behind two measurements
python paper/paper_numbers.py    # data/numbers.json: zero crossings of the margin against drop and damping
python paper/paper_extra.py      # data/extra.json: tolerances in thickness and angle, pump filter, rotated pump
python paper/paper_tables.py     # tables/*.md, one pandoc pipe table per file
python paper/paper_figures.py    # figures/*.png, 300 dpi, 16 cm wide
```

`python src/reproduce.py --stage paper` runs the same five steps and keeps a
log of each in `src/results/`. Each script accepts `--src` (another copy of
`src/`) and, where it reads result files, `--results` (another folder of result
files, for instance `src/results/published/`, the copy of the shipped results
that `reproduce.py` makes before it recomputes anything). Further options:

```
python paper/paper_figures.py fig_route1 fig_ttm --show   # selected figures, also shown in windows
python paper/paper_tables.py --print                       # every table also printed to the console
```

`tests/test_paper.py` checks that every output exists and that the key derived
numbers agree with a fresh computation from the result files.

## Figures

In the order in which the manuscript shows them. The last column names the
tables that give the same data as numbers.

| figure | what it shows | result files | tables |
|---|---|---|---|
| `fig_concept` | geometry; reflectance of the 7.5 nm film at 75° and its absorption dip against the drop of ωp | `data/paper_data.json` | `static_R`, `static_dip`, `budgets` |
| `fig_gas` | drop of ωp against electron temperature, damping laws, electron cooling time | `hot_electron.json`, `src/ttm.py` | `hot`, `gas` |
| `fig_validation` | accuracy of the solvers, residuals of the energy balance, convergence | `tests*.json`, `convergence.json`, the full-wave result files | `tests_kx`, `tests_angle`, `t1_spectrum`, `trace`, `residuals`, `convergence` |
| `fig_benchmarks` | the three published benchmarks B1–B3 | `a4.json`, `a5.json`, `a6.json` | `a4`, `a4_trace`, `a5`, `a5_all`, `a6`, `a6_broad` |
| `fig_mirror` | chirp-matched switching of the ITO–gold mirror against pump delay and angle | `route3*.json` | `route3`, `route3_angles`, `route3_angles_ad` |
| `fig_route1` | the 7.5 nm film on a conductor: dip tracking, spectra, margin against the drop, absorbed fractions | `route1_fdtd.json`, `route1_spectra.json`, `route1_threshold_fdtd.json`, `r1_design.json`, `r1_threshold.json` | `route1_fw`, `spectra`, `dip_track`, `threshold_fw`, `threshold_ad` |
| `fig_design` | margin against thickness and angle for drops of 10, 20 and 30 % | `r1_design.json` | `design`, `scope` |
| `fig_damping` | margin against the damping at the largest drop, and with a gold mirror | `r1_real.json`, `r1_ttm_damping_baxter.json`, `route1_ttm_fdtd_baxter*.json`, `data/numbers.json` | `variants_fw`, `variants`, `variants_wide`, `damping_kernel`, `ttm_damping` |
| `fig_experiments` | the two-temperature model against two pump–probe experiments | `ttm_validation.json`, `a4.json` | `ttm_tirole`, `ttm_alam`, `inferred` |
| `fig_ttm` | time course, pump fluence and heating of the 7.5 nm film in the two-temperature model | `r1_ttm_baxter.json`, `fluence_check_baxter.json`, `route1_spectra.json` | `ttm_drop`, `ttm_profiles`, `ttm_designs`, `ttm_fw`, `heat_r1`, `heat_exp`, `ttm_route1` |
| `fig_requirements` | margin inferred with an error of thickness or angle, and inside the band of a pump filter | `data/extra.json` | `sensitivity`, `passband` |

## Tables

Six tables appear in the main text: `materials`, `numerics`, `inferred`,
`route1_fw`, `variants_fw` and `ttm_fw`. The other 41 form the appendix of
complete datasets. Each file starts with its caption in the form
`Table: TAB:<name> <caption>`, followed by a pipe table whose symbols are
written as inline LaTeX, so pandoc converts it to Word or LaTeX unchanged.

| appendix section | tables |
|---|---|
| optical constants and hot electrons | `gold`, `hot`, `gas` |
| tests of the solvers | `t1_spectrum`, `tests_kx`, `trace`, `tests_angle`, `residuals` |
| benchmarks | `a4_trace`, `a4`, `a5`, `a5_all`, `a6`, `a6_broad` |
| the ITO–gold mirror under a chirped probe | `route3`, `route3_angles`, `route3_angles_ad` |
| the thin film on a conductor, kernel model | `static_dip`, `static_R`, `budgets`, `dip_track`, `spectra`, `threshold_fw`, `threshold_ad`, `design`, `scope` |
| damping and the gold mirror | `variants_wide`, `damping_kernel`, `variants` |
| the two-temperature model | `ttm_tirole`, `ttm_alam`, `ttm_drop`, `ttm_route1`, `ttm_profiles`, `ttm_designs`, `ttm_damping`, `heat_r1`, `heat_exp` |
| convergence | `convergence` |
| measurement requirements | `passband`, `sensitivity` |

## Names in the code and in the paper

| code | paper |
|---|---|
| route 1, `r1_*.py`, `route1_*.py` | thin ITO film on a perfect conductor or gold at a fixed angle in a glass prism |
| route 3, `route3*.py` | ITO–gold mirror of Tirole et al. under a chirped probe |
| A4, A5, A6 (`bench_a4.py` and so on) | benchmarks B1, B2, B3 |
| `pec`, `au` | perfect conductor, gold mirror |
| `fixed`, `hot` | fixed damping, damping that rises with the electron temperature (law of Baxter et al.) |
| `T0_30k`, `T0_-60k` | damping laws with T0 = 30 000 K (half the rise) and T0 = −60 000 K (falling) |
| kernel model | measured response of Tirole et al. (30 fs rise, 210 fs decay) |
| `depth` | largest drop of the plasma frequency |
