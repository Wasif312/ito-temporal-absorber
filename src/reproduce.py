"""reproduce.py -- run the study scripts in order and check the results
against the shipped ones.

Before the first script runs, the shipped results/*.json are copied to
results/published/ (once; later runs keep that copy). Each script then runs
in its own process with its output written to results/<script>.log, and at
the end every result file that a script rewrote is compared with its shipped
copy (compare_results.py). The last stage, paper, derives the data, tables
and figures of the manuscript from the result files (folder ../paper).

    python3 reproduce.py --list                    the steps, their outputs and run times
    python3 reproduce.py --quick                   checks and validations (about 20 min on 2 cores)
    python3 reproduce.py --stage route1 ttm        selected stages
    python3 reproduce.py --all --workers 8         everything (about 4.5 h on 2 cores)
    python3 reproduce.py --figures                 redraw every figure and table from the stored results

Run times are those of a 2-core machine; scripts that take --workers scale
with the number of cores.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
PUBLISHED = os.path.join(RESULTS, "published")

# (stage, script, extra arguments, takes --workers, minutes on 2 cores, outputs)
STEPS = [
    ("materials", "check_step1.py", [], False, 0.1, []),
    ("materials", "hot_electron.py", [], False, 3, ["hot_electron.json"]),
    ("materials", "gold_fit.py", [], False, 0.1, ["gold_fit.json"]),
    ("materials", "ttm_validate.py", [], False, 2, ["ttm_validation.json"]),
    ("validation", "validate_fdtd.py", [], False, 2, ["tests.json"]),
    ("validation", "validate_angle1d.py", [], False, 6, ["tests_angle.json"]),
    ("validation", "validate_angle_gold.py", [], False, 3, ["tests_angle_gold.json"]),
    ("validation", "appendix_data.py", [], False, 0.5, ["fig1b_trace.json"]),
    ("benchmarks", "bench_a4.py", [], True, 20, ["a4.json"]),
    ("benchmarks", "spec_a4.py", [], True, 2.5, ["a4_spectra.json"]),
    ("benchmarks", "spec_fast.py", [], True, 1, ["a4_fast.json"]),
    ("benchmarks", "bench_a5.py", [], False, 5, ["a5.json"]),
    ("benchmarks", "bench_a6.py", [], True, 7, ["a6.json"]),
    ("route3", "bound_scope.py", [], False, 0.3, []),
    ("route3", "chirp_scope.py", [], False, 1, []),
    ("route3", "angle_scan.py", [], False, 0.2, []),
    ("route3", "route3_fdtd.py", [], True, 8, ["route3.json"]),
    ("route3", "route3_angles.py", [], True, 5, ["route3_angles.json"]),
    ("route3", "route3_angles_adiabatic.py", [], False, 0.5, ["route3_angles_adiabatic.json"]),
    ("route1", "r1_chirp_scope.py", [], False, 1.5, ["r1_chirp_scope.json"]),
    ("route1", "r1_design.py", [], False, 13, ["r1_design.json"]),
    ("route1", "r1_threshold.py", [], True, 3.5, ["r1_threshold.json"]),
    ("route1", "route1_fdtd.py", [], True, 10, ["route1_fdtd.json"]),
    ("route1", "route1_threshold_fdtd.py", [], True, 9, ["route1_threshold_fdtd.json"]),
    ("route1", "route1_spectra.py", [], False, 1, ["route1_spectra.json"]),
    ("route1", "r1_real.py", [], True, 10, ["r1_real.json"]),
    ("route1", "r1_real_wide.py", [], True, 9, ["r1_real_wide.json"]),
    ("route1", "r1_damping.py", [], True, 3, ["r1_damping.json"]),
    ("route1", "r1_fixed_wide.py", [], True, 1, ["r1_fixed_wide.json"]),
    ("route1", "route1_real_fdtd.py", [], True, 25, ["route1_real_fdtd.json"]),
    ("ttm", "r1_ttm.py", [], True, 8, ["r1_ttm_baxter.json"]),
    ("ttm", "r1_ttm_damping.py", [], True, 6, ["r1_ttm_damping_baxter.json"]),
    ("ttm", "route1_ttm_fdtd.py", [], True, 13, ["route1_ttm_fdtd_baxter.json"]),
    ("ttm", "route1_ttm_fdtd.py", ["--source", "r1_ttm_damping", "--pick", "pec_T0_30k,7.5,75,0.30",
                                   "--pick", "pec_T0_-60k,7.5,75,0.30", "--tag", "_damping"], True, 6,
     ["route1_ttm_fdtd_baxter_damping.json"]),
    ("ttm", "fluence_check.py", [], False, 0.2, ["fluence_check_baxter.json"]),
    ("convergence", "convergence.py", [], True, 65, ["convergence.json"]),
] + [("figures", f"fig_step{n}.py", [], False, 0.3, []) for n in range(1, 9)] + [
    ("figures", "fig_ttm.py", [], False, 0.3, []),
] + [("paper", os.path.join(os.pardir, "paper", f"{name}.py"), [], False, minutes, [])
     for name, minutes in (("paper_data", 1), ("paper_numbers", 0.1), ("paper_extra", 0.1), ("paper_tables", 0.1),
                           ("paper_figures", 0.2))]
STAGES = list(dict.fromkeys(s[0] for s in STEPS))
QUICK = ["materials", "validation"]


def snapshot():
    """Keep a copy of the shipped results, once."""
    if os.path.isdir(PUBLISHED):
        return
    os.makedirs(PUBLISHED)
    for path in glob.glob(os.path.join(RESULTS, "*.json")):
        if not path.endswith("_partial.json"):          # checkpoints of interrupted runs
            shutil.copy2(path, PUBLISHED)
    print(f"copied the shipped results to {os.path.relpath(PUBLISHED, HERE)}/")


def run_step(step, workers):
    stage, script, extra, takes_workers, minutes, outputs = step
    cmd = [sys.executable, script] + extra + (["--workers", str(workers)] if takes_workers else [])
    name = os.path.splitext(os.path.basename(outputs[0] if outputs else script))[0]
    log = os.path.join(RESULTS, f"{name}.log")
    print(f"[{stage}] {' '.join(cmd[1:])}  (about {minutes:g} min on 2 cores) ...", flush=True)
    t0 = time.time()
    env = dict(os.environ, PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    with open(log, "w", encoding="utf-8") as fh:
        code = subprocess.call(cmd, cwd=HERE, stdout=fh, stderr=subprocess.STDOUT, env=env)
    print(f"    {'done' if code == 0 else f'FAILED (exit {code})'} in {time.time() - t0:.0f} s, log in "
          f"{os.path.relpath(log, HERE)}", flush=True)
    return code


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", action="store_true", help="list the steps and exit")
    ap.add_argument("--quick", action="store_true", help="stages " + ", ".join(QUICK))
    ap.add_argument("--all", action="store_true", help="every stage")
    ap.add_argument("--figures", action="store_true",
                    help="redraw the figures, and the figures and tables of the paper, from the stored results")
    ap.add_argument("--stage", nargs="+", choices=STAGES, default=[], help="stages to run")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                    help="worker processes for the scripts that take them (default: one per core)")
    ap.add_argument("--no-compare", action="store_true", help="skip the comparison with the shipped results")
    args = ap.parse_args()

    if args.list:
        for stage in STAGES:
            steps = [s for s in STEPS if s[0] == stage]
            print(f"{stage} (about {sum(s[4] for s in steps):g} min on 2 cores)")
            for _, script, extra, w, minutes, outs in steps:
                print(f"    {script + (' ' + ' '.join(extra) if extra else ''):60s} {minutes:6g} min"
                      f"{'  [--workers]' if w else ''}  {', '.join(outs)}")
        return 0
    stages = STAGES if args.all else (["figures", "paper"] if args.figures else
                                      list(dict.fromkeys((QUICK if args.quick else []) + args.stage)))
    if not stages:
        ap.print_help()
        return 2
    snapshot()
    failed, written = [], []
    for step in STEPS:
        if step[0] in stages:
            if run_step(step, args.workers):
                failed.append(step[1])
            else:
                written += step[5]
    if written and not args.no_compare:
        print("\ncomparison with the shipped results (results/published/):", flush=True)
        code = subprocess.call([sys.executable, "compare_results.py"] + written, cwd=HERE)
        if code:
            print("some numbers differ from the shipped results; see above")
    if failed:
        print("\nFAILED: " + ", ".join(failed))
        return 1
    print("\nall steps finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
