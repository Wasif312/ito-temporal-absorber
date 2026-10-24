"""compare_results.py -- compare result files with a reference copy.

After a rerun, compare every JSON file in results/ with the same file in a
reference folder (by default results/published/, which reproduce.py fills with
the shipped results before it runs anything). For each file the script walks
both trees and reports the largest absolute difference between matching
numbers, the largest difference relative to the size of the reference value,
and any key or length that does not match (entries that only the new file
has are listed but do not count as a difference). Numbers that agree to within
--atol or --rtol count as equal; the exit status is 1 if any file differs.

Usage: python3 compare_results.py [--ref DIR] [--new DIR] [--atol A] [--rtol R] [FILE ...]
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def walk(a, b, path, out):
    """Collect (path, |a-b|, |a-b|/|a|) for matching numbers, and mismatches."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(a.keys() | b.keys(), key=str):
            if k not in b:
                out["structure"].append(f"{path}/{k}: missing in the new file")
            elif k not in a:
                out["added"].append(f"{path}/{k}")
            else:
                walk(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out["structure"].append(f"{path}: length {len(a)} (reference) against {len(b)} (new)")
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            walk(x, y, f"{path}[{i}]", out)
    elif isinstance(a, bool) or isinstance(b, bool) or a is None or b is None or isinstance(a, str):
        if a != b:
            out["structure"].append(f"{path}: {a!r} against {b!r}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isnan(a) and math.isnan(b):
            return
        d = abs(a - b)
        out["numbers"].append((d, d / abs(a) if a != 0 else (0.0 if d == 0 else math.inf), path))
    else:
        out["structure"].append(f"{path}: type {type(a).__name__} against {type(b).__name__}")


def compare(ref_path, new_path, atol, rtol):
    with open(ref_path) as fh:
        a = json.load(fh)
    with open(new_path) as fh:
        b = json.load(fh)
    out = dict(numbers=[], structure=[], added=[])
    walk(a, b, "", out)
    nums = out["numbers"]
    worst_abs = max(nums, default=(0.0, 0.0, ""))
    worst_rel = max(nums, key=lambda x: x[1], default=(0.0, 0.0, ""))
    bad = [x for x in nums if x[0] > atol and x[1] > rtol]
    return dict(n=len(nums), worst_abs=worst_abs, worst_rel=worst_rel, bad=bad, structure=out["structure"],
                added=out["added"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="*", help="file names to compare (default: every JSON file in the reference)")
    ap.add_argument("--ref", default=os.path.join(HERE, "results", "published"))
    ap.add_argument("--new", default=os.path.join(HERE, "results"))
    ap.add_argument("--atol", type=float, default=1e-9, help="absolute tolerance (default 1e-9)")
    ap.add_argument("--rtol", type=float, default=1e-6, help="relative tolerance (default 1e-6)")
    args = ap.parse_args()
    names = args.files or sorted(f for f in os.listdir(args.ref) if f.endswith(".json"))
    differ = 0
    for name in names:
        ref_path, new_path = os.path.join(args.ref, name), os.path.join(args.new, name)
        if not os.path.exists(new_path):
            print(f"{name:40s} not in {args.new}")
            continue
        r = compare(ref_path, new_path, args.atol, args.rtol)
        ok = not r["bad"] and not r["structure"]
        differ += not ok
        print(f"{name:40s} {'same' if ok else 'DIFFERS':8s} {r['n']:7d} numbers, largest difference "
              f"{r['worst_abs'][0]:.1e} absolute, {r['worst_rel'][1]:.1e} relative")
        for d, rel, path in sorted(r["bad"], reverse=True)[:5]:
            print(f"      {path}: {d:.2e} ({rel:.1e} relative)")
        for s in r["structure"][:5]:
            print(f"      {s}")
        if r["added"]:
            print(f"      new entries, not in the reference: {', '.join(r['added'][:5])}")
    print(f"{len(names) - differ} of {len(names)} files agree within atol {args.atol:g} or rtol {args.rtol:g}")
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
