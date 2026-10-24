"""Folders and command line shared by the paper scripts.

The scripts import the modules of src/ and read the result files of
src/results/. They write to data/, tables/ and figures/ next to this file.
--src points them at another copy of src/, and --results at another folder of
result files, for instance src/results/published/, the copy of the shipped
results that src/reproduce.py keeps before it recomputes anything.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data")
TABLES = os.path.join(HERE, "tables")
FIGURES = os.path.join(HERE, "figures")


def arguments(description, results=True, extra=None):
    """Parse --src (and --results, plus whatever extra(parser) adds), put src/
    on the import path and return the arguments with absolute folders."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--src", default=os.path.join(REPO, "src"),
                    help="src folder of the repository (default: ../src)")
    if results:
        ap.add_argument("--results", help="folder of the result files (default: SRC/results)")
    if extra is not None:
        extra(ap)
    args = ap.parse_args()
    args.src = os.path.abspath(args.src)
    folders = [args.src]
    if results:
        args.results = os.path.abspath(args.results or os.path.join(args.src, "results"))
        folders.append(args.results)
    for folder in folders:
        if not os.path.isdir(folder):
            ap.error(f"no such folder: {folder}")
    sys.path.insert(0, args.src)
    return args


def read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_json(name, obj):
    """Write obj to data/<name> and return the path."""
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)
    return path
