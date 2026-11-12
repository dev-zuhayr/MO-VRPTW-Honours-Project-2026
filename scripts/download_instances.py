#!/usr/bin/env python3
"""Download Solomon VRPTW instances (text format) into data/instances/.

The canonical source is Marius Solomon's page and the SINTEF benchmark site:

    http://web.cba.neu.edu/~msolomon/problems.htm
    https://www.sintef.no/projectweb/top/vrptw/solomon-benchmark/

For convenience this script pulls the 100-customer text files from a public
GitHub mirror. Always cite Solomon (1987) and SINTEF for the data itself.

Examples
--------
    python scripts/download_instances.py                  # the 6 canonical *01 instances
    python scripts/download_instances.py --all            # all 56 instances
    python scripts/download_instances.py C101 R101 RC201  # a specific subset
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

# A public mirror of Solomon's text files (100 customers).
MIRRORS = [
    "https://raw.githubusercontent.com/iRB-Lab/py-ga-VRPTW/master/data/text/{name}.txt",
    "https://raw.githubusercontent.com/ScorpJD/python-ga-VRPTW/master/data/text/{name}.txt",
]

GROUPS = {
    "C1": [f"C1{n:02d}" for n in range(1, 10)],
    "C2": [f"C2{n:02d}" for n in range(1, 9)],
    "R1": [f"R1{n:02d}" for n in range(1, 13)],
    "R2": [f"R2{n:02d}" for n in range(1, 12)],
    "RC1": [f"RC1{n:02d}" for n in range(1, 9)],
    "RC2": [f"RC2{n:02d}" for n in range(1, 9)],
}
ALL = [n for g in GROUPS.values() for n in g]
CANONICAL = ["C101", "C201", "R101", "R201", "RC101", "RC201"]


def fetch(name: str, dest: Path) -> bool:
    for tmpl in MIRRORS:
        url = tmpl.format(name=name)
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = resp.read()
            if b"CUSTOMER" in data:
                dest.write_bytes(data)
                return True
        except Exception:
            continue
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="Instance names (e.g. C101 R101).")
    ap.add_argument("--all", action="store_true", help="Download all 56 instances.")
    ap.add_argument("--out", default="data/instances", help="Output directory.")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.all:
        names = ALL
    elif args.names:
        names = [n.upper() for n in args.names]
    else:
        names = CANONICAL

    ok, fail = 0, []
    for name in names:
        dest = out / f"{name}.txt"
        if dest.exists():
            print(f"skip   {name} (already present)")
            ok += 1
            continue
        if fetch(name, dest):
            print(f"ok     {name}")
            ok += 1
        else:
            print(f"FAILED {name}")
            fail.append(name)

    print(f"\n{ok}/{len(names)} available in {out}.")
    if fail:
        print("Could not fetch:", ", ".join(fail), file=sys.stderr)
        print("Download manually from Solomon's page / SINTEF and drop into", out)


if __name__ == "__main__":
    main()
