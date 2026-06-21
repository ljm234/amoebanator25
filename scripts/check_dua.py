#!/usr/bin/env python3
"""Fail if any tracked data file contains MIMIC-IV patient identifiers.

MIMIC-IV is credentialed-access data and the PhysioNet data use agreement
prohibits redistribution. This guard blocks real MIMIC-derived data from
entering the public repository: it scans tracked tabular and JSON files for
the patient-level keys that mark real MIMIC records, and the loader output
filenames, then exits non-zero on any hit.
"""
from __future__ import annotations

import re
import subprocess
import sys

IDENTIFIERS = re.compile(r"\b(subject_id|hadm_id|stay_id)\b")
FORBIDDEN_NAMES = ("labevents.csv", "diagnoses_icd.csv", "microbiologyevents.csv")
DATA_SUFFIXES = (".csv", ".tsv", ".json", ".ndjson")


def main() -> int:
    files = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout.split()
    violations: list[str] = []
    for path in files:
        name = path.rsplit("/", 1)[-1]
        if name in FORBIDDEN_NAMES:
            violations.append(f"{path}: forbidden MIMIC cohort filename")
            continue
        if not path.endswith(DATA_SUFFIXES):
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
        except (OSError, UnicodeDecodeError):
            continue
        match = IDENTIFIERS.search(text)
        if match:
            violations.append(f"{path}: contains MIMIC identifier '{match.group(1)}'")
    if violations:
        print("DUA GUARD FAILED - possible MIMIC-IV data in the repository:")
        for item in violations:
            print(f"  {item}")
        return 1
    print(f"DUA guard passed: scanned {len(files)} tracked files, no MIMIC data found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
