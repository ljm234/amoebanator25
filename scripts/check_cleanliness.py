#!/usr/bin/env python3
"""Fail if any tracked text file carries AI-authoring tells or process language.

This guard keeps the public repository free of two classes of artifact:

1. Authoring and formatting tells - checked in EVERY tracked text file:
   emoji, typographic dashes / smart quotes / box-drawing / ellipsis, the
   phrase "master prompt", named AI tools, and the old hyphenated repository
   slug "amoebanator-25".

2. Internal process language - checked in every tracked text file EXCEPT the
   vignette data lineage: phase / mini / sprint / closure-gate / spec-gap
   labels, dotted and bare question ids, task ids, "Subphase N" labels, the
   version-pinning word "locked", acceptance-criteria and gate numbering
   ("criterion #N", "gate #N"), and "Day-N" data-batch labels.

The vignette data lineage is exempt from the process-language check only. Those
provenance labels live in the immutable vignette corpus and in the code that
generates and validates it; stripping them from the code while the committed
data keeps them would desynchronise the two. The exemption is an explicit
allow-list (CARVE_OUT below), not a blanket skip - the tell and formatting
checks still apply to every file, carved out or not.

This guard exempts only itself (GUARD_FILES): its source necessarily spells out
the very tokens it forbids, so scanning it would always fail.
"""
from __future__ import annotations

import re
import subprocess
import sys

# Binary suffixes that are never scanned.
BINARY_SUFFIXES = (
    ".pt", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".pkl",
    ".npy", ".npz", ".gz", ".zip", ".woff", ".woff2", ".ttf", ".so", ".bin",
)

# Checks applied to EVERY tracked text file.
GLOBAL_PATTERNS = {
    "emoji": re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F]"),
    "typography": re.compile(r"[\u2014\u2013\u2500-\u257F\u201C\u201D\u2026]"),
    "master-prompt": re.compile(
        r"master[ _-]?prompt|AMOEBANATOR_MASTER", re.IGNORECASE
    ),
    "ai-tool": re.compile(r"\b(claude|copilot|cursor|chatgpt|gemini)\b", re.IGNORECASE),
    "old-repo-slug": re.compile(r"amoebanator-25"),
}

# Internal process language - applied to every text file EXCEPT the carve-out.
PROCESS_PATTERNS = {
    "phase-label": re.compile(r"\bPhase[ -][0-9]"),
    "mini-label": re.compile(r"\bMini-\s*[0-9]"),
    "question-id": re.compile(r"\bQ[0-9]+"),
    "task-id": re.compile(r"\bT[0-9]\.[0-9]"),
    "subphase-label": re.compile(r"\bSubphase\s+[0-9]"),
    "closure-gate": re.compile(r"closure[ -]gate"),
    "sprint": re.compile(r"\b[Ss]print\b"),
    "spec-gap": re.compile(r"spec-?gap"),
    "version-locked": re.compile(r"\b[Ll]ocked\b"),
    "criterion-id": re.compile(r"\bcriterion\s+#[0-9]"),
    "gate-id": re.compile(r"\bgate\s+#[0-9]"),
    "day-label": re.compile(r"\bDay-[0-9]"),
}

# Vignette data lineage: the immutable corpus, the generators that emit it, and
# the tests that validate it. Exempt from PROCESS_PATTERNS only.
CARVE_OUT_PREFIXES = (
    "data/",
    "tests/_snapshots/",
    "tests/vignettes/",
    "scripts/vignettes/",
    "tests/schemas/fixtures/",
)
CARVE_OUT_REGEXES = (
    re.compile(r"^tests/test_subphase_"),
    re.compile(r"^tests/test_stage_.*lockin"),
)
CARVE_OUT_FILES = ("tests/schemas/test_vignette_migration.py",)

# This guard's own source spells out every forbidden token as a pattern, so it
# is the one file exempt from all checks.
GUARD_FILES = ("scripts/check_cleanliness.py",)


def is_carved_out(path: str) -> bool:
    if path in CARVE_OUT_FILES:
        return True
    if path.startswith(CARVE_OUT_PREFIXES):
        return True
    return any(rx.search(path) for rx in CARVE_OUT_REGEXES)


def main() -> int:
    files = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout.split()
    violations: list[str] = []
    scanned = 0
    for path in files:
        if path in GUARD_FILES:
            continue
        if path.endswith(BINARY_SUFFIXES):
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
        except (OSError, UnicodeDecodeError):
            continue
        scanned += 1
        for name, pattern in GLOBAL_PATTERNS.items():
            if pattern.search(text):
                violations.append(f"{path}: {name}")
        if not is_carved_out(path):
            for name, pattern in PROCESS_PATTERNS.items():
                if pattern.search(text):
                    violations.append(f"{path}: {name}")
    if violations:
        print("CLEANLINESS GUARD FAILED:")
        for item in violations:
            print(f"  {item}")
        return 1
    print(
        f"Cleanliness guard passed: scanned {scanned} tracked text files, "
        "no tells or process language found."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
