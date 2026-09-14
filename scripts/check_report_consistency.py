#!/usr/bin/env python
"""Check a report's numbers against EACH OTHER. Complements scripts/audit_cells.py.

The two checks answer different questions and neither substitutes for the other:

    audit_cells.py            "does this figure match the logs?"       (provenance)
    check_report_consistency  "do the figures in this document agree?" (internal coherence)

Both defects a reader caught in this post were invisible to the provenance check, because every
individual number was true and the document still contradicted itself:

1. "8/30 episodes" beside "all ten mentions" -- two different units presented as one count.
2. A pooled total of 0/503 printed above a table whose rows summed to 220. Both figures were
   correct about the data; only their juxtaposition was wrong.

So this script re-derives, from the text alone:

- every `a/n vs b/m` -> `X pp [lo, hi]` claim, against Newcombe on those same four numbers;
- every one-sided bound quoted next to a `0/n`, against Clopper-Pearson;
- every table `subtotal` / `total` row, against the sum of the data rows above it;
- that no numerator exceeds its denominator anywhere.

    uv run --no-sync python scripts/check_report_consistency.py <report.md>

Exit status is non-zero if anything fails, so it can gate a publish step.
"""

from __future__ import annotations

import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from evalrep.stats import cp_upper_zero, newcombe_diff  # noqa: E402

#: The report rounds interval endpoints to whole percentage points, so allow that much slack.
TOL_PP = 0.6

FRAC = re.compile(r"(\d+)\s*/\s*(\d+)")
VS = re.compile(r"(\d+)\s*/\s*(\d+)\s*(?:vs|versus)\s*(\d+)\s*/\s*(\d+)")
DIFF = re.compile(r"([+\-−]?\d+(?:\.\d+)?)\s*pp\s*\[\s*([+\-−]?\d+(?:\.\d+)?)\s*,"
                  r"\s*([+\-−]?\d+(?:\.\d+)?)\s*\]")
BOUND = re.compile(r"bound(?:s)?[^.\d]{0,24}(\d+(?:\.\d+)?)\s*%")


def num(s: str) -> float:
    """Parse a signed number that may use a Unicode minus."""
    return float(s.replace("−", "-").replace("+", ""))


def strip_md(s: str) -> str:
    return re.sub(r"[*`]", "", s)


def check_differences(lines: list[str]) -> list[str]:
    """Every `a/n vs b/m ... X pp [lo, hi]` must match Newcombe on those numbers."""
    out = []
    for i, raw in enumerate(lines, 1):
        line = strip_md(raw)
        v, d = VS.search(line), DIFF.search(line)
        if not (v and d):
            continue
        a, n, b, m = (int(x) for x in v.groups())
        claim_d, claim_lo, claim_hi = (num(x) for x in d.groups())
        if not (n and m):
            continue
        real_d = (a / n - b / m) * 100
        lo, hi = (x * 100 for x in newcombe_diff(a, n, b, m))
        bad = []
        if abs(real_d - claim_d) > TOL_PP:
            bad.append(f"diff {claim_d:+.1f} but {a}/{n}-{b}/{m} = {real_d:+.1f}")
        if abs(lo - claim_lo) > TOL_PP or abs(hi - claim_hi) > TOL_PP:
            bad.append(f"CI [{claim_lo:+.0f},{claim_hi:+.0f}] but Newcombe = [{lo:+.1f},{hi:+.1f}]")
        if bad:
            out.append(f"L{i}: " + "; ".join(bad))
    return out


def check_bounds(lines: list[str]) -> list[str]:
    """A one-sided bound quoted near a zero cell must match Clopper-Pearson for that n."""
    out = []
    for i, raw in enumerate(lines, 1):
        line = strip_md(raw)
        b = BOUND.search(line)
        if not b:
            continue
        claim = num(b.group(1))
        zeros = [int(m.group(2)) for m in FRAC.finditer(line) if m.group(1) == "0"]
        if not zeros:
            continue
        # the bound belongs to one of the zero cells named on the line
        if not any(abs(cp_upper_zero(n) * 100 - claim) <= 0.05 for n in zeros):
            opts = ", ".join(f"0/{n}->{cp_upper_zero(n)*100:.2f}%" for n in sorted(set(zeros)))
            out.append(f"L{i}: bound {claim}% matches no zero cell on the line ({opts})")
    return out


def check_table_totals(lines: list[str]) -> list[str]:
    """`subtotal` sums the data rows since the last subtotal; `total` sums the whole table.

    Two accumulators, because a grand total must not be reset by the subtotals it follows --
    that mistake is how the first version of this check reported a false positive on a table
    that was in fact correct.
    """
    out = []
    sub_n = sub_d = all_n = all_d = 0
    sub_rows = all_rows = 0
    for i, raw in enumerate(lines, 1):
        line = strip_md(raw).strip()
        if not line.startswith("|"):
            sub_n = sub_d = all_n = all_d = sub_rows = all_rows = 0
            continue
        if set(line) <= set("|-: "):
            continue
        low = line.lower()
        is_sub = "subtotal" in low
        is_grand = not is_sub and re.search(r"\|\s*\*{0,2}total\*{0,2}\s*\|", low)
        fracs = FRAC.findall(line)
        if not fracs:
            continue
        a, n = int(fracs[-1][0]), int(fracs[-1][1])
        if is_sub:
            if (a, n) != (sub_n, sub_d):
                out.append(f"L{i}: subtotal says {a}/{n} but the {sub_rows} rows since the "
                           f"last subtotal sum to {sub_n}/{sub_d}")
            sub_n = sub_d = sub_rows = 0
        elif is_grand:
            if (a, n) != (all_n, all_d):
                out.append(f"L{i}: total says {a}/{n} but all {all_rows} data rows in the "
                           f"table sum to {all_n}/{all_d}")
        else:
            sub_n += a
            sub_d += n
            all_n += a
            all_d += n
            sub_rows += 1
            all_rows += 1
    return out


def check_fractions(lines: list[str]) -> list[str]:
    out = []
    for i, raw in enumerate(lines, 1):
        for a, n in FRAC.findall(strip_md(raw)):
            if int(n) and int(a) > int(n):
                out.append(f"L{i}: {a}/{n} has numerator > denominator")
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = pathlib.Path(sys.argv[1])
    lines = path.read_text().splitlines()
    checks = [
        ("differences vs Newcombe", check_differences),
        ("zero-cell bounds vs Clopper-Pearson", check_bounds),
        ("table subtotals/totals", check_table_totals),
        ("numerator <= denominator", check_fractions),
    ]
    failures = 0
    for label, fn in checks:
        problems = fn(lines)
        status = "OK" if not problems else f"{len(problems)} PROBLEM(S)"
        print(f"{label:38} {status}")
        for p in problems:
            print(f"    {p}")
        failures += len(problems)
    print(f"\n{path}: {'CONSISTENT' if not failures else f'{failures} inconsistency(ies)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
