###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2025, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
#!/usr/bin/env python3
"""
Map (xN,value) CSV to (original GAMS var name,value) using GAMS CONVERT dict.txt.

Example:
  dict.txt contains lines like:
      Variables 1 to 4
        x1  area(wheat)
        x2  area(corn)
        x3  area(sugarbeets)
        x4  z

  Input CSV:
      x1,0.0
      x2,0.0
      x3,0.0

  Output CSV:
      varname,value
      area(wheat),0.0
      area(corn),0.0
      area(sugarbeets),0.0
"""

import argparse
import csv
import re
import sys
from pathlib import Path


def load_dict_mapping(dict_path: str) -> dict[str, str]:
    """
    Parse GAMS CONVERT dict.txt and return a mapping:
        scalar_name_lower -> original_name
    Only entries from the Variables section(s) are returned.
    We intentionally ignore equations (which are usually 'e1', 'e2', ...).
    """
    pat_entry = re.compile(r"\s*([A-Za-z]\d+)\s+(.*\S)\s*$")
    mapping: dict[str, str] = {}
    in_variables_block = False

    with open(dict_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()

            # Detect start of a Variables block
            if re.match(r"^Variables\s+\d+\s+to\s+\d+\s*$", s, flags=re.IGNORECASE):
                in_variables_block = True
                continue

            # Detect start of a different section -> leave variables mode
            if re.match(r"^(Equations|Rows|RHS|Bounds|Nonzero counts|Equation counts|Variable counts)\b",
                        s, flags=re.IGNORECASE):
                in_variables_block = False

            if not in_variables_block:
                continue

            m = pat_entry.match(line)
            if not m:
                continue

            scalar, original = m.group(1), m.group(2)
            # Skip equations just in case (they'd typically be 'eN')
            if scalar[0].lower() == "e":
                continue

            mapping[scalar.lower()] = original

    if not mapping:
        raise ValueError(f"No variable mappings found in {dict_path}")
    return mapping


def parse_scalar_label(label: str) -> str | None:
    """
    Normalize labels like 'x1', 'X0001', 'b12' to a lower-cased compact form: 'x1', 'b12'.
    Returns None if it can't parse.
    """
    if not label:
        return None
    s = label.strip()
    m = re.match(r"^([A-Za-z]+)0*([0-9]+)$", s)
    if not m:
        return None
    prefix = m.group(1).lower()
    idx = m.group(2).lstrip("0")
    if idx == "":
        idx = "0"
    return f"{prefix}{idx}"


def main():
    ap = argparse.ArgumentParser(
        description="Map (xN,value) CSV to (original GAMS var name,value) using dict.txt."
    )
    ap.add_argument("dict_txt", help="Path to GAMS CONVERT dict.txt")
    ap.add_argument("input_csv", help="CSV with rows like: x1,183.33")
    ap.add_argument("output_csv", help="Output CSV with rows: varname,value")
    ap.add_argument("--strict", action="store_true",
                    help="Error out if an input CSV label is missing in the dictionary.")
    args = ap.parse_args()

    try:
        mapping = load_dict_mapping(args.dict_txt)
    except Exception as e:
        print(f"Error reading dict.txt: {e}", file=sys.stderr)
        sys.exit(1)

    missing = 0
    converted = 0

    in_path = Path(args.input_csv)
    out_path = Path(args.output_csv)

    try:
        with open(in_path, "r", encoding="utf-8", newline="") as fin, \
             open(out_path, "w", encoding="utf-8", newline="") as fout:

            reader = csv.reader(fin)
            writer = csv.writer(fout)
            writer.writerow(["varname", "value"])

            for rownum, row in enumerate(reader, start=1):
                if not row:
                    continue
                if len(row) < 2:
                    print(f"Warning: line {rownum} in {in_path} has fewer than 2 columns; skipping.",
                          file=sys.stderr)
                    continue

                raw_label = row[0]
                value = row[1]

                key = parse_scalar_label(raw_label)
                if key is None:
                    print(f"Warning: line {rownum}: cannot parse label '{raw_label}'; skipping.",
                          file=sys.stderr)
                    missing += 1
                    if args.strict:
                        sys.exit(2)
                    continue

                name = mapping.get(key)
                if name is None:
                    # Try literal lowercase (in case dict kept zero padding, rare)
                    name = mapping.get(raw_label.strip().lower())
                if name is None:
                    print(f"Warning: line {rownum}: label '{raw_label}' not in dict; skipping.",
                          file=sys.stderr)
                    missing += 1
                    if args.strict:
                        sys.exit(2)
                    continue

                writer.writerow([name, value])
                converted += 1

    except FileNotFoundError as e:
        print(f"File not found: {e.filename}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error processing files: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Done. Wrote {converted} rows to {out_path}. Skipped {missing}.", file=sys.stderr)


if __name__ == "__main__":
    main()
