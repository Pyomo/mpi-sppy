#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path

def load_colnames(col_path):
    """Return a 1-based list of AMPL variable names in column order."""
    names = []
    with open(col_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            names.append(s)
    if not names:
        raise ValueError(f"No variable names found in {col_path}")
    return names  # index 0 -> C0001 (index+1)

def parse_c_label(label):
    """
    Convert 'C0001' or 'C1' (case-insensitive) to its 1-based integer index: 1.
    Returns None if it can't parse.
    """
    if not label:
        return None
    s = label.strip()
    if len(s) < 2 or (s[0] not in "Cc"):
        return None
    digits = s[1:]
    if not digits.isdigit():
        return None
    return int(digits)

def build_index_to_name(colnames):
    """Map 1-based column index -> AMPL var name."""
    return {i + 1: name for i, name in enumerate(colnames)}

def main():
    ap = argparse.ArgumentParser(
        description="Map (Cxxxx,value) CSV to (AMPL var name,value) using .col order."
    )
    ap.add_argument("col_file", help="Path to .col file (one AMPL var name per line, in column order)")
    ap.add_argument("input_csv", help="CSV with rows like: C0001,183.33")
    ap.add_argument("output_csv", help="Output CSV with rows: AMPL_var_name,value")
    ap.add_argument("--strict", action="store_true",
                    help="Error out if an input CSV C-label does not exist in the .col mapping.")
    args = ap.parse_args()

    try:
        colnames = load_colnames(args.col_file)
    except Exception as e:
        print(f"Error reading .col: {e}", file=sys.stderr)
        sys.exit(1)

    idx_to_name = build_index_to_name(colnames)
    missing = 0
    converted = 0

    # Read input CSV and write output CSV
    in_path = Path(args.input_csv)
    out_path = Path(args.output_csv)

    try:
        with open(in_path, "r", encoding="utf-8", newline="") as fin, \
             open(out_path, "w", encoding="utf-8", newline="") as fout:

            reader = csv.reader(fin)
            writer = csv.writer(fout)
            # Header
            writer.writerow(["varname", "value"])

            for rownum, row in enumerate(reader, start=1):
                if not row:
                    continue
                if len(row) < 2:
                    print(f"Warning: line {rownum} in {in_path} has fewer than 2 columns; skipping.",
                          file=sys.stderr)
                    continue

                c_label = row[0].strip()
                value = row[1].strip()

                idx = parse_c_label(c_label)
                if idx is None:
                    print(f"Warning: line {rownum}: cannot parse C-label '{c_label}'; skipping.",
                          file=sys.stderr)
                    missing += 1
                    if args.strict:
                        sys.exit(2)
                    continue

                name = idx_to_name.get(idx)
                if name is None:
                    print(f"Warning: line {rownum}: C-index {idx} not found in .col (max={len(colnames)}); skipping.",
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

    # Summary to stderr so the CSV stays clean
    print(f"Done. Wrote {converted} rows to {out_path}. Skipped {missing}.", file=sys.stderr)

if __name__ == "__main__":
    main()
