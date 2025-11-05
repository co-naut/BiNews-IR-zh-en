#!/usr/bin/env python3
"""
tsv_t2s_convert.py

Convert a .tsv.gz file from Traditional Chinese to Simplified Chinese.
Input format: <document_id>\t<text>
Output format: same, but with text converted.
"""

import argparse
import gzip
import sys
import opencc


def convert_file(in_path: str, out_path: str, report_every: int = 100000) -> None:
    converter = opencc.OpenCC("t2s.json")  # built-in conversion: Traditional → Simplified
    total = 0
    bad_lines = 0

    with gzip.open(in_path, "rt", encoding="utf-8", errors="replace") as fin, \
         gzip.open(out_path, "wt", encoding="utf-8") as fout:

        for line in fin:
            total += 1
            line = line.rstrip("\r\n")
            if not line:
                fout.write("\n")
                continue

            if "\t" not in line:
                bad_lines += 1
                continue

            doc_id_str, text = line.split("\t", 1)
            try:
                int(doc_id_str)
            except ValueError:
                bad_lines += 1
                continue

            fout.write(f"{doc_id_str}\t{converter.convert(text)}\n")

            if report_every > 0 and total % report_every == 0:
                print(f"[progress] processed {total:,} lines...", file=sys.stderr)

    print(f"[done] wrote {out_path}", file=sys.stderr)
    if bad_lines:
        print(f"[warning] skipped {bad_lines:,} malformed lines", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Traditional → Simplified Chinese in a TSV.GZ file.")
    parser.add_argument("--input", "-i", required=True, help="Input .tsv.gz path")
    parser.add_argument("--output", "-o", required=True, help="Output .tsv.gz path")
    parser.add_argument("--report-every", type=int, default=100000, help="Progress report interval")
    args = parser.parse_args()

    convert_file(args.input, args.output, args.report_every)
