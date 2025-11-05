"""
jl2trec.py

Convert a .jl.gz file with records:
{
  "src_id": <query id>,
  "src_query": <query text>,
  "tgt_results": [["183605", 6], ["616737", 4], ...]
}

Outputs:
- queries TSV:  qid<TAB>query
- qrels TXT:    qid 0 docid rel

Usage:
  python jl2trec.py -i input.jl.gz -q queries.tsv -r qrels.txt
"""

from __future__ import annotations
import argparse
import gzip
import json
import sys
from typing import Optional, Tuple


def sanitize_query(q: str, normalize_tabs: bool) -> str:
    # Remove embedded newlines; optionally convert tabs to spaces
    q = q.replace("\r", " ").replace("\n", " ")
    if normalize_tabs:
        q = q.replace("\t", " ")
    return q


def parse_result_pair(pair) -> Optional[Tuple[str, int]]:
    # Expect ["docid", rel]
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        return None
    docid, rel = pair
    # Coerce docid to str, rel to int
    try:
        docid_str = str(docid)
        rel_int = int(rel)
    except Exception:
        return None
    return docid_str, rel_int


def convert_file(
    in_path: str,
    queries_out: str,
    qrels_out: str,
    report_every: int = 100000,
    normalize_tabs: bool = True,
) -> None:
    total = 0
    bad_lines = 0
    bad_pairs = 0
    dup_qids = 0

    seen_qids = {}  # qid -> query (to detect conflicting duplicates)

    with gzip.open(in_path, "rt", encoding="utf-8", errors="replace") as fin, \
         open(queries_out, "w", encoding="utf-8", newline="") as qout, \
         open(qrels_out, "w", encoding="utf-8", newline="") as rout:

        for line in fin:
            total += 1
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue

            qid = rec.get("src_id")
            query = rec.get("src_query")
            results = rec.get("tgt_results")

            if qid is None or query is None or results is None:
                bad_lines += 1
                continue

            # Normalize qid as string (TREC accepts string IDs, but avoid spaces)
            qid_str = str(qid).strip()
            if not qid_str:
                bad_lines += 1
                continue

            # Sanitize query for TSV
            query_str = sanitize_query(str(query), normalize_tabs=normalize_tabs)

            # Handle duplicate qids across lines
            if qid_str in seen_qids:
                if seen_qids[qid_str] != query_str:
                    dup_qids += 1
                    # Keep the first query; still allow writing qrels below
                # Do NOT rewrite the query row again
            else:
                # First time seeing this qid → write to queries.tsv
                qout.write(f"{qid_str}\t{query_str}\n")
                seen_qids[qid_str] = query_str

            # Write qrels
            if isinstance(results, list):
                for pair in results:
                    parsed = parse_result_pair(pair)
                    if parsed is None:
                        bad_pairs += 1
                        continue
                    docid, rel = parsed
                    # TREC qrels format: qid 0 docid rel
                    rout.write(f"{qid_str} 0 {docid} {rel}\n")
            else:
                bad_lines += 1

            if report_every > 0 and total % report_every == 0:
                print(f"[progress] processed {total:,} lines...", file=sys.stderr)

    # Final stats
    print(f"[done] queries: {queries_out}", file=sys.stderr)
    print(f"[done] qrels:   {qrels_out}", file=sys.stderr)
    if bad_lines:
        print(f"[warning] skipped {bad_lines:,} malformed line(s)", file=sys.stderr)
    if bad_pairs:
        print(f"[warning] skipped {bad_pairs:,} malformed result pair(s)", file=sys.stderr)
    if dup_qids:
        print(f"[note] detected {dup_qids:,} duplicate qid(s) with conflicting queries; kept the first query text.", file=sys.stderr)
    print(f"[stats] total lines read: {total:,}", file=sys.stderr)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert .jl.gz to TREC qrels and queries TSV.")
    p.add_argument("-i", "--input", required=True, help="Path to input .jl.gz (JSON Lines, gzipped)")
    p.add_argument("-q", "--queries-out", required=True, help="Path to output queries .tsv (qid<TAB>query)")
    p.add_argument("-r", "--qrels-out", required=True, help="Path to output qrels .txt (qid 0 docid rel)")
    p.add_argument("--report-every", type=int, default=100000, help="Progress interval (default: 100000)")
    p.add_argument("--keep-tabs", action="store_true",
                   help="Keep tabs inside query text (default replaces tabs with spaces)")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    convert_file(
        in_path=args.input,
        queries_out=args.queries_out,
        qrels_out=args.qrels_out,
        report_every=args.report_every,
        normalize_tabs=not args.keep_tabs,
    )


if __name__ == "__main__":
    main()
