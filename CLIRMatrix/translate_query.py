#!/usr/bin/env python3
"""
queries_translate_to_sc.py

Translate queries.tsv (qid<TAB>query) into Simplified Chinese using OpenAI’s API,
and write the translated file as queries_zh.tsv.

Usage:
  export OPENAI_API_KEY=sk-...
  python queries_translate_to_sc.py \
      --input queries.tsv \
      --output queries_zh.tsv \
      --model gpt-4.1-mini
"""

from __future__ import annotations
import argparse
import csv
import os
import sys
import time
from typing import Optional

from openai import OpenAI, APIStatusError, RateLimitError
from tqdm import tqdm  # progress bar


# ---------- API Helpers ----------

def _extract_text(resp) -> str:
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    try:
        for item in getattr(resp, "output", []):
            for piece in getattr(item, "content", []):
                t = getattr(piece, "text", None)
                if isinstance(t, str) and t.strip():
                    return t.strip()
    except Exception:
        pass
    return str(resp).strip()


def gpt_translate(client: OpenAI, model: str, text: str, max_retries: int = 5) -> str:
    """Translate text to Simplified Chinese with retry/backoff."""
    prompt_user = (
        "Translate the following text to Simplified Chinese (简体中文). "
        "Only return the translation itself, with no quotes or explanations.\n\n"
        f"Text:\n{text}"
    )
    attempt = 0
    backoff = 1.0
    while True:
        attempt += 1
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": "You are a precise professional translator to Simplified Chinese. Keep meaning and named entities."},
                    {"role": "user", "content": prompt_user},
                ],
                temperature=0,
            )
            return _extract_text(resp)
        except (RateLimitError, APIStatusError):
            if attempt >= max_retries:
                raise
            time.sleep(backoff)
            backoff *= 2
        except Exception:
            if attempt >= max_retries:
                raise
            time.sleep(backoff)
            backoff *= 2


# ---------- Translation Pipeline ----------

def translate_file(
    in_path: str,
    out_path: str,
    model: str = "gpt-4.1-mini",
    keep_tabs: bool = False,
) -> None:
    client = OpenAI()
    bad_rows = 0
    total_lines = sum(1 for _ in open(in_path, "r", encoding="utf-8"))  # count first for progress bar

    with open(in_path, "r", encoding="utf-8", newline="") as fin, \
         open(out_path, "w", encoding="utf-8", newline="") as fout:

        reader = csv.reader(fin, delimiter="\t")
        writer = csv.writer(fout, delimiter="\t", lineterminator="\n")

        with tqdm(total=total_lines, desc="Translating queries", ncols=90) as pbar:
            for row in reader:
                pbar.update(1)
                if not row or len(row) < 2:
                    bad_rows += 1
                    continue

                qid, query = row[0], row[1]
                q = query.replace("\r", " ").replace("\n", " ").strip()
                if not keep_tabs:
                    q = q.replace("\t", " ")

                try:
                    zh = gpt_translate(client, model=model, text=q)
                except Exception as e:
                    zh = ""
                    bad_rows += 1
                    tqdm.write(f"[error] qid={qid} translation failed: {e}")
                writer.writerow([qid, zh])

    print(f"\n[done] Translations saved to {out_path}")
    if bad_rows:
        print(f"[warning] {bad_rows:,} row(s) failed or malformed.", file=sys.stderr)


# ---------- CLI ----------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Translate queries.tsv to Simplified Chinese using OpenAI.")
    ap.add_argument("-i", "--input", required=True, help="Path to input queries.tsv (qid<TAB>query)")
    ap.add_argument("-o", "--output", required=True, help="Path to output queries_zh.tsv")
    ap.add_argument("--model", default="gpt-4.1-mini", help="OpenAI model to use (default: gpt-4.1-mini)")
    ap.add_argument("--keep-tabs", action="store_true", help="Keep tabs in query text (default replaces with spaces)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(2)
    translate_file(args.input, args.output, model=args.model, keep_tabs=args.keep_tabs)


if __name__ == "__main__":
    main()
