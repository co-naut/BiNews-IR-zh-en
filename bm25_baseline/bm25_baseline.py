"""
Simple BM25 baseline IR system.
- Corpus input: JSONL (.jsonl or .jsonl.gz) with {"id": str, "text": str}
                 or TSV/CSV (.tsv/.tsv.gz) with columns: id, text
- Queries input: TSV with columns: qid, query
- Qrels input: TREC style: qid <ignored> docid rel   (rel > 0 => relevant)

Commands:
  1) search:
     python bm25_baseline.py search \
        --corpus data/corpus.jsonl.gz \
        --queries data/queries.tsv \
        --run out/run.tsv \
        --topk 100

  2) eval:
     python bm25_baseline.py eval \
        --run out/run.tsv \
        --qrels data/qrels.txt

  3) end-to-end (build+search in one go without saving index separately).

Author: ChatGPT
"""
from __future__ import annotations
import argparse
import csv
import gzip
import io
import json
import math
import re
import sys
from tqdm import tqdm
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Set, Optional

# ----------------------------
# Tokenization / preprocessing
# ----------------------------
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

DEFAULT_STOPWORDS: Set[str] = {
    # Small, conservative list; expand if you like
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "by",
    "is", "are", "was", "were", "be", "been", "with", "as", "at", "from",
    "that", "this", "it", "its", "into", "about", "over", "under", "than",
}

def tokenize(text: str, lowercase: bool = True, stopwords: Optional[Set[str]] = None) -> List[str]:
    if not text:
        return []
    if lowercase:
        text = text.lower()
    toks = _TOKEN_RE.findall(text)
    if stopwords:
        toks = [t for t in toks if t not in stopwords]
    return toks

# ----------------------------
# Data loading
# ----------------------------
def _open_maybe_gz(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "rt", encoding="utf-8", errors="ignore")

def iter_corpus(path: Path) -> Iterable[Tuple[str, str]]:
    """
    Yields (doc_id, text). Supports:
    - JSONL/JSONL.GZ: {"id": "...", "text": "..."}
    - TSV/TSV.GZ: two columns without header: id<TAB>text
    """
    p = Path(path)
    if p.suffix in {".jsonl"} or p.suffixes[-2:] == [".jsonl", ".gz"]:
        with _open_maybe_gz(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                yield str(obj["id"]), str(obj["text"])
        return

    # TSV without header (2 columns: id, text)
    with _open_maybe_gz(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) < 2:
                continue
            doc_id, text = parts[0], parts[1]
            yield doc_id, text


def iter_queries(path: Path) -> Iterable[Tuple[str, str]]:
    """
    Yields (qid, query).
    Supports TSV/TSV.GZ without header: qid<TAB>query
    """
    with _open_maybe_gz(Path(path)) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) < 2:
                continue
            qid, query = parts[0], parts[1]
            yield qid, query

def read_qrels(path: Path) -> Dict[str, Dict[str, int]]:
    """
    TREC qrels: qid <ignored> docid rel
    Returns: {qid: {docid: rel}}
    """
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    with _open_maybe_gz(Path(path)) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, _, docid, rel = parts[0], parts[1], parts[2], int(parts[3])
            qrels[qid][docid] = rel
    return qrels

# ----------------------------
# Index + BM25
# ----------------------------
@dataclass
class BM25Index:
    postings: Dict[str, Dict[str, int]]  # term -> {doc_id: tf}
    doc_len: Dict[str, int]              # doc_id -> length (in tokens)
    avg_dl: float
    N: int                               # number of docs
    df: Dict[str, int]                   # term -> document frequency
    idf: Dict[str, float]                # term -> idf (BM25 variant)

    k1: float = 0.9
    b: float = 0.4

    @classmethod
    def build(cls, corpus: Iterable[Tuple[str, List[str]]], k1: float = 0.9, b: float = 0.4) -> "BM25Index":
        postings: Dict[str, Dict[str, int]] = defaultdict(dict)
        doc_len: Dict[str, int] = {}
        N = 0

        for doc_id, toks in corpus:
            N += 1
            doc_len[doc_id] = len(toks)
            tf = Counter(toks)
            for term, f in tf.items():
                postings[term][doc_id] = f

        df = {t: len(postings[t]) for t in postings}
        avg_dl = (sum(doc_len.values()) / N) if N > 0 else 0.0

        # BM25 idf: log( (N - df + 0.5) / (df + 0.5) + 1 )
        idf = {}
        for t, d in df.items():
            idf[t] = math.log((N - d + 0.5) / (d + 0.5) + 1.0)

        return cls(
            postings=postings, doc_len=doc_len, avg_dl=avg_dl, N=N, df=df, idf=idf, k1=k1, b=b
        )

    def score_query(self, query_toks: List[str]) -> Dict[str, float]:
        """
        Score all candidate documents that contain at least one query term.
        """
        if not query_toks:
            return {}

        # gather candidates
        candidates: Set[str] = set()
        for t in set(query_toks):
            if t in self.postings:
                candidates.update(self.postings[t].keys())

        scores: Dict[str, float] = defaultdict(float)
        for doc_id in candidates:
            dl = self.doc_len[doc_id]
            K = self.k1 * ((1 - self.b) + self.b * (dl / self.avg_dl)) if self.avg_dl > 0 else self.k1

            s = 0.0
            for t in query_toks:
                postings_t = self.postings.get(t)
                if not postings_t:
                    continue
                f = postings_t.get(doc_id, 0)
                if f == 0:
                    continue
                idf_t = self.idf.get(t, 0.0)
                # BM25 term contribution
                s += idf_t * (f * (self.k1 + 1.0)) / (f + K)
            scores[doc_id] = s
        return scores

# ----------------------------
# Ranking / evaluation helpers
# ----------------------------
def topk(scores: Dict[str, float], k: int) -> List[Tuple[str, float]]:
    # stable tie-break by doc_id for reproducibility
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]

def dcg_at_k(rels: List[int], k: int) -> float:
    dcg = 0.0
    for i, rel in enumerate(rels[:k], start=1):
        dcg += (2**rel - 1) / math.log2(i + 1)
    return dcg

def ndcg_at_k(rels: List[int], k: int) -> float:
    ideal = sorted(rels, reverse=True)
    denom = dcg_at_k(ideal, k)
    return dcg_at_k(rels, k) / denom if denom > 0 else 0.0

def average_precision_at_k(rels: List[int], k: int) -> float:
    num_hits = 0
    ap_sum = 0.0
    for i, rel in enumerate(rels[:k], start=1):
        if rel > 0:
            num_hits += 1
            ap_sum += num_hits / i
    return (ap_sum / num_hits) if num_hits > 0 else 0.0

# ----------------------------
# Pipeline
# ----------------------------
def build_index_from_corpus(
    corpus_path: Path,
    lowercase: bool = True,
    stopwords: Optional[Set[str]] = None,
    k1: float = 0.9,
    b: float = 0.4,
) -> Tuple[BM25Index, Dict[str, List[str]]]:
    """
    Builds BM25 index with progress bar.
    """
    doc_tokens: Dict[str, List[str]] = {}

    # Count total lines (optional but helps tqdm estimate)
    total = sum(1 for _ in _open_maybe_gz(corpus_path))
    with _open_maybe_gz(corpus_path) as f:
        for line in tqdm(f, total=total, desc="Building index", unit="docs"):
            line = line.strip()
            if not line:
                continue
            # handle both JSONL and TSV automatically
            if corpus_path.suffix in {".jsonl"} or corpus_path.suffixes[-2:] == [".jsonl", ".gz"]:
                obj = json.loads(line)
                did, text = str(obj["id"]), str(obj["text"])
            else:
                parts = line.split("\t", 1)
                if len(parts) < 2:
                    continue
                did, text = parts[0], parts[1]
            doc_tokens[did] = tokenize(text, lowercase=lowercase, stopwords=stopwords)

    index = BM25Index.build(doc_tokens.items(), k1=k1, b=b)
    return index, doc_tokens


def search(
    index: BM25Index,
    queries_path: Path,
    lowercase: bool = True,
    stopwords: Optional[Set[str]] = None,
    topk_n: int = 100
) -> List[Tuple[str, List[Tuple[str, float]]]]:
    """
    Runs BM25 retrieval with progress bar.
    """
    results: List[Tuple[str, List[Tuple[str, float]]]] = []
    total = sum(1 for _ in _open_maybe_gz(queries_path))
    with _open_maybe_gz(queries_path) as f:
        for line in tqdm(f, total=total, desc="Retrieving", unit="queries"):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) < 2:
                continue
            qid, qtext = parts[0], parts[1]
            q_toks = tokenize(qtext, lowercase=lowercase, stopwords=stopwords)
            scores = index.score_query(q_toks)
            ranked = topk(scores, topk_n)
            results.append((qid, ranked))
    return results


def write_run_trec(run_path: Path, run: List[Tuple[str, List[Tuple[str, float]]]], run_tag: str = "bm25"):
    """
    TREC run format:
    qid Q0 docid rank score tag
    """
    run_path.parent.mkdir(parents=True, exist_ok=True)
    with open(run_path, "w", encoding="utf-8") as f:
        for qid, ranked in run:
            for rank, (docid, score) in enumerate(ranked, start=1):
                f.write(f"{qid}\tQ0\t{docid}\t{rank}\t{score:.6f}\t{run_tag}\n")

def evaluate_run(run: List[Tuple[str, List[Tuple[str, float]]]], qrels: Dict[str, Dict[str, int]], k: int = 10) -> Dict[str, float]:
    """
    Computes MAP@k, NDCG@k, Recall@100 across the provided run.
    """
    map_list, ndcg_list, rec_list = [], [], []
    for qid, ranked in run:
        # build relevance list aligned with ranking
        rels = []
        relevant_docs = {d for d, r in qrels.get(qid, {}).items() if r > 0}
        for docid, _ in ranked:
            rels.append(qrels.get(qid, {}).get(docid, 0))

        # metrics
        ap = average_precision_at_k(rels, k)
        ndcg = ndcg_at_k(rels, k)
        # Recall@100 uses top100 ranked docs
        top100 = {docid for docid, _ in ranked[:100]}
        hits = len(relevant_docs & top100)
        recall = (hits / len(relevant_docs)) if relevant_docs else 0.0

        map_list.append(ap)
        ndcg_list.append(ndcg)
        rec_list.append(recall)

    def _avg(xs): return sum(xs) / len(xs) if xs else 0.0
    return {
        f"MAP@{k}": _avg(map_list),
        f"NDCG@{k}": _avg(ndcg_list),
        "Recall@100": _avg(rec_list),
        "NumQueries": len(run),
    }

# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Simple BM25 baseline IR system.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--lowercase", action="store_true", default=True)
    common.add_argument("--no-lowercase", action="store_false", dest="lowercase")
    common.add_argument("--stopwords", action="store_true", default=False, help="Enable a small default stopword list.")
    common.add_argument("--k1", type=float, default=0.9)
    common.add_argument("--b", type=float, default=0.4)

    # search
    p_search = sub.add_parser("search", parents=[common], help="Build index and retrieve.")
    p_search.add_argument("--corpus", type=Path, required=True)
    p_search.add_argument("--queries", type=Path, required=True)
    p_search.add_argument("--run", type=Path, required=True, help="Path to write TREC run file.")
    p_search.add_argument("--topk", type=int, default=100, help="Top-k per query.")
    p_search.add_argument("--tag", type=str, default="bm25", help="Run tag in TREC output.")

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate a run against qrels.")
    p_eval.add_argument("--run", type=Path, required=True)
    p_eval.add_argument("--qrels", type=Path, required=True)
    p_eval.add_argument("--k", type=int, default=10)

    args = parser.parse_args()

    if args.cmd == "search":
        sw = DEFAULT_STOPWORDS if args.stopwords else None
        index, _ = build_index_from_corpus(args.corpus, lowercase=args.lowercase, stopwords=sw, k1=args.k1, b=args.b)
        run = search(index, args.queries, lowercase=args.lowercase, stopwords=sw, topk_n=args.topk)
        write_run_trec(args.run, run, run_tag=args.tag)
        print(f"Indexed {index.N} docs | avg_dl={index.avg_dl:.1f} | k1={index.k1} b={index.b}")
        print(f"Wrote run to: {args.run}")

    elif args.cmd == "eval":
        # read run
        run_map: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        with open(args.run, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                qid, _q0, docid, rank, score, _tag = line.split()
                run_map[qid].append((docid, float(score)))
        # ensure correct order
        run = [(qid, sorted(items, key=lambda x: (-x[1], x[0]))) for qid, items in run_map.items()]
        qrels = read_qrels(args.qrels)
        metrics = evaluate_run(run, qrels, k=args.k)
        for k_, v_ in metrics.items():
            print(f"{k_}: {v_:.4f}" if isinstance(v_, float) else f"{k_}: {v_}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
