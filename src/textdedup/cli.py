"""Command line entry point: dedup a newline-delimited text file."""

from __future__ import annotations

import argparse
import json
import sys

from .lsh import LSHIndex


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="textdedup",
        description="Find near-duplicate lines in a text file via MinHash + LSH.",
    )
    parser.add_argument("input", help="UTF-8 file, one document per line")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="exact Jaccard cutoff for reporting a pair")
    parser.add_argument("--num-perm", type=int, default=128)
    parser.add_argument("--bands", type=int, default=32)
    parser.add_argument("-k", type=int, default=3, help="word shingle size")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args(argv)

    index = LSHIndex(num_perm=args.num_perm, bands=args.bands,
                     k=args.k, threshold=args.threshold)
    count = 0
    with open(args.input, encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if text:
                index.add(f"line-{count}", text)
                count += 1
    pairs = index.similar_pairs()
    if args.json:
        print(json.dumps({
            "documents": count,
            "pairs": [
                {"a": a, "b": b, "jaccard": round(sim, 6)} for a, b, sim in pairs
            ],
        }, indent=2))
    else:
        print(f"documents: {count}")
        print(f"near-duplicate pairs (jaccard >= {args.threshold}): {len(pairs)}")
        for a, b, sim in pairs:
            print(f"  {a} <-> {b}  jaccard={sim:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
