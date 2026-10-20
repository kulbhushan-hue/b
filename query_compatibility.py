"""Query the local z/OS compatibility matrix using retrieval-augmented search.

Run ``py query_compatibility.py "Which Broadcom products support z/OS 3.2?"``.
Set WATSONX_APIKEY and WATSONX_PROJECT_ID (or WATSONX_SPACE_ID) to optionally
generate a written answer with watsonx.ai. Without credentials, matching rows
are printed directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from compatibility_db import CompatibilityDB
from rag_engine import RagIndex, generate_answer, watsonx_configured


def build_parser() -> argparse.ArgumentParser:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Search the local z/OS compatibility data."
    )
    parser.add_argument("query", nargs="+", help="Question or search phrase")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_dir / "data",
        help="Directory containing vendor *_compat.csv files (default: ./data)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Maximum number of matching rows (default: 12)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Print matching rows without calling watsonx.ai",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")

    question = " ".join(args.query)
    db = CompatibilityDB.load(str(args.data_dir))
    matches = RagIndex.build(db.df).retrieve(question, top_k=args.top_k)

    if matches.empty:
        print("No matching rows found. Try a vendor or product name.")
        return

    if not args.no_llm and watsonx_configured():
        print(generate_answer(question, matches))
        print("\nMatching source rows:")
    elif not args.no_llm:
        print(
            "watsonx.ai is not configured; showing matching rows. Set "
            "WATSONX_APIKEY and WATSONX_PROJECT_ID (or WATSONX_SPACE_ID) "
            "to generate a written answer.\n"
        )

    columns = [
        "vendor",
        "product",
        "vrm",
        "zos32",
        "zos31",
        "zos25",
        "ptf_required",
        "link",
    ]
    available_columns = [column for column in columns if column in matches.columns]
    print(matches[available_columns].to_string(index=False))


if __name__ == "__main__":
    main()
