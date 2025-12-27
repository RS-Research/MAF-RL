# data/preprocess_amazon_books.py
# -*- coding: utf-8 -*-
"""
Preprocess Amazon Books (Amazon Reviews) into RecBole format for MAF-RL.

This script converts the commonly used Amazon review dumps into RecBole files:
  data/amazon_books/amazon_books.inter
  data/amazon_books/amazon_books.item
  (optionally) data/amazon_books/amazon_books.user

Supported raw inputs
--------------------
You can use either:
1) JSON Lines reviews file (recommended; common in research dumps):
   - reviews_file: reviews_Books_5.json (or similar)
   Each line contains at least: reviewerID, asin, unixReviewTime
   Optionally: overall (rating)

2) (Optional) JSON Lines metadata file for items:
   - meta_file: meta_Books.json (or similar)
   Each line contains at least: asin
   Optionally: categories, title, brand, description

Output fields
-------------
- .inter (required):
    user_id   item_id   timestamp   (optional: rating)
- .item (optional but recommended):
    item_id   category   (and optionally title)
- .user (optional):
    user_id   (plus any attributes if you have them)

Important
---------
- This script does NOT download Amazon data.
- Amazon data licensing applies; do not redistribute raw files in this repo.
- RecBole will map token ids internally; we keep original string ids in TSV.

Usage
-----
python data/preprocess_amazon_books.py \
  --reviews_file data/raw/amazon_books/reviews_Books_5.json \
  --meta_file data/raw/amazon_books/meta_Books.json \
  --out_dir data/amazon_books \
  --min_user_inter 5 \
  --min_item_inter 5 \
  --keep_rating

If you already created data/amazon_books/, keep dataset name consistent with your YAML:
  dataset: amazon_books
  data_path: data/
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd


@dataclass
class Args:
    reviews_file: str
    meta_file: Optional[str]
    out_dir: str
    min_user_inter: int
    min_item_inter: int
    keep_rating: bool
    max_lines: Optional[int]


def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Preprocess Amazon Books for RecBole (MAF-RL)")
    p.add_argument("--reviews_file", type=str, required=True, help="Path to reviews JSONL file")
    p.add_argument("--meta_file", type=str, default=None, help="Path to meta JSONL file (optional)")
    p.add_argument("--out_dir", type=str, default="data/amazon_books", help="Output directory")
    p.add_argument("--min_user_inter", type=int, default=5, help="Minimum interactions per user to keep")
    p.add_argument("--min_item_inter", type=int, default=5, help="Minimum interactions per item to keep")
    p.add_argument("--keep_rating", action="store_true", help="Keep rating column in .inter")
    p.add_argument("--max_lines", type=int, default=None, help="Optional cap for debugging (read first N lines)")
    ns = p.parse_args()
    return Args(
        reviews_file=ns.reviews_file,
        meta_file=ns.meta_file,
        out_dir=ns.out_dir,
        min_user_inter=int(ns.min_user_inter),
        min_item_inter=int(ns.min_item_inter),
        keep_rating=bool(ns.keep_rating),
        max_lines=int(ns.max_lines) if ns.max_lines is not None else None,
    )


def ensure_exists(path: Path, what: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")


def read_jsonl_reviews(path: Path, max_lines: Optional[int] = None) -> pd.DataFrame:
    """
    Read Amazon review JSONL. Expected keys:
      reviewerID (user), asin (item), unixReviewTime (timestamp)
      optionally overall (rating)
    """
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Minimal required fields
            user = obj.get("reviewerID", None)
            item = obj.get("asin", None)
            ts = obj.get("unixReviewTime", None)
            if user is None or item is None or ts is None:
                continue
            rating = obj.get("overall", None)
            rows.append(
                {
                    "user_id": str(user),
                    "item_id": str(item),
                    "timestamp": int(ts),
                    "rating": float(rating) if rating is not None else None,
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No valid interactions parsed from {path}")
    # Drop duplicates and sort
    df = df.drop_duplicates(subset=["user_id", "item_id", "timestamp"])
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    return df


def read_jsonl_meta(path: Path, max_lines: Optional[int] = None) -> pd.DataFrame:
    """
    Read Amazon meta JSONL. Common keys:
      asin, categories, title
    We extract:
      item_id, category (string), title (optional)
    """
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            asin = obj.get("asin", None)
            if asin is None:
                continue
            category = _extract_category(obj)
            title = obj.get("title", None)
            rows.append(
                {
                    "item_id": str(asin),
                    "category": category,
                    "title": str(title) if title is not None else "",
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["item_id"]).reset_index(drop=True)
    return df


def _extract_category(meta_obj: Dict[str, Any]) -> str:
    """
    Extract a compact category string from common Amazon meta schema.

    - If 'categories' exists and is nested list: take the deepest last label.
    - Else if 'category' exists: use it.
    - Else empty.
    """
    cats = meta_obj.get("categories", None)
    if isinstance(cats, list) and len(cats) > 0:
        # often: [["Books", "Arts & Photography", "Photography & Video"]]
        first = cats[0]
        if isinstance(first, list) and len(first) > 0:
            return str(first[-1])
        return str(first)
    c = meta_obj.get("category", None)
    return str(c) if c is not None else ""


def iterative_kcore_filter(df: pd.DataFrame, min_user_inter: int, min_item_inter: int) -> pd.DataFrame:
    prev_shape = None
    cur = df
    while prev_shape != cur.shape:
        prev_shape = cur.shape
        u_cnt = cur["user_id"].value_counts()
        i_cnt = cur["item_id"].value_counts()
        keep_users = u_cnt[u_cnt >= min_user_inter].index
        keep_items = i_cnt[i_cnt >= min_item_inter].index
        cur = cur[cur["user_id"].isin(keep_users) & cur["item_id"].isin(keep_items)]
    return cur.reset_index(drop=True)


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)


def main() -> None:
    args = parse_args()

    reviews_path = Path(args.reviews_file)
    ensure_exists(reviews_path, "reviews_file")

    meta_path = Path(args.meta_file) if args.meta_file else None
    if meta_path is not None:
        ensure_exists(meta_path, "meta_file")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = "amazon_books"
    inter_out = out_dir / f"{dataset_name}.inter"
    item_out = out_dir / f"{dataset_name}.item"
    user_out = out_dir / f"{dataset_name}.user"

    print(f"[INFO] Reading reviews: {reviews_path}")
    inter_df = read_jsonl_reviews(reviews_path, max_lines=args.max_lines)

    # Filter by k-core if requested
    before = len(inter_df)
    inter_df = iterative_kcore_filter(inter_df, args.min_user_inter, args.min_item_inter)
    after = len(inter_df)
    print(f"[INFO] K-core filter: interactions {before} -> {after}")

    # Keep rating optionally
    if args.keep_rating:
        inter_cols = ["user_id", "item_id", "rating", "timestamp"]
        # fill missing rating with 1.0 (rare but safe), or drop? we fill to keep shape.
        inter_df["rating"] = inter_df["rating"].fillna(1.0)
    else:
        inter_cols = ["user_id", "item_id", "timestamp"]
    inter_df = inter_df[inter_cols].copy()

    # Build item file (optional)
    item_df = None
    if meta_path is not None:
        print(f"[INFO] Reading meta: {meta_path}")
        item_df = read_jsonl_meta(meta_path, max_lines=args.max_lines)
        if not item_df.empty:
            # Keep only items in filtered interactions
            keep_items = set(inter_df["item_id"].unique().tolist())
            item_df = item_df[item_df["item_id"].isin(keep_items)].reset_index(drop=True)

            # Minimal: item_id + category; optionally keep title
            # RecBole will ignore extra columns unless configured.
            item_df = item_df[["item_id", "category", "title"]].copy()

    # Build user file (minimal)
    user_ids = pd.DataFrame({"user_id": inter_df["user_id"].unique()})
    user_ids = user_ids.sort_values("user_id").reset_index(drop=True)

    print(f"[INFO] Writing RecBole files to: {out_dir}")
    write_tsv(inter_df, inter_out)

    if item_df is not None and not item_df.empty:
        write_tsv(item_df, item_out)
    else:
        # Write a minimal item file only if you want strict consistency.
        # Leaving it absent is acceptable if your config does not require load_col.item.
        print("[WARN] meta_file not provided or empty; skipping .item file.")

    # User file is optional; write minimal user list for completeness
    write_tsv(user_ids, user_out)

    print("[DONE] RecBole-formatted Amazon Books created:")
    print(f"  - {inter_out}")
    if item_df is not None and not item_df.empty:
        print(f"  - {item_out}")
    print(f"  - {user_out}")
    print("\nNext:")
    print("  1) Ensure your config uses:")
    print("       dataset: amazon_books")
    print("       data_path: data/")
    print("  2) Run training:")
    print("       python -m maf_rl.scripts.run_recbole_train --config configs/amazon_books.yaml")


if __name__ == "__main__":
    main()
