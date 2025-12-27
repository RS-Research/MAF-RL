# data/preprocess_movielens1m.py
# -*- coding: utf-8 -*-
"""
Preprocess MovieLens-1M into RecBole format for MAF-RL.

Input (raw)
-----------
Expected MovieLens-1M files (from GroupLens):
  data/raw/ml-1m/ratings.dat
  data/raw/ml-1m/movies.dat
  data/raw/ml-1m/users.dat

Output (RecBole)
----------------
Writes into:
  data/ml-1m/ml-1m.inter
  data/ml-1m/ml-1m.item
  data/ml-1m/ml-1m.user

RecBole format notes
--------------------
- Files are tab-separated with a header row.
- Column typing is inferred by RecBole via config. Use consistent field names:
    user_id, item_id, timestamp
  Optional:
    rating, genre, gender, age, occupation, zip_code

Usage
-----
# From repo root:
python data/preprocess_movielens1m.py --raw_dir data/raw/ml-1m --out_dir data/ml-1m

Optional arguments:
  --min_user_inter 5
  --min_item_inter 5
  --keep_rating (keep rating column in .inter)
  --seed 42

License/Attribution
-------------------
MovieLens data is subject to GroupLens licensing. This script does not redistribute data.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


@dataclass
class Args:
    raw_dir: str
    out_dir: str
    min_user_inter: int
    min_item_inter: int
    keep_rating: bool
    seed: int


def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Preprocess MovieLens-1M for RecBole (MAF-RL)")
    p.add_argument("--raw_dir", type=str, default="data/raw/ml-1m", help="Path to raw ml-1m directory")
    p.add_argument("--out_dir", type=str, default="data/ml-1m", help="Output directory for RecBole files")
    p.add_argument("--min_user_inter", type=int, default=5, help="Minimum interactions per user to keep")
    p.add_argument("--min_item_inter", type=int, default=5, help="Minimum interactions per item to keep")
    p.add_argument("--keep_rating", action="store_true", help="Keep rating column in .inter")
    p.add_argument("--seed", type=int, default=42, help="Random seed (for determinism where applicable)")
    ns = p.parse_args()
    return Args(
        raw_dir=ns.raw_dir,
        out_dir=ns.out_dir,
        min_user_inter=ns.min_user_inter,
        min_item_inter=ns.min_item_inter,
        keep_rating=bool(ns.keep_rating),
        seed=int(ns.seed),
    )


def ensure_exists(path: Path, what: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")


def read_ratings(ratings_path: Path) -> pd.DataFrame:
    # ratings.dat format: UserID::MovieID::Rating::Timestamp
    df = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={"user_id": int, "item_id": int, "rating": int, "timestamp": int},
    )
    return df


def read_movies(movies_path: Path) -> pd.DataFrame:
    # movies.dat format: MovieID::Title::Genres
    df = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        header=None,
        names=["item_id", "title", "genre"],
        encoding="latin-1",
    )
    # Keep only fields we will export by default
    df["item_id"] = df["item_id"].astype(int)
    df["genre"] = df["genre"].astype(str)
    return df[["item_id", "genre"]]


def read_users(users_path: Path) -> pd.DataFrame:
    # users.dat format: UserID::Gender::Age::Occupation::Zip-code
    df = pd.read_csv(
        users_path,
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        encoding="latin-1",
    )
    df["user_id"] = df["user_id"].astype(int)
    df["gender"] = df["gender"].astype(str)
    df["age"] = df["age"].astype(int)
    df["occupation"] = df["occupation"].astype(int)
    df["zip_code"] = df["zip_code"].astype(str)
    return df


def iterative_kcore_filter(
    inter: pd.DataFrame, min_user_inter: int, min_item_inter: int
) -> pd.DataFrame:
    """
    Iteratively filter to satisfy user/item minimum interaction counts.
    """
    prev_shape = None
    cur = inter
    while prev_shape != cur.shape:
        prev_shape = cur.shape
        u_cnt = cur["user_id"].value_counts()
        i_cnt = cur["item_id"].value_counts()
        keep_users = u_cnt[u_cnt >= min_user_inter].index
        keep_items = i_cnt[i_cnt >= min_item_inter].index
        cur = cur[cur["user_id"].isin(keep_users) & cur["item_id"].isin(keep_items)]
    return cur


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)


def main() -> None:
    args = parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    ratings_path = raw_dir / "ratings.dat"
    movies_path = raw_dir / "movies.dat"
    users_path = raw_dir / "users.dat"

    ensure_exists(ratings_path, "ratings.dat")
    ensure_exists(movies_path, "movies.dat")
    ensure_exists(users_path, "users.dat")

    print(f"[INFO] Reading raw MovieLens-1M from: {raw_dir}")
    ratings = read_ratings(ratings_path)
    movies = read_movies(movies_path)
    users = read_users(users_path)

    # Basic cleanup: drop duplicates, keep time order
    ratings = ratings.drop_duplicates(subset=["user_id", "item_id", "timestamp"])
    ratings = ratings.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # Optional k-core filtering
    if args.min_user_inter > 1 or args.min_item_inter > 1:
        before = len(ratings)
        ratings = iterative_kcore_filter(ratings, args.min_user_inter, args.min_item_inter)
        after = len(ratings)
        print(f"[INFO] K-core filter: interactions {before} -> {after}")

    # Keep only users/items present after filtering
    keep_users = set(ratings["user_id"].unique().tolist())
    keep_items = set(ratings["item_id"].unique().tolist())
    users = users[users["user_id"].isin(keep_users)].reset_index(drop=True)
    movies = movies[movies["item_id"].isin(keep_items)].reset_index(drop=True)

    # Build RecBole output names
    dataset_name = "ml-1m"
    inter_out = out_dir / f"{dataset_name}.inter"
    item_out = out_dir / f"{dataset_name}.item"
    user_out = out_dir / f"{dataset_name}.user"

    # Prepare .inter
    inter_cols = ["user_id", "item_id", "timestamp"]
    if args.keep_rating:
        inter_cols.insert(2, "rating")  # user_id, item_id, rating, timestamp
    inter_df = ratings[inter_cols].copy()

    # Prepare .item (minimal)
    item_df = movies[["item_id", "genre"]].copy()

    # Prepare .user (minimal)
    user_df = users[["user_id", "gender", "age", "occupation", "zip_code"]].copy()

    print(f"[INFO] Writing RecBole files to: {out_dir}")
    write_tsv(inter_df, inter_out)
    write_tsv(item_df, item_out)
    write_tsv(user_df, user_out)

    print("[DONE] RecBole-formatted MovieLens-1M created:")
    print(f"  - {inter_out}")
    print(f"  - {item_out}")
    print(f"  - {user_out}")
    print("\nNext:")
    print("  1) Ensure your config uses:")
    print("       dataset: ml-1m")
    print("       data_path: data/")
    print("  2) Run training:")
    print("       python -m maf_rl.scripts.run_recbole_train --config configs/movielens1m.yaml")


if __name__ == "__main__":
    main()
