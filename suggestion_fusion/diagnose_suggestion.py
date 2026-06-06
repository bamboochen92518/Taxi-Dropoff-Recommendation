"""分析 suggestion table 對各 split 的召回率。

核心問題: suggestion table 的高召回率是資料本身的特性，
還是只在 val 上偶然成立？

做法: 對 train、val、test 三個 split 分別分析，
若三者召回率相近，代表這是 suggestion table 的固有性質，
與 val 答案無關。

跑法: python diagnose_suggestion.py
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collections import defaultdict
from data_loader import load_split
from read_parquet import SUGG_PATH, read_parquet_cols


def analyze_split(split_name: str, split_df, sugg_user_ends: dict) -> None:
    arr = split_df[["uid_hash", "end_latlng"]].values
    n = len(arr)
    users = set(split_df["uid_hash"].unique())

    n_in_sugg = sum(
        1 for uid, end in arr
        if end in sugg_user_ends.get(uid, set())
    )
    n_user_in_sugg = len(users & set(sugg_user_ends.keys()))

    counts = [
        len(sugg_user_ends[uid])
        for uid in users
        if uid in sugg_user_ends
    ]
    counts_sorted = sorted(counts)
    median = counts_sorted[len(counts_sorted) // 2] if counts_sorted else 0
    mean = sum(counts) / len(counts) if counts else 0
    max_c = max(counts) if counts else 0

    print(f"\n  [{split_name}]  n={n:,} queries, {len(users):,} users")
    print(f"    User 在 suggestion 中:     {n_user_in_sugg:,} / {len(users):,}"
          f"  ({n_user_in_sugg/len(users):.1%})")
    print(f"    Answer 在 suggestion 中:   {n_in_sugg:,} / {n:,}"
          f"  ({n_in_sugg/n:.1%})  ← 召回率")
    print(f"    每人候選數 (median/mean/max): "
          f"{median} / {mean:.1f} / {max_c}")


def main():
    print("載入資料 ...")
    train = load_split("train")
    val   = load_split("val")
    test  = load_split("test")
    sugg  = read_parquet_cols(SUGG_PATH, ["uid_hash", "end_latlng"])

    sugg_user_ends: dict = defaultdict(set)
    for uid, end in zip(sugg["uid_hash"].values, sugg["end_latlng"].values):
        sugg_user_ends[uid].add(end)
    sugg_user_ends = dict(sugg_user_ends)

    print(f"  suggestion: {len(sugg):,} rows, "
          f"{len(sugg_user_ends):,} unique users")

    print("\n" + "=" * 60)
    print("Suggestion Table 召回率（三個 split 對比）")
    print("=" * 60)
    print("  若三者相近 → 召回率是 suggestion table 的固有特性，")
    print("  與 val/test 答案無關，不存在資料洩漏。")

    analyze_split("train", train, sugg_user_ends)
    analyze_split("val  ", val,   sugg_user_ends)
    analyze_split("test ", test,  sugg_user_ends)

    print("\n" + "=" * 60)
    print("Suggestion vs Training 的資訊量對比")
    print("=" * 60)

    train_user_ends: dict = defaultdict(set)
    for uid, end in zip(train["uid_hash"].values, train["end_latlng"].values):
        train_user_ends[uid].add(end)

    extra_counts = []
    for uid, sugg_dests in sugg_user_ends.items():
        train_dests = train_user_ends.get(uid, set())
        extra = len(sugg_dests - train_dests)
        extra_counts.append(extra)

    extra_sorted = sorted(extra_counts)
    print(f"  Train users 在 suggestion 中「超出 training」的目的地數量:")
    print(f"    median: {extra_sorted[len(extra_sorted)//2]}")
    print(f"    mean:   {sum(extra_counts)/len(extra_counts):.1f}")
    print(f"    > 0 的比例: "
          f"{sum(1 for x in extra_counts if x > 0)/len(extra_counts):.1%}")



if __name__ == "__main__":
    main()