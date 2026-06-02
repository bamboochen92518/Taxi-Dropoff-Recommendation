"""Top-K 推薦評估: Hit@K / MRR / NDCG@K,並支援 per-segment 拆解。"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EvalResult:
    k: int
    n: int
    hit_at_k: float
    mrr: float
    ndcg_at_k: float

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            f"Hit@{self.k}": round(self.hit_at_k, 4),
            "MRR": round(self.mrr, 4),
            f"NDCG@{self.k}": round(self.ndcg_at_k, 4),
        }


def _rank_of_truth(preds: list[str], truth: str) -> int:
    """truth 在 preds 中的 1-indexed 排名,沒命中回 0。"""
    for i, p in enumerate(preds, start=1):
        if p == truth:
            return i
    return 0


def evaluate(predictions: list[list[str]], truths: list[str], k: int) -> EvalResult:
    assert len(predictions) == len(truths), "predictions / truths 長度不一致"
    n = len(truths)
    if n == 0:
        return EvalResult(k=k, n=0, hit_at_k=0.0, mrr=0.0, ndcg_at_k=0.0)

    hits = 0
    mrr_sum = 0.0
    ndcg_sum = 0.0
    log2 = math.log(2)
    for preds, truth in zip(predictions, truths):
        rank = _rank_of_truth(preds[:k], truth)
        if rank > 0:
            hits += 1
            mrr_sum += 1.0 / rank
            ndcg_sum += log2 / math.log(rank + 1)  # = 1/log2(rank+1)
    return EvalResult(
        k=k,
        n=n,
        hit_at_k=hits / n,
        mrr=mrr_sum / n,
        ndcg_at_k=ndcg_sum / n,
    )


def evaluate_by_segment(
    predictions: list[list[str]],
    truths: list[str],
    segments: pd.Series,
    k: int,
) -> pd.DataFrame:
    """按 segments (例如 hour_type / is_holiday / user_freq_bucket) 分組評估。"""
    preds_arr = np.array([p[:k] + [""] * (k - len(p[:k])) for p in predictions], dtype=object)
    truths_arr = np.array(truths, dtype=object)
    segs = segments.values

    rows = []
    for seg_val in pd.unique(segs):
        mask = segs == seg_val
        sub_preds = preds_arr[mask].tolist()
        sub_truths = truths_arr[mask].tolist()
        res = evaluate(sub_preds, sub_truths, k=k)
        rows.append({"segment": seg_val, **res.as_dict()})
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def user_freq_bucket(train_df: pd.DataFrame, query_df: pd.DataFrame) -> pd.Series:
    """根據 train 中該 user 的歷史筆數,把 query 分到 bucket。"""
    counts = train_df.groupby("uid_hash").size()
    q_counts = query_df["uid_hash"].map(counts).fillna(0).astype(int)
    bins = [-1, 0, 5, 20, 100, 10_000_000]
    labels = ["new(0)", "1-5", "6-20", "21-100", "100+"]
    return pd.cut(q_counts, bins=bins, labels=labels)


def trip_count_segment(
    train_df: pd.DataFrame, query_df: pd.DataFrame, min_trips: int = 20,
) -> pd.Series:
    """以 train 中該 user 的歷史筆數為基準,把 query 切成 >=N / <N 兩段。"""
    counts = train_df.groupby("uid_hash").size()
    q_counts = query_df["uid_hash"].map(counts).fillna(0).astype(int)
    return pd.Series(
        np.where(q_counts >= min_trips, f">={min_trips}", f"<{min_trips}"),
        name="trip_segment",
        index=query_df.index,
    )


def evaluate_trip_segment(
    predictions: list[list[str]],
    truths: list[str],
    train_df: pd.DataFrame,
    query_df: pd.DataFrame,
    k: int,
    min_trips: int = 20,
) -> pd.DataFrame:
    """便利包裝: 直接吐出 >=N / <N 兩段的 evaluate_by_segment 結果。"""
    seg = trip_count_segment(train_df, query_df, min_trips=min_trips)
    return evaluate_by_segment(predictions, truths, seg, k=k)
