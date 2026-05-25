"""Train / Val / Test data loader,依 created_at 時間切分 80/10/10。

用法:
    from data_loader import load_split
    df = load_split("train")  # or "val", "test"
"""
from functools import lru_cache
from typing import Literal

import pandas as pd

from read_parquet import TRAIN_PATH, read_parquet_cols

Split = Literal["train", "val", "test"]

TRAIN_RATIO = 0.75
VAL_RATIO = 0.1
# test = 剩下的 0.15


@lru_cache(maxsize=1)
def _load_sorted() -> pd.DataFrame:
    """讀整份 training data,按 created_at 排序後快取在記憶體。"""
    df = read_parquet_cols(TRAIN_PATH)
    df = df.sort_values("created_at", kind="mergesort").reset_index(drop=True)
    return df


@lru_cache(maxsize=1)
def _split_bounds() -> tuple[int, int]:
    """回傳 (train_end, val_end) 兩個 row index 切點。"""
    n = len(_load_sorted())
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    return train_end, val_end


def load_split(split: Split, columns: list[str] | None = None) -> pd.DataFrame:
    """讀取指定 split 的資料。

    Args:
        split: "train" / "val" / "test"
        columns: 若指定,只回傳這些欄位
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be 'train'/'val'/'test', got {split!r}")

    df = _load_sorted()
    train_end, val_end = _split_bounds()

    if split == "train":
        out = df.iloc[:train_end]
    elif split == "val":
        out = df.iloc[train_end:val_end]
    else:
        out = df.iloc[val_end:]

    if columns is not None:
        out = out[columns]
    return out.reset_index(drop=True)


def split_summary() -> None:
    """印出三組 split 的筆數、時間範圍、user 數,方便檢查。"""
    for s in ("train", "val", "test"):
        df = load_split(s, columns=["uid_hash", "created_at"])
        print(f"\n=== {s} ===")
        print(f"  rows  : {len(df):,}")
        print(f"  time  : {df['created_at'].min()} ~ {df['created_at'].max()}")
        print(f"  users : {df['uid_hash'].nunique():,}")


if __name__ == "__main__":
    split_summary()
