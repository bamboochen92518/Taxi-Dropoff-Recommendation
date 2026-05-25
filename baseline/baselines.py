"""下車地點推薦 baseline 模型。

統一介面:
    model.fit(train_df)
    model.predict_topk(query_df, k) -> list[list[str]]
        對 query_df 每一列回傳 top-k 個 end_latlng 字串。

注意: 候選清單會從 start_latlng 排除 (上車點不會是下車點)。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Protocol

import pandas as pd
from tqdm import tqdm


class Recommender(Protocol):
    def fit(self, train_df: pd.DataFrame) -> "Recommender": ...
    def predict_topk(self, query_df: pd.DataFrame, k: int) -> list[list[str]]: ...


def _topk_from_counter(c: Counter, k: int, exclude: str | None = None) -> list[str]:
    if exclude is None:
        return [x for x, _ in c.most_common(k)]
    out: list[str] = []
    for x, _ in c.most_common(k + 1):
        if x == exclude:
            continue
        out.append(x)
        if len(out) == k:
            break
    return out


# ---------------------------------------------------------------------------
# 1. 全體熱門
# ---------------------------------------------------------------------------
class GlobalPopularity:
    name = "GlobalPopularity"

    def __init__(self) -> None:
        self.global_top: list[str] = []
        self._counter: Counter = Counter()

    def fit(self, train_df: pd.DataFrame) -> "GlobalPopularity":
        self._counter = Counter(train_df["end_latlng"].tolist())
        # 預先快取 top-K 候選 (多備幾個以便排除 start_latlng 後仍有足夠候選)。
        self.global_top = [x for x, _ in self._counter.most_common(50)]
        return self

    def predict_topk(self, query_df: pd.DataFrame, k: int) -> list[list[str]]:
        # 直接掃預先排序好的 global_top,逐列把該筆的 start_latlng 過濾掉即可,
        # 避免每列都重新呼叫 Counter.most_common (這是 O(N log K),N = 唯一下車點數)。
        global_top = self.global_top
        out: list[list[str]] = []
        for start in tqdm(query_df["start_latlng"].values, desc=f"{self.name}.predict", leave=False):
            picks: list[str] = []
            for x in global_top:
                if x == start:
                    continue
                picks.append(x)
                if len(picks) == k:
                    break
            out.append(picks)
        return out


# ---------------------------------------------------------------------------
# 2. 該 user 歷史最常去 (fallback: global)
# ---------------------------------------------------------------------------
class UserHistory:
    name = "UserHistory"

    def __init__(self) -> None:
        self.user_counter: dict[int, Counter] = {}
        self.global_top: list[str] = []

    def fit(self, train_df: pd.DataFrame) -> "UserHistory":
        gc: Counter = Counter()
        uc: dict[int, Counter] = defaultdict(Counter)
        it = zip(train_df["uid_hash"].values, train_df["end_latlng"].values)
        for uid, end in tqdm(it, total=len(train_df), desc=f"{self.name}.fit", leave=False):
            uc[uid][end] += 1
            gc[end] += 1
        self.user_counter = dict(uc)
        self.global_top = [x for x, _ in gc.most_common(50)]
        return self

    def predict_topk(self, query_df: pd.DataFrame, k: int) -> list[list[str]]:
        out: list[list[str]] = []
        it = zip(query_df["uid_hash"].values, query_df["start_latlng"].values)
        for uid, start in tqdm(it, total=len(query_df), desc=f"{self.name}.predict", leave=False):
            picks: list[str] = []
            seen: set[str] = set()
            uc = self.user_counter.get(uid)
            if uc is not None:
                for x, _ in uc.most_common():
                    if x == start or x in seen:
                        continue
                    picks.append(x); seen.add(x)
                    if len(picks) == k:
                        break
            if len(picks) < k:
                for x in self.global_top:
                    if x == start or x in seen:
                        continue
                    picks.append(x); seen.add(x)
                    if len(picks) == k:
                        break
            out.append(picks)
        return out


# ---------------------------------------------------------------------------
# 3. 該 user × context 最常去 (fallback: user → global)
# ---------------------------------------------------------------------------
class UserContextHistory:
    """context = (hour_type, is_holiday)。"""
    name = "UserContextHistory"

    def __init__(self) -> None:
        self.uc_ctx: dict[tuple, Counter] = {}
        self.uc_user: dict[int, Counter] = {}
        self.global_top: list[str] = []

    def fit(self, train_df: pd.DataFrame) -> "UserContextHistory":
        gc: Counter = Counter()
        uc_user: dict[int, Counter] = defaultdict(Counter)
        uc_ctx: dict[tuple, Counter] = defaultdict(Counter)
        cols = train_df[["uid_hash", "end_latlng", "hour_type", "is_holiday"]].values
        for uid, end, hour, holiday in tqdm(cols, desc=f"{self.name}.fit", leave=False):
            uc_user[uid][end] += 1
            uc_ctx[(uid, hour, holiday)][end] += 1
            gc[end] += 1
        self.uc_user = dict(uc_user)
        self.uc_ctx = dict(uc_ctx)
        self.global_top = [x for x, _ in gc.most_common(50)]
        return self

    def predict_topk(self, query_df: pd.DataFrame, k: int) -> list[list[str]]:
        out: list[list[str]] = []
        cols = query_df[["uid_hash", "start_latlng", "hour_type", "is_holiday"]].values
        for uid, start, hour, holiday in tqdm(cols, desc=f"{self.name}.predict", leave=False):
            picks: list[str] = []
            seen: set[str] = set()
            # tier 1: user × context
            c = self.uc_ctx.get((uid, hour, holiday))
            if c is not None:
                for x, _ in c.most_common():
                    if x == start or x in seen:
                        continue
                    picks.append(x); seen.add(x)
                    if len(picks) == k: break
            # tier 2: user only
            if len(picks) < k:
                c = self.uc_user.get(uid)
                if c is not None:
                    for x, _ in c.most_common():
                        if x == start or x in seen:
                            continue
                        picks.append(x); seen.add(x)
                        if len(picks) == k: break
            # tier 3: global
            if len(picks) < k:
                for x in self.global_top:
                    if x == start or x in seen:
                        continue
                    picks.append(x); seen.add(x)
                    if len(picks) == k: break
            out.append(picks)
        return out


# ---------------------------------------------------------------------------
# 4. start_latlng → end_latlng 共現 (fallback: global)
# ---------------------------------------------------------------------------
class StartEndCoOccurrence:
    name = "StartEndCoOccurrence"

    def __init__(self) -> None:
        self.start_counter: dict[str, Counter] = {}
        self.global_top: list[str] = []

    def fit(self, train_df: pd.DataFrame) -> "StartEndCoOccurrence":
        gc: Counter = Counter()
        sc: dict[str, Counter] = defaultdict(Counter)
        it = zip(train_df["start_latlng"].values, train_df["end_latlng"].values)
        for start, end in tqdm(it, total=len(train_df), desc=f"{self.name}.fit", leave=False):
            sc[start][end] += 1
            gc[end] += 1
        self.start_counter = dict(sc)
        self.global_top = [x for x, _ in gc.most_common(50)]
        return self

    def predict_topk(self, query_df: pd.DataFrame, k: int) -> list[list[str]]:
        out: list[list[str]] = []
        for start in tqdm(query_df["start_latlng"].values, desc=f"{self.name}.predict", leave=False):
            picks: list[str] = []
            seen: set[str] = set()
            c = self.start_counter.get(start)
            if c is not None:
                for x, _ in c.most_common():
                    if x == start or x in seen:
                        continue
                    picks.append(x); seen.add(x)
                    if len(picks) == k: break
            if len(picks) < k:
                for x in self.global_top:
                    if x == start or x in seen:
                        continue
                    picks.append(x); seen.add(x)
                    if len(picks) == k: break
            out.append(picks)
        return out


# ---------------------------------------------------------------------------
# 5. 混合: user history 優先, 不足補 start→end, 再不足補 global
# ---------------------------------------------------------------------------
class HybridUserStartEnd:
    name = "HybridUserStartEnd"

    def __init__(self) -> None:
        self.user_counter: dict[int, Counter] = {}
        self.start_counter: dict[str, Counter] = {}
        self.global_top: list[str] = []

    def fit(self, train_df: pd.DataFrame) -> "HybridUserStartEnd":
        gc: Counter = Counter()
        uc: dict[int, Counter] = defaultdict(Counter)
        sc: dict[str, Counter] = defaultdict(Counter)
        it = zip(
            train_df["uid_hash"].values,
            train_df["start_latlng"].values,
            train_df["end_latlng"].values,
        )
        for uid, start, end in tqdm(it, total=len(train_df), desc=f"{self.name}.fit", leave=False):
            uc[uid][end] += 1
            sc[start][end] += 1
            gc[end] += 1
        self.user_counter = dict(uc)
        self.start_counter = dict(sc)
        self.global_top = [x for x, _ in gc.most_common(50)]
        return self

    def predict_topk(self, query_df: pd.DataFrame, k: int) -> list[list[str]]:
        out: list[list[str]] = []
        cols = query_df[["uid_hash", "start_latlng"]].values
        for uid, start in tqdm(cols, desc=f"{self.name}.predict", leave=False):
            picks: list[str] = []
            seen: set[str] = set()
            # tier 1: user history
            c = self.user_counter.get(uid)
            if c is not None:
                for x, _ in c.most_common():
                    if x == start or x in seen: continue
                    picks.append(x); seen.add(x)
                    if len(picks) == k: break
            # tier 2: start→end
            if len(picks) < k:
                c = self.start_counter.get(start)
                if c is not None:
                    for x, _ in c.most_common():
                        if x == start or x in seen: continue
                        picks.append(x); seen.add(x)
                        if len(picks) == k: break
            # tier 3: global
            if len(picks) < k:
                for x in self.global_top:
                    if x == start or x in seen: continue
                    picks.append(x); seen.add(x)
                    if len(picks) == k: break
            out.append(picks)
        return out


# ---------------------------------------------------------------------------
# 6. Cascade: user×context → user → start→end → global (在 Hybrid 前面加 context tier)
# ---------------------------------------------------------------------------
class HybridContextCascade:
    name = "HybridContextCascade"

    def __init__(self) -> None:
        self.uc_ctx: dict[tuple, Counter] = {}
        self.user_counter: dict[int, Counter] = {}
        self.start_counter: dict[str, Counter] = {}
        self.global_top: list[str] = []

    def fit(self, train_df: pd.DataFrame) -> "HybridContextCascade":
        gc: Counter = Counter()
        uc_user: dict[int, Counter] = defaultdict(Counter)
        uc_ctx: dict[tuple, Counter] = defaultdict(Counter)
        sc: dict[str, Counter] = defaultdict(Counter)
        cols = train_df[["uid_hash", "start_latlng", "end_latlng",
                         "hour_type", "is_holiday"]].values
        for uid, start, end, hour, holiday in tqdm(cols, desc=f"{self.name}.fit", leave=False):
            uc_user[uid][end] += 1
            uc_ctx[(uid, hour, holiday)][end] += 1
            sc[start][end] += 1
            gc[end] += 1
        self.uc_ctx = dict(uc_ctx)
        self.user_counter = dict(uc_user)
        self.start_counter = dict(sc)
        self.global_top = [x for x, _ in gc.most_common(50)]
        return self

    def predict_topk(self, query_df: pd.DataFrame, k: int) -> list[list[str]]:
        out: list[list[str]] = []
        cols = query_df[["uid_hash", "start_latlng", "hour_type", "is_holiday"]].values
        for uid, start, hour, holiday in tqdm(cols, desc=f"{self.name}.predict", leave=False):
            picks: list[str] = []
            seen: set[str] = set()
            # tier 1: user × context
            c = self.uc_ctx.get((uid, hour, holiday))
            if c is not None:
                for x, _ in c.most_common():
                    if x == start or x in seen: continue
                    picks.append(x); seen.add(x)
                    if len(picks) == k: break
            # tier 2: user 全部歷史
            if len(picks) < k:
                c = self.user_counter.get(uid)
                if c is not None:
                    for x, _ in c.most_common():
                        if x == start or x in seen: continue
                        picks.append(x); seen.add(x)
                        if len(picks) == k: break
            # tier 3: start→end
            if len(picks) < k:
                c = self.start_counter.get(start)
                if c is not None:
                    for x, _ in c.most_common():
                        if x == start or x in seen: continue
                        picks.append(x); seen.add(x)
                        if len(picks) == k: break
            # tier 4: global
            if len(picks) < k:
                for x in self.global_top:
                    if x == start or x in seen: continue
                    picks.append(x); seen.add(x)
                    if len(picks) == k: break
            out.append(picks)
        return out


ALL_BASELINES: list[type] = [
    GlobalPopularity,
    UserHistory,
    UserContextHistory,
    StartEndCoOccurrence,
    HybridUserStartEnd,
    HybridContextCascade,
]
