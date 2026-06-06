"""SuggestionFusion: 用 suggestion table 作為候選集 + training 特徵排序。

核心發現: suggestion table 對 val 的 recall = 100%,
中位數每人只有 6 個候選。只要排序做得好, Hit@5 可達 0.7+。

介面:
    model = SuggestionFusion(sugg_df=sugg_df, weights=[...])
    model.fit(train_df)
    preds = model.predict_topk(val_df, k=5)
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import math
from collections import Counter, defaultdict

import pandas as pd
from tqdm import tqdm


class SuggestionFusion:
    """用 suggestion table 當候選集,training data 當排序特徵。

    候選集: 該 user 在 suggestion table 中的所有 end_latlng
           (recall = 100%, 中位數 6 個候選)

    排序信號 (6 個):
        0. user × (hour, holiday, dow) → end  [from training]
        1. user × (hour, holiday)      → end  [from training]
        2. user × start                → end  [from training]
        3. (start, hour, holiday)      → end  [from training, all users]
        4. start → end                        [from training, all users]
        5. 1/距離 (start → end)               [from suggestion latlng_pin]
    """

    name = "SuggestionFusion"

    def __init__(
        self,
        sugg_df: pd.DataFrame,
        weights: list[float] | None = None,
        decay_rate: float = 0.0,
    ) -> None:
        self.decay_rate = decay_rate
        self.weights = weights or [3.0, 10.0, 10.0, 3.0, 0.5, 0.1]
        assert len(self.weights) == 6, f"需要 6 個權重, 收到 {len(self.weights)}"

        # --- 從 suggestion table 建立候選集 ---
        # user → list of end_latlng
        self._user_sugg_dests: dict = {}
        # end_latlng → end_latlng_pin (完整座標, 算距離用)
        self._latlng_to_pin: dict[str, str] = {}

        # 建 user → set(end_latlng)
        user_dests: dict = defaultdict(set)
        for uid, end in zip(sugg_df["uid_hash"].values, sugg_df["end_latlng"].values):
            user_dests[uid].add(end)
        self._user_sugg_dests = {uid: list(dests) for uid, dests in user_dests.items()}

        # 建 latlng → pin 映射
        if "end_latlng_pin" in sugg_df.columns:
            for latlng, pin in zip(sugg_df["end_latlng"].values,
                                   sugg_df["end_latlng_pin"].values):
                self._latlng_to_pin[latlng] = str(pin)

        # --- training lookup tables (填入 fit) ---
        self._uc3: dict = {}
        self._uc2: dict = {}
        self._us: dict = {}
        self._sc: dict = {}
        self._sa: dict = {}
        # GC 保留作為 fallback 補充預測清單用，不參與權重評分
        self._gc: dict = {}
        self._global_top: list[str] = []

    def fit(self, train_df: pd.DataFrame) -> "SuggestionFusion":
        decay = self.decay_rate

        ref_time = train_df["created_at"].max()
        ref_ns = ref_time.value if hasattr(ref_time, "value") else pd.Timestamp(ref_time).value
        created = train_df["created_at"].values
        days_ago = (ref_ns - created.astype("int64")) / (86400 * 1e9)
        dw = (
            [math.exp(-decay * d) for d in days_ago]
            if decay > 0 else [1.0] * len(days_ago)
        )

        uc3 = defaultdict(lambda: defaultdict(float))
        uc2 = defaultdict(lambda: defaultdict(float))
        us = defaultdict(lambda: defaultdict(float))
        sc = defaultdict(lambda: defaultdict(int))
        sa = defaultdict(lambda: defaultdict(int))
        gc: dict = defaultdict(int)

        cols = train_df[["uid_hash", "start_latlng", "end_latlng",
                         "hour_type", "is_holiday", "dayofweek"]].values

        for i, (uid, start, end, hour, holiday, dow) in enumerate(
            tqdm(cols, desc=f"{self.name}.fit", leave=False)
        ):
            w = dw[i]
            uc3[(uid, hour, holiday, dow)][end] += w
            uc2[(uid, hour, holiday)][end] += w
            us[(uid, start)][end] += w
            sc[(start, hour, holiday)][end] += 1
            sa[start][end] += 1
            gc[end] += 1

        self._uc3 = {k: dict(v) for k, v in uc3.items()}
        self._uc2 = {k: dict(v) for k, v in uc2.items()}
        self._us = {k: dict(v) for k, v in us.items()}
        self._sc = {k: dict(v) for k, v in sc.items()}
        self._sa = {k: dict(v) for k, v in sa.items()}
        self._gc = dict(gc)
        self._global_top = sorted(gc, key=gc.get, reverse=True)[:50]

        return self

    def predict_topk(self, query_df: pd.DataFrame, k: int) -> list[list[str]]:
        w = self.weights
        log1p = math.log1p
        uc3 = self._uc3
        uc2 = self._uc2
        us_dict = self._us
        sc_dict = self._sc
        sa = self._sa
        l2p = self._latlng_to_pin
        sugg_dests = self._user_sugg_dests
        global_top = self._global_top

        cols = query_df[["uid_hash", "start_latlng",
                         "hour_type", "is_holiday", "dayofweek"]].values
        out: list[list[str]] = []

        for uid, start, hour, holiday, dow in tqdm(
            cols, desc=f"{self.name}.predict", leave=False
        ):
            # --- 1. 候選集: suggestion table ---
            candidates = sugg_dests.get(uid, [])

            if not candidates:
                # 極端 fallback (不應該發生)
                out.append(global_top[:k])
                continue

            # --- 2. 評分 ---
            s_uc3 = uc3.get((uid, hour, holiday, dow), {})
            s_uc2 = uc2.get((uid, hour, holiday), {})
            s_us = us_dict.get((uid, start), {})
            s_sc = sc_dict.get((start, hour, holiday), {})
            s_sa = sa.get(start, {})

            # 解析 start 座標 (算距離用)
            try:
                slat, slng = start.split(",")
                slat, slng = float(slat), float(slng)
            except:
                slat, slng = 0.0, 0.0

            scored: list[tuple[float, str]] = []
            for end in candidates:
                if end == start:
                    continue

                # 距離信號: 用 pin 座標算 (更精確)
                dist_score = 0.0
                pin = l2p.get(end)
                if pin and slat != 0:
                    try:
                        elat, elng = pin.split(",")
                        dlat = (float(elat) - slat) * 111.0
                        dlng = (float(elng) - slng) * 101.0
                        dist_km = math.sqrt(dlat*dlat + dlng*dlng)
                        dist_score = 1.0 / (1.0 + dist_km)  # 越近分越高
                    except:
                        pass

                score = (
                    w[0] * log1p(s_uc3.get(end, 0))
                    + w[1] * log1p(s_uc2.get(end, 0))
                    + w[2] * log1p(s_us.get(end, 0))
                    + w[3] * log1p(s_sc.get(end, 0))
                    + w[4] * log1p(s_sa.get(end, 0))
                    + w[5] * dist_score
                )
                scored.append((score, end))

            # --- 3. 排序 ---
            scored.sort(key=lambda x: x[0], reverse=True)
            picks = [end for _, end in scored[:k]]

            # 不夠 k 個 (很少見: 用戶 suggestion 不足 k 個)
            if len(picks) < k:
                seen = set(picks)
                seen.add(start)
                # 補: start co-occurrence
                for d in [s_sa, s_sc]:
                    for x in sorted(d, key=d.get, reverse=True):
                        if x not in seen:
                            picks.append(x); seen.add(x)
                            if len(picks) == k: break
                    if len(picks) == k: break
                # 最後: global fallback
                if len(picks) < k:
                    for x in global_top:
                        if x not in seen:
                            picks.append(x); seen.add(x)
                            if len(picks) == k: break

            out.append(picks)

        return out

    def __repr__(self) -> str:
        return (f"SuggestionFusion(decay={self.decay_rate}, "
                f"w={[round(x,2) for x in self.weights]})")