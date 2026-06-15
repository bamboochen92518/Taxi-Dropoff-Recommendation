"""LightGBM LambdaMART ranker for taxi drop-off recommendation (v3 features).

資料行為 (已用 check_sugg.py 確認)
=================================
- suggestion table 每 (uid,end_latlng) 只 1 筆 → 無重複；候選 = user 去重地址清單。
  造訪「頻率」資訊只在 training data，不在 suggestion。故移除 sugg_freq。
- 候選數 median=2, mean=3.78, p90=8, max=470 → 多數人「2 選 1」，
  難度全在 Hit@1；少數高頻 user 候選極多 (難排)。
- 同一 end_latlng 對應多個精確 pin (mean 7.69, 62% 一對多) →
  距離用「最近 pin」；pin/addr 多樣性本身是信號。

特徵 (v3)
---------
歷史 count:   uc3 uc2 ua us sc sa gc
距離(最近pin): dist_score dist_km
pin 多樣性:   n_pins n_addr
recency:      rec_score rec_last_d rec_first_d rec_span_d
context 條件: holiday_ratio  hour_cond  dow_cond
              (候選在 假日/當前hour/當前dow 的造訪佔該候選總量比例
               → 平日/假日、上班/回家 的資料驅動版，per-candidate 變化)
時段親和(全體): eh_ratio ed_ratio
地理 backoff: sa_g1 sa_g0   (start 取小數1位/0位 grid → 候選造訪量，補稀疏 exact start)
熱門廣度:     nu
rank(curated): 對最有判別力的信號做組內 normalized rank
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **k):
        return x

import lightgbm as lgb


KM_PER_LAT = 111.0
KM_PER_LNG = 101.0
DAY_NS = 86400 * 1e9

RAW_FEATURES = [
    "uc3", "uc2", "ua", "us", "sc", "sa", "gc",            # 0-6
    "dist_score", "dist_km", "dlat_km", "dlng_km",         # 7-10  最近 pin + 方向位移
    "n_pins", "n_addr",                                     # 11-12 pin 多樣性
    "rec_score", "rec_last_d", "rec_first_d", "rec_span_d", # 13-16 recency
    "holiday_ratio", "hour_cond", "dow_cond",              # 17-19 候選自身 context
    "uh_cnt", "uh_ratio", "uhol_ratio", "udow_ratio",     # 20-23 user-context 條件 (對症2選1)
    "sa_g1", "sa_g0",                                      # 24-25 地理 backoff
    "nu",                                                  # 26   熱門廣度
]
_RANK_MAP = [
    (1, "r_uc2"), (2, "r_ua"), (3, "r_us"), (6, "r_gc"),
    (7, "r_dist"), (13, "r_rec"), (17, "r_holiday"), (18, "r_hour"),
    (21, "r_uhratio"), (24, "r_sag1"), (26, "r_nu"),
]
RANK_FEATURES = [name for _, name in _RANK_MAP]
FEATURE_NAMES = RAW_FEATURES + RANK_FEATURES
N_RAW = len(RAW_FEATURES)
N_FEATURES = len(FEATURE_NAMES)


def _frac_rank_desc(vals: np.ndarray) -> np.ndarray:
    n = len(vals)
    if n <= 1:
        return np.ones(n, dtype=np.float32)
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty(n, dtype=np.float32)
    sv = vals[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]:
            j += 1
        avg = (i + j) / 2.0
        for t in range(i, j + 1):
            ranks[order[t]] = avg / (n - 1)
        i = j + 1
    return ranks


def _grid(latlng: str, ndp: int) -> str:
    try:
        a, b = latlng.split(",")
        return f"{round(float(a), ndp)},{round(float(b), ndp)}"
    except Exception:
        return latlng


class FeatureStore:
    def __init__(self, sugg_df: pd.DataFrame) -> None:
        # 候選 = user 去重地址；同一 end_latlng 收集所有 pin / address
        self.user_cands: Dict[str, List[str]] = defaultdict(list)
        _seen: Dict[str, set] = defaultdict(set)
        self.latlng_to_pins: Dict[str, List[str]] = defaultdict(list)
        self.n_pins: Dict[str, int] = {}
        self.n_addr: Dict[str, int] = {}

        uid = sugg_df["uid_hash"].values
        end = sugg_df["end_latlng"].values
        for u, e in zip(uid, end):
            if e not in _seen[u]:
                _seen[u].add(e)
                self.user_cands[u].append(e)

        if "end_latlng_pin" in sugg_df.columns:
            pin_seen: Dict[str, set] = defaultdict(set)
            for e, p in zip(end, sugg_df["end_latlng_pin"].values):
                p = str(p)
                if p not in pin_seen[e]:
                    pin_seen[e].add(p)
                    self.latlng_to_pins[e].append(p)
            self.n_pins = {e: len(v) for e, v in self.latlng_to_pins.items()}
        if "end_address" in sugg_df.columns:
            addr_seen: Dict[str, set] = defaultdict(set)
            for e, a in zip(end, sugg_df["end_address"].values):
                addr_seen[e].add(str(a))
            self.n_addr = {e: len(v) for e, v in addr_seen.items()}

        # count tables
        self.uc3 = {}; self.uc2 = {}; self.ua = {}; self.us = {}
        self.sc = {}; self.sa = {}; self.gc = {}
        self.global_top: List[str] = []
        self.rec = {}
        self.e_hol = {}      # end -> 假日造訪數
        self.eh = {}         # (end,hour) -> 數
        self.ed = {}         # (end,dow) -> 數
        self.sa_g1 = {}      # (grid1, end) -> 數
        self.sa_g0 = {}      # (grid0, end) -> 數
        self.u_hour = {}     # (u,hour) -> {end:數}
        self.u_hol = {}      # (u,holiday) -> {end:數}
        self.u_dow = {}      # (u,dow) -> {end:數}
        self.nu = {}
        self.ref_ns = 0.0

    def fit_counts(self, period_df: pd.DataFrame) -> "FeatureStore":
        uc3 = defaultdict(lambda: defaultdict(int))
        uc2 = defaultdict(lambda: defaultdict(int))
        ua = defaultdict(lambda: defaultdict(int))
        us = defaultdict(lambda: defaultdict(int))
        sc = defaultdict(lambda: defaultdict(int))
        sa = defaultdict(lambda: defaultdict(int))
        gc = defaultdict(int)
        e_hol = defaultdict(int)
        eh = defaultdict(int); ed = defaultdict(int)
        sa_g1 = defaultdict(int); sa_g0 = defaultdict(int)
        u_hour = defaultdict(lambda: defaultdict(int))
        u_hol = defaultdict(lambda: defaultdict(int))
        u_dow = defaultdict(lambda: defaultdict(int))
        rec: Dict[tuple, list] = {}
        nu_sets: Dict[str, set] = defaultdict(set)

        ref_time = period_df["created_at"].max()
        self.ref_ns = (ref_time.value if hasattr(ref_time, "value")
                       else pd.Timestamp(ref_time).value)

        cols = period_df[["uid_hash", "start_latlng", "end_latlng",
                          "hour_type", "is_holiday", "dayofweek",
                          "created_at"]].values
        for u, s, e, h, hol, d, ts in tqdm(cols, desc="fit_counts", leave=False):
            uc3[(u, h, hol, d)][e] += 1
            uc2[(u, h, hol)][e] += 1
            ua[u][e] += 1
            us[(u, s)][e] += 1
            sc[(s, h, hol)][e] += 1
            sa[s][e] += 1
            gc[e] += 1
            if hol:
                e_hol[e] += 1
            eh[(e, h)] += 1
            ed[(e, d)] += 1
            sa_g1[(_grid(s, 1), e)] += 1
            sa_g0[(_grid(s, 0), e)] += 1
            u_hour[(u, h)][e] += 1
            u_hol[(u, hol)][e] += 1
            u_dow[(u, d)][e] += 1
            nu_sets[e].add(u)
            ns = ts.value if hasattr(ts, "value") else pd.Timestamp(ts).value
            key = (u, e)
            r = rec.get(key)
            if r is None:
                rec[key] = [ns, ns]
            else:
                if ns > r[0]:
                    r[0] = ns
                if ns < r[1]:
                    r[1] = ns

        self.uc3 = {k: dict(v) for k, v in uc3.items()}
        self.uc2 = {k: dict(v) for k, v in uc2.items()}
        self.ua = {k: dict(v) for k, v in ua.items()}
        self.us = {k: dict(v) for k, v in us.items()}
        self.sc = {k: dict(v) for k, v in sc.items()}
        self.sa = {k: dict(v) for k, v in sa.items()}
        self.gc = dict(gc)
        self.e_hol = dict(e_hol)
        self.eh = dict(eh); self.ed = dict(ed)
        self.sa_g1 = dict(sa_g1); self.sa_g0 = dict(sa_g0)
        self.u_hour = {k: dict(v) for k, v in u_hour.items()}
        self.u_hol = {k: dict(v) for k, v in u_hol.items()}
        self.u_dow = {k: dict(v) for k, v in u_dow.items()}
        self.rec = rec
        self.nu = {e: len(s) for e, s in nu_sets.items()}
        self.global_top = sorted(gc, key=gc.get, reverse=True)[:50]
        return self

    def candidates(self, uid: str) -> List[str]:
        return self.user_cands.get(uid, [])

    def _dist_nearest(self, slat: float, slng: float, end: str):
        """回傳 (score, km, dlat_km, dlng_km) 最近 pin。"""
        pins = self.latlng_to_pins.get(end)
        if not pins:
            return 0.0, 50.0, 0.0, 0.0
        best = 1e9; b_dlat = 0.0; b_dlng = 0.0
        for pin in pins:
            try:
                elat, elng = pin.split(",")
                dlat = (float(elat) - slat) * KM_PER_LAT
                dlng = (float(elng) - slng) * KM_PER_LNG
                km = math.sqrt(dlat * dlat + dlng * dlng)
                if km < best:
                    best = km; b_dlat = dlat; b_dlng = dlng
            except Exception:
                pass
        if best >= 1e9:
            return 0.0, 50.0, 0.0, 0.0
        return 1.0 / (1.0 + best), best, b_dlat, b_dlng

    def build_features(self, query_df: pd.DataFrame, for_training: bool) -> dict:
        log1p = math.log1p
        uc3, uc2, ua, us = self.uc3, self.uc2, self.ua, self.us
        sc, sa, gc = self.sc, self.sa, self.gc
        e_hol, eh, ed = self.e_hol, self.eh, self.ed
        sa_g1, sa_g0, rec, nu = self.sa_g1, self.sa_g0, self.rec, self.nu
        u_hour, u_hol, u_dow = self.u_hour, self.u_hol, self.u_dow
        n_pins_m, n_addr_m = self.n_pins, self.n_addr
        ref_ns = self.ref_ns

        cols = query_df[["uid_hash", "start_latlng", "end_latlng",
                         "hour_type", "is_holiday", "dayofweek"]].values
        X_rows = []; y_list = []; group = []; row_qidx = []
        query_cands = []; kept_qidx = []; fallback_idx = []; n_no_positive = 0

        for qi, (u, s, true_end, h, hol, d) in enumerate(
            tqdm(cols, desc="build_features", leave=False)
        ):
            cands = self.user_cands.get(u)
            if not cands:
                fallback_idx.append(qi); continue
            if for_training and true_end not in cands:
                n_no_positive += 1; continue

            s_uc3 = uc3.get((u, h, hol, d), {})
            s_uc2 = uc2.get((u, h, hol), {})
            s_ua = ua.get(u, {})
            s_us = us.get((u, s), {})
            s_sc = sc.get((s, h, hol), {})
            s_sa = sa.get(s, {})
            s_uh = u_hour.get((u, h), {})
            s_uhol = u_hol.get((u, hol), {})
            s_udow = u_dow.get((u, d), {})
            g1 = _grid(s, 1); g0 = _grid(s, 0)
            try:
                slat, slng = map(float, s.split(","))
            except Exception:
                slat, slng = 0.0, 0.0

            n = len(cands)
            raw = np.zeros((n, N_RAW), dtype=np.float32)
            for ci, e in enumerate(cands):
                if slat != 0.0:
                    dscore, dkm, ddlat, ddlng = self._dist_nearest(slat, slng, e)
                else:
                    dscore, dkm, ddlat, ddlng = 0.0, 50.0, 0.0, 0.0
                raw[ci, 0] = log1p(s_uc3.get(e, 0))
                raw[ci, 1] = log1p(s_uc2.get(e, 0))
                uae = s_ua.get(e, 0)
                raw[ci, 2] = log1p(uae)
                raw[ci, 3] = log1p(s_us.get(e, 0))
                raw[ci, 4] = log1p(s_sc.get(e, 0))
                raw[ci, 5] = log1p(s_sa.get(e, 0))
                gce = gc.get(e, 0)
                raw[ci, 6] = log1p(gce)
                raw[ci, 7] = dscore
                raw[ci, 8] = dkm
                raw[ci, 9] = ddlat
                raw[ci, 10] = ddlng
                raw[ci, 11] = log1p(n_pins_m.get(e, 0))
                raw[ci, 12] = log1p(n_addr_m.get(e, 0))
                # recency
                r = rec.get((u, e))
                if r is not None:
                    last_d = (ref_ns - r[0]) / DAY_NS
                    first_d = (ref_ns - r[1]) / DAY_NS
                    raw[ci, 13] = 1.0 / (1.0 + max(last_d, 0.0))
                    raw[ci, 14] = last_d
                    raw[ci, 15] = first_d
                    raw[ci, 16] = first_d - last_d
                else:
                    raw[ci, 13] = 0.0
                    raw[ci, 14] = -1.0
                    raw[ci, 15] = -1.0
                    raw[ci, 16] = -1.0
                # 候選自身 context (用候選 baseline 正規化)
                gce1 = gce + 1.0
                raw[ci, 17] = e_hol.get(e, 0) / gce1
                raw[ci, 18] = eh.get((e, h), 0) / gce1
                raw[ci, 19] = ed.get((e, d), 0) / gce1
                # user-context 條件 (對症 2選1)：這個 user 的這個地點，
                # 有多少比例是在當前 hour/holiday/dow 去的
                uae1 = uae + 1.0
                uh = s_uh.get(e, 0)
                raw[ci, 20] = log1p(uh)
                raw[ci, 21] = uh / uae1                         # user-hour 親和
                raw[ci, 22] = s_uhol.get(e, 0) / uae1           # user-holiday 親和
                raw[ci, 23] = s_udow.get(e, 0) / uae1           # user-dow 親和
                # 地理 backoff + 熱門廣度
                raw[ci, 24] = log1p(sa_g1.get((g1, e), 0))
                raw[ci, 25] = log1p(sa_g0.get((g0, e), 0))
                raw[ci, 26] = log1p(nu.get(e, 0))

            ranks = np.zeros((n, len(_RANK_MAP)), dtype=np.float32)
            for k_idx, (src, _) in enumerate(_RANK_MAP):
                ranks[:, k_idx] = _frac_rank_desc(raw[:, src])

            feat = np.concatenate([raw, ranks], axis=1)
            X_rows.append(feat)
            query_cands.append(cands)
            kept_qidx.append(qi)
            row_qidx.extend([qi] * n)
            if for_training:
                y_list.extend([1 if e == true_end else 0 for e in cands])
                group.append(n)

        X = (np.concatenate(X_rows, axis=0)
             if X_rows else np.zeros((0, N_FEATURES), dtype=np.float32))
        return dict(
            X=X,
            y=(np.asarray(y_list, dtype=np.int32) if for_training else None),
            group=(group if for_training else None),
            row_qidx=np.asarray(row_qidx, dtype=np.int64),
            query_cands=query_cands, kept_qidx=kept_qidx,
            fallback_idx=fallback_idx, n_no_positive=n_no_positive,
        )


DEFAULT_PARAMS = dict(
    objective="lambdarank", metric="ndcg", ndcg_eval_at=[1, 3, 5],
    lambdarank_truncation_level=3, boosting_type="gbdt",
    learning_rate=0.05, num_leaves=63, min_data_in_leaf=50,
    feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=1.0, label_gain=[0, 1], verbosity=-1,
)


class GBMRanker:
    def __init__(self, params=None, num_boost_round=1500, early_stopping_rounds=80):
        self.params = dict(DEFAULT_PARAMS)
        if params:
            self.params.update(params)
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.booster = None

    def train(self, train_feat, valid_feat=None, valid_group=None, verbose_eval=50):
        dtrain = lgb.Dataset(train_feat["X"], label=train_feat["y"],
                             group=train_feat["group"], feature_name=FEATURE_NAMES)
        valid_sets = [dtrain]; valid_names = ["train"]
        callbacks = [lgb.log_evaluation(period=verbose_eval)]
        if valid_feat is not None and valid_group is not None:
            dvalid = lgb.Dataset(valid_feat["X"], label=valid_feat["y"],
                                 group=valid_group, reference=dtrain,
                                 feature_name=FEATURE_NAMES)
            valid_sets.append(dvalid); valid_names.append("valid")
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds,
                                                first_metric_only=True, verbose=True))
        self.booster = lgb.train(self.params, dtrain,
                                 num_boost_round=self.num_boost_round,
                                 valid_sets=valid_sets, valid_names=valid_names,
                                 callbacks=callbacks)
        return self

    def predict_scores(self, X):
        assert self.booster is not None
        if len(X) == 0:
            return np.zeros(0, dtype=np.float32)
        return self.booster.predict(X, num_iteration=self.booster.best_iteration)

    def predict_topk(self, feat, store, query_df, k):
        n_queries = len(query_df)
        scores = self.predict_scores(feat["X"])
        per_query = {}
        row_qidx = feat["row_qidx"]
        cursor = 0
        for cands in feat["query_cands"]:
            n = len(cands)
            qi = int(row_qidx[cursor])
            per_query[qi] = sorted(zip(scores[cursor:cursor + n], cands), reverse=True)
            cursor += n
        gt = store.global_top
        out = []
        for qi in range(n_queries):
            if qi in per_query:
                picks = [e for _, e in per_query[qi][:k]]
                if len(picks) < k:
                    seen = set(picks)
                    for x in gt:
                        if x not in seen:
                            picks.append(x); seen.add(x)
                            if len(picks) == k:
                                break
                out.append(picks)
            else:
                out.append(gt[:k])
        return out

    def feature_importance(self):
        assert self.booster is not None
        imp = self.booster.feature_importance(importance_type="gain")
        return sorted(zip(FEATURE_NAMES, imp), key=lambda x: -x[1])