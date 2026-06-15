"""LightGBM 超參數調優 (coordinate-wise)。

效率設計
--------
特徵只建一次 (最花時間)，所有 trial 共用。每個 trial 只重訓。
選擇指標 = 真實 Hit@1 (val)，非 LightGBM 內部 ndcg。同時追蹤 Hit@5。

用法
----
python tune_gbm.py                       # 調 val，最後印最佳參數
python tune_gbm.py --select hit5         # 改用 Hit@5 當選擇指標
python tune_gbm.py --final-test          # 調完用最佳參數跑 test
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from data_loader import load_split
from evaluate import evaluate, user_freq_bucket, evaluate_by_segment
from read_parquet import SUGG_PATH, read_parquet_cols
from gbm_ranker import FeatureStore, GBMRanker, DEFAULT_PARAMS
from run_gbm import time_split, LABEL_WINDOW_DAYS

# 調優搜尋空間 (coordinate-wise)
SEARCH_SPACE = {
    "num_leaves":        [31, 63, 127, 255],
    "min_data_in_leaf":  [20, 50, 100, 200],
    "learning_rate":     [0.03, 0.05, 0.08],
    "feature_fraction":  [0.7, 0.8, 0.9, 1.0],
    "bagging_fraction":  [0.6, 0.8, 1.0],
    "lambda_l2":         [0.0, 1.0, 5.0, 10.0],
    "lambdarank_truncation_level": [1, 3, 5],
}
# 調優順序 (影響大的先調)
ORDER = ["num_leaves", "min_data_in_leaf", "learning_rate",
         "feature_fraction", "bagging_fraction", "lambda_l2",
         "lambdarank_truncation_level"]


def build_all(train_df, infer_period, sugg_df, window_days):
    """建立所有 trial 共用的特徵 (只做一次)。"""
    feat_df, label_df, cutoff = time_split(train_df, window_days)
    print(f"  時間切分 cutoff={cutoff.date()}  "
          f"feat={len(feat_df):,}  label={len(label_df):,}")

    print("  建立訓練特徵 ...")
    store_tr = FeatureStore(sugg_df).fit_counts(feat_df)
    train_feat = store_tr.build_features(label_df, for_training=True)

    print("  建立 val 特徵 (early-stop + 評估共用) ...")
    val_df = load_split("val")
    store_inf = FeatureStore(sugg_df).fit_counts(infer_period)
    es_feat = store_inf.build_features(val_df, for_training=True)    # early stop
    val_inf = store_inf.build_features(val_df, for_training=False)   # 真實評估
    truths = val_df["end_latlng"].tolist()
    return train_feat, es_feat, val_inf, store_inf, val_df, truths


def trial_score(params, train_feat, es_feat, val_inf, store_inf, val_df,
                truths, select="hit1", num_round=2500, early_stop=100):
    r = GBMRanker(params=params, num_boost_round=num_round,
                  early_stopping_rounds=early_stop)
    r.train(train_feat, valid_feat=es_feat, valid_group=es_feat["group"],
            verbose_eval=0)
    preds = r.predict_topk(val_inf, store_inf, val_df, k=5)
    h1 = evaluate(preds, truths, k=1).hit_at_k
    h5 = evaluate(preds, truths, k=5).hit_at_k
    score = h1 if select == "hit1" else h5
    return score, h1, h5, r.booster.best_iteration


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--select", choices=["hit1", "hit5"], default="hit1")
    p.add_argument("--window-days", type=int, default=LABEL_WINDOW_DAYS)
    p.add_argument("--num-round", type=int, default=2500)
    p.add_argument("--early-stop", type=int, default=100)
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--final-test", action="store_true")
    args = p.parse_args()

    print("載入資料 ...")
    t0 = time.time()
    train_df = load_split("train")
    sugg_df = read_parquet_cols(SUGG_PATH)
    print(f"  train:{len(train_df):,}  sugg:{len(sugg_df):,}  "
          f"({time.time()-t0:.1f}s)")

    # val 調優：infer period = 整個 train
    t0 = time.time()
    (train_feat, es_feat, val_inf, store_inf,
     val_df, truths) = build_all(train_df, train_df, sugg_df, args.window_days)
    print(f"  特徵建立完成 ({time.time()-t0:.1f}s)\n")

    cur = {k: DEFAULT_PARAMS.get(k, SEARCH_SPACE[k][0]) for k in ORDER}

    def full_params(overrides):
        pm = dict(DEFAULT_PARAMS)
        pm.update(overrides)
        return pm

    print("baseline ...")
    base_s, base_h1, base_h5, base_it = trial_score(
        full_params(cur), train_feat, es_feat, val_inf, store_inf,
        val_df, truths, args.select, args.num_round, args.early_stop)
    best_score = base_s
    print(f"  baseline: Hit@1={base_h1:.4f}  Hit@5={base_h5:.4f}  "
          f"iter={base_it}  (select={args.select})\n")

    for pass_i in range(args.passes):
        print(f"=== Pass {pass_i+1} ===")
        improved = False
        for param in ORDER:
            best_val = cur[param]
            local_best = best_score
            for val in SEARCH_SPACE[param]:
                if val == cur[param]:
                    continue
                trial = dict(cur); trial[param] = val
                s, h1, h5, it = trial_score(
                    full_params(trial), train_feat, es_feat, val_inf,
                    store_inf, val_df, truths, args.select,
                    args.num_round, args.early_stop)
                flag = ""
                if s > local_best:
                    local_best = s; best_val = val; flag = " *"
                print(f"  {param}={val}: Hit@1={h1:.4f} Hit@5={h5:.4f} "
                      f"iter={it}{flag}")
            if best_val != cur[param]:
                improved = True
                print(f"  → {param}: {cur[param]} → {best_val} "
                      f"({args.select}={local_best:.4f})")
                cur[param] = best_val
                best_score = local_best
        if not improved:
            print("  收斂"); break
        print()

    print("=" * 60)
    print("最佳參數:")
    for k in ORDER:
        print(f"  {k}: {cur[k]}")
    final_s, final_h1, final_h5, final_it = trial_score(
        full_params(cur), train_feat, es_feat, val_inf, store_inf,
        val_df, truths, args.select, args.num_round, args.early_stop)
    print(f"\n  val: Hit@1={final_h1:.4f}  Hit@5={final_h5:.4f}  best_iter={final_it}")

    # 印出可直接貼進 DEFAULT_PARAMS 的覆寫
    print("\n  覆寫 DEFAULT_PARAMS:")
    for k in ORDER:
        print(f"    {k}={cur[k]},")

    if args.final_test:
        print("\n" + "=" * 60)
        print("用最佳參數跑 TEST ...")
        val_df2 = load_split("val")
        test_df = load_split("test")
        infer_period = pd.concat([train_df, val_df2], ignore_index=True)

        feat_df, label_df, _ = time_split(train_df, args.window_days)
        store_tr = FeatureStore(sugg_df).fit_counts(feat_df)
        tr_feat = store_tr.build_features(label_df, for_training=True)
        # early stop 仍用 val (count 由 train)
        es_store = FeatureStore(sugg_df).fit_counts(train_df)
        es_f = es_store.build_features(val_df2, for_training=True)

        store_test = FeatureStore(sugg_df).fit_counts(infer_period)
        test_inf = store_test.build_features(test_df, for_training=False)
        test_truths = test_df["end_latlng"].tolist()

        r = GBMRanker(params=full_params(cur), num_boost_round=args.num_round,
                      early_stopping_rounds=args.early_stop)
        r.train(tr_feat, valid_feat=es_f, valid_group=es_f["group"],
                verbose_eval=0)
        preds = r.predict_topk(test_inf, store_test, test_df, k=5)
        print("\n=== TEST (最佳參數) ===")
        for k in [1, 3, 5]:
            res = evaluate(preds, test_truths, k=k)
            print(f"  K={k}: Hit@{k}={res.hit_at_k:.4f}  "
                  f"NDCG@{k}={res.ndcg_at_k:.4f}  MRR={res.mrr:.4f}")
        seg = user_freq_bucket(train_df, test_df)
        print("\n# TEST by user_freq_bucket (K=5)")
        print(evaluate_by_segment(preds, test_truths, seg, k=5).to_string(index=False))


if __name__ == "__main__":
    main()