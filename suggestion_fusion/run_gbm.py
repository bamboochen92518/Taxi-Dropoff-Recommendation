"""訓練 / 評估 LightGBM LambdaMART ranker。

用法
----
# 訓練 + val 評估 (含 per-segment)
python run_gbm.py --segment

# 跑 test (count 特徵用 train+val 重算)
python run_gbm.py --split test

# 與 SuggestionFusion 做 RRF ensemble
python run_gbm.py --ensemble --segment

# 存 / 載入模型
python run_gbm.py --save model_gbm.txt
python run_gbm.py --load model_gbm.txt --split test

Leakage 防護 (關鍵)
-------------------
ranker 訓練樣本: feature 由 train 的「前段」count，label 由 train 的「最後 14 天」query。
兩段不重疊 → 無 oracle leakage。
val 推論:  count 由整個 train 重算。
test 推論: count 由 train+val 重算 (val 相對 test 是過去，合法)。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_loader import load_split
from evaluate import evaluate, evaluate_by_segment, user_freq_bucket
from read_parquet import SUGG_PATH, read_parquet_cols
from suggestion_fusion import SuggestionFusion
from gbm_ranker import FeatureStore, GBMRanker

BEST_V1 = [3.0, 10.0, 0.0, 10.0, 3.0, 0.5, 0.0, 0.0, 0.1]
LABEL_WINDOW_DAYS = 14   # = val 視窗長度，使訓練結構對齊推論


def time_split(train_df: pd.DataFrame, window_days: int = LABEL_WINDOW_DAYS):
    """train 切成 (feature_period, label_period)。label = 最後 window_days 天。"""
    cutoff = train_df["created_at"].max() - pd.Timedelta(days=window_days)
    feat_df = train_df[train_df["created_at"] < cutoff]
    label_df = train_df[train_df["created_at"] >= cutoff]
    return feat_df, label_df, cutoff


def report(preds, truths, ks=(1, 3, 5), tag=""):
    print(f"  {tag}")
    for k in ks:
        res = evaluate(preds, truths, k=k)
        print(f"    K={k}: Hit@{k}={res.hit_at_k:.4f}  "
              f"NDCG@{k}={res.ndcg_at_k:.4f}  MRR={res.mrr:.4f}")


def rrf_ensemble(rank_lists_a, rank_lists_b, k, c=60, wa=1.0, wb=1.0):
    """Reciprocal Rank Fusion。輸入兩組「每 query 的 ranked candidate list」。"""
    out = []
    for la, lb in zip(rank_lists_a, rank_lists_b):
        score = {}
        for r, e in enumerate(la):
            score[e] = score.get(e, 0.0) + wa / (c + r)
        for r, e in enumerate(lb):
            score[e] = score.get(e, 0.0) + wb / (c + r)
        ranked = sorted(score, key=score.get, reverse=True)
        out.append(ranked[:k])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    p.add_argument("--segment", action="store_true")
    p.add_argument("--ensemble", action="store_true",
                   help="與 SuggestionFusion 做 RRF blend")
    p.add_argument("--ens-wa", type=float, default=1.0, help="GBM 權重")
    p.add_argument("--ens-wb", type=float, default=1.0, help="Fusion 權重")
    p.add_argument("--window-days", type=int, default=LABEL_WINDOW_DAYS)
    p.add_argument("--num-round", type=int, default=1500)
    p.add_argument("--early-stop", type=int, default=80)
    p.add_argument("--save", type=str, default=None)
    p.add_argument("--load", type=str, default=None)
    p.add_argument("--importance", action="store_true")
    args = p.parse_args()

    print("載入資料 ...")
    t0 = time.time()
    train_df = load_split("train")
    eval_df = load_split(args.split)
    sugg_df = read_parquet_cols(SUGG_PATH)
    print(f"  train:{len(train_df):,}  {args.split}:{len(eval_df):,}  "
          f"sugg:{len(sugg_df):,}  ({time.time()-t0:.1f}s)")

    truths = eval_df["end_latlng"].tolist()
    max_k = max(args.k)

    # ---- 推論用 count period ----
    # val: 整個 train; test: train + val
    if args.split == "test":
        val_df = load_split("val")
        infer_period = pd.concat([train_df, val_df], ignore_index=True)
        print(f"  test 推論 count period = train+val ({len(infer_period):,})")
    else:
        infer_period = train_df

    # ---- 載入或訓練模型 ----
    ranker = GBMRanker(num_boost_round=args.num_round,
                       early_stopping_rounds=args.early_stop)

    if args.load:
        import lightgbm as lgb
        ranker.booster = lgb.Booster(model_file=args.load)
        print(f"  已載入模型: {args.load}")
        store_inf = FeatureStore(sugg_df).fit_counts(infer_period)
    else:
        # 時間切分訓練樣本 (leakage-free)
        feat_df, label_df, cutoff = time_split(train_df, args.window_days)
        print(f"  時間切分: feature_period={len(feat_df):,}  "
              f"label_period={len(label_df):,}  cutoff={cutoff.date()}")

        print("  建立訓練特徵 ...")
        store_tr = FeatureStore(sugg_df).fit_counts(feat_df)
        train_feat = store_tr.build_features(label_df, for_training=True)
        print(f"    train rows={len(train_feat['X']):,}  "
              f"groups={len(train_feat['group']):,}  "
              f"pos={int(train_feat['y'].sum()):,}  "
              f"丟棄(無正例)={train_feat['n_no_positive']:,}")

        # val 作為 early-stopping 的 eval set (count 用整個 train)
        store_inf = FeatureStore(sugg_df).fit_counts(infer_period)
        # 注意: early stopping 的 eval 一律用真實 val (不是 test)
        es_df = load_split("val")
        es_store = (store_inf if args.split == "val"
                    else FeatureStore(sugg_df).fit_counts(train_df))
        es_feat = es_store.build_features(es_df, for_training=True)
        print(f"    early-stop eval (val) rows={len(es_feat['X']):,}")

        print("  訓練 LightGBM ...")
        t0 = time.time()
        ranker.train(train_feat,
                     valid_feat=es_feat, valid_group=es_feat["group"],
                     verbose_eval=50)
        print(f"    訓練完成 ({time.time()-t0:.1f}s)  "
              f"best_iter={ranker.booster.best_iteration}")

        if args.save:
            ranker.booster.save_model(args.save,
                                      num_iteration=ranker.booster.best_iteration)
            print(f"    已存模型: {args.save}")

    if args.importance and ranker.booster is not None:
        print("\n  feature importance (gain):")
        for name, imp in ranker.feature_importance():
            print(f"    {name:12s} {imp:.0f}")

    # ---- 推論 ----
    print(f"\n  建立 {args.split} 推論特徵 ...")
    eval_feat = store_inf.build_features(eval_df, for_training=False)
    print(f"    無候選 fallback query={len(eval_feat['fallback_idx']):,}")

    gbm_preds = ranker.predict_topk(eval_feat, store_inf, eval_df, k=max_k)

    print(f"\n=== GBM Ranker ({args.split}) ===")
    report(gbm_preds, truths, args.k, tag="GBMRanker")

    final_preds = gbm_preds
    final_tag = "GBM"

    # ---- Ensemble ----
    if args.ensemble:
        print("\n  跑 SuggestionFusion (取完整排序供 RRF) ...")
        fusion = SuggestionFusion(sugg_df=sugg_df, weights=BEST_V1)
        fusion.fit(infer_period)
        fusion_full = fusion.predict_topk(eval_df, k=50)
        gbm_full = ranker.predict_topk(eval_feat, store_inf, eval_df, k=50)

        ens_preds = rrf_ensemble(gbm_full, fusion_full, k=max_k,
                                 wa=args.ens_wa, wb=args.ens_wb)
        print(f"\n=== RRF Ensemble (GBM×{args.ens_wa} + Fusion×{args.ens_wb}) ===")
        report(ens_preds, truths, args.k, tag="Ensemble")
        final_preds = ens_preds
        final_tag = "Ensemble"

    # ---- Per-segment ----
    if args.segment:
        seg = user_freq_bucket(train_df, eval_df)
        print(f"\n# {final_tag} by user_freq_bucket (K={max_k})")
        df = evaluate_by_segment(final_preds, truths, seg, k=max_k)
        print(df.to_string(index=False))

        if args.ensemble:
            print(f"\n# GBM (單獨) by user_freq_bucket (K={max_k})")
            df2 = evaluate_by_segment(gbm_preds, truths, seg, k=max_k)
            print(df2.to_string(index=False))


if __name__ == "__main__":
    main()