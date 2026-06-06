
"""跑 SuggestionFusion。

用法:
    python run_suggestion.py                          # 快速跑
    python run_suggestion.py --grid-search            # 搜權重
    python run_suggestion.py --split test --weights ...  # 跑 test
    python run_suggestion.py --segment                # per-segment 分析
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_loader import load_split
from evaluate import evaluate, evaluate_by_segment, user_freq_bucket
from read_parquet import SUGG_PATH, read_parquet_cols
from suggestion_fusion import SuggestionFusion


def run_single(model, train_df, eval_df, ks, label=None):
    label = label or repr(model)
    max_k = max(ks)
    t0 = time.time()
    model.fit(train_df)
    t_fit = time.time() - t0
    t0 = time.time()
    preds = model.predict_topk(eval_df, max_k)
    t_pred = time.time() - t0
    truths = eval_df["end_latlng"].tolist()
    print(f"\n  {label}")
    print(f"  fit={t_fit:.1f}s  predict={t_pred:.1f}s")
    for k in ks:
        res = evaluate(preds, truths, k=k)
        print(f"    K={k}: Hit@{k}={res.hit_at_k:.4f}  "
              f"NDCG@{k}={res.ndcg_at_k:.4f}  MRR={res.mrr:.4f}")
    return preds


def grid_search(train_df, eval_df, sugg_df, target_k=5):
    truths = eval_df["end_latlng"].tolist()

    print("\n" + "=" * 60)
    print("Grid search: SuggestionFusion 9 weights")
    print("=" * 60)

    # 初始權重
    current_w = [4.0, 5.0, 5.0, 3.0, 0.3, 0.5, 0.1, 2.0, 0.5]
    names = [
        "w0:user×ctx3", "w1:user×ctx2", "w2:user_all", "w3:user×start",
        "w4:start×ctx", "w5:start_all", "w6:global", "w7:POI", "w8:distance",
    ]
    candidates = [0.0, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]

    for iteration in range(3):
        print(f"\n--- Iteration {iteration + 1} ---")
        improved = False
        for dim in range(9):
            best_val = current_w[dim]
            best_score = -1
            for val in candidates:
                w_try = list(current_w)
                w_try[dim] = val
                model = SuggestionFusion(sugg_df=sugg_df, weights=w_try)
                model.fit(train_df)
                preds = model.predict_topk(eval_df, target_k)
                res = evaluate(preds, truths, k=target_k)
                if res.ndcg_at_k > best_score:
                    best_score = res.ndcg_at_k
                    best_val = val
            if best_val != current_w[dim]:
                improved = True
                print(f"  {names[dim]}: {current_w[dim]} → {best_val}  "
                      f"(NDCG@{target_k}={best_score:.4f})")
                current_w[dim] = best_val
            else:
                print(f"  {names[dim]}: {current_w[dim]} (no change, "
                      f"NDCG@{target_k}={best_score:.4f})")
        if not improved:
            print("  收斂")
            break

    print(f"\n{'='*60}")
    print(f"最佳參數: {current_w}")
    model = SuggestionFusion(sugg_df=sugg_df, weights=current_w)
    model.fit(train_df)
    preds = model.predict_topk(eval_df, 5)
    for k in [1, 3, 5]:
        res = evaluate(preds, truths, k=k)
        print(f"  K={k}: Hit@{k}={res.hit_at_k:.4f}  "
              f"NDCG@{k}={res.ndcg_at_k:.4f}  MRR={res.mrr:.4f}")
    print(f"\n  python run_suggestion.py --weights "
          f"{' '.join(str(x) for x in current_w)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--weights", type=float, nargs=9, default=None)
    parser.add_argument("--grid-search", action="store_true")
    parser.add_argument("--segment", action="store_true")
    args = parser.parse_args()

    print("載入資料 ...")
    t0 = time.time()
    train_df = load_split("train")
    eval_df = load_split(args.split)
    sugg_df = read_parquet_cols(SUGG_PATH)
    print(f"  train: {len(train_df):,}, {args.split}: {len(eval_df):,}, "
          f"sugg: {len(sugg_df):,}  ({time.time()-t0:.1f}s)")

    if args.grid_search:
        grid_search(train_df, eval_df, sugg_df)
        return

    weights = args.weights or [4.0, 5.0, 5.0, 3.0, 0.3, 0.5, 0.1, 2.0, 0.5]
    model = SuggestionFusion(sugg_df=sugg_df, weights=weights)
    preds = run_single(model, train_df, eval_df, args.k)

    if args.segment:
        truths = eval_df["end_latlng"].tolist()
        max_k = max(args.k)
        seg_user = user_freq_bucket(train_df, eval_df)
        for col_name, col_data in [
            ("user_freq_bucket", seg_user),
            ("hour_type", eval_df["hour_type"]),
            ("is_holiday", eval_df["is_holiday"]),
        ]:
            print(f"\n# by {col_name} (K={max_k})")
            df = evaluate_by_segment(preds, truths, col_data, k=max_k)
            print(df.to_string(index=False))


if __name__ == "__main__":
    main()