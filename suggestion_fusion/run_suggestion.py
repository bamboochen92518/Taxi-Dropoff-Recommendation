"""跑 SuggestionFusion

用法:
    python run_suggestion.py                              # val 評估，預設權重
    python run_suggestion.py --grid-search --workers 16 # 平行 grid search
    python run_suggestion.py --split test --weights ...   # 跑 test
    python run_suggestion.py --segment                    # per-segment 分析

    48 CPU，--workers 16~24 最合適。
"""
from __future__ import annotations

import argparse
import time
import math
from multiprocessing import Pool
from collections import defaultdict

import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_loader import load_split
from evaluate import evaluate, evaluate_by_segment, user_freq_bucket
from read_parquet import SUGG_PATH, read_parquet_cols
from suggestion_fusion import SuggestionFusion


# ── 平行化 grid search ────────────────────────────────────────────────────
_shared: dict = {} 

def _init_worker(sd):
    global _shared
    _shared = sd

def _eval_weights(args):
    """單一 trial：用共享的 fit 結果直接 predict，不重新 fit。"""
    weights, target_k = args
    model = SuggestionFusion(sugg_df=_shared["sugg_df"], weights=weights)
    model._uc3        = _shared["uc3"]
    model._uc2        = _shared["uc2"]
    model._ua         = _shared["ua"]
    model._us         = _shared["us"]
    model._sc         = _shared["sc"]
    model._sa         = _shared["sa"]
    model._gc         = _shared["gc"]
    model._u_addr     = _shared["u_addr"]
    model._global_top = _shared["global_top"]
    model._user_sugg_dests = _shared["user_sugg_dests"]
    model._latlng_to_pin   = _shared["latlng_to_pin"]
    model._latlng_to_addr  = _shared["latlng_to_addr"]
    preds = model.predict_topk(_shared["eval_df"], target_k)
    res = evaluate(preds, _shared["truths"], k=target_k)
    return res.ndcg_at_k


def grid_search(train_df, eval_df, sugg_df, target_k=5, workers=8):
    truths = eval_df["end_latlng"].tolist()

    # fit 一次
    print("  Fitting base model（只做一次）...")
    base = SuggestionFusion(sugg_df=sugg_df)
    base.fit(train_df)

    shared = dict(
        sugg_df=sugg_df,
        eval_df=eval_df,
        truths=truths,
        uc3=base._uc3, uc2=base._uc2, ua=base._ua,
        us=base._us, sc=base._sc, sa=base._sa,
        gc=base._gc, u_addr=base._u_addr,
        global_top=base._global_top,
        user_sugg_dests=base._user_sugg_dests,
        latlng_to_pin=base._latlng_to_pin,
        latlng_to_addr=base._latlng_to_addr,
    )

    current_w = [3.0, 10.0, 0.0, 10.0, 3.0, 0.5, 0.0, 0.0, 0.1]
    names = [
        "w0:user×ctx3", "w1:user×ctx2", "w2:user_all", "w3:user×start",
        "w4:start×ctx", "w5:start_all", "w6:global",   "w7:POI",  "w8:distance",
    ]
    candidates = [0.0, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]

    print(f"\n{'='*60}")
    print(f"Grid search: SuggestionFusion 9 weights  (workers={workers})")
    print(f"{'='*60}")

    for iteration in range(3):
        print(f"\n--- Iteration {iteration + 1} ---")
        improved = False
        for dim in range(9):
            # 平行跑這個 dim 的所有候選值
            tasks = []
            for val in candidates:
                w_try = list(current_w)
                w_try[dim] = val
                tasks.append((w_try, target_k))

            with Pool(workers, initializer=_init_worker, initargs=(shared,)) as pool:
                scores = pool.map(_eval_weights, tasks)

            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            best_val = candidates[best_idx]
            best_score = scores[best_idx]

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
    # 最終評估
    model = SuggestionFusion(sugg_df=sugg_df, weights=current_w)
    model.fit(train_df)
    preds = model.predict_topk(eval_df, 5)
    for k in [1, 3, 5]:
        res = evaluate(preds, truths, k=k)
        print(f"  K={k}: Hit@{k}={res.hit_at_k:.4f}  "
              f"NDCG@{k}={res.ndcg_at_k:.4f}  MRR={res.mrr:.4f}")
    print(f"\n  python run_suggestion.py --weights "
          f"{' '.join(str(x) for x in current_w)}")
    return current_w


# ── 單次評估 ─────────────────────────────────────────────────────────────
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


# ── main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",    choices=["val", "test"], default="val")
    parser.add_argument("--k",        type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--weights",  type=float, nargs=9, default=None)
    parser.add_argument("--grid-search", action="store_true")
    parser.add_argument("--workers",  type=int, default=8,
                        help="grid search 平行 worker 數（預設 8，建議 16~24）")
    parser.add_argument("--segment",  action="store_true")
    args = parser.parse_args()

    print("載入資料 ...")
    t0 = time.time()
    train_df = load_split("train")
    eval_df  = load_split(args.split)
    sugg_df  = read_parquet_cols(SUGG_PATH)
    print(f"  train: {len(train_df):,}, {args.split}: {len(eval_df):,}, "
          f"sugg: {len(sugg_df):,}  ({time.time()-t0:.1f}s)")

    if args.grid_search:
        grid_search(train_df, eval_df, sugg_df, workers=args.workers)
        return

    weights = args.weights or [3.0, 10.0, 0.0, 10.0, 3.0, 0.5, 0.0, 0.0, 0.1]
    model = SuggestionFusion(sugg_df=sugg_df, weights=weights)
    preds = run_single(model, train_df, eval_df, args.k)

    if args.segment:
        truths = eval_df["end_latlng"].tolist()
        max_k  = max(args.k)
        seg_user = user_freq_bucket(train_df, eval_df)
        for col_name, col_data in [
            ("user_freq_bucket", seg_user),
            ("hour_type",        eval_df["hour_type"]),
            ("is_holiday",       eval_df["is_holiday"]),
        ]:
            print(f"\n# by {col_name} (K={max_k})")
            df = evaluate_by_segment(preds, truths, col_data, k=max_k)
            print(df.to_string(index=False))


if __name__ == "__main__":
    main()