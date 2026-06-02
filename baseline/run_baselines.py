"""一鍵跑全部 baseline 並比較。

用法 (從 repo 根目錄執行):
    python -m baseline.run_baselines                # 在 val 上跑
    python -m baseline.run_baselines --split test
    python -m baseline.run_baselines --k 1 3 5      # 多個 K
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from baseline.baselines import ALL_BASELINES
from data_loader import load_split
from evaluate import (
    evaluate, evaluate_by_segment, evaluate_trip_segment, user_freq_bucket,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--models", nargs="+", default=None,
                        help="只跑指定模型名稱,例如 HybridContextPlus")
    parser.add_argument("--list-models", action="store_true",
                        help="列出可用模型名稱後結束")
    parser.add_argument("--segment", action="store_true",
                        help="額外印出 per-segment 評估")
    parser.add_argument("--trip-segment", action="store_true",
                        help="額外印出 >=N / <N trips 兩段評估")
    parser.add_argument("--min-trips", type=int, default=20,
                        help="--trip-segment 的切點 (預設 20)")
    args = parser.parse_args()

    model_by_name = {cls.name: cls for cls in ALL_BASELINES}
    if args.list_models:
        print("\n".join(model_by_name))
        return
    if args.models is None:
        baseline_classes = ALL_BASELINES
    else:
        unknown = [name for name in args.models if name not in model_by_name]
        if unknown:
            choices = ", ".join(model_by_name)
            raise ValueError(f"unknown model(s): {unknown}; choices: {choices}")
        baseline_classes = [model_by_name[name] for name in args.models]

    print(f"[1/3] 載入 train + {args.split} ...")
    t0 = time.time()
    train_df = load_split("train")
    eval_df = load_split(args.split)
    print(f"  train: {len(train_df):,} rows, {args.split}: {len(eval_df):,} rows "
          f"({time.time()-t0:.1f}s)")

    truths = eval_df["end_latlng"].tolist()
    max_k = max(args.k)

    print(f"\n[2/3] 訓練 + 預測各 baseline (k={max_k}) ...")
    preds_by_model: dict[str, list[list[str]]] = {}
    per_k_rows: dict[int, list[dict]] = {k: [] for k in args.k}
    for cls in baseline_classes:
        model = cls()
        t0 = time.time()
        model.fit(train_df)
        t_fit = time.time() - t0
        t0 = time.time()
        preds = model.predict_topk(eval_df, max_k)
        t_pred = time.time() - t0
        preds_by_model[model.name] = preds
        print(f"  {model.name:<22} fit={t_fit:5.1f}s  predict={t_pred:5.1f}s")
        for k in args.k:
            res = evaluate(preds, truths, k=k)
            per_k_rows[k].append({"model": model.name, **res.as_dict()})

    print(f"\n[3/3] 結果 (split={args.split})")
    for k in args.k:
        print(f"\n[K={k}]")
        print(pd.DataFrame(per_k_rows[k]).to_string(index=False))

    if args.segment:
        print("\n--- per-segment 評估 (取 K=最大值) ---")
        seg_user = user_freq_bucket(train_df, eval_df)
        for model_name, preds in preds_by_model.items():
            print(f"\n# {model_name} — by user_freq_bucket (K={max_k})")
            df = evaluate_by_segment(preds, truths, seg_user, k=max_k)
            print(df.to_string(index=False))
        for col in ["hour_type", "is_holiday"]:
            for model_name, preds in preds_by_model.items():
                print(f"\n# {model_name} — by {col} (K={max_k})")
                df = evaluate_by_segment(preds, truths, eval_df[col], k=max_k)
                print(df.to_string(index=False))

    if args.trip_segment:
        n = args.min_trips
        print(f"\n--- 行程數 >={n} / <{n} 兩段評估 ---")
        for k in args.k:
            rows = []
            for model_name, preds in preds_by_model.items():
                df = evaluate_trip_segment(
                    preds, truths, train_df, eval_df, k=k, min_trips=n,
                )
                row = {"model": model_name}
                for _, r in df.iterrows():
                    tag = r["segment"]
                    row[f"Hit@{k}({tag})"] = r[f"Hit@{k}"]
                    row[f"NDCG@{k}({tag})"] = r[f"NDCG@{k}"]
                rows.append(row)
            print(f"\n[K={k}]")
            print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
