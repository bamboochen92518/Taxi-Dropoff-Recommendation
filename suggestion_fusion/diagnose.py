"""資料診斷: 找出 Hit@5 天花板在哪、模型失敗的根因。

跑法: python diagnose.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from data_loader import load_split
from read_parquet import SUGG_PATH, read_parquet_cols
from evaluate import evaluate


def main():
    print("=" * 70)
    print("載入資料 ...")
    print("=" * 70)
    train = load_split("train")
    val = load_split("val")
    sugg = read_parquet_cols(SUGG_PATH, ["end_latlng", "end_address"])
    print(f"  train: {len(train):,} rows, {train['uid_hash'].nunique():,} users")
    print(f"  val:   {len(val):,} rows, {val['uid_hash'].nunique():,} users")

    # ================================================================
    # 1. Exact Repeat Rate: val 中有多少比例的 end_latlng 在 train 出現過
    # ================================================================
    print("\n" + "=" * 70)
    print("1. Exact Repeat Rate (val 的 end_latlng 是否在該 user 的 train 中出現過)")
    print("=" * 70)

    train_user_ends = defaultdict(set)
    for uid, end in zip(train["uid_hash"].values, train["end_latlng"].values):
        train_user_ends[uid].add(end)

    val_arr = val[["uid_hash", "end_latlng"]].values
    n_val = len(val_arr)
    n_repeat = 0
    n_new_dest = 0
    n_unknown_user = 0
    for uid, end in val_arr:
        if uid not in train_user_ends:
            n_unknown_user += 1
        elif end in train_user_ends[uid]:
            n_repeat += 1
        else:
            n_new_dest += 1

    print(f"  去過的地方 (exact repeat):  {n_repeat:>7,}  ({n_repeat/n_val:.1%})")
    print(f"  沒去過的新地方:              {n_new_dest:>7,}  ({n_new_dest/n_val:.1%})")
    print(f"  training 完全沒見過的 user:  {n_unknown_user:>7,}  ({n_unknown_user/n_val:.1%})")
    print(f"\n  → 這就是 count-based 模型的天花板: {n_repeat/n_val:.1%}")
    print(f"    (只有去過的地方才可能被 frequency 模型猜對)")

    # ================================================================
    # 2. Oracle Top-K: 如果完美預測 user 最常去的 top-K, Hit@K 是多少
    # ================================================================
    print("\n" + "=" * 70)
    print("2. Oracle Top-K (用 train 的 user 歷史 top-K 作為預測)")
    print("=" * 70)

    # 用 user 的 frequency 排序
    user_end_counter = defaultdict(Counter)
    for uid, end in zip(train["uid_hash"].values, train["end_latlng"].values):
        user_end_counter[uid][end] += 1

    truths = val["end_latlng"].tolist()
    for K in [1, 3, 5, 10, 20]:
        preds = []
        for uid in val["uid_hash"].values:
            c = user_end_counter.get(uid)
            if c:
                preds.append([x for x, _ in c.most_common(K)])
            else:
                preds.append([])
        res = evaluate(preds, truths, k=K)
        print(f"  Oracle Top-{K:>2}: Hit@{K}={res.hit_at_k:.4f}  "
              f"MRR={res.mrr:.4f}  NDCG@{K}={res.ndcg_at_k:.4f}")

    # ================================================================
    # 3. Oracle with context: user × (hour, holiday) top-K
    # ================================================================
    print("\n" + "=" * 70)
    print("3. Oracle with Context: user × (hour_type, is_holiday) top-K")
    print("=" * 70)

    user_ctx_counter = defaultdict(Counter)
    for uid, end, hour, hol in zip(
        train["uid_hash"].values, train["end_latlng"].values,
        train["hour_type"].values, train["is_holiday"].values
    ):
        user_ctx_counter[(uid, hour, hol)][end] += 1

    for K in [1, 3, 5]:
        preds = []
        for uid, hour, hol in zip(
            val["uid_hash"].values, val["hour_type"].values, val["is_holiday"].values
        ):
            c = user_ctx_counter.get((uid, hour, hol))
            if c:
                preds.append([x for x, _ in c.most_common(K)])
            else:
                # fallback to user overall
                c2 = user_end_counter.get(uid)
                if c2:
                    preds.append([x for x, _ in c2.most_common(K)])
                else:
                    preds.append([])
        res = evaluate(preds, truths, k=K)
        print(f"  Oracle ctx  Top-{K}: Hit@{K}={res.hit_at_k:.4f}  "
              f"MRR={res.mrr:.4f}  NDCG@{K}={res.ndcg_at_k:.4f}")

    # ================================================================
    # 4. Oracle user × start top-K
    # ================================================================
    print("\n" + "=" * 70)
    print("4. Oracle: user × start_latlng top-K (+ fallback to user)")
    print("=" * 70)

    user_start_counter = defaultdict(Counter)
    for uid, start, end in zip(
        train["uid_hash"].values, train["start_latlng"].values,
        train["end_latlng"].values
    ):
        user_start_counter[(uid, start)][end] += 1

    for K in [1, 3, 5]:
        preds = []
        for uid, start in zip(val["uid_hash"].values, val["start_latlng"].values):
            seen = set()
            picks = []
            c = user_start_counter.get((uid, start))
            if c:
                for x, _ in c.most_common():
                    if x != start and x not in seen:
                        picks.append(x); seen.add(x)
                        if len(picks) == K: break
            if len(picks) < K:
                c2 = user_end_counter.get(uid)
                if c2:
                    for x, _ in c2.most_common():
                        if x != start and x not in seen:
                            picks.append(x); seen.add(x)
                            if len(picks) == K: break
            preds.append(picks)
        res = evaluate(preds, truths, k=K)
        print(f"  Oracle u×s  Top-{K}: Hit@{K}={res.hit_at_k:.4f}  "
              f"MRR={res.mrr:.4f}  NDCG@{K}={res.ndcg_at_k:.4f}")

    # ================================================================
    # 5. Oracle user × start × context top-K
    # ================================================================
    print("\n" + "=" * 70)
    print("5. Oracle: user × start × (hour, holiday) top-K (cascade fallback)")
    print("=" * 70)

    user_start_ctx_counter = defaultdict(Counter)
    for uid, start, end, hour, hol in zip(
        train["uid_hash"].values, train["start_latlng"].values,
        train["end_latlng"].values, train["hour_type"].values,
        train["is_holiday"].values
    ):
        user_start_ctx_counter[(uid, start, hour, hol)][end] += 1

    for K in [1, 3, 5]:
        preds = []
        for uid, start, hour, hol in zip(
            val["uid_hash"].values, val["start_latlng"].values,
            val["hour_type"].values, val["is_holiday"].values
        ):
            seen = set()
            picks = []
            # tier 1: user × start × context
            c = user_start_ctx_counter.get((uid, start, hour, hol))
            if c:
                for x, _ in c.most_common():
                    if x != start and x not in seen:
                        picks.append(x); seen.add(x)
                        if len(picks) == K: break
            # tier 2: user × start
            if len(picks) < K:
                c = user_start_counter.get((uid, start))
                if c:
                    for x, _ in c.most_common():
                        if x != start and x not in seen:
                            picks.append(x); seen.add(x)
                            if len(picks) == K: break
            # tier 3: user × context
            if len(picks) < K:
                c = user_ctx_counter.get((uid, hour, hol))
                if c:
                    for x, _ in c.most_common():
                        if x != start and x not in seen:
                            picks.append(x); seen.add(x)
                            if len(picks) == K: break
            # tier 4: user overall
            if len(picks) < K:
                c = user_end_counter.get(uid)
                if c:
                    for x, _ in c.most_common():
                        if x != start and x not in seen:
                            picks.append(x); seen.add(x)
                            if len(picks) == K: break
            preds.append(picks)
        res = evaluate(preds, truths, k=K)
        print(f"  Oracle full Top-{K}: Hit@{K}={res.hit_at_k:.4f}  "
              f"MRR={res.mrr:.4f}  NDCG@{K}={res.ndcg_at_k:.4f}")

    # ================================================================
    # 6. POI fragmentation: 同一個 address 有幾個不同的 end_latlng
    # ================================================================
    print("\n" + "=" * 70)
    print("6. POI Fragmentation (同一 address 有幾個不同 end_latlng)")
    print("=" * 70)

    addr_latlngs = sugg.groupby("end_address")["end_latlng"].nunique()
    print(f"  唯一 address 數: {len(addr_latlngs):,}")
    print(f"  每個 address 的 end_latlng 數:")
    print(f"    1 個:  {(addr_latlngs == 1).sum():,}  ({(addr_latlngs == 1).mean():.1%})")
    print(f"    2-3個: {((addr_latlngs >= 2) & (addr_latlngs <= 3)).sum():,}")
    print(f"    4-10:  {((addr_latlngs >= 4) & (addr_latlngs <= 10)).sum():,}")
    print(f"    >10:   {(addr_latlngs > 10).sum():,}")
    print(f"    max:   {addr_latlngs.max()}")

    # ================================================================
    # 7. 用戶分段的 repeat rate 和 oracle
    # ================================================================
    print("\n" + "=" * 70)
    print("7. 用戶分段: 依 training 筆數分群的 repeat rate + oracle Hit@5")
    print("=" * 70)

    train_counts = train.groupby("uid_hash").size()
    edges = [0, 5, 20, 50, 100, 200, 10_000_000]
    labels = ["1-5", "6-20", "21-50", "51-100", "101-200", "200+"]

    for lo, hi, label in zip(edges[:-1], edges[1:], labels):
        uids_in_bucket = set(train_counts[(train_counts > lo) & (train_counts <= hi)].index)
        mask = val["uid_hash"].isin(uids_in_bucket).values
        n_seg = mask.sum()
        if n_seg == 0:
            continue

        # repeat rate
        n_rep = 0
        for uid, end in val_arr[mask]:
            if end in train_user_ends.get(uid, set()):
                n_rep += 1

        # oracle hit@5
        seg_preds = []
        seg_truths = []
        for uid, end in val_arr[mask]:
            c = user_end_counter.get(uid)
            if c:
                seg_preds.append([x for x, _ in c.most_common(5)])
            else:
                seg_preds.append([])
            seg_truths.append(end)
        res = evaluate(seg_preds, seg_truths, k=5)

        print(f"  [{label:>7}] n={n_seg:>6,}  "
              f"repeat={n_rep/n_seg:.1%}  "
              f"oracle_Hit@5={res.hit_at_k:.4f}  "
              f"oracle_NDCG@5={res.ndcg_at_k:.4f}")

    print("\n" + "=" * 70)
    print("診斷完成")
    print("=" * 70)


if __name__ == "__main__":
    main()