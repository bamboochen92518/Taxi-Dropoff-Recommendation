"""確認 suggestion table 是否大幅提升 repeat rate。

如果 suggestion repeat rate 遠高於 training repeat rate (35.1%),
那 suggestion table 就是突破 0.7 的關鍵。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from collections import defaultdict
from data_loader import load_split
from read_parquet import SUGG_PATH, read_parquet_cols
from evaluate import evaluate


def main():
    print("載入資料 ...")
    train = load_split("train")
    val = load_split("val")
    sugg = read_parquet_cols(SUGG_PATH, ["uid_hash", "end_latlng"])

    val_arr = val[["uid_hash", "end_latlng"]].values
    n_val = len(val_arr)

    # --- Training-based repeat rate (baseline) ---
    train_user_ends = defaultdict(set)
    for uid, end in zip(train["uid_hash"].values, train["end_latlng"].values):
        train_user_ends[uid].add(end)

    # --- Suggestion-based repeat rate ---
    sugg_user_ends = defaultdict(set)
    for uid, end in zip(sugg["uid_hash"].values, sugg["end_latlng"].values):
        sugg_user_ends[uid].add(end)

    # --- Combined ---
    n_train_only = 0
    n_sugg_only = 0
    n_both = 0
    n_neither = 0
    n_unknown_user = 0

    for uid, end in val_arr:
        in_train = end in train_user_ends.get(uid, set())
        in_sugg = end in sugg_user_ends.get(uid, set())

        if in_train and in_sugg:
            n_both += 1
        elif in_train:
            n_train_only += 1
        elif in_sugg:
            n_sugg_only += 1
        else:
            n_neither += 1
            if uid not in train_user_ends and uid not in sugg_user_ends:
                n_unknown_user += 1

    n_train_repeat = n_both + n_train_only
    n_sugg_repeat = n_both + n_sugg_only
    n_either_repeat = n_both + n_train_only + n_sugg_only

    print(f"\n{'='*60}")
    print(f"Repeat Rate 比較 (val = {n_val:,} queries)")
    print(f"{'='*60}")
    print(f"  Training only:     {n_train_repeat:>6,}  ({n_train_repeat/n_val:.1%})")
    print(f"  Suggestion only:   {n_sugg_repeat:>6,}  ({n_sugg_repeat/n_val:.1%})")
    print(f"  Either (聯集):     {n_either_repeat:>6,}  ({n_either_repeat/n_val:.1%})")
    print(f"  Neither (全新):    {n_neither:>6,}  ({n_neither/n_val:.1%})")

    print(f"\n{'='*60}")
    print(f"細項拆解")
    print(f"{'='*60}")
    print(f"  在 train AND sugg: {n_both:>6,}  ({n_both/n_val:.1%})")
    print(f"  只在 train:        {n_train_only:>6,}  ({n_train_only/n_val:.1%})")
    print(f"  只在 sugg:         {n_sugg_only:>6,}  ({n_sugg_only/n_val:.1%})")
    print(f"  都沒有:            {n_neither:>6,}  ({n_neither/n_val:.1%})")
    print(f"    (其中完全未知 user: {n_unknown_user:,})")

    # --- User coverage ---
    val_users = set(val["uid_hash"].unique())
    train_users = set(train_user_ends.keys())
    sugg_users = set(sugg_user_ends.keys())
    print(f"\n{'='*60}")
    print(f"User 覆蓋率")
    print(f"{'='*60}")
    print(f"  val users:                {len(val_users):,}")
    print(f"  在 train 中:              {len(val_users & train_users):,}")
    print(f"  在 suggestion 中:         {len(val_users & sugg_users):,}")
    print(f"  在 either:                {len(val_users & (train_users | sugg_users)):,}")

    # --- Suggestion 作為候選集的 oracle ---
    print(f"\n{'='*60}")
    print(f"Oracle: 用 suggestion table 作為候選集")
    print(f"{'='*60}")

    truths = val["end_latlng"].tolist()
    for K in [1, 3, 5, 10, 20]:
        preds = []
        for uid, end in val_arr:
            dests = sugg_user_ends.get(uid, set())
            # 無法排序 (沒有 frequency)，只列出來看 recall
            preds.append(list(dests)[:K])
        # 只看 recall (有在候選集中就算命中)
        n_hit = sum(1 for uid, end in val_arr if end in sugg_user_ends.get(uid, set()))
        print(f"  Recall@∞ (sugg 候選集包含正確答案): {n_hit:,} / {n_val:,} = {n_hit/n_val:.1%}")
        break  # recall 跟 K 無關

    # --- 每個 user 在 suggestion 中的目的地數量 ---
    sugg_dest_counts = {uid: len(dests) for uid, dests in sugg_user_ends.items()
                        if uid in val_users}
    if sugg_dest_counts:
        counts = list(sugg_dest_counts.values())
        print(f"\n  Val users 在 suggestion 中的目的地數量:")
        print(f"    min:    {min(counts)}")
        print(f"    median: {sorted(counts)[len(counts)//2]}")
        print(f"    mean:   {sum(counts)/len(counts):.1f}")
        print(f"    max:    {max(counts)}")


if __name__ == "__main__":
    main()