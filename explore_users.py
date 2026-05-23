"""觀察資料集:用戶筆數分佈、類別欄位分佈、座標覆蓋率。"""
import pandas as pd

from read_parquet import TRAIN_PATH, SUGG_PATH, read_parquet_cols


def user_stats(df: pd.DataFrame, name: str) -> None:
    print(f"\n=== {name} ===")
    print(f"總筆數         : {len(df):,}")
    print(f"唯一 user 數   : {df['uid_hash'].nunique():,}")

    counts = df.groupby("uid_hash").size()
    print(f"\n每位用戶筆數統計:")
    print(f"  min  : {counts.min()}")
    print(f"  mean : {counts.mean():.2f}")
    print(f"  max  : {counts.max()}")

    edges = [0, 1, 5, 10, 20, 50, 100, 500, 1000]
    labels_all = ["1", "2-5", "6-10", "11-20", "21-50", "51-100", "101-500", "501-1000", "1000+"]
    max_count = int(counts.max())
    bins, labels = [], []
    for i, e in enumerate(edges):
        if e < max_count:
            bins.append(e)
            labels.append(labels_all[i])
        else:
            break
    bins.append(max_count + 1)
    buckets = pd.cut(counts, bins=bins, labels=labels, right=True, include_lowest=True)
    dist = buckets.value_counts().sort_index()
    print(f"\n用戶筆數分佈 (人數 / 佔比):")
    total_users = len(counts)
    for label, n in dist.items():
        print(f"  {label:>10}: {n:>8,}  ({n / total_users:.2%})")

    print(f"\n行程 > 20 筆的用戶數: {(counts > 20).sum():,}")


def categorical_stats(df: pd.DataFrame, columns: list[str]) -> None:
    print("\n=== 類別欄位分佈 (training) ===")
    for c in columns:
        print(f"\n--- {c} ---")
        print(df[c].value_counts(dropna=False))


def coverage(name: str, train_col: pd.Series, sugg_set: set) -> None:
    total_rows = len(train_col)
    unique_vals = train_col.unique()
    matched_rows = train_col.isin(sugg_set).sum()
    matched_unique = sum(v in sugg_set for v in unique_vals)

    print(f"\n--- {name} ---")
    print(f"training 總筆數              : {total_rows:,}")
    print(f"  └ 在 suggestion 找得到     : {matched_rows:,}  ({matched_rows / total_rows:.2%})")
    print(f"  └ 找不到                   : {total_rows - matched_rows:,}  ({1 - matched_rows / total_rows:.2%})")
    print(f"training 唯一座標數          : {len(unique_vals):,}")
    print(f"  └ 在 suggestion 找得到     : {matched_unique:,}  ({matched_unique / len(unique_vals):.2%})")
    print(f"  └ 找不到                   : {len(unique_vals) - matched_unique:,}")


if __name__ == "__main__":
    pd.set_option("display.width", 200)

    # 1. 用戶筆數分佈 (training)
    train = read_parquet_cols(
        TRAIN_PATH,
        ["uid_hash", "start_latlng", "end_latlng", "hour_type", "dayofweek", "is_holiday"],
    )
    user_stats(train, "address_v2_training_data")

    # 2. 類別欄位分佈
    categorical_stats(train, ["hour_type", "dayofweek", "is_holiday"])

    # 3. 座標覆蓋率:training 的 latlng 是否都能在 suggestion 找到對應地址
    #    (suggestion 只用來做 座標→地址 mapping,不在意它有多少 user)
    print("\n=== 座標覆蓋率 (training latlng 是否在 suggestion 找得到) ===")
    sugg = read_parquet_cols(SUGG_PATH, ["end_latlng"])
    sugg_set = set(sugg["end_latlng"].unique())
    print(f"suggestion 唯一 end_latlng 數: {len(sugg_set):,}")

    coverage("end_latlng (training, 3 位) vs end_latlng (suggestion, 3 位)",
             train["end_latlng"], sugg_set)
    coverage("start_latlng (training, 2 位) vs end_latlng (suggestion, 3 位)",
             train["start_latlng"], sugg_set)
