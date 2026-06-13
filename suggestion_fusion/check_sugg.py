"""確認 suggestion table 的重複行為與候選集分佈。在 meow2 上跑。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from read_parquet import SUGG_PATH, read_parquet_cols
import pandas as pd

sugg = read_parquet_cols(SUGG_PATH)
print(f"總筆數: {len(sugg):,}  欄位: {list(sugg.columns)}")

# 1. 同一 (uid, end_latlng) 是否出現多筆？
dup = sugg.groupby(["uid_hash", "end_latlng"]).size()
print(f"\n(uid, end_latlng) 組合數: {len(dup):,}")
print(f"  其中出現 >1 次的比例: {(dup > 1).mean():.4f}")
print(f"  重複次數分佈:\n{dup.value_counts().head(10).to_string()}")

# 2. 每個 user 的候選數 (去重後 end_latlng 數)
cand_per_user = sugg.groupby("uid_hash")["end_latlng"].nunique()
print(f"\n每 user 候選數 (去重): mean={cand_per_user.mean():.2f}  "
      f"median={cand_per_user.median():.0f}  "
      f"p90={cand_per_user.quantile(0.9):.0f}  max={cand_per_user.max()}")

# 3. end_latlng vs end_latlng_pin 是否一對多 (同一截斷座標對多個精確座標)
if "end_latlng_pin" in sugg.columns:
    pin_per_latlng = sugg.groupby("end_latlng")["end_latlng_pin"].nunique()
    print(f"\n同一 end_latlng 對應的不同 pin 數: "
          f"mean={pin_per_latlng.mean():.2f}  max={pin_per_latlng.max()}")
    print(f"  一對多比例: {(pin_per_latlng > 1).mean():.4f}")

# 4. end_address 是否一對多
if "end_address" in sugg.columns:
    addr_per_latlng = sugg.groupby("end_latlng")["end_address"].nunique()
    print(f"\n同一 end_latlng 對應的不同 address 數: "
          f"mean={addr_per_latlng.mean():.2f}  max={addr_per_latlng.max()}")