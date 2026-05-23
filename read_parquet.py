"""讀取 parquet 的共用工具。

`address_v2_suggestion.parquet` 由 BigQuery 匯出,schema 內含 dbdate 等
extension type,pandas 預設讀不了,因此用 pyarrow 直接讀並丟掉 metadata。
"""
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).parent / "1.下車地址推薦"
TRAIN_PATH = BASE_DIR / "address_v2_training_data.parquet"
SUGG_PATH = BASE_DIR / "address_v2_suggestion.parquet"


def read_parquet_cols(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """用 pyarrow 讀 parquet 指定欄位,並忽略 schema metadata。"""
    table = pq.read_table(path, columns=columns)
    table = table.replace_schema_metadata(None)
    return table.to_pandas(ignore_metadata=True)


def inspect(path: Path) -> pd.DataFrame:
    """印出 parquet 檔案的 shape、dtypes、head,並回傳 DataFrame。"""
    df = read_parquet_cols(path)
    print(f"\n=== {path.name} ===")
    print(f"shape : {df.shape}")
    print(f"dtypes:\n{df.dtypes}")
    print(f"head  :\n{df.head()}")
    return df


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    for p in (TRAIN_PATH, SUGG_PATH):
        inspect(p)
