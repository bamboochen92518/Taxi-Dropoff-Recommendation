# 下車地址推薦

## 題目
為了節約用戶叫車的時間,希望在用戶輸入上車地址後,能快速給出最有可能的下車地址排序。並考量不同時段、平日/假日、用戶屬性,幫助用戶在不同的叫車場景下更快完成叫車行為。

## 資料

資料未包含在 repo 中,請自行下載 `1.下車地址推薦-20260523T095550Z-3-001.zip` 並依下列步驟放置:

```bash
# 放到專案根目錄後解壓
unzip "1.下車地址推薦-20260523T095550Z-3-001.zip"
```

解壓後目錄結構應為:

```
final/
├── 1.下車地址推薦-20260523T095550Z-3-001.zip
└── 1.下車地址推薦/
    ├── address_v2_training_data.parquet
    └── address_v2_suggestion.parquet
```

兩個 parquet 檔案的內容如下。

### 1. `address_v2_training_data.parquet`
每趟行程的上下車經緯度配對紀錄。

- Data range: 2026/01 ~ 2026/05
- Size: 305 MB
- Rows: 1,000,000
- Cols: 7
- Note: 行程大於 20 筆的用戶,大約還有 5,000 位

| 欄位名稱       | 資料型態   | 欄位說明                                   |
| -------------- | ---------- | ------------------------------------------ |
| `uid_hash`     | INTEGER    | 乘客 user_id 經 SHA256 雜湊後的匿名識別碼  |
| `start_latlng` | STRING     | 上車經緯度(取到小數點 2 位)              |
| `end_latlng`   | STRING     | 下車經緯度(取到小數點 3 位)              |
| `created_at`   | TIMESTAMP  | 叫車時間點                                 |
| `hour_type`    | STRING     | 時段                                       |
| `is_holiday`   | STRING     | 是否為假日                                 |
| `dayofweek`    | STRING     | 星期                                       |

### 2. `address_v2_suggestion.parquet`
下車點經緯度對照實際下車地址。

- Data range: 2026/01 ~ 2026/05
- Size: 254 MB
- Rows: 2,493,639
- Cols: 5

| 欄位名稱         | 資料型態 | 欄位說明                                   |
| ---------------- | -------- | ------------------------------------------ |
| `uid_hash`       | INTEGER  | 乘客 user_id 經 SHA256 雜湊後的匿名識別碼  |
| `end_latlng`     | STRING   | 下車點經緯度(取到小數點 3 位)            |
| `end_address`    | STRING   | 下車地址名稱                               |
| `end_latlng_pin` | STRING   | 下車點經緯度(完整 pin 點)                |
| `updated_date`   | DATE     | 更新日期                                   |

## 環境

```bash
pip install pandas pyarrow
```

## 腳本

- `read_parquet.py` — 讀取兩個 parquet,印出 shape / dtypes / head。
- `explore_users.py` — 統計使用者數量與每個用戶的訂車筆數分佈。
