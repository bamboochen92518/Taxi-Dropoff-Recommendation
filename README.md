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
pip install pandas pyarrow tqdm
```

## 腳本

- `read_parquet.py` — 讀取兩個 parquet,印出 shape / dtypes / head。
- `explore_users.py` — 統計使用者數量與每個用戶的訂車筆數分佈。
- `data_loader.py` — 依 `created_at` 時間切分 train/val/test (75/10/15),提供 `load_split(split)` 介面。
- `evaluate.py` — Top-K 推薦評估指標(Hit@K / MRR / NDCG@K),任何推薦模型都可重用。
- `baseline/` — Baseline 推薦器與評估腳本,詳見 [`baseline/baseline.md`](baseline/baseline.md)。
  - `baseline/baselines.py` — 8 個 baseline 推薦器
  - `baseline/run_baselines.py` — 一鍵跑全部 baseline 並對比結果

```bash
python -m baseline.run_baselines                 # 在 val 上跑,K=1,3,5
python -m baseline.run_baselines --models SuggestionWeightedFusion
python -m baseline.run_baselines --models HybridContextPlus  # train-only 最強版
python -m baseline.run_baselines --split test    # 換到 test
python -m baseline.run_baselines --list-models   # 列出可用模型
python -m baseline.run_baselines --segment       # 加上 per-segment 拆解
```

## 資料切分

依 `created_at` 排序後依筆數切分,**時間切分** 而非隨機切分,避免「用未來預測過去」的資料洩漏。

| split | 比例 | 筆數     |
| ----- | ---- | -------- |
| train | 75%  | 750,000  |
| val   | 10%  | 100,000  |
| test  | 15%  | 150,000  |

## 評估指標

下車地點推薦是典型的 **Top-K Ranking** 任務,我們同時看三個互補的指標:

### Hit@K (= Recall@K,因為每筆 query 只有 1 個正確答案)
- **定義**:top-K 推薦中是否包含真實的下車地點(0 / 1),整體取平均。
- **數值意義**:`Hit@5 = 0.40` 代表「**在前 5 個建議中命中真實下車點的比例為 40%**」。也就是若 APP 顯示 5 個快捷下車點,平均 4 趟有 2 趟使用者能直接點選。
- **範圍**:0 ~ 1,越高越好。

### MRR (Mean Reciprocal Rank)
- **定義**:對每筆 query 算 `1 / 真實答案的排名`(沒命中則為 0),整體取平均。
- **數值意義**:
  - `MRR = 1.0` → 每次都排第 1 名
  - `MRR = 0.5` → 平均排在第 2 名(或一半命中第 1、一半沒命中)
  - `MRR = 0.29` → 平均對命中的部分排在第 3~4 名
- **範圍**:0 ~ 1,越高越好。**懲罰「猜對但排太後面」的情況**,是「使用者要滑幾次才看到正確答案」的代理指標。

### NDCG@K (Normalized Discounted Cumulative Gain)
- **定義**:命中時得分 `1 / log2(rank + 1)`,沒命中為 0;整體取平均。因為每筆只有 1 個正確答案,理想 DCG 為 1,所以 NDCG = DCG。
- **數值意義**:同時考慮「有沒有命中」和「排在前面」,是業界 ranking 任務的標準指標。對於只有 1 個正確答案的場景,**NDCG@K 落在 Hit@K 和 MRR 之間**。
- **範圍**:0 ~ 1,越高越好。

### K 為什麼選 1 / 3 / 5
- **K=1**:對應「APP 直接 autofill 一個地址」的場景,最嚴格。
- **K=3**:對應「顯示 3 個下車點建議」的常見 UI。
- **K=5**:對應「下拉選單顯示前 5 個」,寬鬆上限。

### 數值大致對照(本資料集)

| 區間          | 大致意義              |
| ------------- | --------------------- |
| Hit@5 < 5%    | 隨機猜 / 全體熱門級別 |
| Hit@5 10~20%  | 有粗略的 context 訊號 |
| Hit@5 30~40%  | 個人化 baseline 水準  |
| Hit@5 > 50%   | 強模型,可進產品      |

詳見 [`baseline/baseline.md`](baseline/baseline.md) 各 baseline 的實測結果與分析。
