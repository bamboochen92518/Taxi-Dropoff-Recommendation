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
Taxi-Dropoff-Recommendation/
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

下車點經緯度對照實際下車地址（同時作為推薦候選集來源）。

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
pip install pandas pyarrow lightgbm tqdm
```

## 專案結構

```
Taxi-Dropoff-Recommendation/
├── data_loader.py          # 依 `created_at` 時間切分 train/val/test (75/10/15),提供 `load_split(split)` 介面。
├── evaluate.py             # Top-K 推薦評估指標(Hit@K / MRR / NDCG@K),提供 `evaluate` / `evaluate_by_segment` / `evaluate_trip_segment` / `user_freq_bucket` / `trip_count_segment`,任何推薦模型都可重用。
├── read_parquet.py         # 讀取兩個 parquet,印出 shape / dtypes / head。
├── explore_users.py        # 統計使用者數量與每個用戶的訂車筆數分佈。
├── baseline/               # Baseline 推薦器與評估腳本,詳見 [`baseline/baseline.md`](baseline/baseline.md)。
│   ├── baselines.py        # 8 個 baseline 推薦器
│   └── run_baselines.py    # 一鍵跑全部 baseline 並對比結果
├── suggestion_fusion/      # Fusion 方法，詳見 suggestion_fusion/README.md
│   ├── suggestion_fusion.py
│   └── run_suggestion.py
└── lightGBM/
    ├── lightGBM_2/         # LightGBM LambdaMART，詳見 lightGBM/lightGBM_2/README.md
    │   ├── gbm_ranker.py
    │   ├── run_gbm.py
    │   └── tune_gbm.py     
    ├── linego_good_evaluate.ipynb    # 舊版：Colab 版,GPU 訓練 + checkpoint resume
    └── (其他早期實驗版本)
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


## 執行

### Baseline

```bash
python -m baseline.run_baselines                 # 在 val 上跑,K=1,3,5
python -m baseline.run_baselines --models SuggestionWeightedFusion
python -m baseline.run_baselines --models HybridContextPlus  # train-only 最強版
python -m baseline.run_baselines --split test    # 換到 test
python -m baseline.run_baselines --list-models   # 列出可用模型
python -m baseline.run_baselines --segment       # 加上 hour/holiday/freq 拆解
python -m baseline.run_baselines --trip-segment  # >=20 / <20 trips 兩段比較
python -m baseline.run_baselines --trip-segment --min-trips 50  # 自訂切點
```
### Suggestion Fusion

```bash
# val 評估（最佳權重）
python suggestion_fusion/run_suggestion.py --weights 3.0 10.0 0.0 10.0 3.0 0.5 0.0 0.0 0.1

# 平行 grid search（建議 --workers 16~24）
python suggestion_fusion/run_suggestion.py --grid-search --workers 16

# test 評估
python suggestion_fusion/run_suggestion.py --split test --weights 3.0 10.0 0.0 10.0 3.0 0.5 0.0 0.0 0.1

# per-segment 分析
python suggestion_fusion/run_suggestion.py --weights 3.0 10.0 0.0 10.0 3.0 0.5 0.0 0.0 0.1 --segment
```

### LightGBM Ranker

```bash
# 訓練 + val 評估 + per-segment + feature importance
python lightGBM/lightGBM_2/run_gbm.py --segment --importance

# test 評估
python lightGBM/lightGBM_2/run_gbm.py --split test --segment

# 存 / 載入模型
python lightGBM/lightGBM_2/run_gbm.py --save model_gbm.txt
python lightGBM/lightGBM_2/run_gbm.py --load model_gbm.txt --split test

# 超參數調優（~30–50 分鐘）
python lightGBM/lightGBM_2/tune_gbm.py --select hit1 --final-test
```

### 數值大致對照(本資料集)

| 區間          | 大致意義              |
| ------------- | --------------------- |
| Hit@5 < 5%    | 隨機猜 / 全體熱門級別 |
| Hit@5 10~20%  | 有粗略的 context 訊號 |
| Hit@5 30~40%  | 個人化 baseline 水準  |
| Hit@5 > 50%   | 強模型,可進產品      |

詳見 [`baseline/baseline.md`](baseline/baseline.md) 各 baseline 的實測結果與分析。

## 實驗結果（val split）

### Baseline

| 方法 | Hit@1 | Hit@3 | Hit@5 | NDCG@5 | MRR@5 |
|---|:---:|:---:|:---:|:---:|:---:|
| GlobalPopularity | 0.0094 | 0.0197 | 0.0264 | 0.0180 | 0.0153 |
| UserHistory | 0.2165 | 0.3258 | 0.3513 | 0.2921 | 0.2721 |
| UserContextHistory | 0.2496 | 0.3315 | 0.3541 | 0.3073 | 0.2915 |
| StartEndCoOccurrence | 0.0483 | 0.0887 | 0.1116 | 0.0812 | 0.0711 |
| HybridUserStartEnd | 0.2293 | 0.3565 | 0.3910 | 0.3190 | 0.2947 |
| HybridContextCascade | 0.2625 | 0.3622 | 0.3938 | 0.3342 | 0.3142 |
| HybridContextPlus | 0.2723 | 0.3634 | 0.3941 | 0.3385 | 0.3198 |
| SuggestionWeightedFusion | 0.4115 | 0.6346 | 0.7458 | 0.5867 | 0.5340 |

### 本組方法

| 方法 | Hit@1 | Hit@3 | Hit@5 | NDCG@5 | MRR@5 |
|---|:---:|:---:|:---:|:---:|:---:|
| LightGBM（早期版本） | 0.3617 | 0.5979 | 0.7228 | 0.5498 | 0.4926 |
| Suggestion Fusion | 0.4418 | 0.6815 | 0.7857 | 0.6242 | 0.5705 |
| **LightGBM Ranker（LambdaMART）** | **0.5475** | **0.7724** | **0.8541** | **0.7131** | **0.6659** |

### Test split 最終結果

| 方法 | Hit@1 | Hit@3 | Hit@5 | NDCG@5 | MRR@5 |
|---|:---:|:---:|:---:|:---:|:---:|
| Suggestion Fusion | 0.4224 | 0.6673 | 0.7775 | 0.6103 | 0.5547 |
| **LightGBM Ranker（LambdaMART）** | **0.5468** | **0.7744** | **0.8566** | **0.7143** | **0.6666** |

LightGBM val (0.8541) / test (0.8566) 緊貼，無 overfit。

---

## 各子目錄說明

| 路徑 | 說明 |
|---|---|
| [`baseline/baseline.md`](baseline/baseline.md) | 8 個 baseline 的設計思路與詳細結果分析 |
| [`suggestion_fusion/README.md`](suggestion_fusion/README.md) | Fusion 方法說明與執行方式 |
| [`lightGBM/lightGBM_2/README.md`](lightGBM/lightGBM_2/README.md) | LightGBM LambdaMART 特徵設計、leakage 防護與超參數 |