# Taxi Drop-off Recommendation — suggestion_fusion/

給定 `uid / start_latlng / hour_type / is_holiday / dayofweek`，
從 suggestion table 的候選集做 Top-K 排序預測 `end_latlng`。
評估指標 Hit@K / NDCG@K / MRR（K = 1, 3, 5）。

實作兩個方法，皆基於相同的 Suggestion table 生成策略：

- **Fusion**：multi signal 加權對數融合排序。
- **LightGBM Ranker**：LambdaMART 監督式學習排序。

整體實驗驗證：加入這兩個方法能否超越自訂的 baseline。

---

## 候選生成（兩個方法共用）

對每筆 query，取該 user 在 `address_v2_suggestion.parquet` 中的所有歷史目的地
作為候選集。

- 每個 user 候選數：median = 2、mean = 3.78、p90 = 8、max = 470
- suggestion table 每 (uid, end_latlng) 僅一筆（無重複）；造訪頻率資訊僅在 training data
- 對所有評估 query 的 recall = 100%

---

## 方法一：Fusion 

9 個統計信號取 log 後線性加權，對每個候選計算分數並排序：

```
score = w0·log(1 + count[user, hour, holiday, dow → end])
      + w1·log(1 + count[user, hour, holiday → end])
      + w2·log(1 + count[user → end])
      + w3·log(1 + count[user, start → end])
      + w4·log(1 + count[start, hour, holiday → end])
      + w5·log(1 + count[start → end])
      + w6·log(1 + count[global → end])
      + w7·log(1 + score[user, address → end])
      + w8·1/(1 + distance_km)
```

權重由 coordinate-wise grid search 對 NDCG@5 優化，每個 dim 的候選值
**平行**評估（fit 只做一次，predict 平行跑）。

最佳權重（val grid search）：`[3.0, 10.0, 0.0, 10.0, 3.0, 0.5, 0.0, 0.0, 0.1]`

---

## 方法二：LightGBM Ranker

以 LightGBM LambdaMART 對候選做監督式排序，binary relevance（exact 答案為正例）。

**Leakage 修正**：count 特徵與 label 時間切分。training 切成 feature period（前段）
與 label period（最後 14 天，對齊 val 視窗）；特徵只由 feature period 算、
訓練樣本由 label period 產生，兩段不重疊，故特徵不含 query 自身答案。
推論時 count 由「全部可用過去資料」重算（val 用整個 train；test 用 train+val）。

**特徵**：38 維，全部為組內會變化的 per-candidate 特徵，涵蓋個人歷史 count、
user-context 條件親和、人氣統計、候選時段親和、地理（最近 pin 距離 + 方向位移
+ 多層 grid back-off）、recency。

詳細方法見 `method_lightgbm.md`。

---

## 結果

### val split

| 方法 | Hit@1 | Hit@3 | Hit@5 | NDCG@5 | MRR |
|------|:---:|:---:|:---:|:---:|:---:|
| Fusion (最佳權重) | 0.4418 | 0.6815 | 0.7857 | 0.6242 | 0.5705 |
| **LightGBM Ranker** | **0.5475** | **0.7724** | **0.8541** | **0.7131** | **0.6659** |

### test split

| 方法 | Hit@1 | Hit@3 | Hit@5 | NDCG@5 | MRR |
|------|:---:|:---:|:---:|:---:|:---:|
| Fusion (最佳權重) | 0.4224 | 0.6673 | 0.7775 | 0.6103 | 0.5547 |
| **LightGBM Ranker** | **0.5468** | **0.7744** | **0.8566** | **0.7143** | **0.6666** |

LightGBM val (0.8541) / test (0.8566) 緊貼，無 overfit。

### LightGBM per-segment (test, Hit@5)

| segment | n | Hit@5 | MRR | NDCG@5 |
|---------|:-:|:---:|:---:|:---:|
| 0  | 42,768 | 0.9752 | 0.8025 | 0.8463 |
| 1-5     | 55,804 | 0.8716 | 0.6187 | 0.6819 |
| 6-20    | 37,074 | 0.7412 | 0.5917 | 0.6289 |
| 21-100  | 14,298 | 0.7451 | 0.6433 | 0.6688 |
| 100+    | 56     | 0.2679 | 0.2065 | 0.2215 |

---

## 檔案

| 檔案 | 用途 |
|------|------|
| `suggestion_fusion.py`  | Fusion 模型本體（`SuggestionFusion`） |
| `run_suggestion.py`     | Fusion 評估 / 平行 grid search |
| `gbm_ranker.py`         | LightGBM 核心（`FeatureStore` + `GBMRanker`） |
| `run_gbm.py`            | LightGBM 訓練 / 評估 / test / 存載模型 |
| `tune_gbm.py`           | LightGBM 超參數 coordinate-wise 調優 |
| `check_sugg.py`         | suggestion table 資料行為分析 |
| `diagnose.py`           | training data 資料診斷 |
| `diagnose_suggestion.py`| suggestion recall 確認 |
| `method_lightgbm.md`    | LightGBM 方法詳細說明（report 用） |

依賴：`lightgbm`, `numpy`, `pandas`, `tqdm`

---

## 執行

### Fusion

```bash
# val 評估（最佳權重）
python run_suggestion.py --weights 3.0 10.0 0.0 10.0 3.0 0.5 0.0 0.0 0.1

# 平行 grid search（fit 一次，predict 平行，建議 --workers 16~24）
python run_suggestion.py --grid-search --workers 16

# test 評估
python run_suggestion.py --split test --weights 3.0 10.0 0.0 10.0 3.0 0.5 0.0 0.0 0.1

# per-segment 分析
python run_suggestion.py --weights 3.0 10.0 0.0 10.0 3.0 0.5 0.0 0.0 0.1 --segment
```

### LightGBM

```bash
# 訓練 + val 評估 + per-segment + feature importance
python run_gbm.py --segment --importance

# 超參數調優（特徵只建一次，~30-50 分鐘），調完直接跑 test
python tune_gbm.py --select hit1 --final-test

# test 評估
python run_gbm.py --split test --segment

# 存 / 載入模型
python run_gbm.py --save model_gbm.txt
python run_gbm.py --load model_gbm.txt --split test
```

### LightGBM 最佳超參數

```python
num_leaves=63, min_data_in_leaf=200, learning_rate=0.05,
feature_fraction=0.9, bagging_fraction=0.8, lambda_l2=1.0,
lambdarank_truncation_level=3,
objective="lambdarank", metric="ndcg", ndcg_eval_at=[1,3,5], label_gain=[0,1],
```

---

## 重現注意事項

- LightGBM 的 count 特徵累積期間必須與 split 對應，勿用同一份 data 同時算
  特徵與訓練（leakage）。此邏輯已封裝在 `run_gbm.py` / `tune_gbm.py`。
- suggestion table 每 (uid, end_latlng) 僅一筆，無重複。
- LightGBM 距離使用「最近 pin」：同一截斷座標對應多個精確 pin（平均 7.69，
  62% 一對多），取最近者。