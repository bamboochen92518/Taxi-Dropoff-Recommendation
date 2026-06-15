# GBM Ranker — gbm_ranker/

給定 `uid / start_latlng / hour_type / is_holiday / dayofweek`，
從 suggestion table 的候選集做 Top-K 排序預測 `end_latlng`。
評估指標 Hit@K / NDCG@K / MRR（K = 1, 3, 5）。

---

## 候選生成

對每筆 query，取該 user 在 `address_v2_suggestion.parquet` 中的所有歷史目的地
作為候選集。

- 每個 user 候選數：median = 2、mean = 3.78、p90 = 8、max = 470
- suggestion table 每 (uid, end_latlng) 僅一筆（無重複）；造訪頻率資訊僅在 training data
- 對所有評估 query 的 recall = 100%

---

## 方法：LightGBM Ranker

以 LightGBM LambdaMART 對候選做監督式排序，binary relevance（exact 答案為正例）。

**Leakage 修正**：count 特徵與 label 時間切分。training 切成 feature period（前段）
與 label period（最後 14 天，對齊 val 視窗）；特徵只由 feature period 算、
訓練樣本由 label period 產生，兩段不重疊，故特徵不含 query 自身答案。
推論時 count 由「全部可用過去資料」重算（val 用整個 train；test 用 train+val）。

**特徵（38 維）**，全部為組內會變化的 per-candidate 特徵：

| 類別 | 特徵 |
|------|------|
| 個人歷史 count | `uc3` `uc2` `ua` `us` `sc` `sa` `gc` |
| 距離（最近 pin）+ 方向位移 | `dist_score` `dist_km` `dlat_km` `dlng_km` |
| pin 多樣性 | `n_pins` `n_addr` |
| Recency | `rec_score` `rec_last_d` `rec_first_d` `rec_span_d` |
| 候選自身 context | `holiday_ratio` `hour_cond` `dow_cond` |
| User-context 條件親和 | `uh_cnt` `uh_ratio` `uhol_ratio` `udow_ratio` |
| 地理 backoff | `sa_g1` `sa_g0` |
| 熱門廣度 | `nu` |
| 組內 normalized rank | `r_uc2` `r_ua` `r_us` `r_gc` `r_dist` `r_rec` `r_holiday` `r_hour` `r_uhratio` `r_sag1` `r_nu` |

---

## 結果

### val split

| 方法 | Hit@1 | Hit@3 | Hit@5 | NDCG@5 | MRR |
|------|:---:|:---:|:---:|:---:|:---:|
| **LightGBM Ranker** | **0.5475** | **0.7724** | **0.8541** | **0.7131** | **0.6659** |

### test split

| 方法 | Hit@1 | Hit@3 | Hit@5 | NDCG@5 | MRR |
|------|:---:|:---:|:---:|:---:|:---:|
| **LightGBM Ranker** | **0.5468** | **0.7744** | **0.8566** | **0.7143** | **0.6666** |

val (0.8541) / test (0.8566) 緊貼，無 overfit。

### Per-segment（test, Hit@5）

| segment | n | Hit@5 | MRR | NDCG@5 |
|---------|:-:|:---:|:---:|:---:|
| 0 | 42,768 | 0.9752 | 0.8025 | 0.8463 |
| 1–5 | 55,804 | 0.8716 | 0.6187 | 0.6819 |
| 6–20 | 37,074 | 0.7412 | 0.5917 | 0.6289 |
| 21–100 | 14,298 | 0.7451 | 0.6433 | 0.6688 |
| 100+ | 56 | 0.2679 | 0.2065 | 0.2215 |

---

## 檔案

| 檔案 | 用途 |
|------|------|
| `gbm_ranker.py` | 核心（`FeatureStore` + `GBMRanker`） |
| `run_gbm.py` | 訓練 / 評估 / test / 存載模型 |
| `tune_gbm.py` | 超參數 coordinate-wise 調優 |

依賴：`lightgbm`, `numpy`, `pandas`, `tqdm`

---

## 執行

```bash
# 訓練 + val 評估 + per-segment + feature importance
python run_gbm.py --segment --importance

# test 評估
python run_gbm.py --split test --segment

# 存 / 載入模型
python run_gbm.py --save model_gbm.txt
python run_gbm.py --load model_gbm.txt --split test

# 超參數調優（特徵只建一次，~30–50 分鐘），調完直接跑 test
python tune_gbm.py --select hit1 --final-test
```

---

## 最佳超參數

```python
num_leaves=63, min_data_in_leaf=200, learning_rate=0.05,
feature_fraction=0.9, bagging_fraction=0.8, lambda_l2=1.0,
lambdarank_truncation_level=3,
objective="lambdarank", metric="ndcg", ndcg_eval_at=[1,3,5], label_gain=[0,1],
```

---

## 重現注意事項

- count 特徵累積期間必須與 split 對應，勿用同一份 data 同時算特徵與訓練（leakage）。
  此邏輯已封裝在 `run_gbm.py` / `tune_gbm.py`。
- suggestion table 每 (uid, end_latlng) 僅一筆，無重複。
- 距離使用「最近 pin」：同一截斷座標對應多個精確 pin（平均 7.69，62% 一對多），取最近者。