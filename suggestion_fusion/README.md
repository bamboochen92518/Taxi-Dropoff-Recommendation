# SuggestionFusion

## 方法概述

原本模型（如 HybridContextCascade）的 Hit@5 天花板約在 0.40，根本原因是候選集召回率只有 35%——val 有 65% 的下車點在 training data 裡從未出現過，再好的排序也無濟於事。

本方法改用 `address_v2_suggestion.parquet` 作為候選集來源，recall rate提升至 100%，再以 training data 的多維統計信號加權排序。

**核心設計**：
- **候選集**：該 user 在 suggestion table 中的所有歷史目的地（不截斷）
- **排序**：9 個統計信號的加權對數融合，參數由 val 上的 coordinate-wise grid search 決定

---

## 模型設計

### 候選生成

對每筆 query，取該 user 在 suggestion table 中的所有 `end_latlng` 作為候選。

- 中位數每人 6 個候選、平均 10 個、最多 470 個
- 所有 val 下車點均在候選集內（recall = 100%）

### 排序公式

對每個候選 `end_latlng` 計算分數：

```
score = w0 · log(1 + count[user, hour, holiday, dow → end])
      + w1 · log(1 + count[user, hour, holiday → end])
      + w2 · log(1 + count[user → end])
      + w3 · log(1 + count[user, start → end])
      + w4 · log(1 + count[start, hour, holiday → end])
      + w5 · log(1 + count[start → end])
      + w6 · log(1 + count[global → end])
      + w7 · log(1 + score[user, address → end])
      + w8 · 1/(1 + distance_km)
```

分數由高到低排序，回傳 top-K。

### 最佳權重（val grid search）

```
w = [3.0, 10.0, 0.0, 10.0, 3.0, 0.5, 0.0, 0.0, 0.1]
```

`user × (hour, holiday)` 與 `user × start` 為最強信號（各 10），捕捉時段習慣與通勤模式。

---

## 腳本

- `suggestion_fusion.py` — `SuggestionFusion` 模型本體
- `run_suggestion.py` — 評估與 grid search 腳本
- `diagnose.py` — 資料診斷：repeat rate、oracle 上限、用戶分段
- `diagnose_suggestion.py` — 確認 suggestion table 召回率

```bash
# 資料診斷（建議先跑）
python suggestion_fusion/diagnose.py
python suggestion_fusion/diagnose_suggestion.py

# val 評估，使用最佳參數
python suggestion_fusion/run_suggestion.py \
    --weights 3.0 10.0 0.0 10.0 3.0 0.5 0.0 0.0 0.1

# val 評估，使用預設參數
python suggestion_fusion/run_suggestion.py

# grid search 找最佳權重（約 30-60 分鐘）
python suggestion_fusion/run_suggestion.py --grid-search

# test 評估
python suggestion_fusion/run_suggestion.py \
    --split test \
    --weights 3.0 10.0 0.0 10.0 3.0 0.5 0.0 0.0 0.1

# per-segment 分析
python suggestion_fusion/run_suggestion.py \
    --weights 3.0 10.0 0.0 10.0 3.0 0.5 0.0 0.0 0.1 \
    --segment
```

---

## 資料

與根目錄共用，不另外放置：

- `1.下車地址推薦/address_v2_training_data.parquet`
- `1.下車地址推薦/address_v2_suggestion.parquet`

---

## 在 `val` split 上的實測結果

訓練 750,000 筆，驗證 100,000 筆（時間切分 75/10/15）。

**整體指標** (K=1 / 3 / 5)：

| model | Hit@1 | Hit@3 | Hit@5 | MRR@5 | NDCG@5 |
| ----- | ----- | ----- | ----- | ----- | ------ |
| `SuggestionFusion` (預設權重) | 0.3909 | 0.6277 | 0.7465 | 0.5208 | 0.5769 |
| **`SuggestionFusion` (最佳權重)** | **0.4382** | **0.6787** | **0.7852** | **0.5677** | **0.6219** |

**訓練 / 預測時間**：

| model | fit | predict (100k rows) |
| ----- | --- | ------------------- |
| `SuggestionFusion` | 15s | 10s |

**test split 最終結果**：

| model | Hit@1 | Hit@3 | Hit@5 | MRR@5 | NDCG@5 |
| ----- | ----- | ----- | ----- | ----- | ------ |
| `SuggestionFusion` (最佳權重) | 0.4224 | 0.6673 | 0.7775 | 0.5547 | 0.6103 |

### 結果解讀

1. **候選集是關鍵** — Training data 的 exact repeat rate 僅 35.1%，代表 train-only 模型最多只能命中 35% 的 val 查詢，其餘完全無能為力。suggestion table 將召回率提升至 100%，是分數大幅跳升的根本原因。

2. **問題轉化** — 候選集換掉後，問題從「從數萬個地點中猜對一個」變成「從該用戶 6 個已知目的地中選最可能的 5 個」。後者難度遠低於前者，Hit@5 因此得以突破 0.75。

3. **最強信號是通勤模式** — Grid search 發現 `user × (hour, holiday)` 與 `user × start` 各拿到權重 10，遠高於其他信號。前者捕捉「平日早上去公司、假日晚上去餐廳」的時段規律，後者捕捉「從家出發去公司、從公司回家」的出發地規律，兩者合力大幅提升 top-1 排序品質（Hit@1 從 0.39 → 0.44）。

4. **部分信號最佳權重為 0** — `user_all`、`global`、`poi` 的最佳權重為 0。原因是候選集已來自個人 suggestion，所有候選都是該用戶去過的地方，純頻率或全體熱門度反而引入排序雜訊，不如不用。

5. **val / test 接近** — val Hit@5 = 0.7852，test Hit@5 = 0.7775，差距 0.77%，沒有 overfitting 跡象，代表參數具有良好泛化能力。