# 下車地址推薦系統
## Drop-off Address Recommendation System
### 模型訓練與評估報告 | AI 課程期末專案 × LineGO | 2026 年 6 月

---

## 一、問題描述

### 1.1 題目背景

本題由 LineGO 提出，核心目標為：在用戶輸入上車地址後，系統能即時推薦最有可能的下車地址排序清單，協助用戶快速完成叫車行為。題目要求推薦結果需考量不同時段、平日／假日差異，以及用戶個人屬性，以在不同叫車情境下提供最貼近用戶需求的建議。

傳統叫車流程中，用戶每次都需手動輸入目的地，耗時且容易重複輸入相同地點。若系統能根據歷史行為與當前情境主動預測目的地，可大幅降低叫車摩擦，提升使用體驗。

### 1.2 資料說明

本題提供兩張資料表，時間範圍均為 2026 年 1 月至 5 月：

**表 1｜address_v2_training_data（行程日誌）**

| 欄位名稱 | 資料型態 | 欄位說明 | 備註 |
|---|---|---|---|
| uid_hash | STRING | 乘客 ID 經 SHA256 雜湊後的匿名識別碼 | 64 字元 hex |
| start_latlng | STRING | 上車經緯度（精度至小數點第 2 位，約 1.1 km） | 區域級精度 |
| end_latlng | STRING | 下車經緯度（精度至小數點第 3 位，約 110 m） | 地址級精度 |
| created_at | TIMESTAMP | 叫車時間點（UTC） | |
| hour_type | STRING | 時段分類：凌晨、午夜、早尖峰、早離峰、午離峰、小晚尖、晚尖峰 | 共 7 類 |
| is_holiday | STRING | 是否為假日（"0" 平日 / "1" 假日） | |
| dayofweek | STRING | 星期幾（"1"~"7"） | |

資料規模：**1,000,000 筆**行程、325,717 位用戶，無缺失值，檔案大小約 305 MB。

**表 2｜address_v2_suggestion（地址對照表）**

| 欄位名稱 | 資料型態 | 欄位說明 | 備註 |
|---|---|---|---|
| uid_hash | STRING | 同上，乘客匿名識別碼 | 與訓練資料可 JOIN |
| end_latlng | STRING | 下車點經緯度（精度至小數點第 3 位） | JOIN key |
| end_address | STRING | 下車地址名稱（中文） | 推薦顯示用 |
| end_latlng_pin | STRING | 下車點精確經緯度（完整 pin 點） | 地圖定位用 |
| updated_date | DATE | 資料更新日期 | |

資料規模：**2,493,639 筆**、659,806 位用戶，無缺失值，檔案大小約 254 MB。此表由 LineGO 系統累積的用戶歷史目的地記錄構成，同時作為推薦候選集來源及地址查找表。

### 1.3 目標輸出

給定用戶 ID、當前上車位置、時段、假日狀態及星期，系統輸出該用戶最有可能的 Top-K 下車地址排序清單（以 `end_latlng` 表示）。評估重點在於正確答案是否排在前幾名、排名是否越前越好。

---

## 二、系統設計與演進

### 2.1 資料切分策略

為避免 data leakage，採用**時序切分**而非隨機切分。全量 1,000,000 筆行程依 `created_at` 排序後，按筆數比例切成三個不重疊的時間段：

| Split | 比例 | 筆數 | 用戶數 | 時間範圍 | 用途 |
|---|---|---|---|---|---|
| Train | 75% | 750,000 | 273,683 | 2026/01/01 ~ 2026/04/14 | 模型訓練、特徵計算 |
| Val | 10% | 100,000 | 65,461 | 2026/04/14 ~ 2026/04/28 | 模型選擇與調參 |
| Test | 15% | 150,000 | 89,435 | 2026/04/28 ~ 2026/05/17 | 最終評估（訓練期間完全不碰） |

頻率查找表（lookups）**只用 train 行程計算**，避免 val/test 期間的行程次數洩漏進特徵。

### 2.2 評估方式

評估採用統一介面（`evaluate.py`），以**每一筆行程為一個獨立 query**：

| 指標 | 定義 |
|---|---|
| Hit@K | 正確答案是否出現在前 K 名（= Recall@K） |
| MRR | Mean Reciprocal Rank，正確答案排名倒數的平均值 |
| NDCG@K | Normalized DCG，對排名位置加權後正規化的綜合品質分數 |

### 2.3 方法一：LightGBM LambdaRank

**設計思路**：兩階段推薦架構，Stage 1 從 suggestion 表撈出候選集，Stage 2 用 LightGBM LambdaRank 學習排序。

特徵設計 9 個 log 信號（與 SuggestionFusion 相同來源）：

| 特徵 | 意義 |
|---|---|
| log_uc3 | log(1 + count[user × hour × holiday × dow → end]) |
| log_uc2 | log(1 + count[user × hour × holiday → end]) |
| log_ua | log(1 + count[user → end]) |
| log_us | log(1 + count[user × start → end]) |
| log_sc | log(1 + count[start × hour × holiday → end]) |
| log_sa | log(1 + count[start → end]) |
| log_gc | log(1 + count[global → end]) |
| log_poi | log(1 + count[user × address → end]) |
| dist | 1 / (1 + distance_km)，上下車距離 |

**遇到的問題：log_uc3 主導導致其他特徵無法學習**

訓練後發現 `log_uc3`（Feature importance Gain = 654,930）遠超其他所有特徵（次高僅 9,953），Early stopping 在第 1 輪就觸發（val NDCG@1 = 0.9123）。

根本原因是訓練集的**分佈不一致**：訓練集正樣本天生就有較高的 log_uc3（因為行程本身就是計數的來源），第一棵樹 split 在 log_uc3 就能完美區分正負樣本；但 val/test 的大多數 query 是新情境（log_uc3 = 0），模型退化到只靠幾個 Gain ≈ 0 的特徵排序。

嘗試的改進方向：
- 拿掉 log_uc3：次強特徵 log_uc2 立刻接管，問題完全相同
- 增加 NEG_RATIO（4 → 10）：log_uc3 Gain 反而從 654,930 增至 787,870，問題更嚴重
- 調小 feature_fraction、learning_rate：Early stopping 依然第 1 輪觸發

**結論**：這類「用戶 × 細粒度情境」頻率特徵在訓練集上天生是 oracle，LambdaRank 無法從這種分佈學到有效的 fallback 策略，LightGBM 不適合直接用於此問題。

**LightGBM val 最終結果**：

| K | Hit@K | MRR | NDCG@K |
|---|---|---|---|
| 1 | 0.3617 | 0.3617 | 0.3617 |
| 3 | 0.5979 | 0.4642 | 0.4984 |
| 5 | 0.7228 | 0.4926 | 0.5498 |

### 2.4 方法二：SuggestionFusion + 時間衰減

**候選集召回率是根本瓶頸**

分析發現 training data 的 exact repeat rate 只有 35.1%，代表 val 有 64.9% 的正確答案在 train 行程裡從未出現過。再好的排序對這 64.9% 也無能為力，這才是 train-only 模型的真正天花板。

`address_v2_suggestion` 表的候選集對 val 的 recall 接近 100%，且**中位數每人只有 6 個候選地址**，問題從「從數萬個地址猜一個」縮小成「從 6 個裡選最可能的 5 個」，Hit@5 因此能突破 0.75。

**SuggestionFusion 設計**：以 suggestion 表為候選集，用 9 個 log 信號加權融合排序，權重由 val 上的 coordinate-wise grid search 決定：

```
score = w0·log(1+uc3) + w1·log(1+uc2) + w2·log(1+ua) + w3·log(1+us)
      + w4·log(1+sc) + w5·log(1+sa) + w6·log(1+gc) + w7·poi + w8·dist
最佳權重：w = [3.0, 10.0, 0.0, 10.0, 3.0, 0.5, 0.0, 0.0, 0.1]
```

**時間衰減改進**：在 SuggestionFusion 基礎上加入指數時間衰減，讓近期行程權重更高：

```
weight = exp(-decay_rate × days_ago)
```

在 val 上比較不同 decay_rate：

| decay_rate | Hit@1 | Hit@5 | NDCG@5 | MRR |
|---|---|---|---|---|
| 0.0（無衰減）| 0.4382 | 0.7852 | 0.6219 | 0.5677 |
| **0.01（最佳）** | **0.4387** | **0.7854** | **0.6223** | **0.5681** |
| 0.02 | 0.4353 | 0.7847 | 0.6202 | 0.5656 |
| 0.05 | 0.4216 | 0.7788 | 0.6106 | 0.5547 |

decay=0.01 帶來小幅但一致的提升，decay 過大則會損害效果（過度降低較久之前的穩定習慣行程的權重）。

---

## 三、最終結果

### 3.1 最佳模型：SuggestionFusion（decay=0.01）

**Val split 結果（100,000 筆）：**

| K | Hit@K | MRR | NDCG@K |
|---|---|---|---|
| 1 | 0.4387 | 0.4387 | 0.4387 |
| 3 | 0.6789 | 0.5437 | 0.5784 |
| 5 | 0.7854 | 0.5681 | 0.6223 |

**Test split 最終結果（150,000 筆）：**

| K | Hit@K | MRR | NDCG@K |
|---|---|---|---|
| 1 | 0.4226 | 0.4226 | 0.4226 |
| 3 | 0.6674 | 0.5297 | 0.5651 |
| 5 | 0.7778 | 0.5550 | 0.6106 |

Val 和 test 差距約 1.6pp（Hit@5），沒有過擬合跡象，代表模型具有良好的時間泛化能力。

### 3.2 與所有方法對比（val split）

| 模型 | Hit@1 | Hit@3 | Hit@5 | MRR | NDCG@5 |
|---|---|---|---|---|---|
| GlobalPopularity | 0.0094 | 0.0197 | 0.0264 | 0.0153 | 0.0180 |
| UserHistory | 0.2165 | 0.3258 | 0.3513 | 0.2721 | 0.2921 |
| UserContextHistory | 0.2496 | 0.3315 | 0.3541 | 0.2915 | 0.3073 |
| HybridContextCascade | 0.2625 | 0.3622 | 0.3938 | 0.3142 | 0.3342 |
| HybridContextPlus | 0.2723 | 0.3634 | 0.3941 | 0.3198 | 0.3385 |
| LightGBM LambdaRank（本組）| 0.3617 | 0.5979 | 0.7228 | 0.4926 | 0.5498 |
| SuggestionWeightedFusion | 0.4115 | 0.6346 | 0.7458 | 0.5340 | 0.5867 |
| SuggestionFusion（本組，無衰減）| 0.4382 | 0.6787 | 0.7852 | 0.5677 | 0.6219 |
| **SuggestionFusion + decay=0.01（本組）** | **0.4387** | **0.6789** | **0.7854** | **0.5681** | **0.6223** |

---

## 四、結論與未來方向

### 4.1 結論

本次專案探索了兩條技術路線：

**LightGBM LambdaRank**：超越所有 train-only 規則式 baseline（val Hit@5 = 0.7228），但受限於訓練集分佈不一致問題（log_uc3 特徵主導），無法進一步改善。這個探索過程揭示了「用戶 × 細粒度情境」頻率特徵在機器學習排序中的本質問題。

**SuggestionFusion + 時間衰減**：透過分析候選集召回率（training-only = 35.1%，suggestion = 100%）找到根本瓶頸，改用 suggestion 表作為候選集後大幅提升，最終在 val 上達到 Hit@5 = 0.7854、NDCG@5 = 0.6223，超越組員原版 SuggestionWeightedFusion（Hit@5 = 0.7458）約 4pp。加入時間衰減（decay=0.01）後進一步小幅提升，test Hit@5 = 0.7778。

### 4.2 未來可改進方向

- **Sequential Model**：將用戶行程序列建模，用 GRU 或 Transformer 捕捉行為的時序模式，預測下一個目的地
- **Pointwise 二分類**：在 SuggestionFusion 的 9 個 log 信號上用 XGBoost 做二分類而非 LambdaRank，理論上可讓模型自動學習最佳信號權重且不受分佈問題影響
- **冷啟動優化**：目前新用戶 fallback 全局熱門，可引入協同過濾找相似用戶的偏好改善
- **更細緻的時間衰減**：針對不同信號設計不同的衰減率（如通勤規律衰減慢、娛樂目的地衰減快）

---

## 成員貢獻

（請各組依實際情況填寫）

| 姓名 | 學號 | 貢獻說明 |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
