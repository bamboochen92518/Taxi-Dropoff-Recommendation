# Baseline 模型設計

本檔說明 `baselines.py` 中所有 baseline 的設計思路與 fallback 邏輯,並附上在 `val` split 上的實測結果。

所有 baseline 共用介面:

```python
model.fit(train_df)
model.predict_topk(query_df, k) -> list[list[str]]   # 每列 query 回傳 top-k 個 end_latlng
```

共同規則:
- 預測時會把該筆 query 的 `start_latlng` 從候選清單中排除(上車點不可能是下車點)。
- 所有模型最後都會 fallback 到全體最熱門的下車點,保證 top-k 一定能填滿。

---

## 1. `GlobalPopularity` — 全體熱門
**邏輯**:不看任何 context,直接回傳全體訓練資料中出現次數最多的 `end_latlng`。

**用途**:最弱的 sanity check baseline。如果一個模型贏不了 GlobalPopularity,代表它完全沒學到東西。

---

## 2. `UserHistory` — 該 user 歷史最常去
**邏輯**:統計每個 `uid_hash` 在訓練資料中出現過的 `end_latlng` 次數,推薦排名前 K 個。

**Fallback**:該 user 沒有歷史紀錄(cold start)→ 用 GlobalPopularity。

**為什麼這是強 baseline**:這份資料平均每個 user 有 5+ 筆紀錄,而下車地點高度個人化(住家、公司、常去店家),光是「重複自己去過的地方」就會命中很多次。

---

## 3. `UserContextHistory` — 該 user × (時段, 假日) 最常去
**邏輯**:加入 context,把 user 的歷史依 `(hour_type, is_holiday)` 拆桶。例如「早上 + 平日」最常去的、「晚上 + 假日」最常去的分開算。

**Fallback**:
1. 該 user 在這個 context 沒紀錄 → 退到該 user 全部歷史
2. 該 user 完全沒紀錄 → 退到 GlobalPopularity

**為什麼有用**:同一個人平日早上多半去公司、假日晚上多半去娛樂場所——加入時段/假日後 Hit@1 通常顯著提升。

---

## 4. `StartEndCoOccurrence` — 上車點 → 下車點 共現
**邏輯**:統計 `(start_latlng, end_latlng)` 在訓練資料中的共現次數。給一個 `start_latlng`,推薦最常一起出現的 `end_latlng`。

**Fallback**:該 `start_latlng` 沒見過 → GlobalPopularity。

**用途**:完全不看 user,只看「從這個區域出發的人通常往哪裡去」。可以推薦給冷啟動的新使用者。

**限制**:`start_latlng` 只到小數點 2 位、`end_latlng` 到 3 位,共現空間很稀疏,單獨用效果有限。

---

## 5. `HybridUserStartEnd` — 個人歷史 + 上車點共現 + 全體熱門 (三層)
**邏輯**:按優先順序依序填滿 K 個候選:
1. 該 user 最常去的(去重後不夠 K 個)
2. 從這個 `start_latlng` 出發最常去的(補到 K 個還不夠)
3. 全體熱門補到 K 個

**為什麼這是強 baseline**:重度使用者用個人歷史命中,輕度/冷啟動使用者用 start→end 共現補足,兩種使用者都照顧到。但三層都**沒用到 context**(hour_type / is_holiday),這也是下一層可以改進的點。

---

## 6. `HybridContextCascade` — 加入 context tier 的四層 cascade
**邏輯**:在 `HybridUserStartEnd` 的最前面加上 `user × (hour_type, is_holiday)` tier:
1. 該 user 在這個 context 下最常去的
2. 該 user 全部歷史最常去的
3. 從這個 `start_latlng` 出發最常去的
4. 全體熱門

**設計動機**:`UserContextHistory` 在 Hit@1 贏 `HybridUserStartEnd` 是因為 context;反之在 Hit@5 輸是因為候選不夠多。這個設計把兩者的優勢加起來:用 context 推 top-1,不夠再拿 user / start 訓有來補。

---

## 在 `val` split 上的實測結果

訓練 750,000 筆,驗證 100,000 筆(時間切分 80/10/10)。

**整體指標** (K=1 / 3 / 5):

| model                   | Hit@1  | Hit@3  | Hit@5  | MRR@5  | NDCG@5 |
| ----------------------- | ------ | ------ | ------ | ------ | ------ |
| `GlobalPopularity`      | 0.0094 | 0.0197 | 0.0264 | 0.0153 | 0.0180 |
| `UserHistory`           | 0.2165 | 0.3258 | 0.3513 | 0.2721 | 0.2921 |
| `UserContextHistory`    | 0.2496 | 0.3315 | 0.3541 | 0.2915 | 0.3073 |
| `StartEndCoOccurrence`  | 0.0483 | 0.0887 | 0.1116 | 0.0711 | 0.0812 |
| `HybridUserStartEnd`    | 0.2293 | 0.3565 | 0.3910 | 0.2947 | 0.3190 |
| **`HybridContextCascade`**  | **0.2625** | **0.3622** | **0.3938** | **0.3142** | **0.3342** |

**訓練 / 預測時間**:

| model                   | fit   | predict (100k rows) |
| ----------------------- | ----- | ------------------- |
| `GlobalPopularity`      | 0.1s  | 0.1s                |
| `UserHistory`           | 1.3s  | 0.3s                |
| `UserContextHistory`    | 1.9s  | 0.3s                |
| `StartEndCoOccurrence`  | 0.9s  | 7.3s                |
| `HybridUserStartEnd`    | 1.9s  | 5.0s                |
| `HybridContextCascade`  | 2.4s  | 5.0s                |

> `GlobalPopularity` 的 predict 原本要 154.6s,優化後降到 0.1s(**~1500x**),作法見下方「附錄: GlobalPopularity 預測優化」。

### 結果解讀

1. **`GlobalPopularity` 慘到極點(Hit@5 < 3%)** — 預期之內。下車地點空間極稀疏(每個經緯度 bucket 命中率非常低),不個人化就是不行。它存在的意義只是當作 floor,確認其他模型確實有學到東西。

2. **個人化是最大關鍵跳躍** — 從 `GlobalPopularity` 的 0.9% 跳到 `UserHistory` 的 21.6% Hit@1,單純「記住這個人去過哪」就把命中率拉了 **20 倍以上**。這驗證了「下車地點高度個人化」這個假設,也說明資料裡的重度使用者(行程 > 20 筆的 ~5,000 位)貢獻了大部分命中。

3. **Context 對 top-1 幫助最大,對 top-5 邊際效益小**
   - `UserContextHistory` vs `UserHistory`:Hit@1 從 21.6% → 25.0%(**+3.3pp**)
   - 但 Hit@5 只從 35.1% → 35.4%(**+0.3pp**)
   - 解讀:時段/假日資訊主要是在「重排」該 user 的常去地點(早上排公司在前、晚上排家在前),但候選集合本身沒擴大,所以 K 變大後優勢就被吃掉。

4. **`StartEndCoOccurrence` 單獨用很弱(Hit@5 = 11.2%)但很互補**
   - 單獨看比 `UserHistory` 差 3 倍以上
   - 但放進 Hybrid 之後,Hit@5 從 35.1%(UserHistory)→ 39.1%(Hybrid),**多賺 4pp**
   - 解讀:它能命中的是「該 user 沒去過、但同區域大家會去」的地點,屬於「探索性候選」,跟個人歷史不重疊,所以加進來純賺。

5. **`HybridUserStartEnd` 在 Hit@5 / NDCG@5 及格,但 Hit@1 輸給 `UserContextHistory`**
   - 因為 Hybrid 的 tier 1 沒用 context,排序時較弱
   - 這個觀察促成了下面的 `HybridContextCascade`

6. **`HybridContextCascade` 是新的全面贏家** — 加入 user × context tier 後:
   - Hit@1: 0.2293 → **0.2625**(+3.3pp,甚至超過 `UserContextHistory` 的 0.2496)
   - Hit@5: 0.3910 → **0.3938**(+0.3pp)
   - NDCG@5: 0.3190 → **0.3342**(+1.5pp)
   - 預測時間幾乎不變(5.0s),是全 K 都贏的新 SOTA。

7. **MRR 跟 Hit@K 趨勢一致** — 沒有出現「Hit 高但 MRR 低」這種「猜對但排很後面」的怪狀況,代表模型排序是合理的。

8. **`GlobalPopularity` 原本的 154 秒是實作問題,不是模型慢** — 已優化到 0.1 秒(作法見下方附錄)。`UserHistory` / `UserContextHistory` 因為很多 user 的歷史很短,所以 predict 反而最快。

### 下一步方向

目前 SOTA:**`HybridContextCascade` (Hit@5 = 0.3938 / NDCG@5 = 0.3342)**。

可以嘗試:
- 加權分數融合(把 user × ctx / user / start / global 的 log-count 線性組合後排序),在 val 上 grid search 權重
- 加時間衰減(近期紀錄權重更大)
- 把 `start × context` 也快取起來,加進 Cascade 的 tier 3 之前
- ItemCF / UserCF / Matrix Factorization
- 用 `start_latlng + context` 學 embedding,做 ANN 檢索
