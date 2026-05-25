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

**為什麼這通常是最強的 baseline**:重度使用者用個人歷史命中,輕度/冷啟動使用者用 start→end 共現補足,兩種使用者都照顧到。對「該 user 沒去過但同區域熱門」的場景特別有效——這也是為什麼它的 Hit@5 比純 UserHistory 高一截。

---

## 在 `val` split 上的實測結果

訓練 750,000 筆,驗證 100,000 筆(時間切分 80/10/10)。

**整體指標** (K=1 / 3 / 5):

| model                  | Hit@1  | Hit@3  | Hit@5  | MRR@5  | NDCG@5 |
| ---------------------- | ------ | ------ | ------ | ------ | ------ |
| `GlobalPopularity`     | 0.0094 | 0.0197 | 0.0264 | 0.0153 | 0.0180 |
| `UserHistory`          | 0.2165 | 0.3258 | 0.3513 | 0.2721 | 0.2921 |
| `UserContextHistory`   | **0.2496** | 0.3315 | 0.3541 | 0.2915 | 0.3073 |
| `StartEndCoOccurrence` | 0.0483 | 0.0887 | 0.1116 | 0.0711 | 0.0812 |
| `HybridUserStartEnd`   | 0.2293 | **0.3565** | **0.3910** | **0.2947** | **0.3190** |

**訓練 / 預測時間**:

| model                  | fit   | predict (100k rows) |
| ---------------------- | ----- | ------------------- |
| `GlobalPopularity`     | 0.1s  | 0.1s                |
| `UserHistory`          | 1.3s  | 0.3s                |
| `UserContextHistory`   | 1.9s  | 0.3s                |
| `StartEndCoOccurrence` | 0.9s  | 7.3s                |
| `HybridUserStartEnd`   | 1.9s  | 5.0s                |

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

5. **`HybridUserStartEnd` 在 Hit@5 / NDCG@5 全面最佳,但 Hit@1 輸給 `UserContextHistory`**
   - 因為 Hybrid 的 tier 1 沒用 context,排序時較弱
   - **如果關心「第一個推薦準不準」(直接 autofill 場景)→ 選 `UserContextHistory`**
   - **如果關心「top-5 名單命中率」(下拉選單場景)→ 選 `HybridUserStartEnd`**
   - 下一步可以做的:把 context 加進 Hybrid 的 tier 1,理論上能兩邊都贏

6. **MRR 跟 Hit@K 趨勢一致** — 沒有出現「Hit 高但 MRR 低」這種「猜對但排很後面」的怪狀況,代表模型排序是合理的。

7. **`GlobalPopularity` 原本的 154 秒是實作問題,不是模型慢** — 已優化到 0.1 秒(作法見下方附錄)。`UserHistory` / `UserContextHistory` 因為很多 user 的歷史很短,所以 predict 反而最快。

### 下一步方向

模型要打贏的目標分數:**Hit@5 ≥ 0.39 / NDCG@5 ≥ 0.32**(`HybridUserStartEnd` 的水準)。

可以嘗試:
- 把 context 加進 Hybrid 的 tier 1(預期能拉高 Hit@1)
- 加時間衰減(近期紀錄權重更大),老資料的 weight 衰減
- ItemCF / UserCF / Matrix Factorization
- 用 `start_latlng + context` 學 embedding,做 ANN 檢索

---

## 附錄: `GlobalPopularity` 預測優化

### 原本的慢實作

```python
def predict_topk(self, query_df, k):
    starts = query_df["start_latlng"].tolist()
    return [_topk_from_counter(self._counter, k, exclude=s) for s in starts]
    # _topk_from_counter 內部呼叫 self._counter.most_common(k + 1)
```

**問題**:`Counter.most_common(k+1)` 內部用 heap 從 N 個唯一下車點挑出 top-(k+1)。
- 複雜度 ~ `O(N · log(k+1))`,而 `N ≈ 60k` 個唯一 `end_latlng`。
- 每一筆 query 都重算一次,總成本 `O(R · N · log K)`,R = 10 萬列。
- 實測 154.6 秒,大部分時間都在 heap 排序同一份計數結果。

### 優化後

```python
def fit(self, train_df):
    self._counter = Counter(train_df["end_latlng"].tolist())
    self.global_top = [x for x, _ in self._counter.most_common(50)]  # fit 階段只算一次
    return self

def predict_topk(self, query_df, k):
    global_top = self.global_top  # 已排序好
    out = []
    for start in query_df["start_latlng"].values:
        picks = []
        for x in global_top:        # 線性掃 ≤ 50 個
            if x == start: continue
            picks.append(x)
            if len(picks) == k: break
        out.append(picks)
    return out
```

**核心想法**:全體熱門排序與 query 完全無關,應該在 `fit` 階段算一次就好,`predict` 只是線性掃過排好的清單,把該筆的 `start_latlng` 過濾掉。

### 複雜度比較

|       | 原本                       | 優化後                |
| ----- | -------------------------- | --------------------- |
| fit   | O(R) 計數                  | O(R) 計數 + O(N log K) 排一次 |
| predict (每筆) | O(N log K)         | O(K)                  |
| predict (R 筆) | **O(R · N log K)** | **O(R · K)**          |

以 R=100k、N≈60k、K=5 代入,理論加速約 `N log K / K ≈ 26,000 倍`。

### 實測結果

| 階段       | 原本   | 優化後 |
| ---------- | ------ | ------ |
| fit        | 0.1s   | 0.1s   |
| predict    | 154.6s | **0.1s** |

實測 **~1500x** 加速(沒到理論上限是因為 Python 迴圈與 tqdm overhead),且輸出結果**完全相同**(Hit@K / MRR / NDCG 數字一致)。

### 相同思路也可套用到其他 baseline

- `UserHistory.predict`:目前每次 `Counter.most_common()` 都會把該 user 全部歷史排序;若改成 fit 階段就把每個 user 的 top-50 算好,predict 也可再加速(目前已經 0.3s,優化效益較小)。
- `StartEndCoOccurrence.predict`:目前 7.3s,主因是每個 `start_latlng` 的候選清單在每筆 query 都重新排序。fit 階段預存每個 start 的 top-50 即可降到 < 1s。
- `HybridUserStartEnd.predict`:同上,結合上述兩個快取後預期可降到 ~1s。

這些優化跟模型品質無關,純粹是工程實作,需要的時候再做即可。
