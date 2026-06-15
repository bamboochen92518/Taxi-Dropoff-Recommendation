# Suggestion Fusion — suggestion_fusion/

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

## 方法：Fusion

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

## 結果

### val split

| 方法 | Hit@1 | Hit@3 | Hit@5 | NDCG@5 | MRR |
|------|:---:|:---:|:---:|:---:|:---:|
| Fusion（最佳權重） | 0.4418 | 0.6815 | 0.7857 | 0.6242 | 0.5705 |

### test split

| 方法 | Hit@1 | Hit@3 | Hit@5 | NDCG@5 | MRR |
|------|:---:|:---:|:---:|:---:|:---:|
| Fusion（最佳權重） | 0.4224 | 0.6673 | 0.7775 | 0.6103 | 0.5547 |

---

## 檔案

| 檔案 | 用途 |
|------|------|
| `suggestion_fusion.py` | Fusion 模型本體（`SuggestionFusion`） |
| `run_suggestion.py` | 評估 / 平行 grid search |

依賴：`numpy`, `pandas`, `tqdm`

---

## 執行

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

---

## 重現注意事項

- suggestion table 每 (uid, end_latlng) 僅一筆，無重複。
- grid search 建議 `--workers 16~24`（48 CPU 環境），fit 只做一次，predict 平行跑。