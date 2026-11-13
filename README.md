# FIT 面試 - 范佐搖

三個題目各放在一個資料夾,這一頁先放結果,細節在各題自己的說明裡。

- [Task 1 一般問答](task1_general/README.md)
- [Task 2 機器學習](task2_ml/README.md)
- [Task 3 電腦視覺](task3_cv/README.md)

## Task 2:預測客戶流失的機率

目標:用 `train.csv` 訓練模型,預測 `test.csv` 每位客戶離開銀行(`Exited`)的機率,並且評估模型、畫圖呈現。

資料:

| 項目 | 內容 |
|---|---|
| 一筆資料 | 一位銀行客戶 |
| 筆數 | 訓練 165,034 筆、測試 110,023 筆,沒有缺值 |
| 要預測的目標 | `Exited`,1 代表離開,佔 21.2% |
| 可以用的欄位 | 10 個客戶屬性:信用分數、國家、性別、年齡、往來年數、餘額、持有產品數、有沒有信用卡、是否活躍、估計年薪 |
| 沒有使用 | `id`、`CustomerId`、`Surname`,是識別用的 |

看資料的時候發現,持有產品數和年齡跟會不會離開的關係最明顯,而且都不是一直線(例如持有 2 個產品的人最不會離開,1 個和 3 個以上的都比較會),所以預期樹模型會比線性模型適合。詳細的圖在 [Task 2 的說明](task2_ml/README.md)裡。

看完資料之後先比了四種模型,都用同一套 5-fold 交叉驗證。指標主要看 AUC,因為要交的是機率:

| 模型 | AUC |
|---|---|
| 邏輯迴歸 | 0.8705 |
| Random forest | 0.8869 |
| XGBoost | 0.8897 |
| LightGBM | 0.8899 |

XGBoost 和 LightGBM 一樣好,我選 LightGBM。之後試了調參數、加 class weight、自己做特徵、跟 XGBoost 平均,都沒有再進步,所以就定案了。

定案模型的 AUC 是 0.8899。把機率切成會不會流失的話,門檻用 0.32,F1 是 0.665(precision 0.637、recall 0.695)。

![roc pr](task2_ml/outputs/05_roc_pr_curves.png)

`test.csv` 的預測在 [task2_ml/outputs/predictions.csv](task2_ml/outputs/predictions.csv)。

## Task 3:MNIST 數字分類

目標:用 CNN 做手寫數字分類,並且在模型設計、優化、類別不平衡三件事上有所展現。

資料:

| 項目 | 內容 |
|---|---|
| 一筆資料 | 一張 28x28 的灰階手寫數字圖,像素值 0 到 255 |
| 要預測的目標 | 圖上寫的是哪個數字,0 到 9 共 10 類 |
| 筆數 | 官方訓練 60,000 張、測試 10,000 張 |
| 我的切法 | 訓練 54,000、驗證 6,000(每個數字 600 張)、測試 10,000 |

原本的資料每個數字都有五、六千張,是平衡的。題目要的類別不平衡得自己做:我另外做了一份訓練集,數字 0 留 5,000 張,一路遞減到數字 9 只有 50 張,差 100 倍。

下面是資料的樣子。左上是每個數字的樣本,左下是 augmentation 之後的效果,右邊是兩種訓練集的張數分布。

![data](task3_cv/outputs/01_data_overview.png)

看圖可以發現同一個數字的寫法差很多(像 7 有沒有中間那一橫、4 上面有沒有封起來),有些數字本來就長得像(4 和 9、3 和 5)。後面模型錯的也大多是這幾組。

題目指定的三件事各做一組實驗,每組都從 baseline 開始一次改一個東西,用驗證集比較,每個設定跑 3 個 seed。

| | 試了什麼 | 最後用的 | 效果 |
|---|---|---|---|
| 模型 | LeNet-5、加深的 CNN、加 BN、加 dropout 和 augmentation | DigitNet + BN + dropout + augmentation | 測試錯誤 73 張降到 43 張(每一萬張) |
| 優化 | SGD、momentum、Adam、AdamW,固定學習率和 one-cycle | AdamW + one-cycle | 錯誤率 1.9% 降到 0.5% |
| 類別不平衡 | class weight、平衡抽樣、focal loss、logit adjustment | Logit-adjusted loss + augmentation | 準確率 97.4% 拉回 98.9% |

定案的模型有 72,666 個參數,測試準確率 99.64%(一萬張錯 36 張)。每一組實驗的數字和圖在 [Task 3 的說明](task3_cv/README.md)裡。

類別不平衡的部分,數字 0 留 5,000 張,一路遞減到數字 9 只有 50 張。不處理的話數字 9 的 recall 只有 90.4%,定案的做法可以拉到 96.6%。下圖紅線是不處理,藍線是定案的做法。橫軸是數字,括號裡是訓練張數,越右邊越少;縱軸是測試的 recall。

![imbalance](task3_cv/outputs/07_imbalance_per_class_recall.png)

## 怎麼跑

```bash
pip install -r requirements.txt
cd task2_ml && python main.py                    # 約 1 分半
cd ../task3_cv && python main.py --workers 3    # 有 GPU 約 8 分鐘,只有 CPU 會很久
```

結果我都跑好放在各題的 `outputs/` 了,不重跑也可以直接看。
