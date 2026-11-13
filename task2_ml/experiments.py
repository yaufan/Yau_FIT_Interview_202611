"""定案之前試過的做法。

用法:
    python experiments.py

每個做法都用跟 main.py 一樣的 5-fold 切法,比 out-of-fold 的 AUC / log loss / Brier。
結果存在 outputs/experiments.csv 和 outputs/11_experiments.png。
"""
from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

import main as M


def to_numeric(X: pd.DataFrame) -> pd.DataFrame:
    """category 欄位轉成整數代碼,random forest 要用。"""
    X = X.copy()
    for c in X.columns:
        if isinstance(X[c].dtype, pd.CategoricalDtype):
            X[c] = X[c].cat.codes
    return X


# 每個做法是一個函式:給訓練資料和驗證資料,回傳驗證資料的預測機率
def lgbm(params=None):
    def run(X_tr, y_tr, X_va, seed):
        model, _ = M.fit_lgbm_fold(X_tr, y_tr, seed, params)
        return model.predict_proba(X_va)[:, 1]
    return run


def logreg(X_tr, y_tr, X_va, seed):
    return M.make_logreg().fit(X_tr, y_tr).predict_proba(X_va)[:, 1]


def random_forest(X_tr, y_tr, X_va, seed):
    m = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, n_jobs=M.LGB_PARAMS["n_jobs"], random_state=seed)
    return m.fit(to_numeric(X_tr), y_tr).predict_proba(to_numeric(X_va))[:, 1]


def xgboost(X_tr, y_tr, X_va, seed):
    # 跟 LightGBM 一樣,從訓練資料再切 10% 做 early stopping
    a, b, ya, yb = train_test_split(X_tr, y_tr, test_size=0.1, stratify=y_tr, random_state=seed)
    m = XGBClassifier(n_estimators=3000, learning_rate=0.03, max_depth=6, min_child_weight=5, subsample=0.8,
                      colsample_bytree=0.7, reg_lambda=5.0, tree_method="hist", enable_categorical=True,
                      early_stopping_rounds=150, eval_metric="logloss", n_jobs=M.LGB_PARAMS["n_jobs"], random_state=seed)
    m.fit(a, ya, eval_set=[(b, yb)], verbose=False)
    return m.predict_proba(X_va)[:, 1]


OOF = {}  # 每個做法的 out-of-fold 預測


def score(name, group, oof, y, seconds=0.0):
    """算指標。fold 的切法固定,所以可以直接拿 oof 重算每個 fold 的 AUC。"""
    skf = StratifiedKFold(n_splits=M.N_SPLITS, shuffle=True, random_state=M.SEED)
    aucs = [roc_auc_score(y[va], oof[va]) for _, va in skf.split(oof, y)]
    OOF[name] = oof
    row = dict(group=group, name=name, auc=roc_auc_score(y, oof), fold_auc_sd=np.std(aucs, ddof=1),
               log_loss=log_loss(y, oof), brier=brier_score_loss(y, oof), seconds=seconds)
    print(f"  {name:36s} AUC {row['auc']:.4f} ± {row['fold_auc_sd']:.4f}  logloss {row['log_loss']:.4f}  "
          f"brier {row['brier']:.4f}  ({seconds:.0f}s)", flush=True)
    return row


def cross_validate(name, group, run, X, y):
    skf = StratifiedKFold(n_splits=M.N_SPLITS, shuffle=True, random_state=M.SEED)
    oof, t0 = np.zeros(len(y)), time.time()
    for k, (tr, va) in enumerate(skf.split(X, y)):
        oof[va] = run(X.iloc[tr], y[tr], X.iloc[va], M.SEED + k)
    return score(name, group, oof, y, time.time() - t0)


def main():
    M.OUT_DIR.mkdir(exist_ok=True)
    train, test = M.load_data()
    levels = M.category_levels(train, test)
    y = train[M.TARGET].to_numpy()
    X = M.build_features(train, levels)
    rows = []

    print("[1/3] models")
    rows.append(cross_validate("Logistic regression", "model", logreg, train, y))  # 邏輯迴歸的 pipeline 自己做 one-hot
    rows.append(cross_validate("Random forest", "model", random_forest, X, y))
    rows.append(cross_validate("XGBoost", "model", xgboost, X, y))
    rows.append(cross_validate("LightGBM (final)", "model", lgbm(), X, y))

    print("[2/3] LightGBM settings")
    rows.append(cross_validate("LightGBM, lr 0.1", "setting", lgbm(dict(learning_rate=0.1)), X, y))
    rows.append(cross_validate("LightGBM, num_leaves 63", "setting", lgbm(dict(num_leaves=63)), X, y))
    rows.append(cross_validate("LightGBM, min_child_samples 20", "setting", lgbm(dict(min_child_samples=20)), X, y))
    # 流失 : 留下 大約 1 : 3.7,用這個比例加權
    w = float((y == 0).sum() / (y == 1).sum())
    rows.append(cross_validate("LightGBM, class weight", "setting", lgbm(dict(scale_pos_weight=w)), X, y))

    print("[3/3] features and blending")
    rows.append(cross_validate("LightGBM + engineered features", "features", lgbm(),
                               M.build_features(train, levels, engineered=True), y))
    rows.append(cross_validate("LightGBM + CustomerId + Surname", "features", lgbm(),
                               M.build_features(train, levels, identifiers=True), y))
    rows.append(score("LightGBM + XGBoost average", "blend", (OOF["LightGBM (final)"] + OOF["XGBoost"]) / 2, y))

    df = pd.DataFrame(rows)
    df.to_csv(M.OUT_DIR / "experiments.csv", index=False, float_format="%.5f")

    # 圖:各做法的 AUC,誤差線是 fold 之間的標準差,虛線是定案的模型
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    d = df.iloc[::-1]
    colors = d["group"].map({"model": "#4C72B0", "setting": "#55A868", "features": "#8172B2", "blend": "#CCB974"})
    ax.barh(d["name"], d["auc"], xerr=d["fold_auc_sd"], color=colors, capsize=2)
    final = df.loc[df["name"] == "LightGBM (final)", "auc"].item()
    ax.axvline(final, color="k", ls="--", lw=1)
    ax.set_xlim(df["auc"].min() - 0.004, df["auc"].max() + 0.004)
    ax.set_xlabel("out-of-fold ROC-AUC (error bar = sd over 5 folds, dashed line = final model)")
    fig.tight_layout()
    fig.savefig(M.OUT_DIR / "11_experiments.png", dpi=130)
    print(f"saved {M.OUT_DIR / 'experiments.csv'}")


if __name__ == "__main__":
    main()
