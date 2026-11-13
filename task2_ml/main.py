"""Task 2:預測客戶流失(Exited)的機率

用法:
    python main.py          # 定案的流程:資料探索 -> 5-fold CV -> 評估 -> 輸出 test 的預測
    python experiments.py   # 定案之前試過的模型、參數和特徵(比較表)

輸出都在 outputs/:predictions.csv、metrics.csv / metrics.json、confusion_matrix.csv 和圖。

幾個做法:
  - 指標都用 out-of-fold 的預測算
  - early stopping 用訓練那幾份裡再切出來的 10%,不用驗證那一份
  - 每個 fold 的門檻用另外四個 fold 決定
  - test 的預測是 5 個模型的平均
"""
from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# LightGBM 4.7 會提示 eval_set 要改名,舊版還是得用 eval_set,先把提示關掉
warnings.filterwarnings("ignore", message=".*eval_set.*")

SEED = 42
N_SPLITS = 5
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
OUT_DIR = HERE / "outputs"

TARGET = "Exited"
ID_COL = "id"
# 這三個欄位不放進模型,原因寫在 README
IDENTIFIERS = ["id", "CustomerId", "Surname"]
CATEGORICAL = ["Geography", "Gender"]
NUMERIC = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]

LGB_PARAMS = dict(
    objective="binary",
    n_estimators=3000,       # 上限,實際由 early stopping 決定
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=100,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=5.0,
    random_state=SEED,
    n_jobs=min(8, os.cpu_count() or 1),
    deterministic=True,      # 讓結果可以重現
    force_row_wise=True,
    verbose=-1,
)
EARLY_STOPPING_ROUNDS = 150

C_STAY, C_EXIT = "#4C72B0", "#C44E52"


# --------------------------------------------------------------------------------------
# 資料
# --------------------------------------------------------------------------------------
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """讀資料,順便做幾個基本檢查。"""
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    assert TARGET in train.columns and TARGET not in test.columns
    assert train[ID_COL].is_unique and test[ID_COL].is_unique
    assert not set(train[ID_COL]) & set(test[ID_COL]), "train/test ids overlap"
    assert train.isna().sum().sum() == 0 and test.isna().sum().sum() == 0, "unexpected missing values"
    return train, test


def category_levels(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, list]:
    """類別欄位所有出現過的值(train + test),讓兩邊的編碼一致。"""
    return {c: sorted(set(train[c]) | set(test[c])) for c in CATEGORICAL + ["Surname"]}


def build_features(df: pd.DataFrame, cat_levels: dict[str, list], engineered: bool = False,
                   identifiers: bool = False) -> pd.DataFrame:
    """做特徵。每一筆只用自己的欄位,沒有用到 target 或整份資料的統計量。"""
    X = df[NUMERIC].copy()
    for c in CATEGORICAL:
        # 轉成 category,LightGBM 可以直接吃
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    if engineered:  # experiments.py 才用,試過沒有比較好
        X["BalanceIsZero"] = (df["Balance"] == 0).astype(int)
        X["BalanceToSalary"] = df["Balance"] / (df["EstimatedSalary"] + 1.0)
        X["ProductsPerTenure"] = df["NumOfProducts"] / (df["Tenure"] + 1.0)
        X["ActiveWithCard"] = df["IsActiveMember"] * df["HasCrCard"]
        X["AgeXProducts"] = df["Age"] * df["NumOfProducts"]
    if identifiers:  # experiments.py 才用
        X["CustomerId"] = df["CustomerId"]
        X["Surname"] = pd.Categorical(df["Surname"], categories=cat_levels["Surname"])
    else:
        assert not set(IDENTIFIERS) & set(X.columns), "identifier column leaked into the features"
    return X


# --------------------------------------------------------------------------------------
# 模型
# --------------------------------------------------------------------------------------
def fit_lgbm_fold(X_tr, y_tr, seed, params=None):
    """訓練一個 LightGBM。early stopping 用訓練資料裡再切出來的 10%,外面那份驗證資料不碰。"""
    X_fit, X_es, y_fit, y_es = train_test_split(X_tr, y_tr, test_size=0.1, stratify=y_tr, random_state=seed)
    history: dict = {}
    model = lgb.LGBMClassifier(**{**LGB_PARAMS, **(params or {}), "random_state": seed})
    model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_fit, y_fit), (X_es, y_es)],
        eval_names=["train", "early_stop"],
        eval_metric=["binary_logloss", "auc"],
        callbacks=[
            # 看 early_stop 那份的 logloss 決定什麼時候停
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, first_metric_only=True, verbose=False),
            lgb.record_evaluation(history),  # 畫學習曲線用
        ],
    )
    return model, history


def make_logreg() -> Pipeline:
    """Baseline:one-hot + 標準化 + 邏輯迴歸。NumOfProducts 也做 one-hot,因為它跟流失率不是線性關係。"""
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL + ["NumOfProducts"]),
            ("num", StandardScaler(), [c for c in NUMERIC if c != "NumOfProducts"]),
        ]
    )
    return Pipeline([("pre", pre), ("clf", LogisticRegression(C=1.0, max_iter=1000))])


def run_cv(train: pd.DataFrame, test: pd.DataFrame | None, cat_levels, engineered=False, identifiers=False, verbose=True):
    """Stratified K-fold。回傳 out-of-fold 的預測、test 的平均預測,和每個 fold 的模型。"""
    y = train[TARGET].to_numpy()
    X = build_features(train, cat_levels, engineered, identifiers)
    X_test = build_features(test, cat_levels, engineered, identifiers) if test is not None else None
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    oof = np.zeros(len(train))
    fold_of = np.zeros(len(train), dtype=int)  # 每筆資料在哪個 fold,挑門檻時要用
    test_pred = np.zeros(len(test)) if test is not None else None
    models, histories, fold_auc = [], [], []
    for k, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        model, hist = fit_lgbm_fold(X.iloc[tr_idx], y[tr_idx], seed=SEED + k)
        oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        fold_of[va_idx] = k
        if X_test is not None:
            test_pred += model.predict_proba(X_test)[:, 1] / N_SPLITS
        auc = roc_auc_score(y[va_idx], oof[va_idx])
        fold_auc.append(auc)
        models.append(model)
        histories.append(hist)
        if verbose:
            print(f"  [LightGBM] fold {k}: best_iter={model.best_iteration_:4d}  AUC={auc:.5f}  ({time.time() - t0:.1f}s)")
    return dict(oof=oof, fold_of=fold_of, test_pred=test_pred, models=models, histories=histories,
                fold_auc=fold_auc, X=X, y=y)


def run_cv_logreg(train: pd.DataFrame):
    """Baseline 的 CV,fold 的切法跟 LightGBM 一樣。"""
    y = train[TARGET].to_numpy()
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train))
    fold_auc = []
    for k, (tr_idx, va_idx) in enumerate(skf.split(train, y)):
        m = make_logreg().fit(train.iloc[tr_idx], y[tr_idx])
        oof[va_idx] = m.predict_proba(train.iloc[va_idx])[:, 1]
        fold_auc.append(roc_auc_score(y[va_idx], oof[va_idx]))
    return oof, fold_auc


# --------------------------------------------------------------------------------------
# 評估
# --------------------------------------------------------------------------------------
def best_f1_threshold(y, p) -> float:
    """F1 最高的門檻。"""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[np.argmax(f1[:-1])])  # 最後一個點沒有對應的門檻


def crossfit_labels(y, p, fold_of) -> tuple[np.ndarray, list[float]]:
    """每個 fold 的門檻用其他 fold 的資料決定,再套到這個 fold。避免在同一批資料上又調又評。"""
    labels = np.zeros(len(y), dtype=int)
    thrs = []
    for k in np.unique(fold_of):
        t = best_f1_threshold(y[fold_of != k], p[fold_of != k])
        labels[fold_of == k] = (p[fold_of == k] >= t).astype(int)
        thrs.append(t)
    return labels, thrs


def threshold_metrics(y, labels) -> dict:
    """要先切門檻才能算的指標。"""
    tn, fp, fn, tp = confusion_matrix(y, labels).ravel()
    return dict(
        accuracy=accuracy_score(y, labels),
        precision=precision_score(y, labels),
        recall=recall_score(y, labels),
        specificity=tn / (tn + fp),
        f1=f1_score(y, labels),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


def ranking_metrics(y, p) -> dict:
    """直接用機率算的指標。"""
    return dict(
        roc_auc=roc_auc_score(y, p),
        pr_auc=average_precision_score(y, p),
        log_loss=log_loss(y, p),
        brier=brier_score_loss(y, p),
    )


# --------------------------------------------------------------------------------------
# 畫圖(圖上的字用英文,沒裝中文字型會變方塊)
# --------------------------------------------------------------------------------------
def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, dpi=130)
    plt.close(fig)


def plot_eda(train: pd.DataFrame):
    """圖 01-04:資料探索。"""
    rate = train[TARGET].mean()

    # 圖 01:target 的分布
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    counts = train[TARGET].value_counts().sort_index()
    bars = ax.bar(["Stayed (0)", "Exited (1)"], counts.values, color=[C_STAY, C_EXIT])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({v / len(train):.1%})", ha="center", va="bottom")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.set_ylabel("customers")
    ax.set_title("Target distribution (train)")
    savefig(fig, "01_target_distribution.png")

    # 圖 02:各欄位的流失率,虛線是整體平均
    cats = ["Geography", "Gender", "NumOfProducts", "IsActiveMember", "HasCrCard", "Tenure"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, c in zip(axes.ravel(), cats):
        g = train.groupby(c)[TARGET].agg(["mean", "size"])
        labels = [str(int(i)) if isinstance(i, (float, np.floating)) else str(i) for i in g.index]
        bars = ax.bar(labels, g["mean"], color=C_EXIT, alpha=0.85)
        ax.axhline(rate, color="k", ls="--", lw=1, label=f"overall {rate:.1%}")
        for b, (m, n) in zip(bars, g.values):
            txt = f"{m:.0%}\nn={int(n):,}" if len(g) <= 4 else f"{m:.0%}"  # Tenure 太多根,標人數會疊在一起
            ax.text(b.get_x() + b.get_width() / 2, m, txt, ha="center", va="bottom", fontsize=7)
        ax.set_ylim(0, min(1.0, g["mean"].max() * 1.3 + 0.05))
        ax.set_title(f"Exit rate by {c}")
        ax.set_ylabel("exit rate")
        ax.legend(fontsize=8)
    savefig(fig, "02_exit_rate_by_category.png")

    # 圖 03:數值欄位的分布
    nums = ["Age", "CreditScore", "Balance", "EstimatedSalary"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    for ax, c in zip(axes, nums):
        bins = np.linspace(train[c].min(), train[c].max(), 41)
        for v, col, lab in [(0, C_STAY, "Stayed"), (1, C_EXIT, "Exited")]:
            ax.hist(train.loc[train[TARGET] == v, c], bins=bins, density=True, alpha=0.55, color=col, label=lab)
        ax.set_title(c)
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
    fig.suptitle("Numeric features by class (each class normalised to area 1)")
    savefig(fig, "03_numeric_distributions.png")

    # 圖 04:年齡和產品數一起看
    fig, ax = plt.subplots(figsize=(7, 4.2))
    age_bin = pd.cut(train["Age"], bins=[17, 25, 30, 35, 40, 45, 50, 55, 60, 70, 95])
    for n_prod, sub in train.groupby("NumOfProducts"):
        if len(sub) < 1000:  # 4 個產品的人太少,不畫
            continue
        g = sub.groupby(age_bin.loc[sub.index], observed=True)[TARGET].mean()
        ax.plot([iv.mid for iv in g.index], g.values, marker="o", label=f"{n_prod} product(s), n={len(sub):,}")
    ax.set_xlabel("Age")
    ax.set_ylabel("exit rate")
    ax.set_title("Exit rate by age and number of products")
    ax.legend(fontsize=8)
    savefig(fig, "04_exit_rate_age_x_products.png")


def plot_roc_pr(y, preds: dict[str, np.ndarray]):
    """圖 05:ROC 和 PR 曲線。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, p):.4f})")
        prec, rec, _ = precision_recall_curve(y, p)
        axes[1].plot(rec, prec, label=f"{name} (AP={average_precision_score(y, p):.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (out-of-fold)")
    axes[1].axhline(y.mean(), color="k", ls="--", lw=1, label=f"chance (prevalence {y.mean():.3f})")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve (out-of-fold)")
    for ax in axes:
        ax.legend(loc="lower right" if ax is axes[0] else "upper right", fontsize=8)
        ax.grid(alpha=0.3)
    savefig(fig, "05_roc_pr_curves.png")


def plot_confusion(y, labels_by_name: dict[str, np.ndarray]):
    """圖 06:confusion matrix,列是實際、欄是預測。"""
    fig, axes = plt.subplots(1, len(labels_by_name), figsize=(5.2 * len(labels_by_name), 4.4))
    for ax, (name, lab) in zip(np.atleast_1d(axes), labels_by_name.items()):
        cm = confusion_matrix(y, lab)
        row = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(row, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n{row[i, j]:.1%}", ha="center", va="center",
                        color="white" if row[i, j] > 0.5 else "black")
        ax.set_xticks([0, 1], ["Pred stay", "Pred exit"])
        ax.set_yticks([0, 1], ["True stay", "True exit"])
        ax.set_title(f"{name}\nF1={f1_score(y, lab):.3f}", fontsize=10)
    fig.suptitle("Confusion matrix (out-of-fold; % of each true class)")
    savefig(fig, "06_confusion_matrix.png")


def plot_threshold_sweep(y, p, thr_tuned):
    """圖 07:門檻變動時 precision / recall / F1 的變化。"""
    thr = np.linspace(0.02, 0.98, 97)
    prec = [precision_score(y, p >= t, zero_division=0) for t in thr]
    rec = [recall_score(y, p >= t) for t in thr]
    f1 = [f1_score(y, p >= t) for t in thr]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(thr, prec, label="precision")
    ax.plot(thr, rec, label="recall")
    ax.plot(thr, f1, label="F1", lw=2.2, color="k")
    ax.axvline(0.5, color="grey", ls=":", label="threshold 0.50")
    ax.axvline(thr_tuned, color=C_EXIT, ls="--", label=f"F1-optimal {thr_tuned:.2f}")
    ax.set(xlabel="decision threshold on predicted probability", ylabel="score",
           title="Precision / recall / F1 vs. threshold (out-of-fold)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    savefig(fig, "07_threshold_sweep.png")


def plot_calibration_and_scores(y, p, test_pred):
    """圖 08:校準曲線、兩類的分數分布、train 和 test 的預測分布。"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    frac, mean_pred = calibration_curve(y, p, n_bins=15, strategy="quantile")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    axes[0].plot(mean_pred, frac, marker="o", label=f"LightGBM (Brier={brier_score_loss(y, p):.4f})")
    axes[0].set(xlabel="mean predicted probability", ylabel="observed exit rate", title="Calibration (out-of-fold)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    bins = np.linspace(0, 1, 51)
    axes[1].hist(p[y == 0], bins=bins, alpha=0.6, color=C_STAY, label="true stay", density=True)
    axes[1].hist(p[y == 1], bins=bins, alpha=0.6, color=C_EXIT, label="true exit", density=True)
    axes[1].set(xlabel="predicted probability of exit", ylabel="density", title="Score distribution by true class")
    axes[1].legend(fontsize=8)

    axes[2].hist(p, bins=bins, alpha=0.6, label=f"train out-of-fold (mean {p.mean():.3f})", density=True)
    axes[2].hist(test_pred, bins=bins, alpha=0.6, label=f"test (mean {test_pred.mean():.3f})", density=True)
    axes[2].set(xlabel="predicted probability of exit", ylabel="density",
                title="Train OOF vs. test predictions (drift check)")
    axes[2].legend(fontsize=8)
    savefig(fig, "08_calibration_and_score_distributions.png")


def plot_learning_curves(histories):
    """圖 09:學習曲線,每個 fold 一條。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for metric, ax in zip(["binary_logloss", "auc"], axes):
        for k, h in enumerate(histories):
            ax.plot(h["train"][metric], color=C_STAY, alpha=0.5, lw=1, label="train" if k == 0 else None)
            ax.plot(h["early_stop"][metric], color=C_EXIT, alpha=0.8, lw=1, label="early-stopping split" if k == 0 else None)
        ax.set(xlabel="boosting iteration", ylabel=metric, title=f"LightGBM learning curves - {metric} (5 folds)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    # 前面幾棵樹 AUC 變化太大,y 軸只看第 20 棵之後
    lo = min(min(h["early_stop"]["auc"][20:]) for h in histories)
    hi = max(max(h["train"]["auc"]) for h in histories)
    axes[1].set_ylim(lo - 0.005, hi + 0.003)
    savefig(fig, "09_learning_curves.png")


def plot_importance(models, X, y, fold_of):
    """圖 10:特徵重要度。左邊是 permutation(在驗證那份資料上把欄位打亂,看 AUC 掉多少),右邊是 gain。"""
    names = list(X.columns)
    gain = np.mean([m.booster_.feature_importance(importance_type="gain") for m in models], axis=0)
    gain = gain / gain.sum()

    rng = np.random.default_rng(SEED)
    drops = np.zeros((len(models), len(names)))
    for k, m in enumerate(models):
        Xv, yv = X[fold_of == k], y[fold_of == k]
        base = roc_auc_score(yv, m.predict_proba(Xv)[:, 1])
        for j, c in enumerate(names):
            Xp = Xv.copy()
            Xp[c] = Xv[c].iloc[rng.permutation(len(Xv))].values  # 用 .values 才不會掉 category 型態
            drops[k, j] = base - roc_auc_score(yv, m.predict_proba(Xp)[:, 1])
    perm, perm_sd = drops.mean(0), drops.std(0, ddof=1)

    order = np.argsort(perm)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].barh(np.array(names)[order], perm[order], xerr=perm_sd[order], color=C_STAY)
    axes[0].set(xlabel="AUC drop when the feature is shuffled (held-out folds, mean ± sd)",
                title="Permutation importance")
    order_g = np.argsort(gain)
    axes[1].barh(np.array(names)[order_g], gain[order_g], color="#55A868")
    axes[1].set(xlabel="share of total split gain", title="LightGBM gain importance")
    savefig(fig, "10_feature_importance.png")
    return pd.DataFrame({"feature": names, "permutation_auc_drop": perm, "permutation_sd": perm_sd,
                         "gain_share": gain}).sort_values("permutation_auc_drop", ascending=False)


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(exist_ok=True)
    t_start = time.time()

    print("[1/6] Loading data")
    train, test = load_data()
    cat_levels = category_levels(train, test)
    y = train[TARGET].to_numpy()
    print(f"  train={train.shape}  test={test.shape}  exit rate={y.mean():.4f}")

    print("[2/6] EDA plots")
    plot_eda(train)

    print("[3/6] Baseline: logistic regression (5-fold CV)")
    oof_lr, auc_lr = run_cv_logreg(train)
    print(f"  [LogReg] AUC={roc_auc_score(y, oof_lr):.5f}")

    print("[4/6] LightGBM (5-fold CV, early stopping on an inner split)")
    res = run_cv(train, test, cat_levels)
    oof, fold_of, test_pred = res["oof"], res["fold_of"], res["test_pred"]

    print("[5/6] Evaluation on out-of-fold predictions")
    labels_05 = (oof >= 0.5).astype(int)
    labels_tuned, fold_thrs = crossfit_labels(y, oof, fold_of)
    thr_final = best_f1_threshold(y, oof)  # 套在 test 上的門檻
    metrics = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "exit_rate_train": float(y.mean()),
        "cv": f"stratified {N_SPLITS}-fold, out-of-fold predictions",
        "lightgbm": {
            **ranking_metrics(y, oof),
            "fold_auc": [float(a) for a in res["fold_auc"]],
            "fold_auc_mean": float(np.mean(res["fold_auc"])),
            "fold_auc_std": float(np.std(res["fold_auc"], ddof=1)),
            "best_iterations": [int(m.best_iteration_) for m in res["models"]],
            "threshold_0.50": threshold_metrics(y, labels_05),
            "threshold_f1_optimal_crossfitted": {
                **threshold_metrics(y, labels_tuned),
                "fold_thresholds": [float(t) for t in fold_thrs],
                "threshold_used_for_test": thr_final,
            },
        },
        "logistic_regression_baseline": {
            **ranking_metrics(y, oof_lr),
            "fold_auc_mean": float(np.mean(auc_lr)),
            "fold_auc_std": float(np.std(auc_lr, ddof=1)),
            "threshold_0.50": threshold_metrics(y, (oof_lr >= 0.5).astype(int)),
        },
        "test_predictions": {
            "mean_predicted_probability": float(test_pred.mean()),
            "predicted_exit_share_at_tuned_threshold": float((test_pred >= thr_final).mean()),
        },
    }
    with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    # 整理成一張表存成 metrics.csv
    flat = []
    for model, key, lab in [
        ("LightGBM", "threshold_0.50", "0.50"),
        ("LightGBM", "threshold_f1_optimal_crossfitted", f"F1-optimal (~{thr_final:.2f}, cross-fitted)"),
        ("LogReg baseline", "threshold_0.50", "0.50"),
    ]:
        m = metrics["lightgbm" if model == "LightGBM" else "logistic_regression_baseline"]
        t = m[key]
        flat.append(dict(model=model, threshold=lab, roc_auc=m["roc_auc"], pr_auc=m["pr_auc"],
                         log_loss=m["log_loss"], brier=m["brier"],
                         **{k: t[k] for k in ["accuracy", "precision", "recall", "specificity", "f1"]}))
    metrics_df = pd.DataFrame(flat)
    metrics_df.to_csv(OUT_DIR / "metrics.csv", index=False, float_format="%.5f")
    print(metrics_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    cm_rows = []
    for lab_name, lab in [("threshold_0.50", labels_05), ("threshold_f1_optimal_crossfitted", labels_tuned)]:
        t = threshold_metrics(y, lab)
        cm_rows.append(dict(threshold=lab_name, true_stay_pred_stay=t["tn"], true_stay_pred_exit=t["fp"],
                            true_exit_pred_stay=t["fn"], true_exit_pred_exit=t["tp"]))
    pd.DataFrame(cm_rows).to_csv(OUT_DIR / "confusion_matrix.csv", index=False)

    plot_roc_pr(y, {"LightGBM": oof, "Logistic regression (baseline)": oof_lr})
    plot_confusion(y, {"LightGBM @ threshold 0.50": labels_05,
                       f"LightGBM @ F1-optimal threshold (~{thr_final:.2f})": labels_tuned})
    plot_threshold_sweep(y, oof, thr_final)
    plot_calibration_and_scores(y, oof, test_pred)
    plot_learning_curves(res["histories"])
    imp = plot_importance(res["models"], res["X"], y, fold_of)
    imp.to_csv(OUT_DIR / "feature_importance.csv", index=False, float_format="%.6f")

    print("[6/6] Writing test predictions")
    # Exited 是機率,Exited_label 是用門檻切過的 0/1
    sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_pred, f"{TARGET}_label": (test_pred >= thr_final).astype(int)})
    assert len(sub) == len(test) and sub[TARGET].between(0, 1).all()
    sub.to_csv(OUT_DIR / "predictions.csv", index=False, float_format="%.6f")
    print(f"  outputs/predictions.csv  rows={len(sub):,}  mean p={test_pred.mean():.4f}  "
          f"predicted exits={sub[f'{TARGET}_label'].mean():.1%} (threshold {thr_final:.3f})")

    print(f"Done in {time.time() - t_start:.0f}s. Results are in {OUT_DIR}")


if __name__ == "__main__":
    main()
