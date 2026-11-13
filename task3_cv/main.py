"""Task 3:MNIST 數字分類

分三組實驗:
  Part A  模型:LeNet-5 -> DigitNet -> 加 BN -> 加 dropout 和 augmentation
  Part B  優化:LR range test,比較 SGD / momentum / Adam / AdamW 和學習率排程
  Part C  類別不平衡:做一份 100:1 的長尾訓練集,比較幾種處理方法

用法:
    python main.py              # 全部跑完(第一次會下載 MNIST)
    python main.py --quick      # 快速測試,結果寫到 outputs_quick/
    python main.py --plots-only # 只重畫圖

資料切法:60k 訓練資料切成 54k train / 6k val,選設定只看 val,test 只在最後報數字用。
每個模型都用最後一個 epoch 的結果,不挑 checkpoint。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import BatchSampler, RandomSampler, WeightedRandomSampler
from torchvision import datasets

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
DATA_DIR = HERE / "data"
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
NUM_CLASSES = 10
# Part C 看結果用的分組
HEAD, MID, TAIL = [0, 1, 2], [3, 4, 5, 6], [7, 8, 9]
# one-cycle 的最高學習率,從 LR range test 看出來的
PEAK_LR_ADAMW, PEAK_LR_SGD = 3e-3, 0.2


# ======================================================================================
# 資料
# ======================================================================================
def load_mnist(device):
    """下載 MNIST,標準化後整包放到 device 上。資料很小,這樣比用 DataLoader 快。"""
    tr = datasets.MNIST(DATA_DIR, train=True, download=True)
    te = datasets.MNIST(DATA_DIR, train=False, download=True)

    def prep(ds):
        # 轉 float、標準化,補上 channel 維度
        x = ds.data.float().div(255.0).sub(MNIST_MEAN).div(MNIST_STD).unsqueeze(1)
        return x, ds.targets.clone()

    x_all, y_all = prep(tr)
    x_test, y_test = prep(te)

    # 每個數字抽 600 張當驗證集
    rng = np.random.default_rng(0)
    val_idx = np.concatenate([rng.choice(np.where(y_all.numpy() == c)[0], 600, replace=False) for c in range(NUM_CLASSES)])
    mask = np.ones(len(y_all), dtype=bool)
    mask[val_idx] = False
    tr_idx = np.where(mask)[0]
    d = dict(
        x_train=x_all[tr_idx], y_train=y_all[tr_idx],
        x_val=x_all[val_idx], y_val=y_all[val_idx],
        x_test=x_test, y_test=y_test,
    )
    return {k: v.to(device) for k, v in d.items()}


def longtail_indices(y: torch.Tensor, n_max: int, imbalance_factor: float, seed: int = 0):
    """長尾訓練集的 index。第 c 類留 n_max * (1/IF)^(c/9) 張,所以 0 最多、9 最少。"""
    rng = np.random.default_rng(seed)
    y_np = y.cpu().numpy()
    counts = [int(round(n_max * (1.0 / imbalance_factor) ** (c / (NUM_CLASSES - 1)))) for c in range(NUM_CLASSES)]
    idx = np.concatenate([rng.choice(np.where(y_np == c)[0], counts[c], replace=False) for c in range(NUM_CLASSES)])
    return torch.as_tensor(idx, device=y.device), counts


def random_affine(x: torch.Tensor, gen: torch.Generator, max_rot_deg=10.0, max_shift_px=2.0, max_scale=0.10):
    """隨機旋轉、平移、縮放,每張圖各自抽參數。不做翻轉,數字翻過來就不對了。"""
    b = x.size(0)
    r = torch.rand(b, 4, generator=gen).to(x.device) * 2 - 1  # [-1, 1] 的亂數
    ang = r[:, 0] * math.radians(max_rot_deg)
    scale = 1.0 + r[:, 1] * max_scale
    # affine_grid 的座標是 [-1, 1],所以 1 pixel = 1/14
    tx, ty = r[:, 2] * (max_shift_px / 14.0), r[:, 3] * (max_shift_px / 14.0)
    cos, sin = torch.cos(ang) * scale, torch.sin(ang) * scale
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    # 超出去的地方用邊緣補,MNIST 的邊緣就是背景
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


# ======================================================================================
# 模型
# ======================================================================================
class LeNet5(nn.Module):
    """Baseline。LeNet-5 改用 ReLU 和 max-pool,61,706 個參數。"""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),   # 28x28 -> 14x14
            nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2),             # 14x14 -> 10x10 -> 5x5
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(400, 120), nn.ReLU(), nn.Linear(120, 84), nn.ReLU(),
                                        nn.Linear(84, NUM_CLASSES))

    def forward(self, x):
        return self.classifier(self.features(x))


class DigitNet(nn.Module):
    """三段 [conv3x3 + BN + ReLU] x2,中間 max-pool,最後 global average pooling 接 linear。
    為什麼這樣設計寫在 README。"""

    def __init__(self, widths=(16, 32, 64), use_bn=True, dropout=0.2):
        super().__init__()
        layers, c_in = [], 1
        for i, w in enumerate(widths):
            for _ in range(2):
                # 有 BN 就不用 bias
                layers.append(nn.Conv2d(c_in, w, 3, padding=1, bias=not use_bn))
                if use_bn:
                    layers.append(nn.BatchNorm2d(w))
                layers.append(nn.ReLU(inplace=True))
                c_in = w
            if i < len(widths) - 1:  # 最後一段不 pool
                layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, NUM_CLASSES))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.head(self.features(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ======================================================================================
# 訓練設定
# ======================================================================================
@dataclass
class Config:
    """一次訓練的設定。"""
    name: str
    # --- 模型
    model: str = "digitnet"  # "lenet" | "digitnet"
    use_bn: bool = True
    dropout: float = 0.2
    augment: bool = True
    # --- 優化
    optimizer: str = "adamw"  # "sgd" | "sgdm" | "adam" | "adamw"
    lr: float = PEAK_LR_ADAMW  # onecycle 的話是最高的學習率
    weight_decay: float | None = None  # None 的話看 build_optimizer
    schedule: str = "onecycle"  # "const" | "onecycle"
    epochs: int = 6
    batch_size: int = 128
    # --- Part C 用的
    loss: str = "ce"  # "ce" | "weighted_ce" | "focal" | "logit_adjusted"
    sampler: str = "shuffle"  # "shuffle" | "class_balanced"
    focal_gamma: float = 2.0
    la_tau: float = 1.0


def build_model(cfg: Config) -> nn.Module:
    if cfg.model == "lenet":
        return LeNet5()
    return DigitNet(use_bn=cfg.use_bn, dropout=cfg.dropout)


def build_optimizer(cfg: Config, model: nn.Module):
    """weight decay 只加在 conv / linear 的 weight,BN 和 bias 不加。
    AdamW 的 decay 是直接乘在權重上,跟 SGD / Adam 的 L2 大小不能直接比,所以預設值分開(5e-2 和 5e-4)。"""
    wd = cfg.weight_decay if cfg.weight_decay is not None else (5e-2 if cfg.optimizer == "adamw" else 5e-4)
    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]
    groups = [dict(params=decay, weight_decay=wd), dict(params=no_decay, weight_decay=0.0)]
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(groups, lr=cfg.lr)
    if cfg.optimizer == "sgdm":
        return torch.optim.SGD(groups, lr=cfg.lr, momentum=0.9, nesterov=True)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(groups, lr=cfg.lr)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=cfg.lr)
    raise ValueError(cfg.optimizer)


def build_criterion(cfg: Config, class_counts: torch.Tensor):
    """Part C 的 loss。class_counts 是訓練集各類別的張數。"""
    counts = class_counts.float()
    prior = counts / counts.sum()  # 各類別佔的比例
    if cfg.loss == "ce":
        return lambda logits, y: F.cross_entropy(logits, y)
    if cfg.loss == "weighted_ce":
        # 權重跟類別數量成反比。cross_entropy 會除以權重總和,loss 的大小不會差太多
        w = 1.0 / counts
        w = w / w.sum() * NUM_CLASSES
        return lambda logits, y: F.cross_entropy(logits, y, weight=w)
    if cfg.loss == "focal":
        # 用 (1 - p_t)^gamma 把已經分對的樣本權重壓低
        def focal(logits, y):
            logp = F.log_softmax(logits, 1).gather(1, y[:, None]).squeeze(1)
            return (-(1 - logp.exp()) ** cfg.focal_gamma * logp).mean()

        return focal
    if cfg.loss == "logit_adjusted":
        # 訓練時 logit 加上 tau * log(類別比例),預測時用原本的 logit
        log_prior = prior.log()
        return lambda logits, y: F.cross_entropy(logits + cfg.la_tau * log_prior, y)
    raise ValueError(cfg.loss)


# ======================================================================================
# 訓練 / 評估
# ======================================================================================
@torch.no_grad()
def predict_logits(model, x, batch_size=2000):
    model.eval()
    return torch.cat([model(x[i:i + batch_size]) for i in range(0, len(x), batch_size)])


def classification_metrics(logits: torch.Tensor, y: torch.Tensor) -> dict:
    """算各種指標。cm[i, j] = 實際是 i、預測成 j 的張數。"""
    pred = logits.argmax(1)
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    cm.index_put_((y.cpu(), pred.cpu()), torch.ones(len(y), dtype=torch.long), accumulate=True)
    cm_f = cm.double()
    recall = cm_f.diag() / cm_f.sum(1).clamp(min=1)
    precision = cm_f.diag() / cm_f.sum(0).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
    return dict(
        loss=F.cross_entropy(logits, y).item(),
        accuracy=(cm_f.diag().sum() / cm_f.sum()).item(),
        balanced_accuracy=recall.mean().item(),  # 各類別 recall 的平均
        macro_f1=f1.mean().item(),
        recall=recall.tolist(),
        precision=precision.tolist(),
        f1=f1.tolist(),
        head_recall=recall[HEAD].mean().item(),
        mid_recall=recall[MID].mean().item(),
        tail_recall=recall[TAIL].mean().item(),
        confusion=cm.tolist(),
    )


def train(cfg: Config, x_tr, y_tr, x_val, y_val, seed: int, verbose=True):
    """訓練一個模型,回傳 (model, history)。用最後一個 epoch 的模型,val 只拿來記錄曲線。"""
    torch.manual_seed(seed)
    device = x_tr.device
    model = build_model(cfg).to(device)
    opt = build_optimizer(cfg, model)
    n = len(x_tr)
    steps_per_epoch = math.ceil(n / cfg.batch_size)
    sched = None
    if cfg.schedule == "onecycle":
        # 前 25% 的 step 把學習率拉到最高,之後用 cosine 降到接近 0。
        # momentum 不跟著變,這樣跟固定學習率的對照組就只差在排程
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.epochs * steps_per_epoch,
                                                    pct_start=0.25, anneal_strategy="cos", cycle_momentum=False)
    class_counts = torch.bincount(y_tr, minlength=NUM_CLASSES)
    criterion = build_criterion(cfg, class_counts)

    gen = torch.Generator().manual_seed(seed)  # batch 順序和 augmentation 用
    if cfg.sampler == "class_balanced":
        # 讓每個類別被抽到的機率一樣,少的類別會重複抽。一個 epoch 的張數不變
        w = (1.0 / class_counts.float())[y_tr].cpu()
        sampler = WeightedRandomSampler(w, num_samples=n, replacement=True, generator=gen)
    else:
        sampler = RandomSampler(range(n), generator=gen)
    # 資料已經在記憶體裡,只需要 sampler 給 index
    batches = BatchSampler(sampler, cfg.batch_size, drop_last=False)

    hist = dict(epoch=[], train_loss=[], train_acc=[], val_loss=[], val_acc=[], val_bal_acc=[], val_tail_recall=[],
                lr=[], step_loss=[], step_lr=[], seconds=[])
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tot_loss, tot_correct, tot_n = 0.0, 0, 0
        for idx in batches:
            idx = torch.as_tensor(idx, device=device)
            xb, yb = x_tr[idx], y_tr[idx]
            if cfg.augment:
                xb = random_affine(xb, gen)
            logits = model(xb)
            loss = criterion(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            hist["step_loss"].append(loss.item())
            hist["step_lr"].append(opt.param_groups[0]["lr"])
            if sched is not None:
                sched.step()  # 每個 step 更新
            tot_loss += loss.item() * len(yb)
            tot_correct += (logits.argmax(1) == yb).sum().item()
            tot_n += len(yb)
        # train_loss / train_acc 是訓練中累計的(有 augmentation 和 dropout),一開始會比 val 差
        val = classification_metrics(predict_logits(model, x_val), y_val)
        hist["epoch"].append(epoch)
        hist["train_loss"].append(tot_loss / tot_n)
        hist["train_acc"].append(tot_correct / tot_n)
        hist["val_loss"].append(val["loss"])
        hist["val_acc"].append(val["accuracy"])
        hist["val_bal_acc"].append(val["balanced_accuracy"])
        hist["val_tail_recall"].append(val["tail_recall"])
        hist["lr"].append(hist["step_lr"][-1])
        hist["seconds"].append(time.time() - t0)
        if verbose:
            print(f"    [{cfg.name}] epoch {epoch:2d}/{cfg.epochs}  train loss {hist['train_loss'][-1]:.4f} "
                  f"acc {hist['train_acc'][-1]:.4f} | val loss {val['loss']:.4f} acc {val['accuracy']:.4f}  "
                  f"({hist['seconds'][-1]:.0f}s)", flush=True)
    return model, hist


def lr_range_test(cfg: Config, x_tr, y_tr, lr_min=1e-5, lr_max=1.0, steps=300, seed=0):
    """LR range test:學習率每個 step 乘一個倍數往上調,記錄 loss。只用訓練集。"""
    torch.manual_seed(seed)
    model = build_model(cfg).to(x_tr.device)
    opt = build_optimizer(replace(cfg, lr=lr_min), model)
    gen = torch.Generator().manual_seed(seed)
    gamma = (lr_max / lr_min) ** (1 / (steps - 1))
    lrs, losses, ema = [], [], None
    model.train()
    for step in range(steps):
        idx = torch.randint(0, len(x_tr), (cfg.batch_size,), generator=gen).to(x_tr.device)
        xb = random_affine(x_tr[idx], gen) if cfg.augment else x_tr[idx]
        loss = F.cross_entropy(model(xb), y_tr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        ema = loss.item() if ema is None else 0.9 * ema + 0.1 * loss.item()  # 平滑一下
        lrs.append(opt.param_groups[0]["lr"])
        losses.append(ema)
        if not math.isfinite(ema) or ema > 4 * min(losses):  # loss 爆掉就停
            break
        for g in opt.param_groups:
            g["lr"] *= gamma
    return dict(lr=lrs, loss=losses)


# ======================================================================================
# 實驗
#
# 每次訓練包成一個 job(dict)交給 run_job。CPU 上會開多個 process 一起跑,
# 這種小模型一個 process 開太多 thread 也不會變快。
# ======================================================================================
def slim(hist: dict) -> dict:
    """每個 step 的紀錄太長,平均壓到 400 點左右再存。"""
    h = dict(hist)
    k = max(1, len(h["step_loss"]) // 400)
    if k > 1:
        m = len(h["step_loss"]) // k * k
        h["step_loss"] = np.asarray(h["step_loss"])[:m].reshape(-1, k).mean(1).tolist()
        h["step_lr"] = np.asarray(h["step_lr"])[:m].reshape(-1, k).mean(1).tolist()
    h["step_every"] = k
    return h


_DATA = {}


def get_data(device, subset=None):
    """資料只載一次。subset 是 --quick 用的。"""
    key = (device, subset)
    if key not in _DATA:
        d = load_mnist(device)
        if subset:
            d["x_train"], d["y_train"] = d["x_train"][:subset], d["y_train"][:subset]
        _DATA[key] = d
    return _DATA[key]


def _init_worker(threads):
    torch.set_num_threads(threads)


def run_job(job: dict) -> dict:
    """跑一次訓練,回傳可以存成 JSON 的結果。"""
    t0 = time.time()
    data = get_data(job["device"], job.get("subset"))
    cfg = Config(**job["cfg"])
    x_tr, y_tr = data["x_train"], data["y_train"]
    if job.get("longtail"):
        # Part C 用長尾訓練集。seed 固定,所有方法用的是同一份
        idx, counts = longtail_indices(y_tr, seed=0, **job["longtail"])
        x_tr, y_tr = x_tr[idx], y_tr[idx]
    model, hist = train(cfg, x_tr, y_tr, data["x_val"], data["y_val"], seed=job["seed"], verbose=job.get("verbose", False))
    val_logits = predict_logits(model, data["x_val"])
    logits = predict_logits(model, data["x_test"])
    y_val, y_test = data["y_val"], data["y_test"]
    out = dict(group=job["group"], name=cfg.name, seed=job["seed"], history=slim(hist), val_acc=hist["val_acc"][-1],
               val=classification_metrics(val_logits, y_val), test=classification_metrics(logits, y_test))
    out["test_acc"] = out["test"]["accuracy"]
    if job.get("posthoc"):
        # C1b:同一個模型,logit 減掉 log(類別比例) 再預測
        prior = torch.bincount(y_tr, minlength=NUM_CLASSES).float()
        log_prior = (prior / prior.sum()).log()
        out["posthoc_val"] = classification_metrics(val_logits - log_prior, y_val)
        out["posthoc_test"] = classification_metrics(logits - log_prior, y_test)
    if job.get("final"):
        # 記下信心最高的錯誤和信心分布,畫圖用
        conf, pred = logits.softmax(1).max(1)
        wrong = (pred != y_test).nonzero().squeeze(1)
        wrong = wrong[conf[wrong].argsort(descending=True)][:24]
        out["wrong"] = dict(index=wrong.tolist(), pred=pred[wrong].tolist(), true=y_test[wrong].tolist(), conf=conf[wrong].tolist())
        out["confidence"] = dict(correct=conf[pred == y_test].tolist()[::10], wrong=conf[pred != y_test].tolist())
        # 轉成 CPU 再存,沒有 GPU 也能 load
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, Path(job["out_dir"]) / "digitnet_final.pt")
    out["seconds"] = time.time() - t0
    return out


def build_jobs(ep_ab, ep_final, ep_lt, seeds, lt_kwargs):
    """列出所有要跑的訓練。"""
    reg = dict(use_bn=True, dropout=0.2, augment=True)  # 最後用的設定
    # Part A:一次加一個東西
    part_a = [
        Config("A1 LeNet-5 (baseline)", model="lenet", use_bn=False, dropout=0.0, augment=False, epochs=ep_ab),
        Config("A2 DigitNet, no BN", use_bn=False, dropout=0.0, augment=False, epochs=ep_ab),
        Config("A3 DigitNet + BN", use_bn=True, dropout=0.0, augment=False, epochs=ep_ab),
        Config("A4 DigitNet + BN + dropout + augmentation", epochs=ep_ab, **reg),
    ]
    # Part B:模型固定用 A4。B3、B7 是 B4、B8 的對照組(最高學習率一樣,但固定不變)
    b = dict(epochs=ep_ab, **reg)
    part_b = [
        Config("B1 SGD, constant LR 0.05", optimizer="sgd", lr=0.05, schedule="const", **b),
        Config("B2 SGD + momentum, constant LR 0.05", optimizer="sgdm", lr=0.05, schedule="const", **b),
        Config("B3 SGD + momentum, constant LR 0.2", optimizer="sgdm", lr=PEAK_LR_SGD, schedule="const", **b),
        Config("B4 SGD + momentum, one-cycle (peak 0.2)", optimizer="sgdm", lr=PEAK_LR_SGD, schedule="onecycle", **b),
        Config("B5 Adam, constant LR 1e-3", optimizer="adam", lr=1e-3, schedule="const", **b),
        Config("B6 AdamW, constant LR 1e-3", optimizer="adamw", lr=1e-3, schedule="const", **b),
        Config("B7 AdamW, constant LR 3e-3", optimizer="adamw", lr=PEAK_LR_ADAMW, schedule="const", **b),
        Config("B8 AdamW, one-cycle (peak 3e-3)", optimizer="adamw", lr=PEAK_LR_ADAMW, schedule="onecycle", **b),
    ]
    # 最後的模型:A4 的設定訓練久一點
    final = [Config("Final DigitNet", epochs=ep_final, **reg)]
    # Part C:長尾訓練集
    lt = dict(use_bn=True, dropout=0.2, epochs=ep_lt)
    part_c = [
        Config("C1 Cross-entropy (no handling)", augment=False, **lt),
        Config("C2 Class-weighted cross-entropy", augment=False, loss="weighted_ce", **lt),
        Config("C3 Class-balanced sampling", augment=False, sampler="class_balanced", **lt),
        Config("C4 Focal loss (gamma=2)", augment=False, loss="focal", **lt),
        Config("C5 Logit-adjusted loss (tau=1)", augment=False, loss="logit_adjusted", **lt),
        Config("C6 Class-balanced sampling + augmentation", augment=True, sampler="class_balanced", **lt),
        Config("C7 Logit-adjusted loss + augmentation", augment=True, loss="logit_adjusted", **lt),
    ]
    jobs = []
    for group, cfgs, extra, group_seeds in [("A", part_a, {}, seeds), ("B", part_b, {}, seeds),
                                            ("final", final, dict(final=True), seeds[:1]),
                                            ("C", part_c, dict(longtail=lt_kwargs), seeds)]:
        for cfg in cfgs:
            for s in group_seeds:
                job = dict(group=group, cfg=asdict(cfg), seed=s, **extra)
                if cfg.name.startswith("C1"):
                    job["posthoc"] = True  # C1 順便算 C1b
                jobs.append(job)
    order = dict(A=[c.name for c in part_a], B=[c.name for c in part_b], C=[c.name for c in part_c])
    return jobs, order


def run_all(jobs, device, n_workers, threads):
    """跑所有 job,久的先跑。"""
    for j in jobs:
        j["device"] = device
    # 粗估每個 job 要跑多久
    cost = lambda j: j["cfg"]["epochs"] * (0.25 if j.get("longtail") else 1.0) * (0.2 if j["cfg"]["model"] == "lenet" else 1.0)
    jobs = sorted(jobs, key=cost, reverse=True)
    results, t0 = [], time.time()

    def report(r):
        results.append(r)
        print(f"  [{len(results):3d}/{len(jobs)}] {r['name']:50s} seed {r['seed']}  val {r['val_acc'] * 100:6.2f}%  "
              f"test {r['test_acc'] * 100:6.2f}%  ({r['seconds']:.0f}s, elapsed {(time.time() - t0) / 60:.1f} min)", flush=True)

    if n_workers <= 1:
        for j in jobs:
            report(run_job(j))
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # CUDA 不能用 fork
        with ProcessPoolExecutor(n_workers, mp_context=mp.get_context("spawn"), initializer=_init_worker, initargs=(threads,)) as pool:
            futures = {pool.submit(run_job, j): j for j in jobs}
            for fut in as_completed(futures):
                try:
                    report(fut.result())
                except Exception:
                    j = futures[fut]
                    print(f"  !! job failed: {j['cfg']['name']} (seed {j['seed']})", flush=True)
                    raise
    return results


def assemble(results, order, jobs):
    """把結果整理成畫圖和做表格要用的格式。"""
    cfg_of = {j["cfg"]["name"]: j["cfg"] for j in jobs}
    by_name = {}
    for r in sorted(results, key=lambda r: r["seed"]):
        by_name.setdefault(r["name"], []).append(r)

    def entry(name):
        e = dict(config=cfg_of[name], runs=by_name[name])
        e["params"] = count_params(build_model(Config(**cfg_of[name])))
        return e

    res = dict(part_a=[entry(n) for n in order["A"]], part_b=dict(variants=[entry(n) for n in order["B"]]))
    f = by_name["Final DigitNet"][0]
    res["final"] = dict(config=cfg_of["Final DigitNet"], params=count_params(DigitNet()), **f)
    methods = []
    for n in order["C"]:
        methods.append(entry(n))
        if n.startswith("C1"):
            # C1b 是 C1 的模型換預測方式,沒有自己的訓練曲線
            cfg = dict(cfg_of[n], name="C1b Cross-entropy + post-hoc logit adjustment")
            methods.append(dict(config=cfg, runs=[dict(seed=r["seed"], val=r["posthoc_val"], test=r["posthoc_test"], history=None)
                                                  for r in by_name[n]]))
    res["part_c"] = dict(methods=methods)
    return res


def lr_range_tests(data, quick):
    base = dict(use_bn=True, dropout=0.2, augment=True)
    steps = 60 if quick else 300
    return {
        "AdamW": lr_range_test(Config("lr-range AdamW", optimizer="adamw", **base), data["x_train"], data["y_train"], steps=steps),
        "SGD + momentum": lr_range_test(Config("lr-range SGDm", optimizer="sgdm", **base), data["x_train"], data["y_train"],
                                        lr_min=1e-4, lr_max=10.0, steps=steps),
    }


# ======================================================================================
# 畫圖(圖上的字用英文,沒裝中文字型會變方塊)
# ======================================================================================
def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, dpi=130)
    plt.close(fig)


def denorm(x):
    """還原成 0~1 的像素值。"""
    return (x * MNIST_STD + MNIST_MEAN).clamp(0, 1)


def mean_std(runs, key):
    """各 seed 的平均和標準差。"""
    v = np.array([r[key] for r in runs], dtype=float)
    return v.mean(0), (v.std(0, ddof=1) if len(v) > 1 else np.zeros_like(v[0]))


def plot_data_overview(data, lt):
    """圖 01:訓練樣本、資料增強的效果、原始與長尾的類別分布。"""
    x, y = data["x_train"].cpu(), data["y_train"].cpu()
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[3.3, 1])
    gs_right = gs[:, 1].subgridspec(2, 1, height_ratios=[1, 2.2], hspace=0.35)
    # 樣本格:10 個類別 x 每類 12 張
    ax = fig.add_subplot(gs[0, 0])
    rows = [denorm(x[(y == c).nonzero().squeeze(1)[:12]]).squeeze(1) for c in range(NUM_CLASSES)]
    grid = torch.cat([torch.cat(list(r), 1) for r in rows], 0)
    ax.imshow(grid, cmap="gray_r")
    ax.set_yticks(np.arange(NUM_CLASSES) * 28 + 14, [str(c) for c in range(NUM_CLASSES)])
    ax.set_xticks([])
    ax.set_title("Training samples (one row per class)")
    # 資料增強:每條的第一張是原圖,後面 7 張是隨機仿射變換後的結果
    ax = fig.add_subplot(gs[1, 0])
    gen = torch.Generator().manual_seed(0)
    base = x[[int((y == c).nonzero()[0]) for c in (2, 4, 7, 9)]]
    aug_rows = [torch.cat([denorm(b[None])] + [denorm(random_affine(b[None], gen)) for _ in range(7)], 3).squeeze() for b in base]
    ax.imshow(torch.cat([torch.cat(aug_rows[:2], 1), torch.cat(aug_rows[2:], 1)], 0), cmap="gray_r")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Augmentation: original (first image of each strip) + 7 random affine versions\n(rotation ±10°, shift ±2 px, scale ±10%)", fontsize=10)
    # 類別張數
    ax = fig.add_subplot(gs_right[0])
    bal = torch.bincount(y, minlength=NUM_CLASSES).numpy()
    ax.bar(range(NUM_CLASSES), bal, color="#4C72B0")
    ax.set_xticks(range(NUM_CLASSES)); ax.set_ylabel("images")
    ax.set_title(f"Original training split: nearly balanced ({bal.min():,}-{bal.max():,} per class)", fontsize=10)
    ax = fig.add_subplot(gs_right[1])
    counts = lt["class_counts"]
    bars = ax.bar(range(NUM_CLASSES), counts, color=["#4C72B0"] * 3 + ["#8172B2"] * 4 + ["#C44E52"] * 3)
    for b, v in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.08, f"{v:,}", ha="center", fontsize=8)
    ax.set_yscale("log"); ax.set_xticks(range(NUM_CLASSES)); ax.set_xlabel("digit"); ax.set_ylabel("images (log scale)")
    ax.set_ylim(top=max(counts) * 2.2)
    ax.set_title(f"Long-tailed training set for Part C: {counts[0]:,} -> {counts[-1]:,} "
                 f"({lt['imbalance_factor']:.0f}:1), total {sum(counts):,}\nblue = head (0-2), purple = middle (3-6), red = tail (7-9)", fontsize=10)
    savefig(fig, "01_data_overview.png")


def plot_training_comparison(entries, title, fname):
    """圖 02 / 04。左邊訓練 loss,右邊驗證錯誤率。線是各 seed 的平均,色帶是最小到最大。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for e in entries:
        runs = [r["history"] for r in e["runs"]]
        loss, _ = mean_std(runs, "step_loss")
        steps = np.arange(len(loss)) * runs[0]["step_every"]
        k = max(1, len(loss) // 60)
        sm = np.convolve(loss, np.ones(k) / k, mode="valid")
        line, = axes[0].plot(steps[k - 1:], sm, label=e["config"]["name"])
        errs = 100 * (1 - np.array([r["val_acc"] for r in runs], dtype=float))
        err = errs.mean(0)
        axes[1].plot(runs[0]["epoch"], err, marker="o", ms=3, color=line.get_color(), label=f"{e['config']['name']}  ({err[-1]:.2f}%)")
        if len(runs) > 1:
            axes[1].fill_between(runs[0]["epoch"], errs.min(0), errs.max(0), color=line.get_color(), alpha=0.15)
    axes[0].set(xlabel="training step", ylabel="training loss (moving average)", yscale="log", title="Training loss")
    axes[1].set(xlabel="epoch", ylabel="validation error % (log scale)", yscale="log",
                title="Validation error (line = mean, band = min-max over seeds)")
    for ax in axes:
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7.5)
    fig.suptitle(title)
    savefig(fig, fname)


def plot_lr(range_tests, variants):
    """圖 03。左:學習率範圍測試;右:Part B 實際比較的學習率排程。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for name, rt in range_tests.items():
        axes[0].plot(rt["lr"], rt["loss"], label=name)
    axes[0].axvline(PEAK_LR_ADAMW, color="C0", ls="--", lw=1, label=f"chosen peak LR, AdamW ({PEAK_LR_ADAMW:g})")
    axes[0].axvline(PEAK_LR_SGD, color="C1", ls="--", lw=1, label=f"chosen peak LR, SGD+momentum ({PEAK_LR_SGD:g})")
    axes[0].set(xscale="log", xlabel="learning rate (log scale)", ylabel="training loss (EMA)", title="LR range test", ylim=(0, 3))
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3, which="both")
    # 排程一樣的畫成同一條線(例如 B1、B2),不然會疊在一起
    groups = {}
    for e in variants:
        c = e["config"]
        groups.setdefault((c["schedule"], c["lr"]), []).append(e)
    for (schedule, lr), es in groups.items():
        h = es[0]["runs"][0]["history"]
        ids = ", ".join(e["config"]["name"].split()[0] for e in es)
        label = f"{ids}: " + (f"one-cycle, peak {lr:g}" if schedule == "onecycle" else f"constant {lr:g}")
        axes[1].plot(np.arange(len(h["step_lr"])) * h["step_every"], h["step_lr"], label=label,
                     ls="-" if schedule == "onecycle" else "--")
    axes[1].set(yscale="log", xlabel="training step", ylabel="learning rate (log scale)", title="Learning-rate schedules compared in Part B")
    axes[1].legend(fontsize=7.5); axes[1].grid(alpha=0.3, which="both")
    savefig(fig, "03_lr_range_test_and_schedules.png")


def offdiag_max(cm):
    """非對角線的最大值,用來統一色階。"""
    off = np.asarray(cm, dtype=float)
    off = off / off.sum(1, keepdims=True)
    np.fill_diagonal(off, 0)
    return off.max()


def plot_confusion(ax, cm, title, normalise=True, vmax=None):
    """對角線是 recall %,其他格子是錯誤張數。vmax 讓多張圖的色階一致。"""
    cm = np.asarray(cm, dtype=float)
    shown = cm / cm.sum(1, keepdims=True) if normalise else cm
    off = shown.copy()
    np.fill_diagonal(off, 0)
    vmax = vmax or max(off.max(), 1e-9)
    ax.imshow(off, cmap="Reds", vmin=0, vmax=vmax)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            v = cm[i, j]
            if i == j:
                ax.text(j, i, f"{shown[i, j] * 100:.1f}" if normalise else f"{int(v)}", ha="center", va="center", fontsize=7, color="#1a7a1a", fontweight="bold")
            elif v > 0:
                ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=7, color="white" if off[i, j] > 0.6 * vmax else "black")
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(title, fontsize=9)


def plot_final(final, data):
    """圖 05:最終模型的學習曲線與信心分布。圖 06:測試集混淆矩陣與最有把握的錯誤。"""
    h, t = final["history"], final["test"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    axes[0].plot(h["epoch"], h["train_loss"], marker="o", ms=3, label="train (augmented, dropout on)")
    axes[0].plot(h["epoch"], h["val_loss"], marker="o", ms=3, label="validation")
    axes[0].set(xlabel="epoch", ylabel="cross-entropy loss", yscale="log", title="Final model: loss")
    axes[1].plot(h["epoch"], 100 * (1 - np.array(h["train_acc"])), marker="o", ms=3, label="train (augmented, dropout on)")
    axes[1].plot(h["epoch"], 100 * (1 - np.array(h["val_acc"])), marker="o", ms=3, label="validation")
    axes[1].set(xlabel="epoch", ylabel="error % (log scale)", yscale="log", title="Final model: error rate")
    for ax in axes[:2]:
        ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    bins = np.linspace(0, 1, 26)
    axes[2].hist(final["confidence"]["correct"], bins=bins, alpha=0.6, density=True, label="correct predictions")
    axes[2].hist(final["confidence"]["wrong"], bins=bins, alpha=0.6, density=True, label=f"wrong predictions (n={len(final['confidence']['wrong'])})")
    axes[2].set(xlabel="confidence (max softmax probability)", ylabel="density (log scale)", yscale="log", title="Test set: confidence of correct vs. wrong predictions")
    axes[2].legend(fontsize=8)
    savefig(fig, "05_final_learning_curves.png")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), width_ratios=[1, 1.05])
    plot_confusion(axes[0], t["confusion"], f"Final model, test set: accuracy {t['accuracy'] * 100:.2f}%\n"
                   "diagonal (green) = recall %, off-diagonal = number of errors", normalise=True)
    # 把信心最高的錯誤拼成一張圖
    x_test = data["x_test"].cpu()
    w = final["wrong"]
    n_show = min(24, len(w["index"]))
    cols = 8
    rows = math.ceil(n_show / cols)
    canvas = torch.zeros(rows * 28, cols * 28)
    for i in range(n_show):
        r, c = divmod(i, cols)
        canvas[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = denorm(x_test[w["index"][i]]).squeeze(0)
    axes[1].imshow(canvas, cmap="gray_r")
    for i in range(n_show):
        r, c = divmod(i, cols)
        axes[1].text(c * 28 + 1, r * 28 + 5, f"{w['true'][i]}→{w['pred'][i]}", color="#C44E52", fontsize=9, fontweight="bold")
    axes[1].set_xticks([]); axes[1].set_yticks([])
    axes[1].set_title(f"Most confident mistakes on the test set (true→predicted); {int(round((1 - t['accuracy']) * len(x_test)))} errors in total", fontsize=9)
    savefig(fig, "06_final_confusion_and_errors.png")


def best_imbalance_method(methods) -> int:
    """用驗證集的 balanced accuracy 選 Part C 最好的方法。"""
    score = [np.mean([r["val"]["balanced_accuracy"] for r in m["runs"]]) for m in methods]
    return int(np.argmax(score))


def plot_imbalance(lt):
    """圖 07-10:Part C 的結果。"""
    methods = lt["methods"]
    names = [m["config"]["name"] for m in methods]
    colors = plt.cm.tab10(np.arange(len(methods)))
    best = best_imbalance_method(methods)
    # --- 圖 07:每個數字認對的比例,不處理和定案的做法並排
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = np.arange(NUM_CLASSES)
    for k, (i, col, label) in enumerate([(0, "#C44E52", "no handling"), (best, "#4C72B0", "final method")]):
        rec, _ = mean_std([r["test"] for r in methods[i]["runs"]], "recall")
        bars = ax.bar(xs + (k - 0.5) * 0.4, rec * 100, 0.4, color=col, label=label)
        for b, v in zip(bars, rec * 100):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.1f}", ha="center", fontsize=7.5)
    ax.set_ylim(88, 101.2)
    ax.set(xlabel="digit (number of training images)", ylabel="% of test images recognised correctly",
           title="How well each digit is recognised when the training set is imbalanced")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{c}\n({n:,})" for c, n in enumerate(lt["class_counts"])], fontsize=9)
    ax.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False); ax.grid(alpha=0.3, axis="y")
    savefig(fig, "07_imbalance_per_class_recall.png")

    # --- 圖 10:各方法的測試準確率,一個方法一條
    fig, ax = plt.subplots(figsize=(8, 4.2))
    acc = np.array([[r["test"]["accuracy"] for r in m["runs"]] for m in methods]) * 100
    order = np.arange(len(methods))[::-1]
    bar_colors = ["#4C72B0" if i == best else "#C44E52" if i == 0 else "#BBBBBB" for i in order]
    ax.barh([names[i] for i in order], acc.mean(1)[order], xerr=acc.std(1, ddof=1)[order] if acc.shape[1] > 1 else None,
            color=bar_colors, capsize=2)
    for y, i in enumerate(order):
        ax.text(acc.mean(1)[i] + (acc.std(1, ddof=1)[i] if acc.shape[1] > 1 else 0) + 0.04, y, f"{acc.mean(1)[i]:.2f}", va="center", fontsize=9)
    ax.set_xlim(acc.mean(1).min() - 0.5, acc.mean(1).max() + 0.6)
    ax.set(xlabel="test accuracy %", title="Long-tailed training set: test accuracy of each method")
    ax.grid(alpha=0.3, axis="x")
    savefig(fig, "10_imbalance_methods.png")

    # --- 圖 08:驗證曲線(C1b 沒有訓練過程,所以沒有曲線)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for m, col in zip(methods, colors):
        if m["runs"][0]["history"] is None:
            continue
        hs = [r["history"] for r in m["runs"]]
        for ax, key in zip(axes, ["val_bal_acc", "val_tail_recall"]):
            v, _ = mean_std(hs, key)
            ax.plot(hs[0]["epoch"], 100 * (1 - v), color=col, label=m["config"]["name"])
    axes[0].set(xlabel="epoch", ylabel="validation error % (log scale)", yscale="log", title="Balanced validation set: overall error")
    axes[1].set(xlabel="epoch", ylabel="1 - recall on digits 7-9, % (log scale)", yscale="log", title="Balanced validation set: tail-class error")
    for ax in axes:
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7.5)
    savefig(fig, "08_imbalance_validation_curves.png")

    # --- 圖 09:基準(C1)與最佳方法的測試集混淆矩陣;「最佳」是用驗證集選的
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    cms = [np.sum([r["test"]["confusion"] for r in methods[i]["runs"]], axis=0) for i in (0, best)]
    vmax = max(offdiag_max(cm) for cm in cms)  # 兩張圖用同一個色階
    for ax, i, cm in zip(axes, [0, best], cms):
        n_err = int(cm.sum() - np.trace(cm))
        plot_confusion(ax, cm, f"{names[i]}\nsum over {len(methods[i]['runs'])} seeds: {n_err} errors; diagonal = recall %, off-diagonal = errors", vmax=vmax)
    fig.suptitle("Long-tailed training, test-set confusion (both panels use the same colour scale)", fontsize=10)
    savefig(fig, "09_imbalance_confusion_baseline_vs_best.png")
    return names[best]


# ======================================================================================
# 結果表格(summary.md),README 的數字從這裡來
# ======================================================================================
def pm(values, scale=100, digits=2):
    """平均 ± 標準差。"""
    v = np.asarray(values, dtype=float) * scale
    return f"{v.mean():.{digits}f} ± {v.std(ddof=1):.{digits}f}" if len(v) > 1 else f"{v.mean():.{digits}f}"


def write_summary(res):
    L = []
    meta = res["meta"]
    L.append(f"由 `python main.py`{' --quick(快速測試,數字沒有意義)' if meta['quick'] else ''} 產生;"
             f"裝置 {meta['device']}({meta.get('workers', 1)} 個行程,torch {meta['torch']}),共 {meta['minutes']:.1f} 分鐘。\n")
    ns = len(res["part_a"][0]["runs"])
    L.append(f"### Part A - 模型設計({res['part_a'][0]['config']['epochs']} epochs,AdamW + one-cycle,{ns} 個種子的平均 ± 標準差)\n")
    L.append("| 模型 | 參數量 | 驗證準確率 % | 測試準確率 % | 測試錯誤數 / 10,000 |\n|---|---:|---:|---:|---:|")
    for e in res["part_a"]:
        ta = [r["test_acc"] for r in e["runs"]]
        L.append(f"| {e['config']['name']} | {e['params']:,} | {pm([r['val_acc'] for r in e['runs']])} | {pm(ta)} | {np.mean([(1 - a) * 1e4 for a in ta]):.0f} |")
    L.append(f"\n### Part B - 優化(DigitNet,{res['part_b']['variants'][0]['config']['epochs']} epochs,{ns} 個種子的平均 ± 標準差)\n")
    L.append("| 優化器 / 學習率排程 | 第 1 個 epoch 後驗證準確率 % | 最終驗證準確率 % | 測試準確率 % |\n|---|---:|---:|---:|")
    for e in res["part_b"]["variants"]:
        L.append(f"| {e['config']['name']} | {pm([r['history']['val_acc'][0] for r in e['runs']])} | {pm([r['val_acc'] for r in e['runs']])} | {pm([r['test_acc'] for r in e['runs']])} |")
    f = res["final"]
    L.append(f"\n### 最終模型(完整平衡資料,{f['config']['epochs']} epochs)\n")
    L.append(f"- 參數量:{f['params']:,}\n- 驗證準確率:{f['val_acc'] * 100:.2f}%\n- **測試準確率:{f['test']['accuracy'] * 100:.2f}%**"
             f"(10,000 張錯 {int(round((1 - f['test']['accuracy']) * 1e4))} 張),macro-F1 {f['test']['macro_f1'] * 100:.2f}%")
    L.append("- 各數字的 recall %:" + "、".join(f"{c}: {r * 100:.1f}" for c, r in enumerate(f["test"]["recall"])))
    lt = res["part_c"]
    L.append(f"\n### Part C - 類別不平衡(各類訓練張數 {lt['class_counts']},{lt['methods'][0]['config']['epochs']} epochs,"
             f"{len(lt['methods'][0]['runs'])} 個種子的平均 ± 標準差;「驗證」欄用來選方法,其餘各欄是官方測試集)\n")
    L.append("| 方法 | 驗證 balanced acc. % | 測試準確率 % | Macro-F1 % | 頭部 recall %(0-2) | 中段 recall %(3-6) | 尾部 recall %(7-9) | 數字 9 的 recall % |\n|---|---:|---:|---:|---:|---:|---:|---:|")
    for m in lt["methods"]:
        g = lambda k: pm([r["test"][k] for r in m["runs"]])
        L.append(f"| {m['config']['name']} | {pm([r['val']['balanced_accuracy'] for r in m['runs']])} | {g('accuracy')} | {g('macro_f1')} | "
                 f"{g('head_recall')} | {g('mid_recall')} | {g('tail_recall')} | {pm([r['test']['recall'][9] for r in m['runs']])} |")
    best = lt["methods"][best_imbalance_method(lt["methods"])]["config"]["name"]
    L.append(f"\n依驗證集 balanced accuracy 選出的最佳方法:**{best}**")
    (OUT_DIR / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n" + "\n".join(L))


# ======================================================================================
def make_plots(res, data):
    plot_data_overview(data, res["part_c"])
    plot_training_comparison(res["part_a"], "Part A - modeling: architecture and regularisation", "02_modeling_comparison.png")
    plot_lr(res["part_b"]["range_tests"], res["part_b"]["variants"])
    plot_training_comparison(res["part_b"]["variants"], "Part B - optimization: optimizer and LR schedule (same model, same data)", "04_optimizer_comparison.png")
    plot_final(res["final"], data)
    best = plot_imbalance(res["part_c"])
    print(f"Best imbalance method by validation balanced accuracy: {best}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny smoke test")
    ap.add_argument("--plots-only", action="store_true", help="redraw plots from outputs/results.json")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    ap.add_argument("--imbalance-factor", type=float, default=100.0)
    ap.add_argument("--workers", type=int, default=0, help="parallel training processes (0 = auto: cores/threads on CPU, 1 on GPU)")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads per training process (when several processes are used)")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):  # summary 有中文,避免 Windows 終端機印不出來
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    global OUT_DIR
    if args.quick:  # 不要蓋掉正式結果
        OUT_DIR = HERE / "outputs_quick"
    OUT_DIR.mkdir(exist_ok=True)

    if args.device == "auto":
        # 舊版 PyTorch 的 MPS 不支援 grid_sample 的 border padding,所以不自動選 mps
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # CPU 每 4 核開一個 process;GPU 預設 1 個,可以用 --workers 指定
    n_workers = args.workers or (max(1, (os.cpu_count() or 4) // args.threads) if device == "cpu" else 1)
    print(f"device: {device} | torch {torch.__version__} | {n_workers} worker process(es) x {args.threads} threads")
    subset = 6000 if args.quick else None
    data = get_data(device, subset)  # 先下載好,不然子行程會同時去抓
    print({k: tuple(v.shape) for k, v in data.items() if k.startswith("x_")})

    if args.plots_only:
        with open(OUT_DIR / "results.json", encoding="utf-8") as fh:
            res = json.load(fh)
    else:
        t0 = time.time()
        if args.quick:
            ep_ab, ep_final, ep_lt, seeds, n_max = 1, 1, 2, [0], 500
        else:
            # 長尾的資料比較少,所以多跑幾個 epoch
            ep_ab, ep_final, ep_lt, seeds, n_max = 6, 15, 30, [0, 1, 2], 5000
        lt_kwargs = dict(n_max=n_max, imbalance_factor=args.imbalance_factor)
        _, counts = longtail_indices(data["y_train"], seed=0, **lt_kwargs)
        print(f"long-tailed class counts for Part C: {counts} (total {sum(counts)})")

        print("\nLR range tests")
        torch.set_num_threads(min(8, os.cpu_count() or 8))
        range_tests = lr_range_tests(data, args.quick)

        jobs, order = build_jobs(ep_ab, ep_final, ep_lt, seeds, lt_kwargs)
        for j in jobs:
            j["subset"], j["out_dir"] = subset, str(OUT_DIR)
        print(f"\nTraining {len(jobs)} runs (Part A: architectures, Part B: optimizers, final model, Part C: imbalance)")
        results = run_all(jobs, device, n_workers, args.threads)
        res = assemble(results, order, jobs)
        res["part_b"]["range_tests"] = range_tests
        res["part_c"].update(imbalance_factor=args.imbalance_factor, class_counts=counts)
        res["meta"] = dict(quick=args.quick, device=device, torch=torch.__version__, seeds=seeds, workers=n_workers,
                           threads=args.threads, minutes=(time.time() - t0) / 60)
        with open(OUT_DIR / "results.json", "w", encoding="utf-8") as fh:
            json.dump(res, fh)

    make_plots(res, data)
    write_summary(res)
    print(f"\nAll outputs are in {OUT_DIR}")


if __name__ == "__main__":
    main()
