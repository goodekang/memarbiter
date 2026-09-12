from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.memarbiter.nli import LABELS, load_mnli_backbone
from src.paths import resolve_path
from src.train.nli_dataset import PairDataset, load_pairs, split_pairs


def train_conflict_nli(cfg: dict) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pairs = load_pairs(resolve_path(cfg["pairs_path"]))
    if len(pairs) < 2:
        raise FileNotFoundError(f"need labelled pairs at {cfg['pairs_path']}")
    if cfg.get("n_pairs"):
        pairs = pairs[: int(cfg["n_pairs"])]
    train_rows, val_rows = split_pairs(pairs, float(cfg["train"]["val_fraction"]), int(cfg["train"]["seed"]))
    model, tokenizer = load_mnli_backbone(cfg["base_model"], num_labels=len(LABELS))
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    model.to(device)
    train_loader = DataLoader(
        PairDataset(train_rows, tokenizer, int(cfg["max_length"])),
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
    )
    val_loader = DataLoader(
        PairDataset(val_rows, tokenizer, int(cfg["max_length"])),
        batch_size=int(cfg["train"]["eval_batch_size"]),
        shuffle=False,
    )
    optim = AdamW(
        model.parameters(),
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
        betas=(0.9, float(cfg["train"]["adam_beta2"])),
    )
    total_steps = max(1, len(train_loader) * int(cfg["train"]["epochs"]))
    warmup = int(total_steps * float(cfg["train"]["warmup_ratio"]))
    sched = get_linear_schedule_with_warmup(optim, warmup, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["train"]["amp"]) and device == "cuda")
    out_dir = resolve_path(cfg["output_dir"])
    best_dir = out_dir / "best"
    last_dir = out_dir / "last"
    best_dir.mkdir(parents=True, exist_ok=True)
    last_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    global_step = 0
    history = []
    print(f"device={device} train={len(train_rows)} val={len(val_rows)} steps={total_steps}")
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                out = model(**batch)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
            scaler.step(optim)
            scaler.update()
            sched.step()
            running += float(loss.item())
            global_step += 1
            if global_step % int(cfg["train"]["eval_every_steps"]) == 0:
                metrics = evaluate(model, val_loader, device)
                metrics.update({"epoch": epoch, "step": global_step, "train_loss": running / max(1, global_step)})
                history.append(metrics)
                print(
                    f"epoch={epoch} step={global_step} loss={metrics['train_loss']:.4f} "
                    f"val_acc={metrics['acc']:.3f} val_f1={metrics['macro_f1']:.3f}"
                )
                if metrics["macro_f1"] > best_f1:
                    best_f1 = metrics["macro_f1"]
                    _save(model, tokenizer, best_dir, metrics)
        epoch_metrics = evaluate(model, val_loader, device)
        epoch_metrics.update({"epoch": epoch, "step": global_step, "train_loss": running / max(1, len(train_loader))})
        history.append(epoch_metrics)
        _save(model, tokenizer, last_dir, epoch_metrics)
        if epoch_metrics["macro_f1"] > best_f1:
            best_f1 = epoch_metrics["macro_f1"]
            _save(model, tokenizer, best_dir, epoch_metrics)
        print(f"epoch={epoch} done val_f1={epoch_metrics['macro_f1']:.3f} best={best_f1:.3f}")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"best_macro_f1": best_f1, "steps": global_step, "device": device}


@torch.no_grad()
def evaluate(model, loader, device: str) -> dict:
    model.eval()
    gold: list[int] = []
    pred: list[int] = []
    losses = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        out = model(**batch)
        losses.append(float(out.loss.item()))
        pred.extend(out.logits.argmax(dim=-1).cpu().tolist())
        gold.extend(batch["labels"].cpu().tolist())
    gold_a = np.asarray(gold)
    pred_a = np.asarray(pred)
    acc = float((gold_a == pred_a).mean()) if gold_a.size else 0.0
    f1s = []
    for label in range(len(LABELS)):
        tp = int(((pred_a == label) & (gold_a == label)).sum())
        fp = int(((pred_a == label) & (gold_a != label)).sum())
        fn = int(((pred_a != label) & (gold_a == label)).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return {"acc": acc, "macro_f1": float(np.mean(f1s)), "val_loss": float(np.mean(losses) if losses else 0.0)}


def _save(model, tokenizer, path: Path, metrics: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    payload = dict(metrics)
    payload["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    payload["labels"] = LABELS
    (path / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
