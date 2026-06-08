"""
Training loop for LeJEPA on the flotation plant dataset.
"""

from __future__ import annotations

import time
import math
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader


from .config import LeJEPAConfig
from .data import build_dataloaders
from .model import LeJEPA


def _warmup_cosine_schedule(warmup_steps: int, total_steps: int):
    """Produce a callable ``step → lr_factor`` with linear warmup + cosine decay."""
    def fn(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        # Cosine decay
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return fn


def train_one_epoch(
    model: LeJEPA,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    device: torch.device,
    epoch: int,
    max_steps: int,
    log_every: int,
) -> dict[str, float]:
    """Run one training epoch.  Returns averaged losses."""
    model.train()

    total_loss = 0.0
    total_inv = 0.0
    total_var = 0.0
    total_cov = 0.0
    num_batches = 0
    t0 = time.perf_counter()

    for step, (ctx, tgt) in enumerate(loader):
        ctx = ctx.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        # Model runs both ctx and tgt through encoder, then predicts
        losses = model(ctx, tgt)
        loss = losses["loss"]

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to stabilise training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        global_step = epoch * len(loader) + step

        total_loss += losses["loss"].item()
        total_inv += losses["inv"].item()
        total_var += losses["var"].item()
        total_cov += losses["cov"].item()
        num_batches += 1

        if step % log_every == 0 or step == len(loader) - 1:
            lr = scheduler.get_last_lr()[0]
            print(
                f"E{epoch:3d} | batch {step:5d}/{len(loader)} | "
                f"loss {losses['loss'].item():.4f}  "
                f"inv {losses['inv'].item():.4f}  "
                f"var {losses['var'].item():.4f}  "
                f"cov {losses['cov'].item():.4f}  "
                f"lr {lr:.2e}"
            )

    elapsed = time.perf_counter() - t0
    print(
        f"─── Epoch {epoch} finished — "
        f"{elapsed:.0f}s | "
        f"loss {total_loss / num_batches:.4f}  "
        f"inv {total_inv / num_batches:.4f}  "
        f"var {total_var / num_batches:.4f}  "
        f"cov {total_cov / num_batches:.4f}"
    )

    return {
        "loss": total_loss / num_batches,
        "inv": total_inv / num_batches,
        "var": total_var / num_batches,
        "cov": total_cov / num_batches,
    }


@torch.no_grad()
def validate(
    model: LeJEPA,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Compute loss on validation set."""
    model.eval()

    total_loss = 0.0
    total_inv = 0.0
    total_var = 0.0
    total_cov = 0.0
    num = 0

    for ctx, tgt in loader:
        ctx = ctx.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        losses = model(ctx, tgt)

        total_loss += losses["loss"].item()
        total_inv += losses["inv"].item()
        total_var += losses["var"].item()
        total_cov += losses["cov"].item()
        num += 1

    return {
        "loss": total_loss / num,
        "inv": total_inv / num,
        "var": total_var / num,
        "cov": total_cov / num,
    }


def run(cfg: LeJEPAConfig) -> LeJEPA:
    """Full training run.  Returns the trained model."""
    # ── Device ────────────────────────────────────────────────────────────
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = build_dataloaders(cfg)
    print(
        f"Train: {len(train_loader.dataset)} windows  "
        f"Val: {len(val_loader.dataset)} windows"
    )

    # ── Model ─────────────────────────────────────────────────────────────
    sample_ctx, _ = next(iter(train_loader))
    num_features = sample_ctx.shape[-1]

    model = LeJEPA(
        num_features,
        encoder_dims=cfg.encoder_dims,
        encoder_act=cfg.encoder_act,
        predictor_dims=cfg.predictor_dims,
        predictor_act=cfg.predictor_act,
        inv_weight=cfg.inv_weight,
        var_weight=cfg.var_weight,
        cov_weight=cfg.cov_weight,
        var_gamma=cfg.var_gamma,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # ── Optimiser & scheduler ─────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg.max_epochs
    scheduler = LambdaLR(
        optimizer,
        _warmup_cosine_schedule(cfg.warmup_steps, total_steps),
    )

    # ── Checkpoint dir ────────────────────────────────────────────────────
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Train ─────────────────────────────────────────────────────────────
    best_val_loss = float("inf")

    for epoch in range(cfg.max_epochs):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            epoch, total_steps, cfg.log_every,
        )

        val_metrics = validate(model, val_loader, device)
        print(
            f"  val loss {val_metrics['loss']:.4f}  "
            f"inv {val_metrics['inv']:.4f}  "
            f"var {val_metrics['var']:.4f}  "
            f"cov {val_metrics['cov']:.4f}"
        )

        # Save best checkpoint
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            ckpt_path = ckpt_dir / "lejepa_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_metrics["loss"],
                    "config": cfg,
                },
                ckpt_path,
            )
            print(f"  ✓ checkpoint saved to {ckpt_path}")

    # Save final checkpoint
    ckpt_path = ckpt_dir / "lejepa_final.pt"
    torch.save(
        {
            "epoch": cfg.max_epochs - 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_metrics["loss"],
            "config": cfg,
        },
        ckpt_path,
    )
    print(f"Final checkpoint saved to {ckpt_path}")
    print(f"Best val loss: {best_val_loss:.4f}")

    return model
