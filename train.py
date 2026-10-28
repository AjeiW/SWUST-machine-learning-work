"""训练 MNIST CNN，并保存验证集最优权重和训练曲线。"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from model import MNISTCNN
from utils import (
    CHECKPOINT_DIR,
    DATA_DIR,
    OUTPUT_DIR,
    ensure_project_dirs,
    get_device,
    get_mnist_transform,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="训练 MNIST 手写数字 CNN")
    parser.add_argument("--epochs", type=int, default=8, help="训练轮数，默认 8")
    parser.add_argument("--batch-size", type=int, default=64, help="批大小，默认 64")
    parser.add_argument("--lr", type=float, default=0.001, help="Adam 学习率，默认 0.001")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例，默认 0.1")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader 子进程数，Windows 默认 0")
    return parser.parse_args()


def run_epoch(model, loader, criterion, device, optimizer=None):
    """执行一个训练或验证 epoch，返回按样本数加权的 loss 和 accuracy。"""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def save_training_curves(history, output_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train loss")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["train_accuracy"], marker="o", label="Train accuracy")
    axes[1].plot(epochs, history["val_accuracy"], marker="o", label="Validation accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        raise ValueError("epochs 和 batch-size 必须为正数，lr 必须大于 0。")
    if not 0 < args.val_ratio < 1:
        raise ValueError("val-ratio 必须在 0 和 1 之间。")
    if args.num_workers < 0:
        raise ValueError("num-workers 不能小于 0。")

    ensure_project_dirs()
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    # 只从官方训练集划分训练与验证数据；官方测试集留给 test.py。
    full_train_set = datasets.MNIST(
        root=str(DATA_DIR), train=True, download=True, transform=get_mnist_transform()
    )
    val_size = max(1, int(len(full_train_set) * args.val_ratio))
    train_size = len(full_train_set) - val_size
    split_generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(
        full_train_set, [train_size, val_size], generator=split_generator
    )

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        **loader_options,
    )
    val_loader = DataLoader(val_set, shuffle=False, **loader_options)

    model = MNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = {"train_loss": [], "train_accuracy": [], "val_accuracy": []}
    best_val_accuracy = -1.0
    checkpoint_path = CHECKPOINT_DIR / "best_mnist_cnn.pth"
    started_at = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        _, val_accuracy = run_epoch(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"Epoch [{epoch:02d}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy * 100:.2f}% | "
            f"Val Acc: {val_accuracy * 100:.2f}%"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_accuracy": val_accuracy,
                    "config": {
                        "epochs": args.epochs,
                        "batch_size": args.batch_size,
                        "learning_rate": args.lr,
                        "seed": args.seed,
                    },
                },
                checkpoint_path,
            )

        (OUTPUT_DIR / "training_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        save_training_curves(history, OUTPUT_DIR / "training_curves.png")

    elapsed = time.perf_counter() - started_at
    print(f"Best validation accuracy: {best_val_accuracy * 100:.2f}%")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Training curves: {OUTPUT_DIR / 'training_curves.png'}")
    print(f"Elapsed: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
