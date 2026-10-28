"""使用官方 MNIST 测试集评估训练好的 CNN。"""

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets

from model import MNISTCNN
from utils import (
    CHECKPOINT_DIR,
    DATA_DIR,
    OUTPUT_DIR,
    ensure_project_dirs,
    get_device,
    get_mnist_transform,
    load_model_weights,
    set_seed,
)


def main():
    ensure_project_dirs()
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    test_set = datasets.MNIST(
        root=str(DATA_DIR), train=False, download=True, transform=get_mnist_transform()
    )
    test_loader = DataLoader(
        test_set,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = MNISTCNN().to(device)
    checkpoint = load_model_weights(
        model, CHECKPOINT_DIR / "best_mnist_cnn.pth", device
    )
    model.eval()

    all_predictions = []
    all_labels = []
    with torch.inference_mode():
        for images, labels in test_loader:
            logits = model(images.to(device, non_blocking=True))
            all_predictions.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.tolist())

    labels_array = np.asarray(all_labels, dtype=np.int64)
    predictions_array = np.asarray(all_predictions, dtype=np.int64)
    accuracy = float(np.mean(labels_array == predictions_array))
    matrix = confusion_matrix(labels_array, predictions_array, labels=list(range(10)))
    per_class = []
    for digit in range(10):
        class_total = int(matrix[digit].sum())
        class_correct = int(matrix[digit, digit])
        per_class.append(
            {
                "digit": digit,
                "correct": class_correct,
                "total": class_total,
                "accuracy": class_correct / class_total if class_total else None,
            }
        )

    metrics = {
        "test_accuracy": accuracy,
        "test_samples": len(test_set),
        "best_validation_accuracy": checkpoint.get("val_accuracy"),
        "best_epoch": checkpoint.get("epoch"),
        "per_class": per_class,
    }
    (OUTPUT_DIR / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    with (OUTPUT_DIR / "per_class_accuracy.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=["digit", "correct", "total", "accuracy"])
        writer.writeheader()
        writer.writerows(per_class)

    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        title="MNIST Test Confusion Matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(10),
        yticks=np.arange(10),
    )
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(10):
        for column in range(10):
            axis.text(
                column, row, f"{matrix[row, column]}",
                ha="center", va="center",
                color="white" if matrix[row, column] > threshold else "black",
                fontsize=8,
            )
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close(figure)

    # 固定随机种子，便于答辩时复现同一组样本图。
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(test_set), size=16, replace=False)
    figure, axes = plt.subplots(4, 4, figsize=(8, 8))
    for axis, index in zip(axes.flat, sample_indices):
        true_label = int(labels_array[index])
        predicted_label = int(predictions_array[index])
        axis.imshow(test_set.data[index].numpy(), cmap="gray")
        axis.set_title(f"True: {true_label} | Pred: {predicted_label}", fontsize=9)
        axis.axis("off")
    figure.suptitle("Random MNIST Test Predictions", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(OUTPUT_DIR / "test_predictions.png", dpi=160)
    plt.close(figure)

    correct_count = int(np.count_nonzero(labels_array == predictions_array))
    print(f"Test accuracy: {accuracy * 100:.2f}% ({correct_count}/{len(test_set)})")
    print("Per-class accuracy:")
    for item in per_class:
        class_accuracy = item["accuracy"]
        value = f"{class_accuracy * 100:.2f}%" if class_accuracy is not None else "N/A"
        print(f"  {item['digit']}: {value} ({item['correct']}/{item['total']})")
    print(f"Saved evaluation outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
