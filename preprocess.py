"""将摄像头截取区域处理成 MNIST 风格的 28×28 输入。"""

from dataclasses import dataclass
import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from utils import MNIST_MEAN, MNIST_STD


@dataclass
class PreprocessResult:
    """预处理结果；digit 是黑底白字 uint8 图，tensor 已按 MNIST 归一化。"""

    valid: bool
    digit: Optional[np.ndarray]
    tensor: Optional[torch.Tensor]
    message: str


def preprocess_digit(image: np.ndarray) -> PreprocessResult:
    """清理 ROI、提取前景并缩放居中；无效输入返回可显示的原因。"""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return PreprocessResult(False, None, None, "Empty image")
    if image.ndim not in (2, 3):
        return PreprocessResult(False, None, None, "Unsupported image shape")
    if image.shape[0] < 8 or image.shape[1] < 8:
        return PreprocessResult(False, None, None, "ROI is too small")

    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        return PreprocessResult(False, None, None, "Unsupported channel count")

    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # MNIST 是黑底白字。纸面通常亮像素占多数，需反相；深色背景的白字则保留。
    if np.count_nonzero(binary) > binary.size / 2:
        binary = cv2.bitwise_not(binary)

    # 连通域过滤小噪声；阈值随 ROI 面积变化，避免固定像素门槛影响不同分辨率。
    count, component_map, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    min_area = max(4, int(binary.size * 0.0005))
    foreground = np.zeros_like(binary)
    valid_components = []
    for component_id in range(1, count):  # 0 是背景
        x, y, width, height, area = stats[component_id]
        if area >= min_area:
            foreground[component_map == component_id] = 255
            valid_components.append((x, y, width, height))

    if not valid_components:
        return PreprocessResult(False, None, None, "No digit found")

    x_min = min(item[0] for item in valid_components)
    y_min = min(item[1] for item in valid_components)
    x_max = max(item[0] + item[2] for item in valid_components)
    y_max = max(item[1] + item[3] for item in valid_components)
    digit_crop = foreground[y_min:y_max, x_min:x_max]
    height, width = digit_crop.shape
    if height == 0 or width == 0:
        return PreprocessResult(False, None, None, "No digit found")

    # 留出 4 像素边界，保持数字宽高比。
    scale = min(20.0 / width, 20.0 / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(digit_crop, (new_width, new_height), interpolation=interpolation)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    left = (28 - new_width) // 2
    top = (28 - new_height) // 2
    canvas[top : top + new_height, left : left + new_width] = resized

    normalized = (canvas.astype(np.float32) / 255.0 - MNIST_MEAN) / MNIST_STD
    tensor = torch.from_numpy(normalized).unsqueeze(0)
    return PreprocessResult(True, canvas, tensor, "OK")


def main() -> None:
    """独立检查图片预处理：python preprocess.py input.png。"""
    parser = argparse.ArgumentParser(description="预处理一张数字图片为 MNIST 28x28")
    parser.add_argument("image", help="输入图片路径")
    parser.add_argument(
        "--output",
        default="outputs/preprocessed_digit.png",
        help="28x28 预览图路径，默认 outputs/preprocessed_digit.png",
    )
    args = parser.parse_args()

    image = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    result = preprocess_digit(image)
    if not result.valid:
        raise SystemExit(f"Preprocessing failed: {result.message}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), result.digit):
        raise SystemExit(f"Could not write output image: {output_path}")
    print(f"Saved 28x28 preview: {output_path.resolve()}")


if __name__ == "__main__":
    main()
