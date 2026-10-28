"""项目通用配置和辅助函数。"""

import random
from pathlib import Path

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
OUTPUT_DIR = PROJECT_DIR / "outputs"

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def ensure_project_dirs() -> None:
    """创建项目运行所需的目录。"""
    for directory in (DATA_DIR, CHECKPOINT_DIR, OUTPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    """固定 Python、NumPy 和 PyTorch 随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 提高实验可复现性；确定性算法可能略微降低训练速度。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """有可用 CUDA 时使用 GPU，否则回退到 CPU。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_mnist_transform():
    """MNIST 与摄像头图像共用的张量化和归一化流程。"""
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )


def load_model_weights(model: torch.nn.Module, path: Path, device: torch.device):
    """加载 state_dict，也兼容本项目保存的带训练信息的 checkpoint。"""
    if not path.is_file():
        raise FileNotFoundError(
            f"找不到模型权重：{path}\n请先运行 python train.py 完成训练。"
        )
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return checkpoint
