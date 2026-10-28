# 基于 CNN 的 MNIST 手写数字识别系统

这是一个基于 PyTorch CNN 的 MNIST 手写数字识别项目，包含：

- MNIST 数据集训练、验证和模型保存；
- 官方测试集评估、逐类准确率和混淆矩阵分析；
- Flask 本地网页识别，支持摄像头、图片上传和 ROI 框选；
- OpenCV 单窗口备用演示；
- 摄像头或普通图片到 MNIST 风格 `28×28` 输入的预处理流程。

项目只使用 CNN 分类模型。网页和 OpenCV 程序都加载训练生成的同一个 checkpoint。

## 目录结构

```text
SWUST-machine-learning-work/
├── model.py                 # CNN 网络结构
├── utils.py                 # 路径、随机种子、设备和公共预处理配置
├── train.py                 # 训练、验证、保存最佳权重和训练曲线
├── test.py                  # 官方测试集评估和可视化
├── preprocess.py            # 图片/摄像头 ROI 预处理
├── app.py                   # Flask 本地网页识别程序
├── app_cv.py                # OpenCV 单窗口备用演示
├── web/
│   ├── index.html           # 网页结构
│   ├── app.js               # 摄像头、上传和识别交互
│   └── style.css            # 网页样式
├── requirements.txt         # Python 依赖
├── data/MNIST/              # 自动下载的 MNIST 数据集
├── checkpoints/             # 训练生成的模型权重
└── outputs/                 # 训练、测试和预处理输出
```

`data/`、`checkpoints/` 和 `outputs/` 会在运行时自动创建。数据集、模型权重和实验输出属于运行产物，不应和源代码混为一谈。

## 环境准备

建议使用 Windows 10/11、PowerShell 7 和 Python 3.10。项目依赖使用兼容的 PyTorch 2.5.1 / torchvision 0.20.1 组合。

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果 PowerShell 阻止激活脚本，可在当前终端运行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

安装后检查 PyTorch 版本和计算设备：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### CPU 安装

如果使用 CPU，可在安装其余依赖前固定 CPU wheel：

```powershell
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### CUDA 安装

例如使用 CUDA 12.1：

```powershell
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

PyTorch wheel 自带 CUDA runtime，但仍需要兼容的 NVIDIA 驱动。不要把本机 CUDA Toolkit 版本直接当成驱动版本。

## 运行流程

推荐顺序是：训练 → 官方测试 → 应用演示。

### 1. 训练模型

```powershell
python train.py
```

默认配置：

- Epoch：8；
- Batch Size：64；
- Optimizer：Adam；
- Learning Rate：0.001；
- Validation Ratio：0.1，即训练集的 90% / 10% 划分；
- Random Seed：42；
- DataLoader workers：0，适合 Windows。

训练脚本只使用 MNIST 官方训练集，并从其中划分训练集和验证集；官方测试集不会参与训练或最佳模型选择。

可覆盖主要超参数：

```powershell
python train.py --epochs 12 --batch-size 128 --lr 0.0005
```

训练产物：

- `checkpoints/best_mnist_cnn.pth`：验证集准确率最高的模型；
- `outputs/training_history.json`：每轮训练 loss、训练准确率和验证准确率；
- `outputs/training_curves.png`：训练 loss、训练准确率和验证准确率曲线。

### 2. 测试与评估

训练完成后运行：

```powershell
python test.py
```

脚本会使用官方 MNIST 测试集输出整体准确率和 `0–9` 各类别准确率，并生成：

- `outputs/test_metrics.json`：整体指标、最佳 epoch 和逐类指标；
- `outputs/per_class_accuracy.csv`：逐类正确数、样本数和准确率；
- `outputs/confusion_matrix.png`：混淆矩阵；
- `outputs/test_predictions.png`：固定随机种子的 16 张测试图片及预测结果。

测试结果必须以实际运行生成的文件为准。98% 只是目标，不是代码提前承诺的售后服务。

### 3. 网页摄像头与图片识别

```powershell
python app.py
```

程序会加载 `checkpoints/best_mnist_cnn.pth`，启动仅监听本机的 Flask 服务，并尝试打开：

```text
http://127.0.0.1:5000/
```

也可以指定端口，或只启动服务而不自动打开浏览器：

```powershell
python app.py --port 5050
python app.py --no-browser
```

网页功能：

- 打开摄像头并切换摄像头设备；
- 上传 PNG、JPG、BMP 或 WebP 图片；
- 在摄像头画面或图片上拖动鼠标框选 ROI；
- 识别选区，未框选时识别整张图片；
- 显示预测数字、置信度和实际送入模型的 `28×28` 预处理结果；
- 保存最近 5 条识别记录。

网页快捷操作：

- `SPACE`：识别；
- `R`：清除框选；
- `C`：清空识别记录；
- `ESC`：清除框选；
- `Q`：请求停止本机服务。

网页图片只发送到本机 `127.0.0.1` 的 Python 服务，不会上传到外网。

### 4. OpenCV 备用演示

启动摄像头演示：

```powershell
python app_cv.py
```

直接识别本地图片：

```powershell
python app_cv.py --image .\digit.png
```

OpenCV 窗口支持摄像头、图片上传、ROI 框选、设备切换和快捷键操作。它与网页程序共用 `model.py`、`preprocess.py` 和 `utils.py`。

### 5. 单独检查预处理

```powershell
python preprocess.py .\digit.png
```

默认输出：

```text
outputs/preprocessed_digit.png
```

也可以指定输出路径：

```powershell
python preprocess.py .\digit.png --output outputs\preprocessed_digit_01.png
```

## 模型与输入约定

### 模型接口

`MNISTCNN` 的输入形状为 `(N, 1, 28, 28)`，输出为 10 个 logits，类别索引对应数字 `0` 到 `9`。

训练、测试和应用推理必须加载同一模型结构和兼容的 checkpoint。

### 预处理流程

`preprocess_digit()` 会：

1. 检查输入是否为空、尺寸是否过小以及通道数是否支持；
2. 转换为灰度图并进行高斯模糊；
3. 使用 Otsu 方法二值化，并根据前景比例判断是否反相；
4. 过滤小连通域噪声；
5. 裁剪数字前景，保持宽高比缩放；
6. 将数字居中放入黑底 `28×28` 画布；
7. 使用 MNIST 的均值 `0.1307` 和标准差 `0.3081` 归一化。

空白图片、过小 ROI 或无法提取数字的输入会返回明确的失败信息，而不是强行送进模型。

## 超参数实验建议

每次实验只修改一个主要因素，并完整记录命令和输出。示例：

```powershell
# Epoch
python train.py --epochs 4
python train.py --epochs 8
python train.py --epochs 12

# Batch Size
python train.py --batch-size 32
python train.py --batch-size 64
python train.py --batch-size 128

# Learning Rate
python train.py --lr 0.0005
python train.py --lr 0.001
python train.py --lr 0.002
```

每次实验建议记录：

- 实验编号和完整训练命令；
- Python、PyTorch、torchvision 版本以及 CPU/GPU；
- Epoch、Batch Size、Learning Rate 和 Random Seed；
- 最佳验证准确率和最佳 epoch；
- 官方测试准确率；
- 训练耗时；
- 不收敛、过拟合或其他异常现象。

重要实验应及时备份 `outputs/` 中的结果和对应 checkpoint，因为重复运行可能覆盖旧文件。

## 常见问题

- **`py -3.10` 找不到解释器：** 安装 Python 3.10 并确认 `py -0p` 中存在对应版本。
- **MNIST 下载失败：** 检查网络和数据集镜像访问；下载完成后重跑即可，已有数据不会重复下载。
- **`torch.cuda.is_available()` 是 `False`：** 检查是否安装 CUDA 版 wheel 和兼容的 NVIDIA 驱动；CPU 模式仍可运行，只是更慢。
- **找不到 `best_mnist_cnn.pth`：** 先运行 `python train.py`，确保训练结束并生成 checkpoint。
- **网页摄像头打不开：** 检查浏览器站点权限、Windows 相机隐私设置，并关闭占用摄像头的软件；也可切换设备或使用 `python app_cv.py --camera 1`。
- **预测结果不稳定：** 使用较高对比度、较粗笔迹，减少 ROI 中的背景，并尽量让单个数字占据主要区域。摄像头图像和 MNIST 数据存在域差异。
- **图片预处理失败：** 检查 ROI 是否至少为 8×8，图片中是否存在足够清晰的数字前景。
- **导入时报缺少模块：** 确认已激活 `.venv`，然后重新执行 `pip install -r requirements.txt`。
- **5000 端口被占用：** 使用 `python app.py --port 5050`，再访问终端显示的本机地址。

## 运行检查清单

- [ ] 依赖安装成功，PyTorch 能识别 CPU 或 GPU。
- [ ] `python train.py` 能生成 checkpoint 和训练曲线。
- [ ] `python test.py` 能生成测试指标、逐类准确率和混淆矩阵。
- [ ] `python preprocess.py <图片路径>` 能生成 28×28 预览。
- [ ] `python app.py --no-browser` 能启动网页服务。
- [ ] 网页端能上传图片、框选 ROI 并显示预测结果。
- [ ] `python app_cv.py --image <图片路径>` 能完成备用推理。

