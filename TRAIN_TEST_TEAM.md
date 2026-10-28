# 测试训练组项目说明

## 1. 小组目标

测试训练组负责确认项目“训练得出来、测试可信、结果可复现、演示能工作”。本组主要负责执行训练和测试、设计可比较的超参数实验、检查结果、验证应用端并反馈问题。

发现代码错误时，先记录复现步骤、输入和完整错误信息，再交给写代码组处理。不要一边跑实验一边临时改十几个地方，最后得到一份谁也解释不了的结果。

## 2. 负责文件和产物

### 2.1 主要执行文件

| 文件 | 本组工作 | 关注点 |
|---|---|---|
| `train.py` | 执行训练、调整超参数、记录指标 | 数据划分、收敛、最佳验证准确率 |
| `test.py` | 使用官方测试集评估 | 测试准确率、逐类准确率、混淆矩阵 |
| `preprocess.py` | 单独验证预处理 | 不同背景、尺寸和笔迹下的输入质量 |
| `app.py` | 验证网页推理 | 模型加载、上传、摄像头、ROI、错误提示 |
| `app_cv.py` | 验证 OpenCV 演示 | 摄像头、图片、ROI、快捷键、设备切换 |

### 2.2 运行生成的目录

| 路径 | 生成者 | 用途 |
|---|---|---|
| `data/MNIST/` | `train.py` 或 `test.py` | MNIST 数据 |
| `checkpoints/best_mnist_cnn.pth` | `train.py` | 验证集最佳模型权重 |
| `outputs/training_history.json` | `train.py` | 每轮训练指标 |
| `outputs/training_curves.png` | `train.py` | 训练/验证曲线 |
| `outputs/test_metrics.json` | `test.py` | 测试指标和逐类结果 |
| `outputs/per_class_accuracy.csv` | `test.py` | 0–9 各类别准确率 |
| `outputs/confusion_matrix.png` | `test.py` | 真实标签与预测标签关系 |
| `outputs/test_predictions.png` | `test.py` | 固定随机样本预测图 |
| `outputs/preprocessed_digit.png` | `preprocess.py` | 单张图片预处理结果 |

这些是实验产物，不是源代码。重新运行会覆盖部分结果，重要实验应立即复制到独立实验目录或改名保存。

## 3. 标准工作流程

### 第一步：检查环境

```powershell
pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

记录 Python、PyTorch、torchvision、CPU/GPU、操作系统、训练命令和测试命令。

### 第二步：执行基线训练

```powershell
python train.py
```

默认配置：Epoch 8、Batch Size 64、Learning Rate 0.001、Validation Ratio 0.1、Random Seed 42、DataLoader workers 0。

训练后检查：

- 是否生成 `checkpoints/best_mnist_cnn.pth`；
- 是否生成 `outputs/training_history.json` 和 `outputs/training_curves.png`；
- loss 是否总体下降；
- 训练准确率和验证准确率是否正常上升；
- 是否明显过拟合或不收敛。

### 第三步：执行官方测试

```powershell
python test.py
```

检查 checkpoint 是否成功加载、`test_accuracy` 是否合理、0–9 各类是否都有结果、混淆矩阵是否存在明显易混淆数字，以及预测图标题是否正确。

官方测试集只能用于最终评估，不能参与训练或选择最佳 epoch。验证集和测试集不是一回事，混用会让结果看起来很努力，实际上不严谨。

## 4. 超参数实验

每次只修改一个主要因素。

### Epoch

```powershell
python train.py --epochs 4
python train.py --epochs 8
python train.py --epochs 12
```

### Batch Size

```powershell
python train.py --batch-size 32
python train.py --batch-size 64
python train.py --batch-size 128
```

### Learning Rate

```powershell
python train.py --lr 0.0005
python train.py --lr 0.001
python train.py --lr 0.002
```

每次记录：

| 字段 | 内容 |
|---|---|
| 实验编号 | 如 `E01`、`LR02` |
| 训练命令 | 完整命令 |
| Seed | 默认 42 |
| Epoch / Batch Size / Learning Rate | 实际参数 |
| 最佳验证准确率 | 来自训练输出或 checkpoint |
| 最佳 epoch | 最佳模型出现的轮次 |
| 测试准确率 | 来自 `test_metrics.json` |
| 训练耗时 | 便于比较效率 |
| 异常现象 | 不收敛、过拟合、速度异常等 |

重要实验至少备份：`training_history.json`、`training_curves.png`、`test_metrics.json`、`per_class_accuracy.csv`、`confusion_matrix.png` 和 checkpoint。

## 5. 应用端功能测试

### 网页应用 `app.py`

```powershell
python app.py
python app.py --no-browser
```

检查：

- [ ] 服务能启动并加载 checkpoint。
- [ ] 能访问 `http://127.0.0.1:5000`。
- [ ] `/api/health` 显示模型已加载。
- [ ] 能打开摄像头并切换设备。
- [ ] 能上传 PNG、JPG、BMP 或 WebP 图片。
- [ ] 能拖动框选数字 ROI。
- [ ] 过小 ROI 能显示错误提示。
- [ ] 识别后显示数字、置信度和 28×28 预览。
- [ ] 最近记录最多保留 5 条。
- [ ] 无法解码图片、模型未加载或服务关闭时提示清楚。
- [ ] 退出按钮能停止服务。

### OpenCV 应用 `app_cv.py`

```powershell
python app_cv.py
python app_cv.py --image .\digit.png
```

检查：

- [ ] 摄像头能打开，无法打开时能扫描或提示设备。
- [ ] 能拖动 ROI，按 `SPACE` 识别。
- [ ] `R` 能重置 ROI，`C` 能清空历史，`Q` 能退出。
- [ ] 图片路径包含中文时能读取，或给出明确错误。

## 6. 预处理专项测试

```powershell
python preprocess.py .\digit.png
python preprocess.py .\digit.png --output outputs\preprocessed_digit_01.png
```

至少准备：黑底白字、白底黑字、粗细不同笔画、数字偏移、过小数字、空白图片、带噪声图片和多个数字同框。

记录输入特征、是否成功生成预览、是否正确反相、是否误保留噪声、是否因 ROI 太小被拒绝，以及最终识别结果和置信度。

## 7. 问题反馈格式

```text
问题标题：
环境：Python / PyTorch / CPU 或 GPU / Windows 版本
复现命令：
输入文件或输入类型：
实际结果：
预期结果：
完整错误信息：
相关输出文件：
是否稳定复现：
```

不要只说“模型不太准”或“网页坏了”。这种反馈对定位问题的帮助，和把 bug 报告写在许愿池里差不多。

## 8. 交付标准

最终交付应包含：

1. 至少一组完整基线训练结果。
2. 至少一组可比较的超参数实验结果。
3. 官方测试集整体准确率和逐类准确率。
4. 混淆矩阵及易混淆数字分析。
5. 网页应用、OpenCV 应用和预处理专项测试记录。
6. 完整命令、环境信息和随机种子。
7. 对异常结果和失败实验的说明。

## 9. 与写代码组的边界

测试训练组可以提出修改建议，但网络结构、数据预处理、公共配置、模型加载、Flask API 和 OpenCV 交互逻辑应由写代码组修改。

修改 `model.py`、`utils.py` 或 `preprocess.py` 后，旧 checkpoint 和旧测试结果不能直接作为新代码的证据；必须重新执行受影响的训练或测试。

