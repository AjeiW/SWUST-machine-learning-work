# 写代码组项目说明

## 1. 小组目标

写代码组负责把项目做成一个可运行、可复现、可交接的 MNIST 手写数字识别系统。项目包含三部分：PyTorch CNN 训练、官方测试集评估，以及 Flask 网页/OpenCV 窗口识别。

重点是保证训练、测试和实际推理使用同一套模型接口与预处理规则。否则训练结果再漂亮，网页端也可能一本正经地认错数字。

## 2. 负责文件

| 文件 | 主要职责 | 交付要求 |
|---|---|---|
| `model.py` | 定义 `MNISTCNN` 网络结构 | 输入 `(N, 1, 28, 28)`，输出 10 类 logits |
| `utils.py` | 公共路径、随机种子、设备、MNIST transform、权重加载 | 训练、测试、网页推理使用统一配置 |
| `preprocess.py` | 把摄像头或上传图片处理成 MNIST 风格的 28×28 输入 | 输出有效 tensor 和预览图 |
| `app.py` | Flask 网页后端、模型加载、预测 API、本地服务关闭 | `/api/predict` 返回数字、置信度和预览图 |
| `app_cv.py` | OpenCV 单窗口备用演示 | 支持摄像头、图片、ROI 框选和快捷键 |
| `web/index.html` | 网页页面结构 | 控件 ID 与 `web/app.js` 保持一致 |
| `web/app.js` | 摄像头、上传、ROI、请求预测和结果展示 | 正确调用 API 并处理异常 |
| `web/style.css` | 网页布局和样式 | 不影响控件和提示信息 |
| `requirements.txt` | Python 依赖和版本范围 | 保持 PyTorch/torchvision 兼容 |
| `README.md` | 项目运行说明和问题排查 | 命令、目录和实际行为一致 |

## 3. 核心接口

### `model.py`

`MNISTCNN` 是训练、测试和推理共同使用的模型，最终输出 10 个类别的 logits，对应数字 `0` 到 `9`。

修改网络结构时必须确认：

- 输入仍为单通道 `28×28`，除非同步修改所有调用方；
- 输出维度仍为 10；
- `train.py`、`test.py`、`app.py` 和 `app_cv.py` 都能加载新权重；
- 旧 checkpoint 是否还能使用，并在交接说明中写清楚。

### `utils.py`

这是训练和推理之间的公共契约。当前归一化参数为：

- `MNIST_MEAN = 0.1307`
- `MNIST_STD = 0.3081`

`get_mnist_transform()` 用于 MNIST 数据，`preprocess.py` 使用相同参数处理摄像头输入。不要在不同应用中复制另一套归一化配置。

### `preprocess.py`

`preprocess_digit()` 的流程是：检查输入、转灰度、二值化、反相判断、连通域去噪、裁剪前景、保持比例缩放、居中到 `28×28`、归一化并转成 tensor。

无效输入应返回 `PreprocessResult(valid=False, ..., message=...)`，不要让 OpenCV 异常直接穿透到用户界面。

### `app.py`

主要接口：

| 接口 | 方法 | 用途 |
|---|---|---|
| `/` | GET | 返回网页首页 |
| `/api/health` | GET | 检查模型是否加载 |
| `/api/predict` | POST | 接收图片并返回预测 |
| `/api/shutdown` | POST | 关闭本地服务 |

预测流程必须是：接收图片 → `preprocess_digit()` → `torch.inference_mode()` 推理 → 返回数字、置信度和 28×28 预览。

### `app_cv.py`

它是 Flask 之外的备用演示，用于验证摄像头、图片、ROI 和模型推理。它必须复用 `model.py`、`preprocess.py`、`utils.py`，不要维护第二套推理逻辑。

## 4. 与测试训练组的接口

### 输入输出约定

- 模型输入：单通道、归一化后的 `28×28` 图片。
- 模型输出：10 个 logits，类别索引 `0` 到 `9`。
- 网页预测结果：`digit`、`confidence`、`preview`。
- checkpoint：`checkpoints/best_mnist_cnn.pth`。

### 必须交接的变更

修改以下内容后，必须通知测试训练组重新训练或测试：

- `model.py` 网络结构；
- `utils.py` 归一化、随机种子或数据配置；
- `preprocess.py` 图像处理流程；
- `train.py` 训练参数或数据划分；
- checkpoint 保存格式。

交接说明至少包含：修改文件、修改原因、运行命令、是否需要删除旧 checkpoint、预期影响。

## 5. 自检清单

- [ ] `python -m py_compile model.py utils.py preprocess.py app.py app_cv.py` 通过。
- [ ] `python preprocess.py <图片路径>` 能生成 28×28 预览。
- [ ] `python app.py --no-browser` 能启动服务。
- [ ] `/api/health` 能反映模型是否加载成功。
- [ ] `/api/predict` 能处理正常图片、空请求和无法解码的图片。
- [ ] 网页端能打开摄像头或上传图片、框选 ROI、显示结果。
- [ ] OpenCV 备用程序至少能完成图片识别。
- [ ] 修改公共接口后已通知测试训练组。

## 6. 交付标准

交付时提供：

1. 可运行源代码和依赖安装说明。
2. 关键接口及输入输出说明。
3. 已知限制，例如摄像头权限、图片背景和非 MNIST 图片的泛化能力。
4. 变更记录，说明测试训练组需要重新执行哪些命令。

