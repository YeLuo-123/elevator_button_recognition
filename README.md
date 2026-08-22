# 基于 YOLO26 的电梯按钮识别

本项目使用 Ultralytics YOLO26 训练电梯面板按钮目标检测模型，可识别楼层数字、开门、关门、上行、下行、报警、钥匙孔等按钮或面板元素。项目包含完整的数据集、预训练模型、训练脚本、图片批量测试脚本、视频预测脚本以及已经生成的训练和可视化结果。

## 1. 当前项目状态

- 训练设备：NVIDIA GeForce RTX 3080 10GB
- Conda 环境：`elevator-yolo26`
- Python：3.11
- Ultralytics：8.4.123
- PyTorch：2.13.0 + CUDA 13.0
- 初始权重：`yolo26m.pt`
- 训练图片：4386 张
- 验证图片：540 张
- 类别数：345
- 训练轮次：100 epochs
- 最佳权重：`runs/detect/elevator_yolo26m/weights/best.pt`

本次训练的最佳验证结果出现在第 86 轮附近：

| 指标 | 最佳值 |
| --- | ---: |
| Precision | 约 0.49 |
| Recall | 约 0.27 |
| mAP@0.5 | 约 0.289 |
| mAP@0.5:0.95 | 约 0.218 |

> 实际推理应优先使用 `best.pt`，而不是 `last.pt`。

## 2. 项目目录结构

```text
elevator/
├── README.md                         # 项目总说明文档
├── README_TRAIN.md                   # 早期的简版训练说明
├── environment.yml                   # Conda 环境及 Python 依赖配置
├── train_yolo26.py                   # 模型训练脚本
├── train_yolo26_optimized.py         # 第二轮优化训练启动脚本
├── test_yolo26.py                    # 图片/图片目录批量预测脚本
├── predict_video.py                  # 视频逐帧预测和可视化脚本
├── yolo26m.pt                        # YOLO26m COCO 预训练权重，默认训练起点
├── yolo26n.pt                        # 更轻量的 YOLO26n 预训练权重
├── fa476d4327584b23108fbfccce7f07e3.mp4  # 示例测试视频
├── dianti(1)/
│   └── dianti/                       # 24 张实际电梯场景测试图片
├── Elevator Buttons - Original Version.v10-augmented-dataset-size-3x.yolo26/
│   ├── data.yaml                     # YOLO 数据集配置及全部类别名称
│   ├── README.dataset.txt            # Roboflow 数据集来源说明
│   ├── README.roboflow.txt           # 预处理和数据增强说明
│   ├── train/
│   │   ├── images/                   # 训练图片
│   │   └── labels/                   # YOLO 格式训练标签
│   └── valid/
│       ├── images/                   # 验证图片
│       └── labels/                   # YOLO 格式验证标签
└── runs/
    ├── detect/elevator_yolo26m/      # 训练日志、曲线、验证结果和权重
    └── predict/                      # 图片与视频预测结果
```

### 2.1 根目录主要文件

#### `environment.yml`

记录可复现的 Conda 环境。Python 包安装源已经配置为清华 PyPI 镜像，包含：

- Python 3.11
- Ultralytics 8.4.123
- TensorBoard 2.21.0
- Typeguard

当前已经创建好的环境中，`pip.conf` 同样固定使用清华源。

#### `train_yolo26.py`

训练入口。默认加载 `yolo26m.pt`，使用项目数据集训练 100 轮，并将结果保存到 `runs/detect/`。脚本会在使用 GPU 时检查 CUDA 是否可用。

主要默认参数：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--model` | `yolo26m.pt` | 初始预训练权重 |
| `--data` | 数据集中的 `data.yaml` | 数据集配置 |
| `--epochs` | `100` | 最大训练轮次 |
| `--imgsz` | `640` | 训练输入尺寸 |
| `--batch` | `-1` | 自动使用约 60% GPU 显存 |
| `--device` | `0` | 使用第 0 块 CUDA GPU |
| `--workers` | `8` | 数据加载进程数 |
| `--patience` | `30` | Early Stopping 等待轮次 |
| `--name` | `elevator_yolo26m` | 实验名称和结果目录名称 |
| `--seed` | `42` | 随机种子 |

脚本还默认启用 AMP 混合精度、余弦学习率、验证、训练曲线和权重保存。

#### `train_yolo26_optimized.py`

第二轮优化训练入口，不会覆盖第一轮实验。默认重新从 `yolo26m.pt` 训练 200 轮，使用固定 batch size 4、AdamW、较低初始学习率，并针对当前已经离线增强 3 倍的数据集减弱在线几何增强。脚本关闭水平翻转，避免改变开门/关门图标的方向语义；启动训练前还会检查数据路径、标签范围和长尾类别分布。

#### `test_yolo26.py`

用于测试单张图片或整个图片目录。默认读取：

```text
dianti(1)/dianti
```

默认使用训练后的 `best.pt`，输出：

- 带检测框、类别和置信度的逐张图片
- `contact_sheet.jpg`：全部结果的汇总拼图
- `detections.csv`：每个检测框的类别、置信度和坐标

#### `predict_video.py`

用于视频预测。脚本逐帧调用模型，将检测框、类别名称和置信度绘制到画面上，再以原视频的分辨率和帧率编码为 MP4。

> 当前脚本通过 OpenCV 写入新视频，只保留画面，不会复制原视频音轨。

#### `yolo26m.pt` 与 `yolo26n.pt`

- `yolo26m.pt`：中型模型，精度潜力较高，是当前训练脚本的默认模型。
- `yolo26n.pt`：轻量模型，训练和推理速度更快、显存占用更低，但精度通常低于 m 模型。

这两个文件是训练起点，不是本项目训练完成后的最终权重。最终权重位于 `runs/detect/elevator_yolo26m/weights/`。

### 2.2 数据集文件

标签采用标准 YOLO 检测格式，每一行代表一个目标：

```text
class_id x_center y_center width height
```

坐标和宽高均为相对于图片尺寸归一化后的 `0–1` 浮点数。例如：

```text
9 0.512 0.431 0.120 0.098
```

图片和标签文件必须同名：

```text
images/example.jpg
labels/example.txt
```

`data.yaml` 定义训练集、验证集、345 个类别及类别名称。该数据集没有独立的带标签 test 集，因此训练过程中使用 valid 集评估。

### 2.3 训练结果目录

`runs/detect/elevator_yolo26m/` 中的主要文件：

| 文件 | 作用 |
| --- | --- |
| `weights/best.pt` | 验证指标最佳的模型，推荐用于推理 |
| `weights/last.pt` | 最后一轮模型，主要用于中断恢复 |
| `args.yaml` | 本次训练使用的完整参数 |
| `results.csv` | 每个 epoch 的损失、Precision、Recall 和 mAP |
| `results.png` | 训练损失及验证指标曲线 |
| `confusion_matrix.png` | 混淆矩阵 |
| `confusion_matrix_normalized.png` | 归一化混淆矩阵 |
| `BoxPR_curve.png` | Precision-Recall 曲线 |
| `BoxF1_curve.png` | F1 与置信度阈值曲线 |
| `BoxP_curve.png` | Precision 与置信度阈值曲线 |
| `BoxR_curve.png` | Recall 与置信度阈值曲线 |
| `train_batch*.jpg` | 训练批次及增强效果预览 |
| `val_batch*_labels.jpg` | 验证集真实标签预览 |
| `val_batch*_pred.jpg` | 验证集模型预测预览 |

## 3. 环境配置

### 3.1 使用已经配置好的环境

```bash
cd /home/fq/elevator
conda activate elevator-yolo26
```

检查 CUDA：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

预期输出包含：

```text
True
NVIDIA GeForce RTX 3080
```

### 3.2 从 `environment.yml` 重新创建环境

如果需要在另一台机器上复现：

```bash
cd /home/fq/elevator
conda env create -f environment.yml
conda activate elevator-yolo26
```

检查依赖：

```bash
python -m pip check
python -c "import torch, ultralytics; print(torch.__version__); print(ultralytics.__version__)"
```

`environment.yml` 中的普通 Conda 包和 pip 包均使用清华镜像。不同显卡或驱动环境下，可能需要根据本机 CUDA 兼容情况单独安装合适版本的 PyTorch。

## 4. 启动训练

### 4.1 默认训练

```bash
cd /home/fq/elevator
conda activate elevator-yolo26
python train_yolo26.py
```

默认输出目录：

```text
runs/detect/elevator_yolo26m/
```

如果该目录已经存在，Ultralytics 会避免直接覆盖已有实验。再次训练时建议设置新的实验名。

### 4.2 启动第二轮优化训练

推荐先从官方预训练权重重新训练，以便和第一轮结果公平比较：

```bash
python train_yolo26_optimized.py
```

默认输出目录：

```text
runs/detect/elevator_yolo26m_optimized/
```

如果希望从第一轮的 `best.pt` 开始微调，并重新开启完整学习率周期：

```bash
python train_yolo26_optimized.py \
  --from-best \
  --name elevator_yolo26m_finetune
```

优化版默认调整包括：

- `epochs=200`、`patience=40`、`batch=4`
- `optimizer=AdamW`、`lr0=0.001`、余弦学习率
- `fliplr=0.0`，禁止方向敏感图标水平翻转
- `mosaic=0.5`，并在最后 20 轮关闭 Mosaic
- `translate=0.05`、`scale=0.3`
- 降低 HSV 色彩增强，关闭 MixUp、CutMix 和 Copy-Paste
- 每 10 轮保存一次阶段权重

自定义轮次和实验名称：

```bash
python train_yolo26_optimized.py \
  --epochs 250 \
  --batch 4 \
  --patience 50 \
  --name elevator_yolo26m_optimized_v2
```

### 4.3 自定义原始训练参数

例如训练 200 轮、固定 batch size 为 4，并使用新实验名：

```bash
python train_yolo26.py \
  --epochs 200 \
  --batch 4 \
  --imgsz 640 \
  --patience 40 \
  --name elevator_yolo26m_v2
```

使用轻量的 YOLO26n：

```bash
python train_yolo26.py \
  --model yolo26n.pt \
  --name elevator_yolo26n
```

使用 CPU 训练（速度会非常慢）：

```bash
python train_yolo26.py --device cpu --workers 4 --batch 2
```

查看全部参数：

```bash
python train_yolo26.py --help
```

### 4.4 恢复被中断的训练

恢复默认实验：

```bash
python train_yolo26.py --resume
```

恢复指定实验：

```bash
python train_yolo26.py --name elevator_yolo26m_v2 --resume
```

恢复功能读取对应实验目录中的 `weights/last.pt`。它适合继续一次意外中断的训练；如果原训练已经正常完成、只是希望增加总轮次，建议从权重启动一个新的实验并重新设置学习率周期。

优化版实验的恢复方法相同：

```bash
python train_yolo26_optimized.py --resume
```

### 4.5 查看训练过程

启动 TensorBoard：

```bash
tensorboard --logdir runs --port 6006
```

然后在浏览器中打开：

```text
http://localhost:6006
```

也可以直接查看：

```text
runs/detect/elevator_yolo26m/results.png
runs/detect/elevator_yolo26m/results.csv
```

## 5. 图片测试与可视化

### 5.1 测试默认图片目录

```bash
cd /home/fq/elevator
conda activate elevator-yolo26
python test_yolo26.py
```

默认结果目录：

```text
runs/predict/dianti_results/
```

其中：

```text
contact_sheet.jpg   # 24 张预测结果的汇总大图
detections.csv      # 检测明细
*.jpg               # 每张图片对应的可视化结果
```

### 5.2 测试其他图片或目录

测试单张图片：

```bash
python test_yolo26.py \
  --source /path/to/image.jpg \
  --output runs/predict/single_image
```

测试另一个图片目录：

```bash
python test_yolo26.py \
  --source /path/to/images \
  --output runs/predict/new_images
```

调整置信度阈值和 NMS IoU：

```bash
python test_yolo26.py --conf 0.40 --iou 0.60
```

阈值说明：

- 降低 `--conf`：可能找回更多目标，但误检通常也会增加。
- 提高 `--conf`：保留更可信的目标，但可能增加漏检。
- `--iou`：控制重叠候选框在 NMS 阶段的合并强度。

指定其他模型：

```bash
python test_yolo26.py \
  --model runs/detect/elevator_yolo26m_v2/weights/best.pt
```

## 6. 视频预测与可视化

### 6.1 预测默认视频

```bash
cd /home/fq/elevator
conda activate elevator-yolo26
python predict_video.py
```

脚本默认输入：

```text
fa476d4327584b23108fbfccce7f07e3.mp4
```

默认输出：

```text
runs/predict/video_results/fa476d4327584b23108fbfccce7f07e3_predicted.mp4
```

### 6.2 预测其他视频

```bash
python predict_video.py \
  --source /path/to/input.mp4 \
  --output runs/predict/video_results/output.mp4
```

提高置信度阈值：

```bash
python predict_video.py --conf 0.40
```

使用其他权重：

```bash
python predict_video.py \
  --model runs/detect/elevator_yolo26m_v2/weights/best.pt \
  --source /path/to/input.mp4 \
  --output runs/predict/video_results/v2_output.mp4
```

脚本结束后会打印：

- 处理帧数
- 视频帧率和分辨率
- 各类别在全部帧中的检测次数

这里的类别次数是“逐帧检测框总数”，不是视频中不同物体的数量。同一按钮连续出现 100 帧会被统计约 100 次。

## 7. 模型效果与后续优化

当前模型能在实拍图片和视频中识别常见楼层数字、开关门按钮、上下行按钮等目标，但整个数据集存在明显的长尾问题：

- 345 个类别对于当前数据量来说过多。
- 训练集中有 29 类没有实例。
- 129 类少于 5 个训练实例。
- 验证集中只有 182 类实际出现。
- 部分类别存在相近名称、细粒度文字类别或疑似拼写重复。

因此，单纯把训练轮次从 100 增加到 200 或 300，提升预计有限。推荐优化顺序：

1. 合并重复、拼写错误和业务上不需要区分的类别。
2. 为稀有类别补充真实图片，每类尽量达到至少 50–100 个实例。
3. 建立独立、带人工标签的真实场景测试集。
4. 按原始图片划分 train/valid/test 后再做增强，避免增强版本跨集合。
5. 对方向敏感的开门/关门图标关闭水平翻转增强。
6. 在数据改进后重新训练 150–200 轮，并用新的实验名保存结果。

## 8. 常见问题

### CUDA 不可用

检查：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

如果返回 `False`，确认已经激活 `elevator-yolo26` 环境，并检查显卡驱动与 PyTorch CUDA 版本是否兼容。

### 显存不足（CUDA out of memory）

减小 batch size：

```bash
python train_yolo26.py --batch 4
```

仍然不足时继续减小：

```bash
python train_yolo26.py --batch 2
```

也可以使用更小的模型：

```bash
python train_yolo26.py --model yolo26n.pt --name elevator_yolo26n
```

### 找不到模型文件

图片和视频脚本默认寻找：

```text
runs/detect/elevator_yolo26m/weights/best.pt
```

如果使用了其他实验名，需要通过 `--model` 指定实际路径。

### 找不到测试图片

默认图片目录的实际名称是：

```text
dianti(1)/dianti
```

也可以通过 `--source` 指定其他目录。

### 预测结果误检较多

先尝试提高置信度阈值：

```bash
python test_yolo26.py --conf 0.5
python predict_video.py --conf 0.5
```

如果仍然存在稳定误检，应补充这些真实场景作为负样本或重新检查相应类别标注，而不应只依赖提高阈值。

## 9. 快速命令汇总

```bash
# 进入项目并激活环境
cd /home/fq/elevator
conda activate elevator-yolo26

# 默认训练
python train_yolo26.py

# 新实验训练 200 轮
python train_yolo26_optimized.py

# 恢复中断的训练
python train_yolo26.py --name elevator_yolo26m_v2 --resume

# 批量测试图片
python test_yolo26.py

# 测试视频
python predict_video.py

# 查看训练曲线
tensorboard --logdir runs --port 6006
```

## 10. 数据集来源与许可

数据集来自 Roboflow Universe 的 **Elevator Buttons - Original Version**，版本 10，数据集说明标注为 MIT License。详细来源、导出时间、预处理和增强方式请查看：

```text
Elevator Buttons - Original Version.v10-augmented-dataset-size-3x.yolo26/README.dataset.txt
Elevator Buttons - Original Version.v10-augmented-dataset-size-3x.yolo26/README.roboflow.txt
```

Ultralytics 软件和预训练模型的使用还应遵循其对应的软件许可及商业使用条款。
