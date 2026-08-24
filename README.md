# 电梯按钮识别与机械臂联动

本项目使用 Ultralytics YOLO 检测电梯按钮，把检测框中心的像素坐标转换为机械臂世界坐标，并通过可替换的回调接口触发按键动作。默认机械臂实现是 `MockArmController`，只打印指令，不会驱动真实设备。

## 已完成的模型训练工作

本项目已经完成模型训练，并非只提供训练代码。当前可直接部署的权重是 `models/best.pt`，它来自 `elevator_yolo26m_optimized` 优化实验，已通过 Git LFS 上传到仓库。

- 基础网络：YOLO26m 预训练模型。
- 数据集：Roboflow Elevator Buttons Original Version v10，配置在 `configs/dataset.yaml`。
- 类别数量：345 类。
- 训练输入尺寸：640×640。
- 训练设备：NVIDIA RTX 3080，batch size 为 4。
- 优化方式：AdamW、余弦学习率、AMP 混合精度；针对电梯开关门等方向敏感图标关闭水平和垂直翻转，并降低在线几何增强强度。
- 训练过程：最大计划 200 epoch，实际完成 147 epoch，并按照验证集表现保存最佳权重。
- 最佳验证结果：第 107 epoch 的 Precision 为 0.5908、Recall 为 0.4374、mAP@0.5 为 0.4944、mAP@0.5:0.95 为 0.3779。

以上指标来自本机保留的 `runs/detect/elevator_yolo26m_optimized/results.csv`。`runs/` 属于可再生成的训练产物，因此不提交 Git；部署使用的最佳权重单独保存在 `models/best.pt`。

## 模型功能

`models/best.pt` 是电梯面板目标检测模型，主要提供以下能力：

- 在图片、视频或摄像头画面中定位一个或多个电梯按钮，并输出检测框坐标。
- 为每个检测目标输出类别名称、类别编号和置信度。
- 识别数据集定义的楼层按钮，例如数字、字母及地下楼层组合标签。
- 识别 `open`、`close`、`up`、`down`、`alarm`、`call`、`stop`、`switch` 等常见功能按钮或图标。
- 检测部分面板文字、指示灯、钥匙孔、扬声器等数据集中已标注的元素。
- 计算检测框中心像素 `(u,v)`，供手眼标定模块转换为机械臂世界坐标。
- 通过目标类别和最低置信度过滤识别结果，再调用机械臂回调执行按键动作。

模型属于目标检测模型，不是通用 OCR，也不会可靠识别训练类别之外的任意文字或按钮。实际效果会受到拍摄角度、反光、遮挡、按钮尺寸和现场面板样式影响；真机部署前应使用目标电梯的现场图片重新验证，必要时补充数据进行微调。

## 目录结构

```text
.
├── configs/                 # 数据集、推理、训练超参数和手眼标定
├── models/                  # .pt/.onnx 权重（Git 忽略；共享时使用 Git LFS）
├── inference/               # YOLO 检测、像素坐标转换、图片/视频/相机入口
├── arm_control/             # 机械臂协议、Mock 实现和检测回调
├── scripts/                 # 训练与一键启动脚本
├── docs/                    # 训练交接文档、测试图片/视频
├── tests/                   # 坐标、回调、摄像头和串口测试
├── requirements.txt         # pip 环境
└── environment.yml          # Conda 环境
```

训练数据目前保留在原 Roboflow 导出目录，入口为 `configs/dataset.yaml`；训练和推理输出统一写入 `runs/`，不提交 Git。

## 环境一键配置

推荐 Python 3.11、NVIDIA 驱动 550 或更高版本。当前环境按 PyTorch 2.7.x + CUDA 12.8、cuDNN 9.x 编写。PyTorch wheel 自带所需 CUDA 运行库，通常不必另外安装完整 Toolkit，但宿主 NVIDIA 驱动必须兼容。CPU 也可运行，将参数设为 `--device cpu`。

Conda 一键创建：

```bash
conda env create -f environment.yml
conda activate elevator-yolo26
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

也可以使用 venv：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

若需严格指定 GPU 版 PyTorch，请先按目标机器的 CUDA/驱动从 PyTorch 官方索引安装匹配 wheel，再安装 `requirements.txt`。

## 模型文件与 Git LFS

所有部署权重放在 `models/`：

```text
models/best.pt       # 已完成训练的推荐部署权重
models/model.onnx    # 可选 ONNX 导出
```

`models/` 只存放本项目训练或导出的模型，不存放 YOLO 官方基础模型。训练脚本默认使用 `yolo26m.pt` 模型名，由 Ultralytics 自动下载到本机缓存。其他模型文件默认忽略；`models/best.pt` 作为部署模型通过 Git LFS 共享。GitHub 普通文件硬限制为 100 MB：

```bash
git lfs install
git lfs track "models/*.pt" "models/*.onnx" "models/*.engine"
git add .gitattributes
git add -f models/best.pt
git commit -m "Add deployment model via Git LFS"
git push
```

新机器克隆后运行 `git lfs pull`。仓库历史中的媒体和权重也使用 LFS，请确保 GitHub LFS 配额足够。

## 模型推理

```bash
git lfs pull
python -m inference.detect_images --model models/best.pt --source docs/test_media/images --device 0
python -m inference.detect_video --model models/best.pt --source docs/test_media/sample_video.mp4 --device 0
```

摄像头与机械臂回调（Mock 模式）：

```bash
./scripts/run_detection.sh --model models/best.pt --camera 0 --target 3
./scripts/run_detection.sh --model models/best.pt --camera 0 --target 3 --trigger-arm
```

按 `q` 退出。第二条会启用回调，但当前 `MockArmController` 仍只打印动作。接入真机时，在 `arm_control/` 新增实现 `press_at(x, y, z, label)` 的 SDK/串口/网络适配器，并在 `inference/inference_with_arm.py` 中替换 Mock；必须同时增加急停、工作空间限位、去抖和二次确认。

## 像素坐标转机械臂世界坐标

参数文件位于 `configs/hand_eye_calibration.yaml`，实现位于 `inference/coordinate_transform.py`。当前适用于按钮面板近似为平面的单应变换。

```text
u = (x1 + x2) / 2,  v = (y1 + y2) / 2
[X' Y' W']ᵀ = H · [u v 1]ᵀ
X_world = X' / W',  Y_world = Y' / W',  Z_world = fixed_z
```

其中 `H` 是 3×3 单应矩阵，`fixed_z` 是按钮平面在机械臂基坐标系中的高度，默认单位为 mm。仓库内单位矩阵仅是占位示例，严禁直接用于真机。

现场标定步骤：固定相机和面板；采集至少 4 个不共线特征点的像素坐标 `(u,v)`；让机械臂记录对应基坐标 `(X,Y)`；用 `cv2.findHomography(pixel_points, world_points)` 求 `H`；写入 YAML 后用未参与拟合的点验证误差。若面板不共面、相机会移动或需要真实深度，应改用相机内参反投影与手眼外参：`P_base = T_base_camera · depth · K⁻¹[u,v,1]ᵀ`。

## 训练

```bash
python scripts/train_detector.py --data configs/dataset.yaml --model yolo26m.pt
python scripts/train_detector_optimized.py --data configs/dataset.yaml --model yolo26m.pt
```

训练默认值位于 `configs/model_hyperparameters.yaml`，实验说明见 `docs/training.md`。

## 测试与硬件排查

```bash
pytest -q
RUN_HARDWARE_TESTS=1 pytest -q tests/test_camera_connectivity.py
SERIAL_PORT=/dev/ttyUSB0 pytest -q tests/test_serial_connectivity.py
```

摄像头失败时检查 `ls -l /dev/video*` 和 `video` 组权限。串口失败时检查设备名、波特率、线缆及 `dialout` 组权限。真机联调必须先用 Mock 验证坐标误差和目标过滤。

## 配置索引

- `configs/inference.yaml`：模型、阈值、设备和相机默认值。
- `configs/model_hyperparameters.yaml`：训练超参数。
- `configs/dataset.yaml`：数据路径与 345 类标签映射。
- `configs/hand_eye_calibration.yaml`：手眼标定矩阵、单位、平面高度和工具姿态。

## GitHub 发布

```bash
git status
git lfs status
git add -A
git commit -m "Refactor project for modular deployment"
git push origin main
```

推送前务必确认训练缓存、临时输出和非 LFS 大文件未进入提交。
