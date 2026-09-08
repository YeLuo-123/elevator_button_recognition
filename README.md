# 电梯按钮识别、亮暗判断与视频防跳变

本项目使用 Ultralytics YOLO 检测电梯按钮，把检测框中心的像素坐标转换为机械臂世界坐标，并通过可替换的回调接口触发按键动作。默认机械臂实现是 `MockArmController`，只打印指令，不会驱动真实设备。

另外提供 Qwen3-VL-2B 按钮亮暗微调、离线视频测试及带标注框的视频导出。当前流程为：

```text
图片 / 视频 → YOLO 检测按钮位置与名称 → 按钮裁剪图
          → Qwen 基模 + LoRA 适配器 → ON / OFF / UNCERTAIN
          → 按钮跟踪与时序防跳变 → 标注视频、前后对比视频、JSONL 结果
```

亮暗识别和时序处理目前是独立的离线流程，尚未接入机械臂实时控制入口。

## 快速开始

```bash
git clone https://github.com/YeLuo-123/elevator_button_recognition.git
cd elevator_button_recognition
git lfs install
git lfs pull
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m inference.detect_images --model models/best.pt --source docs/test_media/images --device cpu
```

上述命令运行 YOLO 图片检测。GPU 推理使用 `--device 0`，并确保环境安装了兼容驱动的 CUDA 版 PyTorch。Qwen 训练与视频测试还需安装 `requirements-qwen.txt`，首次联网下载基模；本仓库已提供成品微调适配器，见下文。

## 同事直接运行成品亮暗模型

### 从 GitHub 克隆（首次运行联网）

```bash
git clone https://github.com/YeLuo-123/elevator_button_recognition.git
cd elevator_button_recognition
git lfs install
git lfs pull --include="models/best.pt,models/button_state_lora/*"
python3.11 -m venv .venv
source .venv/bin/activate
# 先安装与本机 NVIDIA 驱动兼容的 CUDA 版 PyTorch，再安装项目依赖
python -m pip install -r requirements-qwen.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python scripts/run_button_state.py --source /path/to/video.mp4
```

本项目成品使用 NVIDIA GPU；本机在 RTX 3080 10 GB 上验证。若显存不足可加 `--batch 2`。运行入口自动使用仓库内成品 LoRA，首次从 `Qwen/Qwen3-VL-2B-Instruct` 下载约 4.26 GB 基模，需要能访问 Hugging Face；也可以通过 `--base /path/to/Qwen3-VL-2B-Instruct` 指定本地基模。无需训练数据。

结果默认写入 `runs/predict/button_state_<时间>/`：`temporal/temporal.mp4` 是带框结果，`temporal/comparison.mp4` 是逐帧与防跳变对比。可用 `--output /path/to/new-output` 指定尚不存在的目录。视频较长时内存占用和推理时间会增加，当前仍为离线处理。

### 完整压缩包（无需下载模型）

GitHub 拒绝上传超过该仓库单文件限制的基模，因此另行提供 `elevator-button-state-full-20260908.tar.gz`，由项目持有人直接转交。压缩包包含运行代码、YOLO 成品、LoRA 成品、Qwen 基模、基模许可说明、依赖清单与逐文件 SHA-256；不包含训练数据、虚拟环境或训练中间检查点。

```bash
tar -xzf elevator-button-state-full-20260908.tar.gz
cd elevator-button-state-full-20260908
sha256sum -c SHA256SUMS
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-qwen.txt
python scripts/run_button_state.py --source /path/to/video.mp4
```

仍需自行安装 NVIDIA 驱动、兼容的 CUDA 版 PyTorch 和 Python 依赖。脚本优先使用压缩包内 `models/Qwen3-VL-2B-Instruct/`，无需再下载模型。需要隔离网络验证模型加载时可设置 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`。本包未附带离线依赖安装包。

GitHub 的 “Download ZIP” 不应视为完整权重包；如果其中是 LFS 指针，请改用 `git clone` 加 `git lfs pull`，或使用上述完整版压缩包。

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

### 检测效果示例

下图为 `models/best.pt` 对电梯面板测试图片的实际检测结果。检测框上显示模型识别出的按钮类别和置信度：

![电梯按钮模型检测结果](docs/test_media/results/detection_example.jpg)

## 目录结构

```text
.
├── configs/                 # 数据集、推理、训练超参数和手眼标定
├── models/                  # best.pt 使用 Git LFS；其他权重默认忽略
├── inference/               # YOLO 检测、像素坐标转换、图片/视频/相机入口
├── arm_control/             # 机械臂协议、Mock 实现和检测回调
├── scripts/                 # 训练与一键启动脚本
├── tools/                   # 浏览器按钮亮暗标注工具
├── datasets/                # 本机亮暗标注与导出裁剪图，不提交 Git
├── runs/                    # 本机训练权重、评估与视频结果，不提交 Git
├── docs/                    # 训练交接文档、测试图片/视频
├── tests/                   # 坐标、回调、摄像头和串口测试
├── requirements.txt         # pip 环境
├── requirements-qwen.txt    # Qwen 训练和视频测试附加依赖
└── environment.yml          # Conda 环境
```

训练数据目前保留在原 Roboflow 导出目录，入口为 `configs/dataset.yaml`；训练和推理输出统一写入 `runs/`，不提交 Git。

## 环境一键配置

使用 Python 3.11。本次 Qwen 实验在 RTX 3080 10 GB、驱动 580.173.02、PyTorch 2.13.0+cu130 上完成；这是本机实测环境，并非最低版本要求。应根据目标机器驱动选择匹配的 PyTorch。YOLO 支持 `--device cpu`；本项目 Qwen 脚本使用 CUDA 设备 0 和 bitsandbytes 4-bit 量化，需要可用的 NVIDIA GPU。

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

### 使用 docs 文件夹测试模型

`docs/test_media/` 已提供可直接用于验收的测试素材：

- `docs/test_media/images/`：电梯面板测试图片。
- `docs/test_media/sample_video.mp4`：测试视频。
- `docs/test_media/results/detection_example.jpg`：已生成的检测效果示例。

拉取模型和 LFS 测试素材后，可直接运行：

```bash
git lfs pull
python -m inference.detect_images \
  --model models/best.pt \
  --source docs/test_media/images \
  --output runs/predict/docs_image_results \
  --device 0

python -m inference.detect_video \
  --model models/best.pt \
  --source docs/test_media/sample_video.mp4 \
  --output runs/predict/docs_video_result.mp4 \
  --device 0
```

图片检测会输出标注图片、`detections.csv` 和汇总图；视频检测会输出带检测框的视频。生成结果保存在 `runs/predict/`，该目录已被 Git 忽略，不会污染仓库。

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

## 按钮亮暗状态标注

使用已有 YOLO 框补充 `ON`、`OFF`、`UNCERTAIN` 标签：

```bash
python tools/annotate_button_state.py
```

浏览器会打开 `http://127.0.0.1:8765`。快捷键为 `1=ON`、`2=OFF`、
`3=UNCERTAIN`、`0=跳过`、`Z=撤销`。结果自动保存到
`datasets/button_state/annotations.json`，不会修改原 YOLO 标注，可随时继续。

默认排除 `led`、`indicator`、`text_*`、钥匙孔等非按钮组件；如需查看所有已有框：

```bash
python tools/annotate_button_state.py --all-boxes
```

## 测试与硬件排查

```bash
python -m pytest -q tests
RUN_HARDWARE_TESTS=1 pytest -q tests/test_camera_connectivity.py
SERIAL_PORT=/dev/ttyUSB0 pytest -q tests/test_serial_connectivity.py
```

摄像头失败时检查 `ls -l /dev/video*` 和 `video` 组权限。串口失败时检查设备名、波特率、线缆及 `dialout` 组权限。真机联调必须先用 Mock 验证坐标误差和目标过滤。

若本机 ROS 环境自动注入 pytest 插件并导致导入失败，可使用 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests` 运行项目测试。硬件测试默认跳过。

## 配置索引

- `configs/inference.yaml`：模型、阈值、设备和相机默认值。
- `configs/model_hyperparameters.yaml`：训练超参数。
- `configs/dataset.yaml`：数据路径与 345 类标签映射。
- `configs/hand_eye_calibration.yaml`：手眼标定矩阵、单位、平面高度和工具姿态。

## Qwen 按钮亮暗训练

状态定义：`ON` 表示点亮，`OFF` 表示熄灭，`UNCERTAIN` 表示图片不足以确定。模型接收单个按钮的裁剪图，输出如 `{"state":"ON"}` 的文本。无法解析的输出在评估中计为 `INVALID`。

基模标识为 `Qwen/Qwen3-VL-2B-Instruct`。基模权重提供通用图文理解能力；LoRA 适配器保存本项目微调得到的参数，推理时两者必须一起加载。QLoRA 使用量化基模训练少量 LoRA 参数，减少显存占用；本项目冻结视觉编码器，仅训练语言侧适配器。

### 数据准备

先通过前述标注工具得到 `datasets/button_state/annotations.json`，再导出：

```bash
python -m pip install -r requirements-qwen.txt
python scripts/export_button_state_vlm.py \
  --annotations datasets/button_state/annotations.json \
  --data configs/dataset.yaml \
  --output datasets/button_state/vlm_v1
```

导出会扩展检测框 25% 后裁剪，并按原始图片来源分组划分 train / val / test，避免同一来源的增强图片跨集合。输出包含裁剪图、分组清单、标签快照和 JSON 数据。输出目录应为新目录；`--replace` 会替换已有导出，使用前确认路径。

2026-09-07 实验使用 **8,157 条人工亮暗标注**：

| 集合 | 条数 | OFF | ON | UNCERTAIN |
| --- | ---: | ---: | ---: | ---: |
| 训练 | 5,800 | 5,334 | 442 | 24 |
| 验证 | 1,229 | 1,131 | 93 | 5 |
| 测试 | 1,128 | 1,030 | 92 | 6 |

训练对少数类别过采样，每个 epoch 实际使用 8,121 个样本；验证和测试保持原分布。基模自动复核改标方案已取消，当前流程继续使用原人工标签。

### 正式训练与后台运行

以下命令从模型仓库加载基模；已有完整本地模型时，将 `--model` 改为本地目录。输出使用新实验目录，避免覆盖完成的实验。

```bash
mkdir -p runs/button_state
nohup python -u scripts/train_qwen_vl_button_state.py \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --data datasets/button_state/vlm_v1 \
  --output runs/button_state/qwen3_vl_2b_lora_v1 \
  --epochs 3 --batch-size 4 --gradient-accumulation 4 \
  --dataloader-workers 4 \
  > runs/button_state/qwen3_vl_2b_lora_v1.log 2>&1 &
echo $! > runs/button_state/qwen3_vl_2b_lora_v1.pid
```

```bash
tail -f runs/button_state/qwen3_vl_2b_lora_v1.log
nvidia-smi
```

`epoch` 是遍历一次训练采样集；`batch-size` 是单次送入 GPU 的样本数；梯度累积 4 次、batch 为 4 时，单卡有效 batch 为 16。进度条中的剩余时间会随速度变化，后续验证和保存也需要时间。显存不足时可改为 batch 2、累积 8，保持有效 batch 不变。

本次参数为 3 epoch、学习率 `2e-4`、LoRA rank 16 / alpha 32、NF4 4-bit 量化、FP16 计算，启用梯度检查点。共 1,524 次优化更新，训练约 51 分钟；这是本机此次实验耗时，不保证其他环境相同。正式训练结束会加载验证 loss 最低的 checkpoint，再保存到 `final_adapter/`；此次最佳为 epoch 2 的 checkpoint-1016，验证 loss 为 0.03638。

可先在独立输出目录使用 `--max-steps 2` 做 smoke test（短流程检查，验证加载、前向、反向和保存能否执行，不代表模型效果）。断点续训使用 `--resume-from-checkpoint <checkpoint目录>`。

### 评估与实测结果

```bash
python scripts/evaluate_qwen_vl_button_state.py \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --adapter runs/button_state/qwen3_vl_2b_lora_v1/final_adapter \
  --data datasets/button_state/vlm_v1 --split test \
  --output runs/button_state/evaluation_test.json
```

将 `--split test` 改为 `--split val` 并更换输出文件即可评估验证集。请保留 `--adapter`，省略时测试的是基模。

| 指标 | 验证集 | 测试集 |
| --- | ---: | ---: |
| 总体准确率 | 95.85% | 93.71% |
| ON 精确率 | 77.91% | 64.52% |
| ON 召回率 | 72.04% | 65.22% |
| ON F1 | 74.86% | 64.86% |
| 三类宏平均 F1 | 57.54% | 53.81% |

以上来自本机 `qwen3_vl_2b_lora_20260907` 实验。测试集 OFF 占比较高，总体准确率不能代表亮灯识别质量；92 个 ON 中识别正确 60 个、漏判 32 个。UNCERTAIN 类别尚未有效识别，需要补充与复核样本。精确率衡量预测 ON 中有多少真的亮；召回率衡量实际亮灯中找回了多少；F1 综合两者。

## 视频测试与防跳变可视化

先逐帧检测并运行训练后的 Qwen 模型。将 `test.mp4` 替换为自己的输入路径：

```bash
python scripts/test_button_state_video.py \
  --source test.mp4 \
  --base Qwen/Qwen3-VL-2B-Instruct \
  --adapter runs/button_state/qwen3_vl_2b_lora_v1/final_adapter \
  --batch 4 --output runs/predict/test_qwen_state

python scripts/test_temporal_video.py \
  --source test.mp4 \
  --predictions runs/predict/test_qwen_state/predictions.jsonl \
  --output runs/predict/test_qwen_temporal \
  --off-delay 0.35 --sustained-off-delay 1.2
```

两个输出目录均需尚不存在。第一步输出原始逐帧 `predictions.jsonl` 和 `annotated.mp4`（无音轨）；第二步只重放预测，不重复模型推理，输出：

- `temporal.mp4`：防跳变后的标注框视频。
- `comparison.mp4`：左侧逐帧判断，右侧时序结果；可用 `--baseline-predictions` 对比旧版时序结果。
- `predictions.jsonl`：包含原始状态、时序状态和跟踪信息。
- `summary.json` 与对比截图：辅助检查状态变化。

时序视频使用 H.264 编码，输入含兼容音轨时保留音频。绿色框表示 ON，蓝色表示 OFF，黄色表示 UNCERTAIN；`HOLD` 表示沿用近期状态。

时序模块通过 IoU 匹配相邻帧的按钮框：首次 ON 立即生效；0.4 秒内再次出现 ON 后，进入持续亮灯状态，需要连续 1.2 秒 OFF 才确认熄灭；孤立 ON 使用 0.35 秒的短保持。跟踪丢失超过 0.2 秒会失效，不凭空补框。该策略只使用当前和历史帧。

本机 5.37 秒测试视频中，“2”号按钮首次 ON 仍在约 1.61 秒；从固定延迟版到持续亮灯版，主要轨迹的相邻帧状态切换由 4 次降至 0 次。该结果仅说明这一片段更稳定，未提供逐帧真值准确率；真实熄灭可能延迟最多约 1.2 秒确认，跟踪重建也可能产生跳变。应再使用包含真实熄灭、遮挡和移动的独立视频验证。

当前视频脚本将整段视频读入内存，适用于短片离线测试。本次 160 帧、1,593 个按钮裁剪的模型测试约耗时 215 秒，尚不具备实时处理速度。

## 文件共享与复现边界

仓库包含代码、依赖说明、已通过 Git LFS 管理的 YOLO 权重、原 Roboflow 检测数据与已有测试素材。亮暗标注 `datasets/`、训练结果 `runs/`、虚拟环境、临时文件及本机输入 `test.mp4` 不纳入此次提交。

**已训练 LoRA 适配器现已通过 Git LFS 分发在 `models/button_state_lora/`，无需重新标注或训练即可运行亮暗检测。** Qwen 基模不在 Git 仓库中：其单文件为 4,255,140,312 字节，2026-09-08 GitHub LFS 实际返回 422，要求不超过 2,147,483,648 字节。联网版首次运行自动下载基模；完整版压缩包提供本地基模。检测数据来源和使用条款见原数据目录的 `README.dataset.txt`、`README.roboflow.txt`。
