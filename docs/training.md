# YOLO26 电梯按钮训练

环境已经安装为 `elevator-yolo26`，训练脚本默认使用项目根目录下的
YOLO 官方 `yolo26m.pt` 预训练权重，在 RTX 3080 上训练 100 个 epoch。基础权重由 Ultralytics 自动下载，不放入 `models/`；批量大小由 Ultralytics
按可用显存自动选择。

启动训练：

```bash
cd /home/fq/elevator
conda activate elevator-yolo26
python scripts/train_detector.py
```

常用参数示例：

```bash
# 改为 200 epoch，并固定 batch size 为 8
python scripts/train_detector.py --epochs 200 --batch 8

# 从同名实验的 last.pt 继续训练
python scripts/train_detector.py --resume
```

输出保存在 `runs/detect/elevator_yolo26m/`。最佳权重位于
`runs/detect/elevator_yolo26m/weights/best.pt`。若不激活环境，也可直接运行：

```bash
conda run -n elevator-yolo26 python scripts/train_detector.py
```

查看训练曲线：

```bash
tensorboard --logdir runs
```
