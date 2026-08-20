# YOLO26 电梯按钮训练

环境已经安装为 `elevator-yolo26`，训练脚本默认使用项目根目录下的
`yolo26m.pt`，在 RTX 3080 上训练 100 个 epoch。批量大小由 Ultralytics
按可用显存自动选择。

启动训练：

```bash
cd /home/fq/elevator
conda activate elevator-yolo26
python train_yolo26.py
```

常用参数示例：

```bash
# 改为 200 epoch，并固定 batch size 为 8
python train_yolo26.py --epochs 200 --batch 8

# 从同名实验的 last.pt 继续训练
python train_yolo26.py --resume
```

输出保存在 `runs/detect/elevator_yolo26m/`。最佳权重位于
`runs/detect/elevator_yolo26m/weights/best.pt`。若不激活环境，也可直接运行：

```bash
conda run -n elevator-yolo26 python train_yolo26.py
```

查看训练曲线：

```bash
tensorboard --logdir runs
```
