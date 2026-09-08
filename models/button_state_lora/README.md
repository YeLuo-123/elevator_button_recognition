# 按钮亮暗 LoRA 成品

配套基模：`Qwen/Qwen3-VL-2B-Instruct`。本目录为适配器及对应处理器文件，必须与基模同时加载，不能单独作为完整 Qwen 模型使用。

- 来源：`qwen3_vl_2b_lora_20260907`，8,157 条人工标注。
- 训练：3 epoch，按验证 loss 选择 epoch 2 的 checkpoint-1016，保存为 final_adapter。
- 测试集准确率 93.71%，ON 召回率 65.22%；详细限制见仓库 README。
- 分发时仅把 adapter_config.json 的本机绝对基模路径改为公开模型标识；权重保持原样。
- `adapter_model.safetensors` 使用 Git LFS。克隆后请执行 `git lfs pull`。

在仓库根目录运行：

```bash
python scripts/run_button_state.py --source /path/to/video.mp4
```

基模由 Qwen 发布，其条款适用于基模的使用和再分发。此成品说明不另行授予项目其他代码或数据的许可。
