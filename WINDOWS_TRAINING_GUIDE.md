# LLaDA Windows从头训练指南

这份指南将帮助您在Windows上从头开始训练LLaDA模型。

## 🚀 快速开始

### 1. 系统要求

**最低配置:**
- Windows 10/11
- Python 3.8+
- 6GB GPU内存 (GTX 1660或更好)
- 16GB RAM
- 10GB可用存储空间

**推荐配置:**
- Windows 11
- Python 3.9+
- 12GB+ GPU内存 (RTX 3060或更好)
- 32GB RAM
- 50GB可用存储空间

### 2. 安装依赖

```bash
# 安装PyTorch (CUDA版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install transformers pyyaml tqdm numpy
```

### 3. 快速测试

在开始完整训练前，建议先运行快速测试：

```bash
python quick_test.py
```

这会运行一个100步的小规模训练来验证环境配置。

## 📁 训练方式选择

### 方式1: 一键批处理脚本 (推荐初学者)

```bash
train_windows.bat
```

这个脚本会：
- 自动检查依赖
- 生成示例数据
- 运行预训练
- 可选择运行SFT训练

### 方式2: PowerShell脚本

```powershell
.\training\train_windows.ps1
```

更灵活的PowerShell脚本，支持更多自定义选项。

### 方式3: Python训练管道

```bash
python training\train_pipeline.py --config training\config_windows.yaml --stage all
```

最灵活的方式，支持完全自定义配置。

### 方式4: 分步训练

```bash
# 1. 生成数据
python training\generate_sample_data.py

# 2. 预训练
python training\pretraining_from_scratch.py \
    --model_name_or_path "microsoft/DialoGPT-small" \
    --train_data_path "data\pretrain\train.jsonl" \
    --output_dir "checkpoints\pretraining" \
    --max_steps 10000 \
    --batch_size 2

# 3. SFT训练
python training\sft_training.py \
    --model_name_or_path "checkpoints\pretraining\best_model.pt" \
    --train_data_path "data\sft\train.jsonl" \
    --output_dir "checkpoints\sft" \
    --max_steps 2000

# 4. 测试推理
python training\inference.py \
    --model_name_or_path "checkpoints\sft\best_sft_model.pt" \
    --prompt "Hello, how are you?"
```

## 🔧 配置说明

### Windows优化配置 (`config_windows.yaml`)

```yaml
model:
  name_or_path: "microsoft/DialoGPT-small"  # 小模型，适合Windows
  max_length: 1024                          # 减少内存使用

pretraining:
  max_steps: 10000      # 较短的训练步数
  batch_size: 2         # 小批次大小
  num_workers: 0        # Windows兼容性
```

### 内存优化设置

如果遇到内存不足错误，尝试以下设置：

```bash
# 减少批次大小
--batch_size 1

# 减少序列长度
--max_length 512

# 使用CPU（如果GPU内存不足）
set CUDA_VISIBLE_DEVICES=""
```

## 📊 训练监控

### 实时监控

训练过程中会在控制台输出日志：

```
Step: 100, Loss: 3.2456, LR: 2.98e-4
Step: 200, Loss: 3.1234, LR: 2.96e-4
Eval Loss: 3.0987
```

### 日志文件

详细日志保存在：
- `logs/training.log` - 预训练日志
- `logs/sft_training.log` - SFT训练日志

### 检查点

模型检查点保存在：
- `checkpoints/pretraining/` - 预训练检查点
- `checkpoints/sft/` - SFT检查点

## 🎯 关键参数说明

### 预训练参数

- `--max_steps`: 训练步数 (Windows建议: 5000-20000)
- `--batch_size`: 批次大小 (Windows建议: 1-4)
- `--learning_rate`: 学习率 (建议: 3e-4)
- `--max_length`: 序列长度 (Windows建议: 512-1024)

### SFT参数

- `--max_steps`: SFT步数 (建议: 1000-5000)
- `--learning_rate`: 学习率 (建议: 2e-5, 比预训练低)

### 推理参数

- `--method`: 采样方法 (fixed_length/semi_autoregressive_padding)
- `--gen_length`: 生成长度
- `--remasking`: 重掩码策略 (low_confidence/random)

## 📈 性能优化

### GPU优化

```bash
# 启用混合精度训练
--dtype float16

# 使用梯度累积模拟大批次
--gradient_accumulation_steps 4
```

### 内存优化

```python
# 在代码中设置
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
```

## 🐛 常见问题解决

### 1. CUDA内存不足

```
RuntimeError: CUDA out of memory
```

**解决方案:**
- 减少batch_size到1
- 减少max_length到512
- 关闭其他GPU程序

### 2. 权限错误

```
PermissionError: [WinError 5] Access is denied
```

**解决方案:**
- 以管理员身份运行
- 检查文件夹权限
- 使用不同的输出目录

### 3. 模型加载失败

```
OSError: Can't load tokenizer
```

**解决方案:**
- 检查网络连接
- 使用本地模型路径
- 清除Hugging Face缓存

### 4. 训练中断

**解决方案:**
- 使用`--resume_from_checkpoint`恢复
- 检查最新的检查点文件
- 确保有足够的存储空间

## 📚 数据格式

### 预训练数据格式

```jsonl
{"text": "This is a sample text for pre-training."}
{"text": "Another piece of text for language modeling."}
```

### SFT数据格式

```jsonl
{
  "conversations": [
    {"role": "user", "content": "What is AI?"},
    {"role": "assistant", "content": "AI is artificial intelligence."}
  ]
}
```

## 🔍 训练验证

### 检查训练效果

```bash
# 查看损失下降
grep "Loss:" logs/training.log | tail -20

# 测试推理
python training\inference.py \
    --model_name_or_path "checkpoints\pretraining\best_model.pt" \
    --prompt "The capital of France is" \
    --gen_length 32
```

### 评估指标

- 训练损失应该稳定下降
- 评估损失不应该持续上升（过拟合）
- 生成的文本应该连贯

## 📞 获取帮助

如果遇到问题：

1. 首先运行 `python quick_test.py` 诊断环境
2. 检查 `logs/` 目录中的详细日志
3. 尝试减少批次大小和序列长度
4. 参考GitHub Issues或文档

## 🎉 成功标志

训练成功的标志：
- ✅ 损失稳定下降
- ✅ 生成检查点文件
- ✅ 推理能产生合理输出
- ✅ 没有CUDA错误

完成训练后，您就拥有了自己的LLaDA模型！