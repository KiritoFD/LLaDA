# LLaDA 训练流程与官方指南对齐情况

本文档用于梳理当前仓库中的离线预训练流水线是否遵循 LLaDA 官方指南，并列出已经落地的要点、偏离之处以及后续建议。

## 结论速览

- ✅ **核心流程已对齐**：当前的 `training/train_completely_offline.py` 基于 Transformer Encoder（无因果 Mask）、固定的 `<MASK>` token（ID 126336）、扩散式前向噪声注入及 1% 随机裁剪，均与指南一致。
- ⚙️ **工程增强**：实现了断点续训（`resume_from`）、训练/评估损失持久化、基于配置的评估采样等增补能力，方便小规模实验迭代。
- ⚠️ **资源取舍**：为适配本地 GPU/数据规模，对序列长度、模型尺寸与数据集规模做了压缩；这些属于偏离指南的地方，后续可视资源放宽逐步补齐。

## 指南对齐映射

| 指南要点 | 当前实现 | 备注 |
| --- | --- | --- |
| 使用 Transformer Encoder，移除自注意力中的因果 Mask | `SimpleTransformer`（见 `training/train_completely_offline.py`，第 206~257 行）使用 `nn.TransformerEncoder`，未设置 causal mask | ✅ 完全符合 |
| 预留 `<MASK>` token，ID=126336 | `SimpleTokenizer` 初始化时写死 `mask_token_id=126336`，并在嵌入层大小中预留 | ✅ 完全符合 |
| 扩散式前向噪声注入 (`forward_process`) | `forward_process()` 与指南一致：采样 `p_mask`、随机掩码、用 `<MASK>` 替换 | ✅ 完全符合 |
| 1% batch 采用随机截断长度 | `compute_loss()` 中 `torch.rand(1) < 0.01` 后随机切片 | ✅ 完全符合 |
| 训练数据张量尺寸 (指南示例 `b × 4096`) | 目前配置 `max_length = 256`（`configs/offline_train.json`）以兼容显存 | ⚠️ 偏离：可在显存允许时提升 |
| 词表来源 | 自构建 `SimpleTokenizer`，从 `data/xad_full_text.jsonl` 抽取词汇，`vocab_size=20000` | ✅ 合理，但与官方语料不同 |
| 训练数据准备 | `training/prepare_full_text_dataset.py` 将《计算机网络安全教程》拆分段落生成 JSONL | ✅ 自行实现，可替换为其他语料 |
| 评估流程 | 未提供官方验证集，当前使用 `eval_subset_ratio=0.2` 从训练集中抽样 | ⚠️ 偏离：仅用于监控 Loss |
| 断点续训 | `resume_from` 读取 `best_model.pt`，恢复模型/优化器/调度器/词表 | ⚙️ 自定义增强 |
| 损失曲线记录 | 训练与评估损失写入 `outputs_offline/train_loss.jsonl`、`eval_loss.jsonl` | ⚙️ 自定义增强 |

## 训练配置摘要

- **配置文件**：`configs/offline_train.json`
  - `max_length`: 256 （可按需求拉长）
  - `vocab_size`: 20000（启动时会与检查点词表取较大值）
  - `d_model`: 192、`nhead`: 12、`num_layers`: 12（相较指南规模缩小）
  - `learning_rate`: 3e-5，余弦退火至 1e-6
  - `eval_subset_ratio`: 0.2（每次评估随机采样 20% 训练数据）
  - `resume_from`: `outputs_offline/best_model.pt`
- **数据集**：
  - `data/xad_full_text.jsonl`：通过 `training/prepare_full_text_dataset.py` 从教材原文自动切分（349 条）

## 特殊能力与使用说明

1. **断点续训**
   - 调整 `configs/offline_train.json` 中的 `resume_from` 即可指定起点（默认 `best_model.pt`）。
   - 运行示例：
     ```powershell
     python training\train_completely_offline.py --config_path configs\offline_train.json --max_steps 5000
     ```
   - 日志中会有 `Resumed training from checkpoint at step ...` 提示。

2. **损失曲线持久化**
   - `train_loss.jsonl` / `eval_loss.jsonl` 采用 JSONL，每行含 `run_id` 与 UTC 时间戳，可直接用 pandas/matplotlib 绘制。

3. **评估抽样**
   - 若拥有独立验证集，可将 `eval_data_path` 指向对应 JSONL 并将 `eval_subset_ratio` 设为 0。

## 偏离项与后续建议

- **序列长度与模型规模**：指南示例使用 4K token 长度与较大隐藏维；当前受限于显存以 256 × 192 × 12 层运行。未来可按以下顺序扩展：提高 `max_length` → 增加 `d_model`/`num_layers` → 调整 batch。
- **数据规模**：教材段落仅 349 条，远低于真实预训练需求。后续可追加更多章节或使用网络语料；若引入新语料，记得重新运行 `prepare_full_text_dataset.py` 并考虑增大词表。
- **评估策略**：目前评估仅做 Loss 监控，未实现论文中更完整的指标。若需要对齐官方流程，可准备开发集并关闭随机抽样。

## 参考文件

- `training/train_completely_offline.py`
- `training/prepare_full_text_dataset.py`
- `configs/offline_train.json`
- `outputs_offline/train_loss.jsonl`, `outputs_offline/eval_loss.jsonl`
- `GUIDELINES.md`

如需进一步对齐 LLaDA 完整指南（例如 SFT、采样策略等），可在此文档基础上持续更新。