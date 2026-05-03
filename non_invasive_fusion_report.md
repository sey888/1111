# PanDA 算法：LoRA 与 Polar Attention 的“非侵入式”深度融合方案

## 1. 方案背景与消融实验一致性

在学术研究中，为了保证消融实验（Ablation Study）的严谨性，`base+attn` 与 `base+lora+attn` 两个实验组所使用的 `PolarAttention` 模块代码必须保持完全一致。

针对之前实验中出现的精度下降问题，本方案放弃了对 `polar_attention.py` 源码的修改，转而采用**“外部包装适配”**和**“软启动训练调度”**的策略。这种方法在不改变模块内部逻辑的前提下，通过优化模块间的交互界面，解决了 LoRA 与 Attention 之间的数值冲突。

## 2. 核心改进思想：外部尺度对齐与软性嵌入

### 2.1 外部尺度对齐 (External Scale Alignment)
LoRA 在编码器中改变了特征的统计分布（均值和方差）。当这些特征进入解码器并直接经过 `PolarAttention` 时，由于 `PolarAttention` 内部的 `BatchNorm` 和 `Sigmoid` 是基于原始分布初始化的，这种不匹配会导致数值溢出或饱和，从而引发震荡。

**改进措施**：在 `dpt.py` 的解码器融合路径中，我们在调用 `PolarAttention` 之前引入了外部的 `GroupNorm` 层 (`attn_ln`)。
*   **作用**：在不改变 `PolarAttention` 内部结构的情况下，将 LoRA 处理后的特征重新映射到 `PolarAttention` 易于处理的数值区间。
*   **位置**：放置在 `FeatureFusionBlock` (refinenet) 的输出端，对融合后的特征进行归一化，再输入 Polar 模块。

### 2.2 软启动训练调度 (Soft-Start Training)
之前的失败经验显示，如果 `PolarAttention` 的学习率过高，其 `gamma` 参数会迅速从 0 增长到一个较大的值，在 LoRA 还没来得及适应新的解码器环境时就强行改变了特征流。

**改进措施**：
*   **极低初始学习率**：在 `train.yaml` 中将 `polar_attn_lr` 设置为 `1e-5`（主学习率的 1/10）。
*   **逻辑**：这让 Polar 模块在训练初期几乎处于“隐身”状态，给 LoRA 留出足够的 Epoch（通常是前 1-2 个）来稳定全景特征。随着训练进行，Polar 模块会以极其温和的方式介入，实现平滑的精度提升。

## 3. 具体实施细节

### 3.1 模块还原
`polar_attention.py` 已完全还原为您的原始版本，确保消融实验的单一变量原则。

### 3.2 解码器适配 (`dpt.py`)
我们在 `DPTHead` 中添加了 `attn_ln` 序列，并修改了 `forward` 逻辑：
```python
# 外部包装逻辑示例
path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
if self.use_polar_attention:
    # 先归一化再进入 Polar 模块，保证数值稳定性
    path_4 = self.polar_attn4(self.attn_ln4(path_4))
```

### 3.3 优化器配置 (`train.yaml`)
```yaml
optimizer:
  lr: 1.e-4
  lora_lr: 1.e-4 
  polar_attn_lr: 1.e-5 # 关键：软启动
```

## 4. 预期效果与实验建议

1.  **稳定性**：引入 `GroupNorm` 后，特征流在每一层的均值和方差将保持稳定，a1 指标不应再出现剧烈下滑。
2.  **收敛性**：由于采用了软启动，Loss 曲线在初期应与 `base+lora` 接近，后期则会因为 Polar 模块对极点畸变的修正而实现更低的误差。
3.  **建议**：如果 10 个 Epoch 后精度提升不明显，可以尝试在第 5 个 Epoch 时手动将 `polar_attn_lr` 提高到 `5e-5`，这种“分阶段升温”的方法通常对这类融合任务非常有效。

本方案在严格遵守学术规范的前提下，通过工程化的手段解决了算法冲突，是目前最符合您论文需求的解决方案。
