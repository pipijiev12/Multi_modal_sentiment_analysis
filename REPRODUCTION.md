# CMU-MOSI 论文模型复现

本项目统一复现并训练以下十二个模型：EF-LSTM、LF-LSTM、MARN、MFN、TFN、LMF、MulT、QMF、QRSAN、MEGAKAN、M3SA 和 ALMT。

## 数据划分

当前缓存的 CMU-MOSI 数据使用标准划分：

| 划分 | 样本数 |
| --- | ---: |
| 训练集 | 1284 |
| 验证集 | 229 |
| 测试集 | 686 |

输入包含 textual、visual、acoustic 三个对齐模态，序列长度为 50；输入维度分别为 300、20、5。

## 模型入口

各模型的配置位于 `config/reproduction/`。QMF 使用仓库原有的局部 n-gram 量子混合实现 `LocalMixtureNN`；QRSAN 使用仓库原有的量子自注意力加残差实现 `uQDNN_ATTENTION`。现在二者分别通过 `network_type = qmf` 和 `network_type = qrsan` 显式注册。

## 运行

先激活环境：

```powershell
conda activate multimodal-sa
```

检查九个模型能否完成一个 batch 的前向和反向传播：

```powershell
python scripts/reproduce_models.py --check
```

按固定顺序训练全部模型（默认读取各配置中的 20 epochs）：

```powershell
python scripts/reproduce_models.py --continue-on-error
```

只训练部分模型：

```powershell
python scripts/reproduce_models.py --models mult qmf qrsan --continue-on-error
```

临时指定 epoch 数，不修改正式配置：

```powershell
python scripts/reproduce_models.py --epochs 1 --continue-on-error
```

每个模型的日志写入 `eval/reproduction/<model>.log`，运行状态写入 `eval/reproduction/training_state.json`，测试指标写入相应的 CSV 文件。
