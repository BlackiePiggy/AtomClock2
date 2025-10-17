# 原子钟预后与健康管理(PHM)框架

## 📋 项目简介

这是一个完整的原子钟预后与健康管理(Prognostic Health Management, PHM)框架，实现了从数据生成、特征提取、质量评估、时序分解、深度学习建模到异常检测的完整流程。

## 🏗️ 项目结构

```
AtomicClock_PHM/
├── config.py                    # 全局配置文件
├── utils.py                     # 通用工具函数
├── step0_data_generation.py     # 步骤0: 数据生成
├── step1_feature_extraction.py  # 步骤1: 特征提取
├── step2_quality_assessment.py  # 步骤2: 质量评估
├── step3_decomposition.py       # 步骤3: 时序分解
├── step4_build_model.py         # 步骤4: 构建模型
├── step5_train_model.py         # 步骤5: 训练模型
├── step6_latent_space.py        # 步骤6: 潜在空间分析
├── step7_anomaly_detection.py   # 步骤7: 异常检测
├── run_pipeline.py              # 完整流程执行脚本
├── README.md                    # 本文档
├── data/                        # 数据目录（自动创建）
├── models/                      # 模型目录（自动创建）
├── figures/                     # 图表目录（自动创建）
└── output/                      # 输出目录（自动创建）
```

## 🚀 快速开始

### 1. 环境要求

```bash
Python >= 3.8
numpy >= 1.20.0
pandas >= 1.3.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
scipy >= 1.7.0
torch >= 1.9.0
scikit-learn >= 0.24.0
statsmodels >= 0.13.0
```

### 2. 安装依赖

```bash
pip install numpy pandas matplotlib seaborn scipy torch scikit-learn statsmodels
```

### 3. 运行完整流程

```bash
python run_pipeline.py
```

选择选项1运行所有步骤。

### 4. 运行单个步骤

```bash
# 例如：只运行数据生成
python step0_data_generation.py

# 或运行特征提取
python step1_feature_extraction.py
```

## 📊 各步骤详解

### 步骤0: 数据生成 (step0_data_generation.py)

**功能**: 生成模拟原子钟遥测数据

**输入**: 配置参数

**输出**:
- `data/normal_units_raw.pkl` - 正常单元原始数据
- `data/anomaly_units_raw.pkl` - 异常单元原始数据

**特点**:
- 模拟7维遥测通道
- 包含缓变漂移、周期性成分和噪声
- 模拟18小时数据间隙
- 异常单元具有加速退化模式

---

### 步骤1: 特征提取 (step1_feature_extraction.py)

**功能**: 从原始遥测段提取统计和频域特征

**输入**: 
- `data/normal_units_raw.pkl`
- `data/anomaly_units_raw.pkl`

**输出**:
- `data/normal_features.pkl` - 正常单元特征矩阵
- `data/anomaly_features.pkl` - 异常单元特征矩阵
- `figures/step1_features.png` - 特征提取可视化

**提取的特征** (每个通道12个特征 × 7通道 = 84维):
- 时域: 均值、标准差、RMS、方差、峭度、偏度、峰峰值、IQR
- 频域: 低/中/高频能量、主频率

---

### 步骤2: 质量评估 (step2_quality_assessment.py)

**功能**: 评估特征的预后质量，选择高质量特征

**输入**:
- `data/normal_features.pkl`
- `data/anomaly_features.pkl`

**输出**:
- `data/quality_assessment.csv` - 特征质量评分表
- `data/filtered_normal_features.pkl` - 筛选后正常特征
- `data/filtered_anomaly_features.pkl` - 筛选后异常特征
- `figures/step2_quality.png` - 质量评估可视化

**评估指标**:
- **单调性** (Monotonicity): 特征是否随时间单调变化
- **趋势性** (Trendability): 不同单元间轨迹相似度
- **可预示性** (Prognosability): 终点收敛程度

---

### 步骤3: 时序分解 (step3_decomposition.py)

**功能**: 分解时序数据为趋势和残差（无季节性）

**输入**:
- `data/filtered_normal_features.pkl`
- `data/filtered_anomaly_features.pkl`

**输出**:
- `data/imputed_normal_features.pkl` - 分解后正常特征
- `data/imputed_anomaly_features.pkl` - 分解后异常特征
- `figures/step3_decomposition.png` - 分解可视化

**支持的分解方法**:
- **HP滤波** (Hodrick-Prescott): 平滑长期趋势，适合缓慢退化
- **LOWESS**: 局部加权回归，适合非线性趋势
- **Savitzky-Golay**: 保持数据特征
- **移动平均**: 简单快速
- **指数加权MA**: 对近期数据权重更高

---

### 步骤4: 构建模型 (step4_build_model.py)

**功能**: 构建SS-MST-VAE模型架构

**输入**:
- `data/imputed_normal_features.pkl`

**输出**:
- 模型架构信息打印
- 未训练模型保存

**模型特点**:
- **多尺度时序卷积网络** (Multi-Scale TCN): 捕捉不同时间尺度特征
- **变分自编码器** (VAE): 学习紧凑的潜在表示
- **时间对比学习** (TCL): 增强时序建模能力

---

### 步骤5: 训练模型 (step5_train_model.py)

**功能**: 训练SS-MST-VAE模型

**输入**:
- `data/imputed_normal_features.pkl`

**输出**:
- `models/ss_mst_vae.pth` - 训练好的模型
- `data/training_history.pkl` - 训练历史
- `figures/step5_training.png` - 训练过程可视化

**损失函数**:
- 重构损失 (MSE)
- KL散度损失
- 时间对比学习损失

**训练配置** (可在config.py修改):
- Epochs: 30
- Learning Rate: 0.001
- Batch Size: 32
- Beta (KL权重): 0.1

---

### 步骤6: 潜在空间分析 (step6_latent_space.py)

**功能**: 提取并可视化潜在空间退化轨迹

**输入**:
- `models/ss_mst_vae.pth`
- `data/imputed_normal_features.pkl`

**输出**:
- `data/latent_trajectories.pkl` - 潜在轨迹数据
- `figures/step6_latent_space.png` - 潜在空间可视化

**可视化内容**:
- 2D/3D潜在空间轨迹
- 时间演进热图
- 潜在维度时序图
- PCA方差解释率
- 轨迹长度分析

---

### 步骤7: 异常检测 (step7_anomaly_detection.py)

**功能**: 综合异常检测与亚健康预警

**输入**:
- `models/ss_mst_vae.pth`
- `data/imputed_normal_features.pkl`
- `data/imputed_anomaly_features.pkl`

**输出**:
- `data/anomaly_scores.pkl` - 异常分数
- `models/anomaly_detector.pkl` - 检测器统计信息
- `figures/step7_anomaly_detection.png` - 异常检测可视化

**异常检测指标**:
1. **重构误差** (30%): 模型重构质量
2. **潜在偏差** (30%): 潜在向量范数
3. **对比发散** (40%): 与正常轨迹的距离

**预警功能**:
- 综合健康指标 (HI)
- 预警提前期计算
- 滑动窗口异常率
- 多层次异常分解

---

## ⚙️ 配置说明

所有配置参数都在 `config.py` 中定义：

### 数据生成配置
```python
DATA_CONFIG = {
    'n_channels': 7,           # 遥测通道数
    'seq_len': 100,            # 段长度
    'n_normal_units': 10,      # 正常单元数
    'n_anomaly_units': 1,      # 异常单元数
    'n_segments': 150,         # 段数
}
```

### 特征工程配置
```python
FEATURE_CONFIG = {
    'window_size': 100,        # 特征窗口大小
    'top_k_features': 12,      # 选择特征数
}
```

### 模型配置
```python
MODEL_CONFIG = {
    'latent_dim': 16,          # 潜在空间维度
    'hidden_dim': 64,          # 隐藏层维度
    'epochs': 30,              # 训练轮数
    'learning_rate': 0.001,    # 学习率
}
```

### 异常检测配置
```python
ANOMALY_CONFIG = {
    'alpha_recon': 0.3,        # 重构权重
    'alpha_latent': 0.3,       # 潜在权重
    'alpha_contrast': 0.4,     # 对比权重
}
```

---

## 📈 使用示例

### 示例1: 运行完整流程

```bash
python run_pipeline.py
# 选择 1
```

### 示例2: 单独运行某个步骤

```bash
# 假设已经完成步骤0-2，现在只想重新训练模型
python step5_train_model.py
```

### 示例3: 修改配置后重新运行

```python
# 修改 config.py
MODEL_CONFIG['epochs'] = 50
MODEL_CONFIG['learning_rate'] = 0.0005

# 重新运行训练
python step5_train_model.py
```

### 示例4: 使用不同的分解方法

```python
# 修改 config.py
DECOMPOSE_CONFIG['method'] = 'lowess'  # 改为LOWESS方法

# 重新运行步骤3
python step3_decomposition.py
```

---

## 🔍 输出文件说明

### 数据文件 (data/)

| 文件 | 说明 | 大小 |
|------|------|------|
| `*_raw.pkl` | 原始遥测数据 | ~MB |
| `*_features.pkl` | 特征矩阵 | ~KB |
| `imputed_*.pkl` | 分解后特征 | ~KB |
| `latent_trajectories.pkl` | 潜在轨迹 | ~KB |
| `anomaly_scores.pkl` | 异常分数 | ~KB |

### 模型文件 (models/)

| 文件 | 说明 |
|------|------|
| `ss_mst_vae.pth` | 训练好的VAE模型 |
| `anomaly_detector.pkl` | 异常检测器统计信息 |

### 图表文件 (figures/)

| 文件 | 说明 |
|------|------|
| `step1_features.png` | 特征提取可视化 |
| `step2_quality.png` | 质量评估可视化 |
| `step3_decomposition.png` | 时序分解可视化 |
| `step5_training.png` | 训练过程可视化 |
| `step6_latent_space.png` | 潜在空间可视化 |
| `step7_anomaly_detection.png` | 异常检测可视化 |

---

## 🛠️ 故障排除

### 问题1: 导入错误
```
ModuleNotFoundError: No module named 'torch'
```
**解决**: 安装PyTorch
```bash
pip install torch
```

### 问题2: 文件未找到
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/...'
```
**解决**: 确保按顺序运行步骤，或先运行前面的步骤生成数据

### 问题3: 内存不足
```
RuntimeError: CUDA out of memory
```
**解决**: 
- 减小batch_size
- 减少数据量 (n_segments)
- 使用CPU训练 (会较慢)

### 问题4: 中文显示乱码
**解决**: 
- Windows: 安装SimHei字体
- Linux/Mac: 使用默认英文显示

---

## 📚 技术细节

### 核心算法

1. **特征提取**: 基于帧的统计和频域特征
2. **质量评估**: 单调性、趋势性、可预示性三维评估
3. **时序分解**: HP滤波等5种方法，专为非周期数据设计
4. **深度学习**: Multi-Scale TCN + VAE + 时间对比学习
5. **异常检测**: 三层异常分数融合 + 自适应阈值

### 关键创新点

- ✅ 模块化设计，每步独立运行
- ✅ 完整的数据持久化
- ✅ 丰富的可视化
- ✅ 灵活的配置系统
- ✅ 健壮的错误处理

---

## 📖 参考文献

1. Hodrick-Prescott Filter (1997)
2. VAE: Kingma & Welling (2014)
3. Temporal Contrastive Learning
4. Multi-Scale TCN
5. Prognostic Metrics

---

## 👥 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License

---

## 📞 联系方式

如有问题，请通过以下方式联系：
- Email: [your-email]
- GitHub Issues

---

**更新日期**: 2024-01-15
**版本**: v1.0.0