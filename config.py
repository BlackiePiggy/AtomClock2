"""
配置文件 - 全局参数设置
Configuration file for Atomic Clock PHM Framework
"""

import os

# ==================== 目录配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

# 创建必要的目录
for dir_path in [DATA_DIR, OUTPUT_DIR, MODELS_DIR, FIGURES_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ==================== 数据生成配置 ====================
DATA_CONFIG = {
    'n_channels': 7,           # 遥测数据通道数
    'seq_len': 1500,            # 每个连续段的长度
    'n_normal_units': 10,      # 正常单元数量（增加到50）
    'n_anomaly_units': 2,      # 异常单元数量（增加到2）
    'n_segments': 100,         # 每个单元的段数（增加到500，更长时间尺度）

    # 缓变特征参数
    'drift_rate_base': 0.0001,     # 基础漂移率（更缓慢）
    'drift_rate_std': 0.00002,     # 漂移率标准差
    'drift_timescale': 300000,      # 漂移时间尺度

    # 噪声参数
    'noise_level_base': 0.15,      # 基础噪声水平（高噪声）
    'noise_level_std': 0.05,       # 噪声水平标准差

    # 趋势模式参数（无周期）
    'trend_patterns': [
        'monotonic_increase',      # 单调递增
        'monotonic_decrease',      # 单调递减
        'increase_then_decrease',  # 先增后减
        'decrease_then_increase',  # 先减后增
        'increase_then_stable',    # 先增后稳定
        'decrease_then_stable',    # 先减后稳定
    ],

    # 间隙参数
    'gap_duration': 3000,           # 18小时间隙（时间步）
    'gap_probability': 0.9,        # 间隙出现概率

    # 异常模式参数
    'anomaly_start_ratio': 0.7,    # 异常从70%处开始
    'anomaly_acceleration': 0.002, # 加速退化率
}

# ==================== 特征工程配置 ====================
FEATURE_CONFIG = {
    'window_size': 500,        # 特征提取窗口大小
    'top_k_features': 12,      # 选择的top特征数量
}

# ==================== 时序分解配置 ====================
DECOMPOSE_CONFIG = {
    'method': 'hp',            # 默认方法: 'hp', 'lowess', 'savgol', 'ma', 'ewma'
    'compare_methods': True,   # 是否对比所有方法
    'hp_lambda': 1600,         # HP滤波平滑参数
    'lowess_frac': 0.1,        # LOWESS窗口比例
    'savgol_window': 51,       # Savitzky-Golay窗口长度
    'savgol_polyorder': 3,     # Savitzky-Golay多项式阶数
    'ma_window': 50,           # 移动平均窗口
    'ewma_alpha': 0.1,         # 指数加权移动平均alpha
}

# ==================== 模型配置 ====================
MODEL_CONFIG = {
    'latent_dim': 16,          # 潜在空间维度
    'hidden_dim': 64,          # 隐藏层维度
    'epochs': 1000,              # 训练轮数
    'batch_size': 32,          # 批次大小
    'learning_rate': 0.001,    # 学习率
    'beta_vae': 0.1,           # VAE的KL散度权重
    'contrast_weight': 0.1,    # 对比学习权重
}

# ==================== 异常检测配置 ====================
ANOMALY_CONFIG = {
    'alpha_recon': 0.3,        # 重构误差权重
    'alpha_latent': 0.3,       # 潜在偏差权重
    'alpha_contrast': 0.4,     # 对比发散权重
    'percentile_threshold': 99.9,  # 百分位阈值
}

# ==================== 文件路径配置 ====================
FILE_PATHS = {
    # Step 0: 数据生成
    'raw_normal_data': os.path.join(DATA_DIR, 'normal_units_raw.pkl'),
    'raw_anomaly_data': os.path.join(DATA_DIR, 'anomaly_units_raw.pkl'),
    'data_visualization': os.path.join(FIGURES_DIR, 'step0_data_longterm.png'),

    # Step 1: 特征提取
    'normal_features': os.path.join(DATA_DIR, 'normal_features.pkl'),
    'anomaly_features': os.path.join(DATA_DIR, 'anomaly_features.pkl'),
    'feature_visualization': os.path.join(FIGURES_DIR, 'step1_features.png'),

    # Step 2: 质量评估
    'quality_df': os.path.join(DATA_DIR, 'quality_assessment.csv'),
    'filtered_normal_features': os.path.join(DATA_DIR, 'filtered_normal_features.pkl'),
    'filtered_anomaly_features': os.path.join(DATA_DIR, 'filtered_anomaly_features.pkl'),
    'quality_visualization': os.path.join(FIGURES_DIR, 'step2_quality.png'),

    # Step 3: 时序分解
    'imputed_normal': os.path.join(DATA_DIR, 'imputed_normal_features.pkl'),
    'imputed_anomaly': os.path.join(DATA_DIR, 'imputed_anomaly_features.pkl'),
    'decompose_visualization': os.path.join(FIGURES_DIR, 'step3_decomposition.png'),
    'methods_comparison': os.path.join(FIGURES_DIR, 'step3_methods_comparison.png'),

    # Step 4-5: 模型训练
    'trained_model': os.path.join(MODELS_DIR, 'ss_mst_vae.pth'),
    'training_history': os.path.join(DATA_DIR, 'training_history.pkl'),
    'training_visualization': os.path.join(FIGURES_DIR, 'step5_training.png'),

    # Step 6: 潜在空间
    'latent_trajectories': os.path.join(DATA_DIR, 'latent_trajectories.pkl'),
    'latent_visualization': os.path.join(FIGURES_DIR, 'step6_latent_space.png'),

    # Step 7: 异常检测
    'anomaly_scores': os.path.join(DATA_DIR, 'anomaly_scores.pkl'),
    'anomaly_detector': os.path.join(MODELS_DIR, 'anomaly_detector.pkl'),
    'anomaly_visualization': os.path.join(FIGURES_DIR, 'step7_anomaly_detection.png'),
}

print(f"配置加载完成 ✓")
print(f"工作目录: {BASE_DIR}")