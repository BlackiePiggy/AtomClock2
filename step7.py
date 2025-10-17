"""
步骤7: 异常检测与亚健康预警 - 修复版
综合异常检测器：重构误差 + 潜在偏差 + 对比发散
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from config import ANOMALY_CONFIG, FILE_PATHS
from utils import (save_pickle, load_pickle, print_step_header, print_completion)
from step4_build_model import SS_MST_VAE


class AnomalyDetector:
    """综合异常检测器"""

    def __init__(self, model, alpha_recon=0.3, alpha_latent=0.3, alpha_contrast=0.4):
        self.model = model
        self.alpha_recon = alpha_recon
        self.alpha_latent = alpha_latent
        self.alpha_contrast = alpha_contrast
        self.normal_stats = {}

    def fit_normal_distribution(self, normal_data_dict):
        """在正常数据上拟合分布"""
        recon_errors = []
        latent_deviations = []

        self.model.eval()
        with torch.no_grad():
            for unit_id, features in normal_data_dict.items():
                x = torch.FloatTensor(features).unsqueeze(0)
                x_recon, mu, logvar, z, z_proj = self.model(x)

                # 重构误差
                recon_err = F.mse_loss(x_recon, x, reduction='mean')
                recon_errors.append(recon_err.cpu().numpy())

                # 潜在偏差
                latent_dev = torch.norm(z, dim=1).mean()
                latent_deviations.append(latent_dev.cpu().numpy())

        recon_errors = np.array(recon_errors)
        latent_deviations = np.array(latent_deviations)

        # 计算统计量
        self.normal_stats['recon_mean'] = np.mean(recon_errors)
        self.normal_stats['recon_std'] = max(np.std(recon_errors), 1e-6)
        self.normal_stats['recon_99p'] = np.percentile(recon_errors,
                                                       ANOMALY_CONFIG['percentile_threshold'])

        self.normal_stats['latent_mean'] = np.mean(latent_deviations)
        self.normal_stats['latent_std'] = max(np.std(latent_deviations), 1e-6)
        self.normal_stats['latent_99p'] = np.percentile(latent_deviations,
                                                        ANOMALY_CONFIG['percentile_threshold'])

        print("正常数据分布拟合完成:")
        print(f"  重构误差: μ={self.normal_stats['recon_mean']:.4f}, "
              f"σ={self.normal_stats['recon_std']:.4f}, "
              f"99.9%={self.normal_stats['recon_99p']:.4f}")
        print(f"  潜在偏差: μ={self.normal_stats['latent_mean']:.4f}, "
              f"σ={self.normal_stats['latent_std']:.4f}, "
              f"99.9%={self.normal_stats['latent_99p']:.4f}")

    def compute_segment_anomaly_scores(self, features, reference_latents=None):
        """
        逐段计算异常分数
        features: (n_segments, n_features)
        返回: (n_segments,) 的异常分数数组
        """
        n_segments = len(features)
        total_scores = []
        recon_scores = []
        latent_scores = []
        contrast_scores = []

        self.model.eval()
        with torch.no_grad():
            # 逐段处理
            for seg_idx in range(n_segments):
                # 取当前段
                segment = features[seg_idx:seg_idx+1]  # (1, n_features)
                x = torch.FloatTensor(segment).unsqueeze(0)  # (1, 1, n_features)

                x_recon, mu, logvar, z, z_proj = self.model(x)

                # 1. 重构误差
                recon_error = F.mse_loss(x_recon, x, reduction='mean')
                recon_score = (recon_error - self.normal_stats['recon_mean']) / self.normal_stats['recon_std']

                # 2. 潜在偏差
                latent_deviation = torch.norm(z, dim=1).mean()
                latent_score = (latent_deviation - self.normal_stats['latent_mean']) / self.normal_stats['latent_std']

                # 3. 对比发散度
                if reference_latents is not None and len(reference_latents) > 0:
                    ref_latents_tensor = torch.tensor(reference_latents,
                                                     dtype=z.dtype, device=z.device)
                    distances = torch.cdist(z, ref_latents_tensor)
                    contrast_score = torch.min(distances, dim=1)[0]

                    # 标准化
                    contrast_mean = torch.tensor(0.0)
                    contrast_std = torch.tensor(1.0)
                    if len(reference_latents) > 1:
                        all_dists = []
                        for i in range(len(ref_latents_tensor)):
                            for j in range(i+1, len(ref_latents_tensor)):
                                dist = torch.norm(ref_latents_tensor[i] - ref_latents_tensor[j])
                                all_dists.append(dist.item())
                        if len(all_dists) > 0:
                            contrast_mean = np.mean(all_dists)
                            contrast_std = max(np.std(all_dists), 1e-6)

                    contrast_score = (contrast_score - contrast_mean) / contrast_std
                else:
                    contrast_score = torch.tensor(0.0)

                # 综合异常分数
                total_score = (self.alpha_recon * recon_score +
                             self.alpha_latent * latent_score +
                             self.alpha_contrast * contrast_score)

                total_scores.append(total_score.cpu().numpy())
                recon_scores.append(recon_score.cpu().numpy())
                latent_scores.append(latent_score.cpu().numpy())
                contrast_scores.append(contrast_score.cpu().numpy())

        return {
            'total': np.array(total_scores).flatten(),
            'recon': np.array(recon_scores).flatten(),
            'latent': np.array(latent_scores).flatten(),
            'contrast': np.array(contrast_scores).flatten()
        }


def visualize_anomaly_detection(time_indices, anomaly_scores_dict,
                                threshold, failure_time=None, save_path=None):
    """可视化异常检测结果"""
    fig, axes = plt.subplots(3, 2, figsize=(18, 12))
    fig.suptitle('亚健康预警系统：多层次异常检测', fontsize=16, fontweight='bold')

    total_scores = np.nan_to_num(anomaly_scores_dict['total'], nan=0.0, posinf=0.0, neginf=0.0)
    recon_scores = np.nan_to_num(anomaly_scores_dict['recon'], nan=0.0, posinf=0.0, neginf=0.0)
    latent_scores = np.nan_to_num(anomaly_scores_dict['latent'], nan=0.0, posinf=0.0, neginf=0.0)
    contrast_scores = np.nan_to_num(anomaly_scores_dict['contrast'], nan=0.0, posinf=0.0, neginf=0.0)

    # 1. 综合异常分数
    axes[0, 0].plot(time_indices, total_scores, 'b-', linewidth=1.5, label='综合分数')
    axes[0, 0].axhline(y=threshold, color='r', linestyle='--',
                      linewidth=2, label=f'阈值 ({threshold:.2f})')
    if failure_time:
        axes[0, 0].axvline(x=failure_time, color='darkred', linestyle=':',
                          linewidth=2, label='实际宕机')

    warning_indices = time_indices[total_scores > threshold]
    if len(warning_indices) > 0:
        first_warning = warning_indices[0]
        axes[0, 0].axvline(x=first_warning, color='orange', linestyle='--',
                          linewidth=2, label=f'首次预警 (t={first_warning})')

        if failure_time:
            lead_time = failure_time - first_warning
            axes[0, 0].text(0.02, 0.98, f'预警提前期: {lead_time:.0f} 时间步',
                          transform=axes[0, 0].transAxes, fontsize=12,
                          verticalalignment='top', bbox=dict(boxstyle='round',
                          facecolor='yellow', alpha=0.7))

    axes[0, 0].set_xlabel('时间')
    axes[0, 0].set_ylabel('异常分数')
    axes[0, 0].set_title('综合健康指标 (HI)')
    axes[0, 0].legend(loc='best')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 三个分量对比
    axes[0, 1].plot(time_indices, recon_scores, label='重构误差', linewidth=1.5, alpha=0.7)
    axes[0, 1].plot(time_indices, latent_scores, label='潜在偏差', linewidth=1.5, alpha=0.7)
    axes[0, 1].plot(time_indices, contrast_scores, label='对比发散', linewidth=1.5, alpha=0.7)
    if failure_time:
        axes[0, 1].axvline(x=failure_time, color='darkred', linestyle=':',
                          linewidth=2, alpha=0.5)
    axes[0, 1].set_xlabel('时间')
    axes[0, 1].set_ylabel('归一化分数')
    axes[0, 1].set_title('异常分数分量分解')
    axes[0, 1].legend(loc='best')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 累积异常分布
    axes[1, 0].hist(total_scores, bins=50, color='skyblue',
                   edgecolor='black', alpha=0.7, label='总分数')
    axes[1, 0].axvline(x=threshold, color='r', linestyle='--',
                      linewidth=2, label='阈值')
    axes[1, 0].set_xlabel('异常分数')
    axes[1, 0].set_ylabel('频数')
    axes[1, 0].set_title('异常分数分布')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # 4. 分量贡献饼图
    avg_contributions = [
        np.mean(np.abs(recon_scores)),
        np.mean(np.abs(latent_scores)),
        np.mean(np.abs(contrast_scores))
    ]
    axes[1, 1].pie(avg_contributions, labels=['重构误差', '潜在偏差', '对比发散'],
                   autopct='%1.1f%%', startangle=90,
                   colors=['#ff9999', '#66b3ff', '#99ff99'])
    axes[1, 1].set_title('平均分量贡献')

    # 5. 滑动窗口异常率
    window_size = min(50, len(total_scores) // 4)
    if len(total_scores) > window_size and window_size > 0:
        anomaly_rate = []
        rate_time_indices = []

        for i in range(len(total_scores) - window_size + 1):
            window_scores = total_scores[i:i+window_size]
            rate = np.sum(window_scores > threshold) / window_size
            anomaly_rate.append(rate)
            rate_time_indices.append(time_indices[i + window_size // 2])

        if len(anomaly_rate) > 0:
            axes[2, 0].plot(rate_time_indices, anomaly_rate, 'g-', linewidth=2)
            axes[2, 0].axhline(y=0.1, color='orange', linestyle='--',
                              linewidth=1.5, label='警戒线 (10%)')
            axes[2, 0].axhline(y=0.3, color='r', linestyle='--',
                              linewidth=1.5, label='危险线 (30%)')
            if failure_time:
                axes[2, 0].axvline(x=failure_time, color='darkred',
                                  linestyle=':', linewidth=2, alpha=0.5)
            axes[2, 0].set_xlabel('时间')
            axes[2, 0].set_ylabel('异常率')
            axes[2, 0].set_title(f'滑动窗口异常率 (窗口={window_size})')
            axes[2, 0].legend(loc='best')
            axes[2, 0].grid(True, alpha=0.3)
    else:
        axes[2, 0].text(0.5, 0.5, '数据不足以计算滑动窗口异常率',
                       ha='center', va='center', transform=axes[2, 0].transAxes)

    # 6. 时间演进热图
    axes[2, 1].scatter(time_indices, total_scores, c=total_scores,
                      cmap='RdYlGn_r', s=20, alpha=0.6)
    axes[2, 1].axhline(y=threshold, color='r', linestyle='--', linewidth=2)
    if failure_time:
        axes[2, 1].axvline(x=failure_time, color='darkred', linestyle=':', linewidth=2)
    axes[2, 1].set_xlabel('时间')
    axes[2, 1].set_ylabel('异常分数')
    axes[2, 1].set_title('时间-异常分数热图')
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    print_step_header(7, "异常检测与亚健康预警")

    # 加载模型
    print("\n加载模型...")
    checkpoint = torch.load(FILE_PATHS['trained_model'])
    model_config = checkpoint['model_config']

    model = SS_MST_VAE(
        input_dim=model_config['input_dim'],
        hidden_dim=model_config['hidden_dim'],
        latent_dim=model_config['latent_dim'],
        seq_len=model_config['seq_len']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 加载数据
    print("\n加载数据...")
    imputed_normal = load_pickle(FILE_PATHS['imputed_normal'])
    imputed_anomaly = load_pickle(FILE_PATHS['imputed_anomaly'])

    # 创建异常检测器
    print("\n初始化异常检测器...")
    detector = AnomalyDetector(
        model,
        alpha_recon=ANOMALY_CONFIG['alpha_recon'],
        alpha_latent=ANOMALY_CONFIG['alpha_latent'],
        alpha_contrast=ANOMALY_CONFIG['alpha_contrast']
    )

    # 在正常数据上拟合分布
    print("\n拟合正常数据分布...")
    detector.fit_normal_distribution(imputed_normal)

    # 提取参考潜在轨迹
    print("\n提取参考潜在轨迹...")
    reference_latents = []
    with torch.no_grad():
        for feat in imputed_normal.values():
            x = torch.FloatTensor(feat).unsqueeze(0)
            _, _, _, z, _ = model(x)
            reference_latents.append(z.squeeze(0).numpy())
    reference_latents = np.vstack(reference_latents)

    # 检测异常单元（逐段）
    print("\n检测异常单元（逐段分析）...")
    anomaly_scores_dict = {}

    for unit_id, features in imputed_anomaly.items():
        print(f"\n  处理 {unit_id}...")
        print(f"    特征形状: {features.shape}")

        # 逐段计算异常分数
        scores = detector.compute_segment_anomaly_scores(features, reference_latents)

        anomaly_scores_dict[unit_id] = scores

        avg_score = np.mean(scores['total'])
        max_score = np.max(scores['total'])
        print(f"    平均异常分数: {avg_score:.4f}")
        print(f"    最大异常分数: {max_score:.4f}")
        print(f"    时间步数: {len(scores['total'])}")

    # 可视化
    print("\n生成可视化...")
    first_anomaly = list(anomaly_scores_dict.keys())[0]
    scores = anomaly_scores_dict[first_anomaly]

    time_indices = np.arange(len(scores['total']))
    threshold = 3.0  # 使用固定阈值
    failure_time = int(len(scores['total']) * 0.85)  # 假设85%处故障

    visualize_anomaly_detection(
        time_indices, scores, threshold, failure_time,
        save_path=FILE_PATHS['anomaly_visualization']
    )

    # 保存结果
    print("\n保存异常检测结果...")
    save_pickle(anomaly_scores_dict, FILE_PATHS['anomaly_scores'])
    save_pickle(detector.normal_stats, FILE_PATHS['anomaly_detector'])

    print_completion("异常检测")


if __name__ == "__main__":
    main()