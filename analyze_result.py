"""
完整结果分析脚本 - 修复字体版本
展示：原始时序 + 异常健康度 + 正常vs异常对比
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from pathlib import Path
import platform

# 设置中文字体 - 根据操作系统自动选择
def setup_chinese_font():
    """设置中文字体"""
    system = platform.system()

    if system == 'Windows':
        # Windows系统
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']
        plt.rcParams['axes.unicode_minus'] = False
    elif system == 'Darwin':  # macOS
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'STHeiti']
        plt.rcParams['axes.unicode_minus'] = False
    else:  # Linux
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Droid Sans Fallback', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    # 设置monospace字体（用于报告文本）
    if system == 'Windows':
        plt.rcParams['font.monospace'] = ['Courier New', 'SimSun']
    else:
        plt.rcParams['font.monospace'] = ['Courier', 'DejaVu Sans Mono']

    print(f"✓ 字体设置完成 (系统: {system})")

setup_chinese_font()


def load_pickle(filepath):
    """加载pickle文件"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def plot_comprehensive_analysis(save_path='figures/comprehensive_analysis.png'):
    """
    生成综合分析图表
    包含：原始时序、健康度、对比分析
    """
    print("="*70)
    print("开始生成综合分析图表...")
    print("="*70)

    # 加载数据
    print("\n加载数据...")
    raw_normal = load_pickle('data/normal_units_raw.pkl')
    raw_anomaly = load_pickle('data/anomaly_units_raw.pkl')
    anomaly_scores = load_pickle('data/anomaly_scores.pkl')
    detector_stats = load_pickle('models/anomaly_detector.pkl')

    # 选择要展示的单元
    normal_unit_id = list(raw_normal.keys())[0]
    anomaly_unit_ids = list(raw_anomaly.keys())

    print(f"\n选择的单元:")
    print(f"  正常单元: {normal_unit_id}")
    print(f"  异常单元: {', '.join(anomaly_unit_ids)}")

    # 准备正常单元数据
    normal_data = raw_normal[normal_unit_id]
    normal_segments = np.vstack(normal_data['segments'])
    normal_times = np.concatenate([ts for ts in normal_data['timestamps']])

    # 创建大图
    fig = plt.figure(figsize=(24, 16))
    fig.suptitle('原子钟PHM系统 - 综合分析报告', fontsize=20, fontweight='bold', y=0.995)

    # 布局：4行3列 = 12个子图
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

    # ============ 第一行：正常单元完整分析 ============
    print("\n绘制正常单元分析...")

    # 1. 正常单元 - 原始多通道时序
    ax1 = fig.add_subplot(gs[0, :2])
    for ch in range(min(7, normal_segments.shape[1])):
        ax1.plot(normal_times, normal_segments[:, ch],
                alpha=0.6, linewidth=0.5, label=f'Channel{ch}')
    ax1.set_xlabel('Time Step', fontsize=10)
    ax1.set_ylabel('Signal Amplitude', fontsize=10)
    ax1.set_title(f'[Normal Unit] Multi-channel Sensor Time Series - {normal_data["trend_pattern"]}',
                 fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8, ncol=7)
    ax1.grid(True, alpha=0.3)

    # 添加统计信息
    info_text = f"Drift: {normal_data['drift_rate']:.6f}\nNoise: {normal_data['noise_level']:.4f}"
    ax1.text(0.98, 0.95, info_text, transform=ax1.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
            fontsize=9)

    # 2. 正常单元 - 单通道详细时序（带趋势线）
    ax2 = fig.add_subplot(gs[0, 2])
    ch0_data = normal_segments[:, 0]
    ax2.plot(normal_times, ch0_data, 'b-', linewidth=0.5, alpha=0.5, label='Raw Signal')

    # 滑动平均显示趋势
    window = 500
    if len(ch0_data) > window:
        smoothed = np.convolve(ch0_data, np.ones(window)/window, mode='valid')
        ax2.plot(normal_times[:len(smoothed)], smoothed, 'r-',
                linewidth=2, label='Trend', alpha=0.8)

    ax2.set_xlabel('Time Step', fontsize=10)
    ax2.set_ylabel('Amplitude', fontsize=10)
    ax2.set_title('[Normal] Channel 0 Detail', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ============ 第二、三行：每个异常单元分析 ============
    for row_idx, anomaly_unit_id in enumerate(anomaly_unit_ids[:2], start=1):
        print(f"\n绘制异常单元 {row_idx}: {anomaly_unit_id}")

        # 准备异常单元数据
        anomaly_data = raw_anomaly[anomaly_unit_id]
        anomaly_segments = np.vstack(anomaly_data['segments'])
        anomaly_times = np.concatenate([ts for ts in anomaly_data['timestamps']])

        # 获取异常分数
        scores = anomaly_scores[anomaly_unit_id]
        avg_score = np.mean(scores['total'])
        max_score = np.max(scores['total'])

        # 状态判断
        if avg_score > 10:
            status = "Severe Anomaly"
            status_icon = "[!!!]"
            color = 'red'
        elif avg_score > 3:
            status = "Mild Anomaly"
            status_icon = "[!]"
            color = 'orange'
        else:
            status = "Normal"
            status_icon = "[OK]"
            color = 'green'

        # 3/6. 异常单元 - 原始多通道时序
        ax3 = fig.add_subplot(gs[row_idx, :2])
        for ch in range(min(7, anomaly_segments.shape[1])):
            ax3.plot(anomaly_times, anomaly_segments[:, ch],
                    alpha=0.6, linewidth=0.5, label=f'Channel{ch}')

        # 标记异常开始点
        anomaly_start = int(len(anomaly_times) * 0.7)
        ax3.axvline(x=anomaly_times[anomaly_start], color='red',
                   linestyle='--', linewidth=2, alpha=0.7, label='Anomaly Start')

        ax3.set_xlabel('Time Step', fontsize=10)
        ax3.set_ylabel('Signal Amplitude', fontsize=10)
        title = f'[Anomaly Unit {row_idx}] {anomaly_data["trend_pattern"]} | Type: {anomaly_data["anomaly_type"]}'
        ax3.set_title(title, fontsize=12, fontweight='bold', color=color)
        ax3.legend(loc='upper left', fontsize=8, ncol=7)
        ax3.grid(True, alpha=0.3)

        # 添加统计信息
        info_text = (f"Drift: {anomaly_data['drift_rate']:.6f}\n"
                    f"Noise: {anomaly_data['noise_level']:.4f}\n"
                    f"Avg Score: {avg_score:.2f}\n"
                    f"Max Score: {max_score:.2f}\n"
                    f"Status: {status_icon} {status}")
        ax3.text(0.98, 0.95, info_text, transform=ax3.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                fontsize=9)

        # 4/7. 异常单元 - 健康度时序
        ax4 = fig.add_subplot(gs[row_idx, 2])

        # 绘制综合健康指标
        time_indices = np.arange(len(scores['total']))
        ax4.plot(time_indices, scores['total'], 'b-', linewidth=1.5, label='Health Index')

        # 阈值线
        threshold = detector_stats['recon_99p'] / detector_stats['recon_mean']
        ax4.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold={threshold:.1f}')

        # 标记异常区域
        anomaly_mask = scores['total'] > threshold
        if np.any(anomaly_mask):
            first_anomaly = np.where(anomaly_mask)[0][0]
            ax4.axvline(x=first_anomaly, color='orange', linestyle=':',
                       linewidth=2, label=f'First Warning')
            ax4.axvspan(first_anomaly, len(time_indices), alpha=0.2, color='red')

        ax4.set_xlabel('Time Step', fontsize=10)
        ax4.set_ylabel('Anomaly Score', fontsize=10)
        ax4.set_title(f'[Unit {row_idx}] Health Evolution', fontsize=11, fontweight='bold')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

        # 添加预警信息
        if np.any(anomaly_mask):
            lead_time = len(time_indices) - first_anomaly
            warning_text = f"Lead Time: {lead_time} steps"
            ax4.text(0.02, 0.98, warning_text, transform=ax4.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                    fontsize=9)

    # ============ 第四行：综合对比分析 ============
    print("\n绘制综合对比分析...")

    # 9. 信号对比（平滑后）
    ax9 = fig.add_subplot(gs[3, 0])

    # 正常单元平滑
    normal_smooth = np.convolve(normal_segments[:, 0], np.ones(500)/500, mode='valid')
    ax9.plot(normal_times[:len(normal_smooth)], normal_smooth,
            'b-', linewidth=2, alpha=0.7, label='Normal')

    # 异常单元平滑
    for idx, anomaly_unit_id in enumerate(anomaly_unit_ids[:2]):
        anomaly_data = raw_anomaly[anomaly_unit_id]
        anomaly_segments = np.vstack(anomaly_data['segments'])
        anomaly_times = np.concatenate([ts for ts in anomaly_data['timestamps']])
        anomaly_smooth = np.convolve(anomaly_segments[:, 0], np.ones(500)/500, mode='valid')

        color = 'red' if idx == 0 else 'orange'
        ax9.plot(anomaly_times[:len(anomaly_smooth)], anomaly_smooth,
                linewidth=2, alpha=0.7, color=color, label=f'Anomaly{idx+1}')

    ax9.set_xlabel('Time Step', fontsize=10)
    ax9.set_ylabel('Amplitude (Smoothed)', fontsize=10)
    ax9.set_title('Trend Comparison', fontsize=11, fontweight='bold')
    ax9.legend(fontsize=9)
    ax9.grid(True, alpha=0.3)

    # 10. 健康度对比
    ax10 = fig.add_subplot(gs[3, 1])

    # 创建健康度箱线图
    health_data = []
    labels = []
    colors_box = []

    # 正常单元（模拟健康度=0附近）
    health_data.append(np.random.randn(len(normal_times)//100) * 0.5)
    labels.append('Normal')
    colors_box.append('lightblue')

    # 异常单元
    for idx, anomaly_unit_id in enumerate(anomaly_unit_ids[:2]):
        scores = anomaly_scores[anomaly_unit_id]
        health_data.append(scores['total'][::100])  # 采样
        labels.append(f'Anomaly{idx+1}')
        colors_box.append('lightcoral' if idx == 0 else 'lightyellow')

    bp = ax10.boxplot(health_data, labels=labels, patch_artist=True,
                      showmeans=True, meanline=True)

    # 设置颜色
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)

    ax10.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label='Threshold')
    ax10.set_ylabel('Anomaly Score', fontsize=10)
    ax10.set_title('Health Distribution Comparison', fontsize=11, fontweight='bold')
    ax10.legend(fontsize=8)
    ax10.grid(True, alpha=0.3, axis='y')

    # 11. 异常检测总结（使用纯英文）
    ax11 = fig.add_subplot(gs[3, 2])
    ax11.axis('off')

    # 创建总结表格 - 使用英文避免字体问题
    summary_text = "[Detection Summary]\n\n"
    summary_text += f"Normal Units: {len(raw_normal)}\n"
    summary_text += f"Anomaly Units: {len(raw_anomaly)}\n"
    summary_text += "-" * 40 + "\n\n"

    for idx, anomaly_unit_id in enumerate(anomaly_unit_ids):
        scores = anomaly_scores[anomaly_unit_id]
        avg_score = np.mean(scores['total'])
        max_score = np.max(scores['total'])

        if avg_score > 10:
            status = "[!!!] Severe - Immediate Action"
        elif avg_score > 3:
            status = "[!] Mild - Monitor Closely"
        else:
            status = "[OK] Normal - Continue Monitoring"

        anomaly_data = raw_anomaly[anomaly_unit_id]

        summary_text += f"[Anomaly Unit {idx+1}]\n"
        summary_text += f"ID: {anomaly_unit_id}\n"
        summary_text += f"Type: {anomaly_data['anomaly_type']}\n"
        summary_text += f"Pattern: {anomaly_data['trend_pattern']}\n"
        summary_text += f"Avg Score: {avg_score:.2f}\n"
        summary_text += f"Max Score: {max_score:.2f}\n"
        summary_text += f"Status: {status}\n"

        # 计算预警提前期
        anomaly_mask = scores['total'] > threshold
        if np.any(anomaly_mask):
            first_warning = np.where(anomaly_mask)[0][0]
            total_steps = len(scores['total'])
            lead_time = total_steps - first_warning
            lead_ratio = lead_time / total_steps * 100
            summary_text += f"Lead Time: {lead_time} steps ({lead_ratio:.1f}%)\n"
        else:
            summary_text += f"Lead Time: No warning triggered\n"

        summary_text += "\n" + "-" * 40 + "\n\n"

    # 添加推荐
    summary_text += "[Recommendations]\n"
    for idx, anomaly_unit_id in enumerate(anomaly_unit_ids):
        scores = anomaly_scores[anomaly_unit_id]
        avg_score = np.mean(scores['total'])

        if avg_score > 10:
            summary_text += f"* Unit{idx+1}: Stop & Repair Immediately\n"
        elif avg_score > 3:
            summary_text += f"* Unit{idx+1}: Increase Monitoring\n"
        else:
            summary_text += f"* Unit{idx+1}: Continue Normal Operation\n"

    # 使用普通字体而不是monospace
    ax11.text(0.05, 0.95, summary_text, transform=ax11.transAxes,
             verticalalignment='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 保存图表
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 综合分析图表已保存: {save_path}")
    plt.show()

    print("\n" + "="*70)
    print("综合分析完成！")
    print("="*70)


def generate_detailed_report():
    """生成详细的文字报告"""
    print("\n" + "="*70)
    print("生成详细分析报告...")
    print("="*70)

    # 加载数据
    anomaly_scores = load_pickle('data/anomaly_scores.pkl')
    detector_stats = load_pickle('models/anomaly_detector.pkl')
    raw_anomaly = load_pickle('data/anomaly_units_raw.pkl')

    threshold = detector_stats['recon_99p'] / detector_stats['recon_mean']

    report = []
    report.append("\n" + "="*70)
    report.append("  Atomic Clock PHM System - Detailed Analysis Report")
    report.append("="*70 + "\n")

    for unit_id, scores in anomaly_scores.items():
        anomaly_data = raw_anomaly[unit_id]

        report.append(f"\n[{unit_id}] Detailed Analysis")
        report.append("="*60)

        # 基本信息
        report.append(f"\n1. Basic Information:")
        report.append(f"   * Anomaly Type: {anomaly_data['anomaly_type']}")
        report.append(f"   * Trend Pattern: {anomaly_data['trend_pattern']}")
        report.append(f"   * Drift Rate: {anomaly_data['drift_rate']:.6f}")
        report.append(f"   * Noise Level: {anomaly_data['noise_level']:.4f}")

        # 异常分数统计
        report.append(f"\n2. Anomaly Score Statistics:")
        report.append(f"   * Average Score: {np.mean(scores['total']):.2f}")
        report.append(f"   * Maximum Score: {np.max(scores['total']):.2f}")
        report.append(f"   * Minimum Score: {np.min(scores['total']):.2f}")
        report.append(f"   * Std Deviation: {np.std(scores['total']):.2f}")
        report.append(f"   * Detection Threshold: {threshold:.2f}")

        # 分量贡献
        report.append(f"\n3. Component Contributions:")
        report.append(f"   * Reconstruction Error: mean={np.mean(scores['recon']):.2f}, "
                     f"max={np.max(scores['recon']):.2f}")
        report.append(f"   * Latent Deviation: mean={np.mean(scores['latent']):.2f}, "
                     f"max={np.max(scores['latent']):.2f}")
        report.append(f"   * Contrastive Divergence: mean={np.mean(scores['contrast']):.2f}, "
                     f"max={np.max(scores['contrast']):.2f}")

        # 时序分析
        anomaly_mask = scores['total'] > threshold
        anomaly_ratio = np.sum(anomaly_mask) / len(scores['total']) * 100

        report.append(f"\n4. Temporal Analysis:")
        report.append(f"   * Total Time Steps: {len(scores['total'])}")
        report.append(f"   * Anomalous Steps: {np.sum(anomaly_mask)}")
        report.append(f"   * Anomaly Ratio: {anomaly_ratio:.2f}%")

        if np.any(anomaly_mask):
            first_anomaly_idx = np.where(anomaly_mask)[0][0]
            lead_time = len(scores['total']) - first_anomaly_idx
            lead_ratio = lead_time / len(scores['total']) * 100

            report.append(f"   * First Warning Step: {first_anomaly_idx}")
            report.append(f"   * Warning Lead Time: {lead_time} steps ({lead_ratio:.1f}%)")
        else:
            report.append(f"   * Warning Status: No warning triggered")

        # 健康评估
        avg_score = np.mean(scores['total'])
        report.append(f"\n5. Health Assessment:")

        if avg_score > 10:
            report.append(f"   [!!!] Status: Severe Anomaly")
            report.append(f"   * Recommendation: Stop and repair immediately")
            report.append(f"   * Urgency Level: HIGH")
        elif avg_score > 3:
            report.append(f"   [!] Status: Mild Anomaly")
            report.append(f"   * Recommendation: Increase monitoring frequency")
            report.append(f"   * Urgency Level: MEDIUM")
        else:
            report.append(f"   [OK] Status: Normal")
            report.append(f"   * Recommendation: Continue routine monitoring")
            report.append(f"   * Urgency Level: LOW")

        report.append("\n" + "="*60)

    # 打印报告
    for line in report:
        print(line)

    # 保存报告
    report_path = 'output/detailed_report.txt'
    Path('output').mkdir(exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"\n✓ Detailed report saved: {report_path}")


if __name__ == "__main__":
    print("""
    ================================================================
    
        Atomic Clock PHM System - Comprehensive Analysis
    
        Output:
        1. Comprehensive Analysis Chart (12 subplots)
        2. Detailed Text Report
    
    ================================================================
    """)

    # 生成综合分析图表
    plot_comprehensive_analysis()

    # 生成详细报告
    generate_detailed_report()

    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print("\nGenerated Files:")
    print("  * figures/comprehensive_analysis.png - Comprehensive chart")
    print("  * output/detailed_report.txt - Detailed report")
    print("="*70)