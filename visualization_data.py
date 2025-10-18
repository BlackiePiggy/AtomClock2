"""
原子钟遥测数据完整可视化脚本
加载pickle文件并基于真实timestamp展示
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pickle
from pathlib import Path

from config import FILE_PATHS

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 统一中文文本使用的字体（用于避免缺失字形警告）
TEXT_FONT_KWARGS = {
    'fontfamily': plt.rcParams['font.sans-serif'][0]
} if plt.rcParams.get('font.sans-serif') else {}


def load_pickle(filepath):
    """加载pickle文件"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def create_timestamps(unit_data, start_datetime='2024-01-01 00:00:00', sampling_interval=10):
    """
    根据unit_data的timestamps字段生成真实的datetime对象

    参数:
        unit_data: 单元数据字典
        start_datetime: 起始时间字符串
        sampling_interval: 采样间隔（秒）

    返回:
        所有数据点对应的datetime列表
    """
    start_dt = datetime.strptime(start_datetime, '%Y-%m-%d %H:%M:%S')

    all_datetimes = []
    for seg_idx, timestamps in enumerate(unit_data['timestamps']):
        # timestamps是该段每个点的全局时间步
        seg_datetimes = [start_dt + timedelta(seconds=int(t * sampling_interval))
                         for t in timestamps]
        all_datetimes.extend(seg_datetimes)

    return all_datetimes


def prepare_unit_sequence(unit_data, start_datetime, sampling_interval):
    """将单元的所有片段拼接成连续序列及对应的时间戳"""

    if unit_data is None:
        return None, None

    segments = np.vstack(unit_data['segments'])
    datetimes = create_timestamps(unit_data, start_datetime, sampling_interval)
    return segments, datetimes


def compute_channel_statistics(sequence):
    """计算每个通道的统计量"""

    if sequence is None:
        return []

    stats = []
    for ch in range(sequence.shape[1]):
        channel_data = sequence[:, ch]
        stats.append({
            'min': float(np.min(channel_data)),
            'max': float(np.max(channel_data)),
            'mean': float(np.mean(channel_data)),
            'std': float(np.std(channel_data)),
            'median': float(np.median(channel_data)),
            'p05': float(np.percentile(channel_data, 5)),
            'p95': float(np.percentile(channel_data, 95)),
        })
    return stats


def format_stats_text(channel_idx, stats):
    """格式化通道统计信息文本"""

    return (
        f"通道{channel_idx}\n"
        f"  min: {stats['min']:.4f}\n"
        f"  max: {stats['max']:.4f}\n"
        f"  mean: {stats['mean']:.4f}\n"
        f"  std: {stats['std']:.4f}\n"
        f"  median: {stats['median']:.4f}\n"
        f"  p05-p95: [{stats['p05']:.4f}, {stats['p95']:.4f}]"
    )


def build_dataset_summary(unit_id, unit_data, datetimes, stats):
    """生成数据集的文本摘要"""

    if unit_data is None or datetimes is None:
        return "无数据可展示"

    duration_hours = (datetimes[-1] - datetimes[0]).total_seconds() / 3600
    n_points = len(datetimes)
    n_channels = len(stats)
    n_segments = len(unit_data['segments'])

    drift_rate = unit_data.get('drift_rate', 0.0)
    noise_level = unit_data.get('noise_level', 0.0)
    trend_pattern = unit_data.get('trend_pattern', 'unknown')

    summary_lines = [
        f"【单元 {unit_id} 数据摘要】",
        "",
        f"数据点数: {n_points:,}",
        f"通道数: {n_channels}",
        f"分段数: {n_segments}",
        f"起始时间: {datetimes[0].strftime('%Y-%m-%d %H:%M:%S')}",
        f"结束时间: {datetimes[-1].strftime('%Y-%m-%d %H:%M:%S')}",
        f"总时长: {duration_hours:.2f} 小时",
        "",
        f"趋势模式: {trend_pattern}",
        f"漂移率: {drift_rate:.6f}",
        f"噪声水平: {noise_level:.4f}",
        "",
        "每通道均值范围:",
    ]

    for idx, channel_stats in enumerate(stats):
        summary_lines.append(
            f"  通道{idx}: mean={channel_stats['mean']:.4f} ± {channel_stats['std']:.4f}"
        )

    return "\n".join(summary_lines)


def plot_channel_timeseries(ax, datetimes, sequence, channel_idx, stats, title, color):
    """绘制单个通道的时间序列"""

    if sequence is None or datetimes is None:
        ax.text(0.5, 0.5, '无数据', ha='center', va='center', fontsize=12, color='gray')
        ax.axis('off')
        return

    downsample = max(1, len(datetimes) // 6000)
    times = datetimes[::downsample]
    values = sequence[::downsample, channel_idx]

    ax.plot(times, values, color=color, linewidth=1.0, alpha=0.8)
    ax.axhline(stats['mean'], color=color, linestyle='--', linewidth=1, alpha=0.9,
               label='均值')
    ax.axhline(stats['mean'] + stats['std'], color=color, linestyle=':', linewidth=0.8,
               alpha=0.6)
    ax.axhline(stats['mean'] - stats['std'], color=color, linestyle=':', linewidth=0.8,
               alpha=0.6)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('幅值', fontsize=10)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha='right', fontsize=8)

    stats_text = format_stats_text(channel_idx, stats)
    ax.text(0.99, 0.95, stats_text,
            transform=ax.transAxes,
            ha='right', va='top',
            fontsize=8,
            **TEXT_FONT_KWARGS,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor=color))

def visualize_complete_telemetry_data(normal_data_path, anomaly_data_path,
                                     output_dir='figures',
                                     start_datetime='2024-01-01 00:00:00',
                                     sampling_interval=10,
                                     normal_unit_id=None,
                                     anomaly_unit_id=None,
                                     output_path=None):
    """
    完整可视化原子钟遥测数据（基于真实timestamp）

    参数:
        normal_data_path: 正常单元数据pickle文件路径
        anomaly_data_path: 异常单元数据pickle文件路径
        output_dir: 输出目录
        start_datetime: 起始时间
        sampling_interval: 采样间隔（秒）
    """

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("原子钟遥测数据完整可视化")
    print("="*80)

    # 1. 加载数据
    print(f"\n[1/4] 加载数据...")
    print(f"  正常单元数据: {normal_data_path}")
    print(f"  异常单元数据: {anomaly_data_path}")

    normal_data = load_pickle(normal_data_path)
    anomaly_data = load_pickle(anomaly_data_path) if Path(anomaly_data_path).exists() else {}

    print(f"  ✓ 成功加载 {len(normal_data)} 个正常单元")
    print(f"  ✓ 成功加载 {len(anomaly_data)} 个异常单元")

    # 2. 选择展示的单元
    normal_units = list(normal_data.keys())
    anomaly_units = list(anomaly_data.keys())

    if len(normal_units) == 0:
        print("  ✗ 错误：没有正常单元数据！")
        return

    # 选择要展示的正常单元
    if normal_unit_id and normal_unit_id in normal_data:
        selected_normal_id = normal_unit_id
    else:
        selected_normal_id = normal_units[0]
    selected_normal_unit = normal_data[selected_normal_id]

    print(f"\n[2/4] 处理数据...")
    print(f"  选择单元: {selected_normal_id}")
    print(f"  趋势模式: {selected_normal_unit.get('trend_pattern', 'unknown')}")
    print(f"  漂移率: {selected_normal_unit.get('drift_rate', 0):.6f}")
    print(f"  噪声水平: {selected_normal_unit.get('noise_level', 0):.4f}")

    # 合并所有段
    all_segments = np.vstack(selected_normal_unit['segments'])
    all_datetimes = create_timestamps(selected_normal_unit, start_datetime, sampling_interval)

    n_points = len(all_datetimes)
    n_channels = all_segments.shape[1]
    time_span = (all_datetimes[-1] - all_datetimes[0]).total_seconds() / 3600  # 小时

    print(f"  数据点数: {n_points:,}")
    print(f"  通道数: {n_channels}")
    print(f"  时间范围: {all_datetimes[0]} ~ {all_datetimes[-1]}")
    print(f"  总时长: {time_span:.2f} 小时 ({time_span/24:.2f} 天)")

    # 3. 创建可视化
    print(f"\n[3/4] 生成可视化...")

    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(5, 2, hspace=0.35, wspace=0.25)

    # 颜色方案
    colors = ['#ef4444', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6', '#ec4899', '#06b6d4']

    # ==================== 1. 完整时间序列（单通道）====================
    print("  - 绘制完整时间序列...")
    ax1 = fig.add_subplot(gs[0, :])
    channel_idx = 0

    ax1.plot(all_datetimes, all_segments[:, channel_idx],
             'b-', linewidth=0.5, alpha=0.7, label=f'通道{channel_idx}')

    ax1.set_xlabel('时间', fontsize=12, fontweight='bold')
    ax1.set_ylabel('幅值', fontsize=12, fontweight='bold')
    ax1.set_title(f'完整时间序列 - {selected_normal_id} (通道{channel_idx})\n'
                  f'趋势: {selected_normal_unit.get("trend_pattern", "unknown")} | '
                  f'总时长: {time_span:.1f}小时 | 数据点: {n_points:,}',
                  fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')

    # 添加统计信息
    ch0_data = all_segments[:, channel_idx]
    mean_val = np.mean(ch0_data)
    std_val = np.std(ch0_data)
    min_val = np.min(ch0_data)
    max_val = np.max(ch0_data)

    stats_text = f'均值: {mean_val:.4f}\n标准差: {std_val:.4f}\n值域: [{min_val:.4f}, {max_val:.4f}]'
    ax1.text(0.02, 0.98, stats_text,
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85),
             fontsize=10,
             **TEXT_FONT_KWARGS)

    # ==================== 2. 局部放大（显示噪声）====================
    print("  - 绘制局部放大图...")
    ax2 = fig.add_subplot(gs[1, 0])

    zoom_start = len(all_datetimes) // 2
    zoom_end = zoom_start + 1000

    ax2.plot(all_datetimes[zoom_start:zoom_end],
             all_segments[zoom_start:zoom_end, channel_idx],
             'b-', linewidth=1.2)

    ax2.set_xlabel('时间', fontsize=11)
    ax2.set_ylabel('幅值', fontsize=11)
    ax2.set_title('局部放大 - 显示高频噪声特征（无周期）', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=9)

    # ==================== 3. 趋势提取（滑动平均）====================
    print("  - 进行趋势提取...")
    ax3 = fig.add_subplot(gs[1, 1])

    window = 500
    smoothed = np.convolve(ch0_data, np.ones(window)/window, mode='valid')
    smoothed_times = all_datetimes[:len(smoothed)]

    ax3.plot(all_datetimes, ch0_data,
             color='lightblue', linewidth=0.3, alpha=0.4, label='原始数据')
    ax3.plot(smoothed_times, smoothed,
             'r-', linewidth=2.5, label=f'平滑趋势(窗口={window})')

    ax3.set_xlabel('时间', fontsize=11)
    ax3.set_ylabel('幅值', fontsize=11)
    ax3.set_title('趋势提取 - 滑动平均去噪显示纯单调特征', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=9)

    # ==================== 4. 多通道对比 ====================
    print("  - 绘制多通道对比...")
    ax4 = fig.add_subplot(gs[2, :])

    n_channels_display = min(5, n_channels)
    downsample = max(1, len(all_datetimes) // 5000)  # 降采样

    for ch in range(n_channels_display):
        ax4.plot(all_datetimes[::downsample],
                all_segments[::downsample, ch],
                alpha=0.7, linewidth=1.5,
                label=f'通道{ch}',
                color=colors[ch % len(colors)])

    ax4.set_xlabel('时间', fontsize=12, fontweight='bold')
    ax4.set_ylabel('幅值', fontsize=12, fontweight='bold')
    ax4.set_title('多通道对比 - 不同通道的单调趋势模式', fontsize=14, fontweight='bold')
    ax4.legend(loc='best', fontsize=10, ncol=n_channels_display)
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=30, ha='right')

    # ==================== 5. 分段统计 ====================
    print("  - 计算分段统计...")
    ax5 = fig.add_subplot(gs[3, 0])

    n_segments = len(selected_normal_unit['segments'])
    segment_means = []
    segment_stds = []
    segment_labels = []

    for seg_idx, segment in enumerate(selected_normal_unit['segments']):
        seg_data = segment[:, channel_idx]
        segment_means.append(np.mean(seg_data))
        segment_stds.append(np.std(seg_data))
        segment_labels.append(f'S{seg_idx+1}')

    # 只显示前20个段（避免过于拥挤）
    display_segments = min(20, n_segments)
    x_pos = np.arange(display_segments)
    width = 0.35

    ax5_twin = ax5.twinx()
    ax5.bar(x_pos - width/2, segment_means[:display_segments], width,
            label='均值', color='#8b5cf6', alpha=0.8)
    ax5_twin.bar(x_pos + width/2, segment_stds[:display_segments], width,
                 label='标准差', color='#06b6d4', alpha=0.8)

    ax5.set_xlabel('数据段', fontsize=11)
    ax5.set_ylabel('均值', fontsize=11, color='#8b5cf6', fontweight='bold')
    ax5_twin.set_ylabel('标准差', fontsize=11, color='#06b6d4', fontweight='bold')
    ax5.set_title(f'分段统计 - 各段均值与标准差 (前{display_segments}段)',
                  fontsize=12, fontweight='bold')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(segment_labels[:display_segments], fontsize=8, rotation=45)
    ax5.tick_params(axis='y', labelcolor='#8b5cf6')
    ax5_twin.tick_params(axis='y', labelcolor='#06b6d4')
    ax5.grid(True, alpha=0.3, axis='y')

    # ==================== 6. 趋势变化率 ====================
    print("  - 分析趋势变化率...")
    ax6 = fig.add_subplot(gs[3, 1])

    segment_trends = []
    for segment in selected_normal_unit['segments'][:display_segments]:
        seg_data = segment[:, channel_idx]
        trend_rate = (seg_data[-1] - seg_data[0]) / len(seg_data)
        segment_trends.append(trend_rate)

    colors_bar = ['green' if t > 0 else 'red' for t in segment_trends]
    ax6.bar(segment_labels[:display_segments], segment_trends,
            color=colors_bar, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax6.axhline(y=0, color='black', linestyle='--', linewidth=2)

    ax6.set_xlabel('数据段', fontsize=11)
    ax6.set_ylabel('趋势变化率', fontsize=11)
    ax6.set_title(f'趋势变化率分析 (前{display_segments}段)', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, fontsize=8)

    positive_count = sum(1 for t in segment_trends if t > 0)
    negative_count = sum(1 for t in segment_trends if t < 0)
    ax6.text(0.98, 0.98,
             f'上升: {positive_count}/{len(segment_trends)}\n'
             f'下降: {negative_count}/{len(segment_trends)}',
             transform=ax6.transAxes, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
             fontsize=9)

    # ==================== 7. 正常vs异常对比 ====================
    print("  - 绘制正常vs异常对比...")
    ax7 = fig.add_subplot(gs[4, 0])

    if len(anomaly_units) > 0:
        if anomaly_unit_id and anomaly_unit_id in anomaly_data:
            selected_anomaly_id = anomaly_unit_id
        else:
            selected_anomaly_id = anomaly_units[0]
        selected_anomaly_unit = anomaly_data[selected_anomaly_id]

        anomaly_segments = np.vstack(selected_anomaly_unit['segments'])
        anomaly_datetimes = create_timestamps(selected_anomaly_unit, start_datetime, sampling_interval)

        # 平滑处理
        normal_smooth = np.convolve(ch0_data, np.ones(window)/window, mode='valid')
        anomaly_smooth = np.convolve(anomaly_segments[:, channel_idx],
                                     np.ones(window)/window, mode='valid')

        normal_smooth_times = all_datetimes[:len(normal_smooth)]
        anomaly_smooth_times = anomaly_datetimes[:len(anomaly_smooth)]

        ax7.plot(normal_smooth_times, normal_smooth,
                'b-', linewidth=2, alpha=0.7, label='正常')
        ax7.plot(anomaly_smooth_times, anomaly_smooth,
                'r-', linewidth=2, alpha=0.7,
                label=f'异常({selected_anomaly_unit.get("anomaly_type", "unknown")})')

        ax7.set_xlabel('时间', fontsize=11)
        ax7.set_ylabel('幅值（平滑）', fontsize=11)
        ax7.set_title(f'正常 vs 异常 趋势对比\n'
                     f'正常: {selected_normal_unit.get("trend_pattern", "?")} | '
                     f'异常: {selected_anomaly_unit.get("trend_pattern", "?")}',
                     fontsize=12, fontweight='bold')
        ax7.legend(loc='best', fontsize=10)
        ax7.grid(True, alpha=0.3)
        ax7.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.setp(ax7.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=9)
    else:
        ax7.text(0.5, 0.5, '无异常数据',
                ha='center', va='center', fontsize=16, color='gray')
        ax7.set_xlim(0, 1)
        ax7.set_ylim(0, 1)
        ax7.axis('off')

    # ==================== 8. 数据特征总结 ====================
    print("  - 生成数据特征总结...")
    ax8 = fig.add_subplot(gs[4, 1])
    ax8.axis('off')

    sampling_rate = n_points / time_span  # 点/小时

    summary_text = f"""
【原子钟遥测数据特征总结】

【基本信息】
  - 单元ID: {selected_normal_id}
  - 数据点数: {n_points:,} 点
  - 通道数: {n_channels} 个
  - 分段数: {n_segments} 段

【时间特征】
  - 起始时间: {all_datetimes[0].strftime('%Y-%m-%d %H:%M:%S')}
  - 结束时间: {all_datetimes[-1].strftime('%Y-%m-%d %H:%M:%S')}
  - 总时长: {time_span:.2f} 小时 ({time_span/24:.2f} 天)
  - 采样率: ~{sampling_rate:.1f} 点/小时
  - 采样间隔: {sampling_interval} 秒/点

【信号特征】
  - 趋势模式: {selected_normal_unit.get('trend_pattern', 'unknown')}
  - 漂移率: {selected_normal_unit.get('drift_rate', 0):.6f}
  - 噪声水平: {selected_normal_unit.get('noise_level', 0):.4f}
  - 信噪比: {abs(mean_val/std_val):.2f}

【数据质量】(通道{channel_idx})
  - 值域: [{min_val:.4f}, {max_val:.4f}]
  - 均值: {mean_val:.4f}
  - 标准差: {std_val:.4f}
  - 上升段: {positive_count}/{len(segment_trends)} ({positive_count/len(segment_trends)*100:.1f}%)
  - 下降段: {negative_count}/{len(segment_trends)} ({negative_count/len(segment_trends)*100:.1f}%)

【数据集规模】
  - 正常单元: {len(normal_data)} 个
  - 异常单元: {len(anomaly_data)} 个
    """

    ax8.text(0.05, 0.95, summary_text,
            transform=ax8.transAxes,
            verticalalignment='top',
            fontsize=9.5,
            **TEXT_FONT_KWARGS,
            bbox=dict(boxstyle='round', facecolor='#f0f9ff',
                     alpha=0.95, edgecolor='#3b82f6', linewidth=2.5))

    # 总标题
    fig.suptitle('原子钟遥测数据完整展示 - 基于真实Timestamp的连续时间轴',
                fontsize=18, fontweight='bold', y=0.995)

    # 保存图片
    print(f"\n[4/4] 保存可视化...")
    if output_path is None:
        output_path = Path(output_dir) / 'telemetry_data_complete_visualization.png'
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ 已保存至: {output_path}")

    print("\n" + "="*80)
    print("✓ 可视化完成！")
    print("="*80)

    plt.show()


def visualize_channel_feature_overview(normal_data_path, anomaly_data_path,
                                       start_datetime='2024-01-01 00:00:00',
                                       sampling_interval=10,
                                       normal_unit_id=None,
                                       anomaly_unit_id=None,
                                       output_path=None):
    """对比展示正常与异常单元的全部通道特征"""

    print("=" * 80)
    print("原子钟遥测数据通道特征总览")
    print("=" * 80)

    print(f"\n[1/3] 加载数据...")
    print(f"  正常数据: {normal_data_path}")
    print(f"  异常数据: {anomaly_data_path}")

    normal_data = load_pickle(normal_data_path)
    anomaly_data = load_pickle(anomaly_data_path) if Path(anomaly_data_path).exists() else {}

    normal_units = list(normal_data.keys())
    anomaly_units = list(anomaly_data.keys())

    if not normal_units:
        print("  ✗ 错误：无正常单元数据！")
        return

    if normal_unit_id and normal_unit_id in normal_data:
        selected_normal_id = normal_unit_id
    else:
        selected_normal_id = normal_units[0]

    selected_normal_unit = normal_data[selected_normal_id]
    normal_sequence, normal_datetimes = prepare_unit_sequence(
        selected_normal_unit, start_datetime, sampling_interval)
    normal_stats = compute_channel_statistics(normal_sequence)

    if anomaly_units:
        if anomaly_unit_id and anomaly_unit_id in anomaly_data:
            selected_anomaly_id = anomaly_unit_id
        else:
            selected_anomaly_id = anomaly_units[0]

        selected_anomaly_unit = anomaly_data[selected_anomaly_id]
        anomaly_sequence, anomaly_datetimes = prepare_unit_sequence(
            selected_anomaly_unit, start_datetime, sampling_interval)
        anomaly_stats = compute_channel_statistics(anomaly_sequence)
    else:
        selected_anomaly_id = None
        selected_anomaly_unit = None
        anomaly_sequence = None
        anomaly_datetimes = None
        anomaly_stats = []

    n_channels = 0
    if normal_sequence is not None:
        n_channels = normal_sequence.shape[1]
    elif anomaly_sequence is not None:
        n_channels = anomaly_sequence.shape[1]

    if n_channels == 0:
        print("  ✗ 错误：无可用于展示的通道数据！")
        return

    print(f"\n[2/3] 生成可视化... (共 {n_channels} 个通道)")

    colors = ['#ef4444', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6', '#ec4899', '#06b6d4']

    fig = plt.figure(figsize=(22, 3 * n_channels + 4))
    gs = fig.add_gridspec(n_channels + 1, 2,
                          height_ratios=[3] * n_channels + [1.6],
                          hspace=0.4, wspace=0.25)

    for ch in range(n_channels):
        color = colors[ch % len(colors)]

        ax_normal = fig.add_subplot(gs[ch, 0])
        title_normal = f'正常通道{ch} - 单元 {selected_normal_id}'
        if normal_sequence is not None:
            plot_channel_timeseries(
                ax_normal,
                normal_datetimes,
                normal_sequence,
                ch,
                normal_stats[ch],
                title_normal,
                color,
            )
        else:
            ax_normal.text(0.5, 0.5, '无正常数据', ha='center', va='center',
                           fontsize=12, color='gray')
            ax_normal.axis('off')

        ax_anomaly = fig.add_subplot(gs[ch, 1])
        if anomaly_sequence is not None and ch < anomaly_sequence.shape[1]:
            title_anomaly = (
                f'异常通道{ch} - 单元 {selected_anomaly_id} '
                f"({selected_anomaly_unit.get('anomaly_type', 'unknown')})"
            )
            plot_channel_timeseries(
                ax_anomaly,
                anomaly_datetimes,
                anomaly_sequence,
                ch,
                anomaly_stats[ch],
                title_anomaly,
                color,
            )
        else:
            ax_anomaly.text(0.5, 0.5, '无异常数据', ha='center', va='center',
                            fontsize=12, color='gray')
            ax_anomaly.axis('off')

    ax_summary_normal = fig.add_subplot(gs[-1, 0])
    ax_summary_normal.axis('off')
    normal_summary = build_dataset_summary(
        selected_normal_id, selected_normal_unit, normal_datetimes, normal_stats)
    ax_summary_normal.text(
        0.02, 0.95, normal_summary,
        transform=ax_summary_normal.transAxes,
        va='top', ha='left', fontsize=10,
        **TEXT_FONT_KWARGS,
        bbox=dict(boxstyle='round', facecolor='#f1f5f9',
                  edgecolor='#1d4ed8', linewidth=2, alpha=0.95)
    )

    ax_summary_anomaly = fig.add_subplot(gs[-1, 1])
    ax_summary_anomaly.axis('off')
    anomaly_summary = build_dataset_summary(
        selected_anomaly_id, selected_anomaly_unit, anomaly_datetimes, anomaly_stats)
    ax_summary_anomaly.text(
        0.02, 0.95, anomaly_summary,
        transform=ax_summary_anomaly.transAxes,
        va='top', ha='left', fontsize=10,
        **TEXT_FONT_KWARGS,
        bbox=dict(boxstyle='round', facecolor='#fef3c7',
                  edgecolor='#b45309', linewidth=2, alpha=0.95)
    )

    fig.suptitle('正常 vs 异常 单元通道特征概览', fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    print(f"\n[3/3] 保存图像...")
    if output_path is None:
        output_path = FILE_PATHS.get(
            'channel_overview_visualization',
            Path('figures') / 'telemetry_channel_overview.png'
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ 已保存至: {output_path}")

    print("\n" + "=" * 80)
    print("✓ 通道特征可视化完成！")
    print("=" * 80)

    plt.show()


def main():
    """主函数"""

    # 文件路径
    normal_data_path = FILE_PATHS['raw_normal_data']
    anomaly_data_path = FILE_PATHS['raw_anomaly_data']
    output_dir = Path(FILE_PATHS['data_visualization']).parent

    # 时间参数
    start_datetime = '2024-01-01 00:00:00'
    sampling_interval = 10  # 秒

    # 执行可视化
    visualize_complete_telemetry_data(
        normal_data_path=normal_data_path,
        anomaly_data_path=anomaly_data_path,
        output_dir=output_dir,
        start_datetime=start_datetime,
        sampling_interval=sampling_interval
    )

    visualize_channel_feature_overview(
        normal_data_path=normal_data_path,
        anomaly_data_path=anomaly_data_path,
        start_datetime=start_datetime,
        sampling_interval=sampling_interval
    )


if __name__ == "__main__":
    main()
