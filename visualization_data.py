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

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


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


def visualize_complete_telemetry_data(normal_data_path, anomaly_data_path,
                                     output_dir='figures',
                                     start_datetime='2024-01-01 00:00:00',
                                     sampling_interval=10):
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

    # 选择第一个正常单元作为主要展示对象
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
             fontsize=10, family='monospace')

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

📊 基本信息
  • 单元ID: {selected_normal_id}
  • 数据点数: {n_points:,} 点
  • 通道数: {n_channels} 个
  • 分段数: {n_segments} 段

⏱ 时间特征
  • 起始时间: {all_datetimes[0].strftime('%Y-%m-%d %H:%M:%S')}
  • 结束时间: {all_datetimes[-1].strftime('%Y-%m-%d %H:%M:%S')}
  • 总时长: {time_span:.2f} 小时 ({time_span/24:.2f} 天)
  • 采样率: ~{sampling_rate:.1f} 点/小时
  • 采样间隔: {sampling_interval} 秒/点

📈 信号特征
  • 趋势模式: {selected_normal_unit.get('trend_pattern', 'unknown')}
  • 漂移率: {selected_normal_unit.get('drift_rate', 0):.6f}
  • 噪声水平: {selected_normal_unit.get('noise_level', 0):.4f}
  • 信噪比: {abs(mean_val/std_val):.2f}

🔍 数据质量 (通道{channel_idx})
  • 值域: [{min_val:.4f}, {max_val:.4f}]
  • 均值: {mean_val:.4f}
  • 标准差: {std_val:.4f}
  • 上升段: {positive_count}/{len(segment_trends)} ({positive_count/len(segment_trends)*100:.1f}%)
  • 下降段: {negative_count}/{len(segment_trends)} ({negative_count/len(segment_trends)*100:.1f}%)

📦 数据集规模
  • 正常单元: {len(normal_data)} 个
  • 异常单元: {len(anomaly_data)} 个
    """

    ax8.text(0.05, 0.95, summary_text,
            transform=ax8.transAxes,
            verticalalignment='top',
            fontsize=9.5,
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='#f0f9ff',
                     alpha=0.95, edgecolor='#3b82f6', linewidth=2.5))

    # 总标题
    fig.suptitle('原子钟遥测数据完整展示 - 基于真实Timestamp的连续时间轴',
                fontsize=18, fontweight='bold', y=0.995)

    # 保存图片
    print(f"\n[4/4] 保存可视化...")
    output_path = Path(output_dir) / 'telemetry_data_complete_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ 已保存至: {output_path}")

    print("\n" + "="*80)
    print("✓ 可视化完成！")
    print("="*80)

    plt.show()


def main():
    """主函数"""

    # 文件路径（根据你的实际输出调整）
    normal_data_path = r'E:\projects\AtomClock\proj2\src\data\normal_units_raw.pkl'
    anomaly_data_path = r'E:\projects\AtomClock\proj2\src\data\anomaly_units_raw.pkl'
    output_dir = r'E:\projects\AtomClock\proj2\src\figures'

    # 或者使用相对路径
    # normal_data_path = 'data/normal_units_raw.pkl'
    # anomaly_data_path = 'data/anomaly_units_raw.pkl'
    # output_dir = 'figures'

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


if __name__ == "__main__":
    main()