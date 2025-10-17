"""
步骤1: 特征提取
从原始遥测段数据中提取统计和频域特征
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from config import FEATURE_CONFIG, FILE_PATHS
from utils import (save_pickle, load_pickle, print_step_header,
				   print_completion, print_data_info)


class PrognosticFeatureExtractor:
	"""基于帧的特征提取器"""

	def __init__(self, window_size=100):
		self.window_size = window_size

	def extract_frame_features(self, segment):
		"""从一个连续段提取特征"""
		features = []

		for ch in range(segment.shape[1]):
			data = segment[:, ch]

			# 时域特征
			features.extend([
				np.mean(data),
				np.std(data),
				np.sqrt(np.mean(data ** 2)),  # RMS
				np.var(data),
				stats.kurtosis(data), # 峰度
				stats.skew(data), # 偏度，数据分布的不对称性
				np.max(data) - np.min(data),  # 峰峰值
				np.percentile(data, 75) - np.percentile(data, 25)  # IQR
			])

			# 频域特征
			fft_vals = np.abs(fft(data))
			freqs = fftfreq(len(data))

			low_freq = np.sum(fft_vals[np.abs(freqs) < 0.1])
			mid_freq = np.sum(fft_vals[(np.abs(freqs) >= 0.1) & (np.abs(freqs) < 0.3)])
			high_freq = np.sum(fft_vals[np.abs(freqs) >= 0.3])

			features.extend([low_freq, mid_freq, high_freq, np.argmax(fft_vals)])

		return np.array(features)

	def process_all_segments(self, segments_list):
		"""处理所有连续段"""
		features_matrix = []
		for seg in segments_list:
			feat = self.extract_frame_features(seg)
			features_matrix.append(feat)
		return np.array(features_matrix)

	def visualize_features(self, raw_segment, features, save_path=None):
		"""可视化特征提取"""
		fig, axes = plt.subplots(3, 2, figsize=(15, 10))
		fig.suptitle('特征提取可视化', fontsize=16, fontweight='bold')

		# 原始信号
		axes[0, 0].plot(raw_segment)
		axes[0, 0].set_title('原始多维遥测信号')
		axes[0, 0].set_xlabel('时间步')
		axes[0, 0].set_ylabel('幅值')
		axes[0, 0].legend([f'Ch{i}' for i in range(raw_segment.shape[1])],
						  loc='upper right', fontsize=8)

		# FFT频谱
		for ch in range(min(3, raw_segment.shape[1])):
			fft_vals = np.abs(fft(raw_segment[:, ch]))
			freqs = fftfreq(len(raw_segment[:, ch]))
			axes[0, 1].plot(freqs[:len(freqs) // 2], fft_vals[:len(fft_vals) // 2])
		axes[0, 1].set_title('频域分析 (FFT)')
		axes[0, 1].set_xlabel('频率')
		axes[0, 1].set_ylabel('幅度')

		# 特征向量
		axes[1, 0].bar(range(len(features)), features)
		axes[1, 0].set_title('提取的特征向量')
		axes[1, 0].set_xlabel('特征索引')
		axes[1, 0].set_ylabel('特征值')

		# 时域特征热图
		feature_groups = features.reshape(raw_segment.shape[1], -1)
		time_features = feature_groups[:, :8]
		sns.heatmap(time_features, ax=axes[1, 1], cmap='viridis',
					xticklabels=['Mean', 'Std', 'RMS', 'Var', 'Kurt', 'Skew', 'P2P', 'IQR'],
					yticklabels=[f'Ch{i}' for i in range(feature_groups.shape[0])],
					cbar_kws={'label': '特征值'})
		axes[1, 1].set_title('时域特征热图')

		# 频域能量分布
		freq_features = feature_groups[:, 8:11]
		axes[2, 0].bar(range(len(freq_features.flatten())), freq_features.flatten())
		axes[2, 0].set_title('频域能量分布')
		axes[2, 0].set_xlabel('通道 x 频段')
		axes[2, 0].set_ylabel('能量')

		# 信号包络
		for ch in range(min(3, raw_segment.shape[1])):
			analytic_signal = signal.hilbert(raw_segment[:, ch])
			envelope = np.abs(analytic_signal)
			axes[2, 1].plot(envelope, alpha=0.7)
		axes[2, 1].set_title('信号包络线')
		axes[2, 1].set_xlabel('时间步')
		axes[2, 1].set_ylabel('包络幅值')

		plt.tight_layout()
		if save_path:
			plt.savefig(save_path, dpi=300, bbox_inches='tight')
		plt.show()


def main():
	print_step_header(1, "特征提取")

	# 加载原始数据
	print("\n加载原始数据...")
	normal_data = load_pickle(FILE_PATHS['raw_normal_data'])
	anomaly_data = load_pickle(FILE_PATHS['raw_anomaly_data'])

	# 创建特征提取器
	extractor = PrognosticFeatureExtractor(
		window_size=FEATURE_CONFIG['window_size']
	)

	# 提取正常单元特征
	print("\n提取正常单元特征...")
	normal_features = {}
	for unit_id, unit_data in normal_data.items():
		segments = unit_data['segments']
		features = extractor.process_all_segments(segments)
		normal_features[unit_id] = features
		print(f"  {unit_id}: {len(segments)} 段 -> 特征矩阵 {features.shape}")

	# 提取异常单元特征
	print("\n提取异常单元特征...")
	anomaly_features = {}
	for unit_id, unit_data in anomaly_data.items():
		segments = unit_data['segments']
		features = extractor.process_all_segments(segments)
		anomaly_features[unit_id] = features
		print(f"  {unit_id}: {len(segments)} 段 -> 特征矩阵 {features.shape}")

	# 可视化第一个正常单元
	print("\n生成可视化...")
	first_unit_id = list(normal_data.keys())[0]
	sample_segment = normal_data[first_unit_id]['segments'][0]
	sample_features = normal_features[first_unit_id][0]
	extractor.visualize_features(
		sample_segment,
		sample_features,
		save_path=FILE_PATHS['feature_visualization']
	)

	# 保存特征
	print("\n保存特征...")
	save_pickle(normal_features, FILE_PATHS['normal_features'])
	save_pickle(anomaly_features, FILE_PATHS['anomaly_features'])

	# 打印特征统计
	print_data_info(normal_features, "正常单元特征")

	print_completion("特征提取")


if __name__ == "__main__":
	main()