"""
步骤3: 时间序列分解
使用多种方法分解时序数据（趋势+残差，无季节性）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess
from config import DECOMPOSE_CONFIG, FILE_PATHS
from utils import (save_pickle, load_pickle, print_step_header,
				   print_completion, print_data_info)


class TimeSeriesDecomposer:
	"""时间序列分解器（专为非周期性数据设计）"""

	def __init__(self, method='hp', **kwargs):
		self.method = method
		self.kwargs = kwargs

	def hp_filter(self, time_series, lamb=1600):
		"""Hodrick-Prescott滤波器"""
		n = len(time_series)
		D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n))
		I = sparse.eye(n)
		A = I + lamb * D.T @ D
		trend = spsolve(A, time_series)
		return trend

	def lowess_smooth(self, time_series, frac=0.1):
		"""LOWESS局部加权回归"""
		x = np.arange(len(time_series))
		smoothed = lowess(time_series, x, frac=frac, return_sorted=False)
		return smoothed

	def savgol_filter_method(self, time_series, window_length=51, polyorder=3):
		"""Savitzky-Golay滤波器"""
		if window_length % 2 == 0:
			window_length += 1
		window_length = min(window_length, len(time_series))
		if window_length <= polyorder:
			window_length = polyorder + 2
			if window_length % 2 == 0:
				window_length += 1
		trend = savgol_filter(time_series, window_length, polyorder)
		return trend

	def moving_average(self, time_series, window=50):
		"""简单移动平均"""
		trend = pd.Series(time_series).rolling(
			window=window, center=True, min_periods=1
		).mean().values
		return trend

	def ewma_smooth(self, time_series, alpha=0.1):
		"""指数加权移动平均"""
		trend = pd.Series(time_series).ewm(alpha=alpha, adjust=False).mean().values
		return trend

	def decompose(self, time_series):
		"""分解时间序列"""
		if self.method == 'hp':
			lamb = self.kwargs.get('lamb', DECOMPOSE_CONFIG['hp_lambda'])
			trend = self.hp_filter(time_series, lamb=lamb)
		elif self.method == 'lowess':
			frac = self.kwargs.get('frac', DECOMPOSE_CONFIG['lowess_frac'])
			trend = self.lowess_smooth(time_series, frac=frac)
		elif self.method == 'savgol':
			window_length = self.kwargs.get('window_length',
											DECOMPOSE_CONFIG['savgol_window'])
			polyorder = self.kwargs.get('polyorder',
										DECOMPOSE_CONFIG['savgol_polyorder'])
			trend = self.savgol_filter_method(time_series, window_length, polyorder)
		elif self.method == 'ma':
			window = self.kwargs.get('window', DECOMPOSE_CONFIG['ma_window'])
			trend = self.moving_average(time_series, window=window)
		elif self.method == 'ewma':
			alpha = self.kwargs.get('alpha', DECOMPOSE_CONFIG['ewma_alpha'])
			trend = self.ewma_smooth(time_series, alpha=alpha)
		else:
			raise ValueError(f"未知方法: {self.method}")

		residual = time_series - trend
		return trend, residual

	def visualize_decomposition(self, time_series, save_path=None):
		"""可视化分解结果"""
		trend, residual = self.decompose(time_series)
		timestamps = np.arange(len(time_series))

		fig, axes = plt.subplots(3, 2, figsize=(16, 10))
		method_names = {
			'hp': 'Hodrick-Prescott滤波',
			'lowess': 'LOWESS局部加权回归',
			'savgol': 'Savitzky-Golay滤波',
			'ma': '移动平均',
			'ewma': '指数加权移动平均'
		}
		fig.suptitle(f'时间序列分解 ({method_names.get(self.method, self.method)})',
					 fontsize=16, fontweight='bold')

		# 原始序列
		axes[0, 0].plot(timestamps, time_series, 'b-', linewidth=1.5, alpha=0.7)
		axes[0, 0].set_ylabel('幅值')
		axes[0, 0].set_title('原始遥测序列')
		axes[0, 0].grid(True, alpha=0.3)

		# 趋势分量
		axes[0, 1].plot(timestamps, trend, 'r-', linewidth=2.5)
		axes[0, 1].set_ylabel('趋势值')
		axes[0, 1].set_title('长期退化趋势')
		axes[0, 1].grid(True, alpha=0.3)

		# 叠加对比
		axes[1, 0].plot(timestamps, time_series, 'b-', linewidth=1, alpha=0.5, label='原始')
		axes[1, 0].plot(timestamps, trend, 'r-', linewidth=2, label='趋势')
		axes[1, 0].set_ylabel('幅值')
		axes[1, 0].set_title('原始信号与趋势对比')
		axes[1, 0].legend()
		axes[1, 0].grid(True, alpha=0.3)

		# 残差
		axes[1, 1].plot(timestamps, residual, color='gray', linewidth=0.8, alpha=0.7)
		axes[1, 1].axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
		axes[1, 1].set_ylabel('残差')
		axes[1, 1].set_title('残差（去趋势后噪声）')
		axes[1, 1].grid(True, alpha=0.3)

		# 残差分布
		axes[2, 0].hist(residual, bins=50, color='gray', edgecolor='black', alpha=0.7)
		axes[2, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
		axes[2, 0].set_xlabel('残差值')
		axes[2, 0].set_ylabel('频数')
		axes[2, 0].set_title(f'残差分布 (μ={np.mean(residual):.4f}, σ={np.std(residual):.4f})')
		axes[2, 0].grid(True, alpha=0.3, axis='y')

		# 趋势变化率
		trend_diff = np.diff(trend)
		axes[2, 1].plot(timestamps[1:], trend_diff, 'purple', linewidth=1.5)
		axes[2, 1].axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
		axes[2, 1].set_xlabel('时间')
		axes[2, 1].set_ylabel('趋势变化率')
		axes[2, 1].set_title('退化速度（趋势导数）')
		axes[2, 1].grid(True, alpha=0.3)

		plt.tight_layout()
		if save_path:
			plt.savefig(save_path, dpi=300, bbox_inches='tight')
		plt.show()

		# 打印统计
		print(f"\n分解统计:")
		print(f"  原始信号: 均值={np.mean(time_series):.4f}, 标准差={np.std(time_series):.4f}")
		print(f"  趋势分量: 均值={np.mean(trend):.4f}, 标准差={np.std(trend):.4f}")
		print(f"  残差分量: 均值={np.mean(residual):.4f}, 标准差={np.std(residual):.4f}")
		print(f"  信噪比: {np.std(trend) / (np.std(residual) + 1e-8):.2f}")


def main():
	print_step_header(3, "时间序列分解")

	# 加载特征
	print("\n加载筛选后的特征...")
	filtered_normal = load_pickle(FILE_PATHS['filtered_normal_features'])
	filtered_anomaly = load_pickle(FILE_PATHS['filtered_anomaly_features'])

	# 创建分解器
	method = DECOMPOSE_CONFIG['method']
	print(f"\n使用分解方法: {method.upper()}")
	decomposer = TimeSeriesDecomposer(method=method)

	# 分解正常单元
	print("\n分解正常单元特征...")
	imputed_normal = {}
	for unit_id, features in filtered_normal.items():
		n_features = features.shape[1]
		imputed_unit = np.zeros_like(features)

		for feat_idx in range(n_features):
			time_series = features[:, feat_idx]
			trend, residual = decomposer.decompose(time_series)
			imputed_unit[:, feat_idx] = trend + residual

		imputed_normal[unit_id] = imputed_unit
	print(f"  ✓ 完成 {len(imputed_normal)} 个正常单元")

	# 分解异常单元
	print("\n分解异常单元特征...")
	imputed_anomaly = {}
	for unit_id, features in filtered_anomaly.items():
		n_features = features.shape[1]
		imputed_unit = np.zeros_like(features)

		for feat_idx in range(n_features):
			time_series = features[:, feat_idx]
			trend, residual = decomposer.decompose(time_series)
			imputed_unit[:, feat_idx] = trend + residual

		imputed_anomaly[unit_id] = imputed_unit
	print(f"  ✓ 完成 {len(imputed_anomaly)} 个异常单元")

	# 可视化第一个正常单元的第一个特征
	print("\n生成可视化...")
	first_unit_id = list(filtered_normal.keys())[0]
	sample_series = filtered_normal[first_unit_id][:, 0]
	decomposer.visualize_decomposition(
		sample_series,
		save_path=FILE_PATHS['decompose_visualization']
	)

	# 保存结果
	print("\n保存分解结果...")
	save_pickle(imputed_normal, FILE_PATHS['imputed_normal'])
	save_pickle(imputed_anomaly, FILE_PATHS['imputed_anomaly'])

	print_completion("时间序列分解")


if __name__ == "__main__":
	main()