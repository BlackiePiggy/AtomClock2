"""
步骤2: 预后特征质量评估
评估特征的单调性、趋势性、可预示性，选择高质量特征
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import FEATURE_CONFIG, FILE_PATHS
from utils import (save_pickle, load_pickle, print_step_header,
				   print_completion, print_data_info)


class PrognosticQualityMetrics:
	"""预后特征质量评估"""

	@staticmethod
	def monotonicity(trajectory):
		"""计算单调性指标"""
		diffs = np.diff(trajectory)
		if len(diffs) == 0:
			return 0

		positive = np.sum(diffs > 0)
		negative = np.sum(diffs < 0)
		total = len(diffs)

		if total == 0:
			return 0

		return (positive - negative) / total

	@staticmethod
	def trendability(trajectories_list):
		"""计算趋势性：多个单元间轨迹相似度"""
		if len(trajectories_list) < 2:
			return 0

		# 归一化
		normalized = []
		for traj in trajectories_list:
			if np.max(traj) - np.min(traj) > 0:
				norm_traj = (traj - np.min(traj)) / (np.max(traj) - np.min(traj))
			else:
				norm_traj = traj
			normalized.append(norm_traj)

		# 计算相关系数
		correlations = []
		for i in range(len(normalized)):
			for j in range(i + 1, len(normalized)):
				len_i, len_j = len(normalized[i]), len(normalized[j])
				if len_i != len_j:
					x_common = np.linspace(0, 1, max(len_i, len_j))
					traj_i = np.interp(x_common, np.linspace(0, 1, len_i), normalized[i])
					traj_j = np.interp(x_common, np.linspace(0, 1, len_j), normalized[j])
				else:
					traj_i, traj_j = normalized[i], normalized[j]

				corr = np.corrcoef(traj_i, traj_j)[0, 1]
				if not np.isnan(corr):
					correlations.append(corr)

		return np.mean(correlations) if correlations else 0

	@staticmethod
	def prognosability(trajectories_list):
		"""可预示性：终点变异相对于总变化范围"""
		if len(trajectories_list) < 2:
			return 0

		final_vals = [traj[-1] for traj in trajectories_list]
		final_std = np.std(final_vals)

		all_vals = np.concatenate(trajectories_list)
		total_range = np.max(all_vals) - np.min(all_vals)

		if total_range > 0:
			prog = 1 - (final_std / total_range)
		else:
			prog = 0

		return max(0, prog)

	def evaluate_all_features(self, features_by_unit):
		"""评估所有特征"""
		n_features = list(features_by_unit.values())[0].shape[1]
		results = []

		for feat_idx in range(n_features):
			trajectories = [features_by_unit[unit][:, feat_idx]
							for unit in features_by_unit.keys()]

			mono_scores = [self.monotonicity(traj) for traj in trajectories]
			mono_avg = np.mean(np.abs(mono_scores))

			trend_score = self.trendability(trajectories)
			prog_score = self.prognosability(trajectories)

			results.append({
				'Feature_Index': feat_idx,
				'Monotonicity': mono_avg,
				'Trendability': trend_score,
				'Prognosability': prog_score,
				'Combined_Score': (mono_avg + trend_score + prog_score) / 3
			})

		return pd.DataFrame(results).sort_values('Combined_Score', ascending=False)

	def visualize_quality(self, quality_df, top_k=10, save_path=None):
		"""可视化质量评估结果"""
		fig, axes = plt.subplots(2, 2, figsize=(16, 12))
		fig.suptitle('预后特征质量评估', fontsize=16, fontweight='bold')

		top_features = quality_df.head(top_k)

		# 综合得分排名
		axes[0, 0].barh(range(len(top_features)), top_features['Combined_Score'].values)
		axes[0, 0].set_yticks(range(len(top_features)))
		axes[0, 0].set_yticklabels([f'Feat_{idx}' for idx in top_features['Feature_Index']])
		axes[0, 0].set_xlabel('综合得分')
		axes[0, 0].set_title(f'Top-{top_k} 预后特征排名')
		axes[0, 0].invert_yaxis()

		# 3D散点图
		ax_3d = fig.add_subplot(2, 2, 2, projection='3d')
		scatter = ax_3d.scatter(quality_df['Monotonicity'],
								quality_df['Trendability'],
								quality_df['Prognosability'],
								c=quality_df['Combined_Score'],
								cmap='viridis', s=50)
		ax_3d.set_xlabel('单调性')
		ax_3d.set_ylabel('趋势性')
		ax_3d.set_zlabel('可预示性')
		ax_3d.set_title('三维特征质量空间')
		plt.colorbar(scatter, ax=ax_3d, label='综合得分')

		# 雷达图
		top5 = quality_df.head(5)
		categories = ['单调性', '趋势性', '可预示性']
		angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
		angles += angles[:1]

		ax_radar = plt.subplot(2, 2, 3, projection='polar')
		for idx, row in top5.iterrows():
			values = [row['Monotonicity'], row['Trendability'], row['Prognosability']]
			values += values[:1]
			ax_radar.plot(angles, values, 'o-', linewidth=2,
						  label=f"Feat_{row['Feature_Index']}")
			ax_radar.fill(angles, values, alpha=0.15)
		ax_radar.set_xticks(angles[:-1])
		ax_radar.set_xticklabels(categories)
		ax_radar.set_ylim(0, 1)
		ax_radar.set_title('Top-5 特征雷达图')
		ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

		# 热图
		metrics_matrix = quality_df[['Monotonicity', 'Trendability', 'Prognosability']].head(20).T
		sns.heatmap(metrics_matrix, ax=axes[1, 1], cmap='RdYlGn',
					xticklabels=[f'F{i}' for i in quality_df.head(20)['Feature_Index']],
					yticklabels=['单调性', '趋势性', '可预示性'],
					cbar_kws={'label': '得分'}, vmin=0, vmax=1)
		axes[1, 1].set_title('Top-20 特征质量热图')

		plt.tight_layout()
		if save_path:
			plt.savefig(save_path, dpi=300, bbox_inches='tight')
		plt.show()


def main():
	print_step_header(2, "预后特征质量评估")

	# 加载特征
	print("\n加载特征数据...")
	normal_features = load_pickle(FILE_PATHS['normal_features'])
	anomaly_features = load_pickle(FILE_PATHS['anomaly_features'])

	# 创建评估器
	evaluator = PrognosticQualityMetrics()

	# 评估特征质量
	print("\n评估特征质量...")
	quality_df = evaluator.evaluate_all_features(normal_features)

	print(f"\nTop-10 预后特征:")
	print(quality_df.head(10).to_string(index=False))

	# 可视化
	print("\n生成可视化...")
	evaluator.visualize_quality(
		quality_df,
		top_k=FEATURE_CONFIG['top_k_features'],
		save_path=FILE_PATHS['quality_visualization']
	)

	# 选择高质量特征
	print(f"\n选择 Top-{FEATURE_CONFIG['top_k_features']} 特征...")
	selected_indices = quality_df.head(FEATURE_CONFIG['top_k_features'])['Feature_Index'].values

	filtered_normal = {uid: feat[:, selected_indices]
					   for uid, feat in normal_features.items()}
	filtered_anomaly = {uid: feat[:, selected_indices]
						for uid, feat in anomaly_features.items()}

	print(f"  原始特征维度: {list(normal_features.values())[0].shape[1]}")
	print(f"  筛选后维度: {list(filtered_normal.values())[0].shape[1]}")

	# 保存结果
	print("\n保存结果...")
	quality_df.to_csv(FILE_PATHS['quality_df'], index=False)
	save_pickle(filtered_normal, FILE_PATHS['filtered_normal_features'])
	save_pickle(filtered_anomaly, FILE_PATHS['filtered_anomaly_features'])

	print_completion("特征质量评估")


if __name__ == "__main__":
	main()