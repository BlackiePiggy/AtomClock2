"""
步骤6: 潜在空间轨迹提取与分析
提取并可视化潜在空间中的退化轨迹
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from config import FILE_PATHS
from utils import (save_pickle, load_pickle, print_step_header, print_completion)
from step4_build_model import SS_MST_VAE


def visualize_latent_trajectory(latent_vectors, time_indices, unit_labels, save_path=None):
	"""可视化潜在空间轨迹"""
	fig = plt.figure(figsize=(18, 12))
	fig.suptitle('潜在空间退化轨迹分析', fontsize=16, fontweight='bold')

	# PCA降维
	pca_2d = PCA(n_components=2)
	pca_3d = PCA(n_components=3)

	latent_2d = pca_2d.fit_transform(latent_vectors)
	latent_3d = pca_3d.fit_transform(latent_vectors)

	# 2D轨迹图
	ax1 = plt.subplot(2, 3, 1)
	unique_units = np.unique(unit_labels)
	colors = plt.cm.tab10(np.linspace(0, 1, len(unique_units)))

	for i, unit in enumerate(unique_units):
		mask = unit_labels == unit
		ax1.plot(latent_2d[mask, 0], latent_2d[mask, 1],
				 'o-', color=colors[i], label=f'Unit {unit}',
				 markersize=4, linewidth=1.5, alpha=0.7)
	ax1.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})')
	ax1.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})')
	ax1.set_title('2D潜在空间轨迹')
	ax1.legend(loc='best', fontsize=8)
	ax1.grid(True, alpha=0.3)

	# 3D轨迹图
	ax2 = fig.add_subplot(2, 3, 2, projection='3d')
	for i, unit in enumerate(unique_units):
		mask = unit_labels == unit
		ax2.plot(latent_3d[mask, 0], latent_3d[mask, 1], latent_3d[mask, 2],
				 'o-', color=colors[i], label=f'Unit {unit}',
				 markersize=3, linewidth=1.5, alpha=0.7)
	ax2.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})')
	ax2.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})')
	ax2.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})')
	ax2.set_title('3D潜在空间轨迹')
	ax2.legend(loc='best', fontsize=7)

	# 时间演进热图
	ax3 = plt.subplot(2, 3, 3)
	scatter = ax3.scatter(latent_2d[:, 0], latent_2d[:, 1],
						  c=time_indices, cmap='plasma', s=30, alpha=0.6)
	ax3.set_xlabel('PC1')
	ax3.set_ylabel('PC2')
	ax3.set_title('时间演进色彩编码')
	plt.colorbar(scatter, ax=ax3, label='时间步')

	# 潜在维度时序图
	ax4 = plt.subplot(2, 3, 4)
	for dim in range(min(5, latent_vectors.shape[1])):
		ax4.plot(time_indices, latent_vectors[:, dim],
				 label=f'Latent Dim {dim}', alpha=0.7, linewidth=1)
	ax4.set_xlabel('时间')
	ax4.set_ylabel('潜在向量值')
	ax4.set_title('潜在维度时序演进')
	ax4.legend(loc='best', fontsize=8)
	ax4.grid(True, alpha=0.3)

	# 方差解释率
	ax5 = plt.subplot(2, 3, 5)
	n_components = min(10, latent_vectors.shape[1])
	pca_full = PCA(n_components=n_components)
	pca_full.fit(latent_vectors)
	ax5.bar(range(n_components), pca_full.explained_variance_ratio_)
	ax5.set_xlabel('主成分')
	ax5.set_ylabel('方差解释率')
	ax5.set_title('PCA方差解释率')
	ax5.grid(True, alpha=0.3, axis='y')

	# 轨迹长度分析
	ax6 = plt.subplot(2, 3, 6)
	trajectory_lengths = []
	for unit in unique_units:
		mask = unit_labels == unit
		unit_trajectory = latent_2d[mask]
		length = np.sum(np.sqrt(np.sum(np.diff(unit_trajectory, axis=0) ** 2, axis=1)))
		trajectory_lengths.append(length)

	ax6.bar(range(len(unique_units)), trajectory_lengths, color=colors, alpha=0.7)
	ax6.set_xlabel('单元ID')
	ax6.set_ylabel('轨迹长度')
	ax6.set_title('潜在空间轨迹长度')
	ax6.set_xticks(range(len(unique_units)))
	ax6.set_xticklabels([f'U{u}' for u in unique_units])
	ax6.grid(True, alpha=0.3, axis='y')

	plt.tight_layout()
	if save_path:
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
	plt.show()


def main():
	print_step_header(6, "潜在空间轨迹提取")

	# 加载模型
	print("\n加载训练好的模型...")
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
	print("  ✓ 模型加载成功")

	# 加载数据
	print("\n加载数据...")
	imputed_normal = load_pickle(FILE_PATHS['imputed_normal'])

	# 提取潜在轨迹
	print("\n提取潜在空间轨迹...")
	all_latents = []
	all_time_indices = []
	all_unit_labels = []

	with torch.no_grad():
		for unit_idx, (unit_id, features) in enumerate(imputed_normal.items()):
			x = torch.FloatTensor(features).unsqueeze(0)
			seq_len = len(features)
			unit_latents = []

			# 使用滑动窗口提取局部潜在表示
			if seq_len <= 10:
				# 序列短，直接使用全局潜在向量
				_, _, _, z, _ = model(x)
				latent = z.squeeze(0).numpy()
				for t in range(seq_len):
					unit_latents.append(latent)
			else:
				# 使用滑动窗口
				window_size = min(50, seq_len // 3)
				for t in range(seq_len):
					start_idx = max(0, t - window_size // 2)
					end_idx = min(seq_len, t + window_size // 2 + 1)

					window_data = features[start_idx:end_idx]
					x_window = torch.FloatTensor(window_data).unsqueeze(0)

					_, _, _, z_window, _ = model(x_window)
					latent_window = z_window.squeeze(0).numpy()
					unit_latents.append(latent_window)

			unit_latents = np.array(unit_latents)
			all_latents.append(unit_latents)
			all_time_indices.extend(range(seq_len))
			all_unit_labels.extend([unit_idx] * seq_len)

			print(f"  {unit_id}: 提取 {len(unit_latents)} 个时间步的潜在向量")

	all_latents = np.vstack(all_latents)
	all_time_indices = np.array(all_time_indices)
	all_unit_labels = np.array(all_unit_labels)

	print(f"\n潜在轨迹统计:")
	print(f"  总形状: {all_latents.shape}")
	print(f"  时间范围: {all_time_indices.min()} - {all_time_indices.max()}")
	print(f"  单元数: {len(np.unique(all_unit_labels))}")

	# 可视化
	print("\n生成可视化...")
	visualize_latent_trajectory(
		all_latents,
		all_time_indices,
		all_unit_labels,
		save_path=FILE_PATHS['latent_visualization']
	)

	# 保存结果
	print("\n保存潜在轨迹...")
	latent_data = {
		'latent_vectors': all_latents,
		'time_indices': all_time_indices,
		'unit_labels': all_unit_labels
	}
	save_pickle(latent_data, FILE_PATHS['latent_trajectories'])

	print_completion("潜在空间轨迹提取")


if __name__ == "__main__":
	main()