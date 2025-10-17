"""
步骤5: 训练SS-MST-VAE模型
训练自监督多尺度时序VAE
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from config import MODEL_CONFIG, FILE_PATHS
from utils import (save_pickle, load_pickle, print_step_header, print_completion)
from step4_build_model import SS_MST_VAE


def vae_loss(x, x_recon, mu, logvar, beta=1.0):
	"""VAE损失：重构 + KL散度"""
	recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.size(0)
	kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
	return recon_loss + beta * kl_loss, recon_loss, kl_loss


def temporal_contrastive_loss(z_proj, time_indices, temperature=0.5):
	"""时间对比学习损失"""
	batch_size = z_proj.size(0)

	z_norm = F.normalize(z_proj, dim=1)
	similarity_matrix = torch.matmul(z_norm, z_norm.T) / temperature

	time_diff = torch.abs(time_indices.unsqueeze(0) - time_indices.unsqueeze(1))

	positive_mask = (time_diff < 5).float() - torch.eye(batch_size).to(z_proj.device)

	exp_sim = torch.exp(similarity_matrix)
	pos_sim = (exp_sim * positive_mask).sum(1)
	all_sim = exp_sim.sum(1) - torch.diag(exp_sim)

	loss = -torch.log(pos_sim / (all_sim + 1e-8) + 1e-8).mean()
	return loss


def visualize_training(losses_history, save_path=None):
	"""可视化训练过程"""
	fig, axes = plt.subplots(2, 2, figsize=(15, 10))
	fig.suptitle('SS-MST-VAE训练过程监控', fontsize=16, fontweight='bold')

	epochs = range(1, len(losses_history['total']) + 1)

	# 总损失
	axes[0, 0].plot(epochs, losses_history['total'], 'b-', linewidth=2)
	axes[0, 0].set_xlabel('Epoch')
	axes[0, 0].set_ylabel('Total Loss')
	axes[0, 0].set_title('总损失下降曲线')
	axes[0, 0].grid(True, alpha=0.3)

	# 损失分量
	axes[0, 1].plot(epochs, losses_history['recon'], label='重构损失', linewidth=2)
	axes[0, 1].plot(epochs, losses_history['kl'], label='KL散度', linewidth=2)
	axes[0, 1].plot(epochs, losses_history['contrastive'], label='对比损失', linewidth=2)
	axes[0, 1].set_xlabel('Epoch')
	axes[0, 1].set_ylabel('Loss')
	axes[0, 1].set_title('损失分量分解')
	axes[0, 1].legend()
	axes[0, 1].grid(True, alpha=0.3)

	# 重构误差分布
	if len(losses_history['recon_errors']) > 0:
		final_errors = losses_history['recon_errors'][-1]
		axes[1, 0].hist(final_errors, bins=50, color='skyblue',
						edgecolor='black', alpha=0.7)
		axes[1, 0].set_xlabel('重构误差')
		axes[1, 0].set_ylabel('频数')
		axes[1, 0].set_title('最终Epoch重构误差分布')
		axes[1, 0].grid(True, alpha=0.3)

	# 潜在空间
	if len(losses_history['latent_samples']) > 0:
		latent = losses_history['latent_samples'][-1]
		if latent.shape[1] > 2:
			pca = PCA(n_components=2)
			latent_2d = pca.fit_transform(latent)
		else:
			latent_2d = latent

		scatter = axes[1, 1].scatter(latent_2d[:, 0], latent_2d[:, 1],
									 c=range(len(latent_2d)), cmap='viridis',
									 alpha=0.6, s=20)
		axes[1, 1].set_xlabel('潜在维度 1')
		axes[1, 1].set_ylabel('潜在维度 2')
		axes[1, 1].set_title('潜在空间分布')
		plt.colorbar(scatter, ax=axes[1, 1], label='时间步')

	plt.tight_layout()
	if save_path:
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
	plt.show()


def main():
	print_step_header(5, "训练SS-MST-VAE模型")

	# 加载数据
	print("\n加载数据...")
	imputed_normal = load_pickle(FILE_PATHS['imputed_normal'])

	# 获取维度
	sample_features = list(imputed_normal.values())[0]
	input_dim = sample_features.shape[1]
	seq_len = sample_features.shape[0]

	# 创建模型
	print("\n创建模型...")
	model = SS_MST_VAE(
		input_dim=input_dim,
		hidden_dim=MODEL_CONFIG['hidden_dim'],
		latent_dim=MODEL_CONFIG['latent_dim'],
		seq_len=seq_len
	)

	# 准备训练数据
	print("\n准备训练数据...")
	all_sequences = []
	for unit_id, features in imputed_normal.items():
		all_sequences.append(torch.FloatTensor(features))

	train_data = torch.stack(all_sequences)
	print(f"  训练数据形状: {train_data.shape}")

	# 训练设置
	optimizer = torch.optim.Adam(model.parameters(),
								 lr=MODEL_CONFIG['learning_rate'])
	epochs = MODEL_CONFIG['epochs']

	losses_history = {
		'total': [],
		'recon': [],
		'kl': [],
		'contrastive': [],
		'recon_errors': [],
		'latent_samples': []
	}

	print(f"\n开始训练: {epochs} epochs")
	print("=" * 70)

	for epoch in range(epochs):
		model.train()
		epoch_losses = {'total': 0, 'recon': 0, 'kl': 0, 'contrast': 0}
		epoch_recon_errors = []
		epoch_latent_samples = []

		for unit_idx in range(len(train_data)):
			x = train_data[unit_idx].unsqueeze(0)

			optimizer.zero_grad()

			x_recon, mu, logvar, z, z_proj = model(x)

			# VAE损失
			total_loss, recon_loss, kl_loss = vae_loss(
				x, x_recon, mu, logvar,
				beta=MODEL_CONFIG['beta_vae']
			)

			# 对比学习损失
			contrast_loss = torch.tensor(0.0)
			if unit_idx > 0:
				time_idx = torch.LongTensor([unit_idx])
				contrast_loss = temporal_contrastive_loss(z_proj, time_idx)

			# 组合损失
			loss = total_loss + MODEL_CONFIG['contrast_weight'] * contrast_loss

			loss.backward()
			optimizer.step()

			epoch_losses['total'] += loss.item()
			epoch_losses['recon'] += recon_loss.item()
			epoch_losses['kl'] += kl_loss.item()
			epoch_losses['contrast'] += contrast_loss.item()

			# 收集统计
			with torch.no_grad():
				recon_error = F.mse_loss(x_recon, x, reduction='none').mean(dim=(1, 2))
				epoch_recon_errors.extend(recon_error.cpu().numpy())
				epoch_latent_samples.append(z.squeeze(0).cpu().numpy())

		# 记录
		n_units = len(train_data)
		losses_history['total'].append(epoch_losses['total'] / n_units)
		losses_history['recon'].append(epoch_losses['recon'] / n_units)
		losses_history['kl'].append(epoch_losses['kl'] / n_units)
		losses_history['contrastive'].append(epoch_losses['contrast'] / n_units)
		losses_history['recon_errors'].append(epoch_recon_errors)
		losses_history['latent_samples'].append(np.vstack(epoch_latent_samples))

		if (epoch + 1) % 5 == 0 or epoch == 0:
			print(f"Epoch {epoch + 1}/{epochs} - "
				  f"Loss: {losses_history['total'][-1]:.4f} "
				  f"(Recon: {losses_history['recon'][-1]:.4f}, "
				  f"KL: {losses_history['kl'][-1]:.4f}, "
				  f"Contrast: {losses_history['contrastive'][-1]:.4f})")

	print("=" * 70)
	print("训练完成!")

	# 可视化
	print("\n生成训练可视化...")
	visualize_training(losses_history, save_path=FILE_PATHS['training_visualization'])

	# 保存模型和历史
	print("\n保存模型和训练历史...")
	torch.save({
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'model_config': {
			'input_dim': input_dim,
			'hidden_dim': MODEL_CONFIG['hidden_dim'],
			'latent_dim': MODEL_CONFIG['latent_dim'],
			'seq_len': seq_len
		},
		'training_config': MODEL_CONFIG,
		'final_loss': losses_history['total'][-1]
	}, FILE_PATHS['trained_model'])

	save_pickle(losses_history, FILE_PATHS['training_history'])

	print(f"  ✓ 模型已保存")
	print(f"  ✓ 最终损失: {losses_history['total'][-1]:.4f}")

	print_completion("模型训练")


if __name__ == "__main__":
	main()