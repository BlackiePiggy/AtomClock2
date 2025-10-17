"""
步骤4: 构建SS-MST-VAE模型
自监督多尺度时序变分自编码器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import MODEL_CONFIG, FILE_PATHS
from utils import (save_pickle, load_pickle, print_step_header,
				   print_completion)


class MultiScaleTCN(nn.Module):
	"""多尺度时序卷积网络"""

	def __init__(self, input_channels, hidden_channels, kernel_sizes=[3, 5, 7]):
		super().__init__()

		self.branches = nn.ModuleList()
		for k_size in kernel_sizes:
			branch = nn.Sequential(
				nn.Conv1d(input_channels, hidden_channels, kernel_size=k_size,
						  padding=k_size // 2, dilation=1),
				nn.BatchNorm1d(hidden_channels),
				nn.ReLU(),
				nn.Conv1d(hidden_channels, hidden_channels, kernel_size=k_size,
						  padding=k_size // 2 * 2, dilation=2),
				nn.BatchNorm1d(hidden_channels),
				nn.ReLU()
			)
			self.branches.append(branch)

		self.fusion = nn.Conv1d(hidden_channels * len(kernel_sizes),
								hidden_channels, kernel_size=1)

	def forward(self, x):
		branch_outputs = [branch(x) for branch in self.branches]
		concatenated = torch.cat(branch_outputs, dim=1)
		fused = self.fusion(concatenated)
		return fused


class VAE_Encoder(nn.Module):
	"""VAE编码器 with MST-TCN"""

	def __init__(self, input_dim, hidden_dim, latent_dim):
		super().__init__()

		self.tcn = MultiScaleTCN(input_dim, hidden_dim)
		self.pool = nn.AdaptiveAvgPool1d(1)

		self.fc_mu = nn.Linear(hidden_dim, latent_dim)
		self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

	def forward(self, x):
		# x: (batch, seq_len, features) -> (batch, features, seq_len)
		x = x.transpose(1, 2)

		h = self.tcn(x)
		h = self.pool(h).squeeze(-1)

		mu = self.fc_mu(h)
		logvar = self.fc_logvar(h)

		return mu, logvar


class VAE_Decoder(nn.Module):
	"""VAE解码器"""

	def __init__(self, latent_dim, hidden_dim, output_dim, seq_len):
		super().__init__()

		self.seq_len = seq_len
		self.fc = nn.Linear(latent_dim, hidden_dim * seq_len)

		self.tcn = MultiScaleTCN(hidden_dim, hidden_dim)
		self.output_layer = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)

	def forward(self, z):
		h = self.fc(z)
		h = h.view(z.size(0), -1, self.seq_len)

		h = self.tcn(h)
		out = self.output_layer(h)

		return out.transpose(1, 2)


class SS_MST_VAE(nn.Module):
	"""自监督多尺度时序VAE"""

	def __init__(self, input_dim, hidden_dim, latent_dim, seq_len):
		super().__init__()

		self.encoder = VAE_Encoder(input_dim, hidden_dim, latent_dim)
		self.decoder = VAE_Decoder(latent_dim, hidden_dim, input_dim, seq_len)

		# TCL投影头
		self.projection_head = nn.Sequential(
			nn.Linear(latent_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, latent_dim)
		)

	def reparameterize(self, mu, logvar):
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(std)
		return mu + eps * std

	def forward(self, x):
		mu, logvar = self.encoder(x)
		z = self.reparameterize(mu, logvar)
		x_recon = self.decoder(z)

		# TCL投影
		z_proj = self.projection_head(z)

		return x_recon, mu, logvar, z, z_proj


def main():
	print_step_header(4, "构建SS-MST-VAE模型")

	# 加载数据以确定输入维度
	print("\n加载数据...")
	imputed_normal = load_pickle(FILE_PATHS['imputed_normal'])

	# 获取数据维度
	sample_features = list(imputed_normal.values())[0]
	input_dim = sample_features.shape[1]
	seq_len = sample_features.shape[0]

	print(f"\n数据维度:")
	print(f"  输入特征维度: {input_dim}")
	print(f"  序列长度: {seq_len}")

	# 创建模型
	print("\n创建模型...")
	model = SS_MST_VAE(
		input_dim=input_dim,
		hidden_dim=MODEL_CONFIG['hidden_dim'],
		latent_dim=MODEL_CONFIG['latent_dim'],
		seq_len=seq_len
	)

	# 打印模型信息
	print(f"\n模型架构:")
	print(f"  输入维度: {input_dim}")
	print(f"  隐藏维度: {MODEL_CONFIG['hidden_dim']}")
	print(f"  潜在维度: {MODEL_CONFIG['latent_dim']}")
	print(f"  序列长度: {seq_len}")
	print(f"  总参数量: {sum(p.numel() for p in model.parameters()):,}")

	# 测试前向传播
	print("\n测试模型...")
	test_input = torch.randn(1, seq_len, input_dim)
	with torch.no_grad():
		x_recon, mu, logvar, z, z_proj = model(test_input)

	print(f"  输入形状: {test_input.shape}")
	print(f"  重构输出: {x_recon.shape}")
	print(f"  潜在向量: {z.shape}")
	print(f"  投影向量: {z_proj.shape}")

	# 保存模型架构（未训练）
	print("\n保存模型架构...")
	torch.save({
		'model_state_dict': model.state_dict(),
		'model_config': {
			'input_dim': input_dim,
			'hidden_dim': MODEL_CONFIG['hidden_dim'],
			'latent_dim': MODEL_CONFIG['latent_dim'],
			'seq_len': seq_len
		}
	}, FILE_PATHS['trained_model'].replace('.pth', '_untrained.pth'))

	print_completion("模型构建")


if __name__ == "__main__":
	main()