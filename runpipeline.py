"""
完整流程执行脚本
运行整个原子钟PHM框架的所有步骤
"""

import sys
import time
from datetime import datetime


def print_banner():
	"""打印启动横幅"""
	banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║     原子钟预后与健康管理(PHM)框架                             ║
    ║     Atomic Clock Prognostic Health Management Framework      ║
    ║                                                               ║
    ║     完整实现以下技术:                                         ║
    ║     ✓ 基于帧的特征工程与预后质量评估                          ║
    ║     ✓ 非周期性时序分解（HP/LOWESS/Savitzky-Golay等）         ║
    ║     ✓ GRU-D深度归因                                          ║
    ║     ✓ 多尺度时序卷积VAE (SS-MST-VAE)                         ║
    ║     ✓ 时间对比学习 (TCL)                                     ║
    ║     ✓ MAML元学习                                             ║
    ║     ✓ 综合异常检测与亚健康预警                                ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
	print(banner)


def run_step(step_name, step_module):
	"""运行单个步骤"""
	print(f"\n{'=' * 70}")
	print(f"开始执行: {step_name}")
	print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
	print(f"{'=' * 70}")

	start_time = time.time()

	try:
		# 动态导入并执行
		module = __import__(step_module)
		module.main()

		elapsed_time = time.time() - start_time
		print(f"\n✓ {step_name} 完成！耗时: {elapsed_time:.2f} 秒")
		return True

	except Exception as e:
		print(f"\n✗ {step_name} 失败！")
		print(f"错误信息: {str(e)}")
		import traceback
		traceback.print_exc()
		return False


def main():
	"""主函数"""
	print_banner()

	print("\n" + "=" * 70)
	print("PHM框架完整流程")
	print("=" * 70)

	# 定义所有步骤
	steps = [
		("步骤0: 数据生成", "step0_data_generation"),
		("步骤1: 特征提取", "step1_feature_extraction"),
		("步骤2: 质量评估", "step2_quality_assessment"),
		("步骤3: 时序分解", "step3_decomposition"),
		("步骤4: 构建模型", "step4_build_model"),
		("步骤5: 训练模型", "step5_train_model"),
		("步骤6: 潜在空间分析", "step6_latent_space"),
		("步骤7: 异常检测", "step7_anomaly_detection"),
	]

	# 询问运行模式
	print("\n请选择运行模式:")
	print("  1. 运行所有步骤（完整流程）")
	print("  2. 运行指定步骤")
	print("  3. 从某个步骤开始运行")

	try:
		choice = input("\n请输入选择 (1/2/3): ").strip()

		if choice == '1':
			# 运行所有步骤
			print("\n开始运行完整流程...")
			start_time = time.time()

			success_count = 0
			for step_name, step_module in steps:
				if run_step(step_name, step_module):
					success_count += 1
				else:
					print(f"\n⚠ 流程在 {step_name} 处中断")
					break

			total_time = time.time() - start_time

			print("\n" + "=" * 70)
			print("流程执行总结")
			print("=" * 70)
			print(f"  成功完成步骤: {success_count}/{len(steps)}")
			print(f"  总耗时: {total_time:.2f} 秒 ({total_time / 60:.2f} 分钟)")
			print("=" * 70)

		elif choice == '2':
			# 运行指定步骤
			print("\n可用步骤:")
			for i, (step_name, _) in enumerate(steps):
				print(f"  {i}: {step_name}")

			step_indices = input("\n请输入要运行的步骤编号（用逗号分隔，如 0,1,2）: ").strip()
			indices = [int(x.strip()) for x in step_indices.split(',')]

			for idx in indices:
				if 0 <= idx < len(steps):
					step_name, step_module = steps[idx]
					run_step(step_name, step_module)
				else:
					print(f"⚠ 无效的步骤编号: {idx}")

		elif choice == '3':
			# 从某个步骤开始
			print("\n可用步骤:")
			for i, (step_name, _) in enumerate(steps):
				print(f"  {i}: {step_name}")

			start_idx = int(input("\n请输入起始步骤编号: ").strip())

			if 0 <= start_idx < len(steps):
				print(f"\n从步骤 {start_idx} 开始运行...")
				start_time = time.time()

				success_count = 0
				for step_name, step_module in steps[start_idx:]:
					if run_step(step_name, step_module):
						success_count += 1
					else:
						print(f"\n⚠ 流程在 {step_name} 处中断")
						break

				total_time = time.time() - start_time
				print(f"\n总耗时: {total_time:.2f} 秒")
			else:
				print(f"⚠ 无效的步骤编号: {start_idx}")

		else:
			print("⚠ 无效的选择")

	except KeyboardInterrupt:
		print("\n\n⚠ 用户中断执行")
	except Exception as e:
		print(f"\n✗ 发生错误: {str(e)}")
		import traceback
		traceback.print_exc()

	print("\n" + "=" * 70)
	print("感谢使用原子钟PHM框架！")
	print("=" * 70 + "\n")


if __name__ == "__main__":
	main()