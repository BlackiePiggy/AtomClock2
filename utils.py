"""
通用工具函数
Common utility functions for Atomic Clock PHM Framework
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def save_pickle(data, filepath):
    """保存数据为pickle格式"""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"✓ 数据已保存: {filepath}")


def load_pickle(filepath):
    """加载pickle格式数据"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"✓ 数据已加载: {filepath}")
    return data


def print_step_header(step_number, step_name):
    """打印步骤标题"""
    print("\n" + "="*70)
    print(f"步骤 {step_number}: {step_name}")
    print("="*70)


def print_completion(step_name):
    """打印完成信息"""
    print("\n" + "✓"*70)
    print(f"{step_name} 完成!")
    print("✓"*70 + "\n")


def check_file_exists(filepath):
    """检查文件是否存在"""
    import os
    return os.path.exists(filepath)


def get_file_size(filepath):
    """获取文件大小"""
    import os
    size_bytes = os.path.getsize(filepath)
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.2f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/(1024**2):.2f} MB"
    else:
        return f"{size_bytes/(1024**3):.2f} GB"


def print_data_info(data, data_name="Data"):
    """打印数据信息"""
    if isinstance(data, dict):
        print(f"\n{data_name} 信息:")
        print(f"  类型: Dictionary")
        print(f"  键数量: {len(data)}")
        for key in list(data.keys())[:5]:  # 显示前5个键
            value = data[key]
            if isinstance(value, np.ndarray):
                print(f"    {key}: shape {value.shape}, dtype {value.dtype}")
            elif isinstance(value, dict):
                print(f"    {key}: dict with {len(value)} keys")
            else:
                print(f"    {key}: {type(value).__name__}")
        if len(data) > 5:
            print(f"    ... 还有 {len(data)-5} 个键")
    elif isinstance(data, np.ndarray):
        print(f"\n{data_name} 信息:")
        print(f"  类型: NumPy Array")
        print(f"  形状: {data.shape}")
        print(f"  数据类型: {data.dtype}")
        print(f"  统计: min={np.min(data):.4f}, max={np.max(data):.4f}, mean={np.mean(data):.4f}")
    else:
        print(f"\n{data_name} 信息:")
        print(f"  类型: {type(data).__name__}")


if __name__ == "__main__":
    print("工具函数模块加载成功 ✓")