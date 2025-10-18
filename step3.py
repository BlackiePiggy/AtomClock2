"""步骤3: 时间序列分解.

使用多种方法分解时序数据（趋势+残差，无季节性），并对结果进行保存与可视化。
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.signal import savgol_filter
from scipy.sparse.linalg import spsolve
from statsmodels.nonparametric.smoothers_lowess import lowess

from config import DECOMPOSE_CONFIG, FILE_PATHS
from utils import load_pickle, print_completion, print_step_header, save_pickle


class TimeSeriesDecomposer:
    """时间序列分解器（专为非周期性数据设计）"""

    def __init__(self, method: str = "hp", **kwargs) -> None:
        self.method = method
        self.kwargs = kwargs

    def hp_filter(self, time_series: np.ndarray, lamb: float = 1600) -> np.ndarray:
        """Hodrick-Prescott滤波器"""
        n = len(time_series)
        D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n))
        I = sparse.eye(n)
        A = I + lamb * D.T @ D
        trend = spsolve(A, time_series)
        return trend

    def lowess_smooth(self, time_series: np.ndarray, frac: float = 0.1) -> np.ndarray:
        """LOWESS局部加权回归"""
        x = np.arange(len(time_series))
        smoothed = lowess(time_series, x, frac=frac, return_sorted=False)
        return smoothed

    def savgol_filter_method(
        self,
        time_series: np.ndarray,
        window_length: int = 51,
        polyorder: int = 3,
    ) -> np.ndarray:
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

    def moving_average(self, time_series: np.ndarray, window: int = 50) -> np.ndarray:
        """简单移动平均"""
        trend = (
            pd.Series(time_series)
            .rolling(window=window, center=True, min_periods=1)
            .mean()
            .values
        )
        return trend

    def ewma_smooth(self, time_series: np.ndarray, alpha: float = 0.1) -> np.ndarray:
        """指数加权移动平均"""
        trend = pd.Series(time_series).ewm(alpha=alpha, adjust=False).mean().values
        return trend

    def decompose(self, time_series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """分解时间序列"""
        if self.method == "hp":
            lamb = self.kwargs.get("lamb", DECOMPOSE_CONFIG["hp_lambda"])
            trend = self.hp_filter(time_series, lamb=lamb)
        elif self.method == "lowess":
            frac = self.kwargs.get("frac", DECOMPOSE_CONFIG["lowess_frac"])
            trend = self.lowess_smooth(time_series, frac=frac)
        elif self.method == "savgol":
            window_length = self.kwargs.get(
                "window_length", DECOMPOSE_CONFIG["savgol_window"]
            )
            polyorder = self.kwargs.get(
                "polyorder", DECOMPOSE_CONFIG["savgol_polyorder"]
            )
            trend = self.savgol_filter_method(time_series, window_length, polyorder)
        elif self.method == "ma":
            window = self.kwargs.get("window", DECOMPOSE_CONFIG["ma_window"])
            trend = self.moving_average(time_series, window=window)
        elif self.method == "ewma":
            alpha = self.kwargs.get("alpha", DECOMPOSE_CONFIG["ewma_alpha"])
            trend = self.ewma_smooth(time_series, alpha=alpha)
        else:
            raise ValueError(f"未知方法: {self.method}")

        residual = time_series - trend
        return trend, residual

    def visualize_decomposition(
        self,
        time_series: np.ndarray,
        trend: np.ndarray | None = None,
        residual: np.ndarray | None = None,
        save_path: str | None = None,
        title_suffix: str = "",
    ) -> None:
        """可视化分解结果."""

        if trend is None or residual is None:
            trend, residual = self.decompose(time_series)
        timestamps = np.arange(len(time_series))

        fig, axes = plt.subplots(3, 2, figsize=(16, 10))
        method_names = {
            "hp": "Hodrick-Prescott滤波",
            "lowess": "LOWESS局部加权回归",
            "savgol": "Savitzky-Golay滤波",
            "ma": "移动平均",
            "ewma": "指数加权移动平均",
        }
        fig.suptitle(
            f"时间序列分解 ({method_names.get(self.method, self.method)}){title_suffix}",
            fontsize=16,
            fontweight="bold",
        )

        # 原始序列
        axes[0, 0].plot(timestamps, time_series, "b-", linewidth=1.5, alpha=0.7)
        axes[0, 0].set_ylabel("幅值")
        axes[0, 0].set_title("原始遥测序列")
        axes[0, 0].grid(True, alpha=0.3)

        # 趋势分量
        axes[0, 1].plot(timestamps, trend, "r-", linewidth=2.5)
        axes[0, 1].set_ylabel("趋势值")
        axes[0, 1].set_title("长期退化趋势")
        axes[0, 1].grid(True, alpha=0.3)

        # 叠加对比
        axes[1, 0].plot(timestamps, time_series, "b-", linewidth=1, alpha=0.5, label="原始")
        axes[1, 0].plot(timestamps, trend, "r-", linewidth=2, label="趋势")
        axes[1, 0].set_ylabel("幅值")
        axes[1, 0].set_title("原始信号与趋势对比")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 残差
        axes[1, 1].plot(timestamps, residual, color="gray", linewidth=0.8, alpha=0.7)
        axes[1, 1].axhline(y=0, color="k", linestyle="--", linewidth=1, alpha=0.5)
        axes[1, 1].set_ylabel("残差")
        axes[1, 1].set_title("残差（去趋势后噪声）")
        axes[1, 1].grid(True, alpha=0.3)

        # 残差分布
        axes[2, 0].hist(residual, bins=50, color="gray", edgecolor="black", alpha=0.7)
        axes[2, 0].axvline(x=0, color="r", linestyle="--", linewidth=2)
        axes[2, 0].set_xlabel("残差值")
        axes[2, 0].set_ylabel("频数")
        axes[2, 0].set_title(
            f"残差分布 (μ={np.mean(residual):.4f}, σ={np.std(residual):.4f})"
        )
        axes[2, 0].grid(True, alpha=0.3, axis="y")

        # 趋势变化率
        trend_diff = np.diff(trend)
        axes[2, 1].plot(timestamps[1:], trend_diff, "purple", linewidth=1.5)
        axes[2, 1].axhline(y=0, color="k", linestyle="--", linewidth=1, alpha=0.5)
        axes[2, 1].set_xlabel("时间")
        axes[2, 1].set_ylabel("趋势变化率")
        axes[2, 1].set_title("退化速度（趋势导数）")
        axes[2, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        # 打印统计
        print("\n分解统计:")
        print(f"  原始信号: 均值={np.mean(time_series):.4f}, 标准差={np.std(time_series):.4f}")
        print(f"  趋势分量: 均值={np.mean(trend):.4f}, 标准差={np.std(trend):.4f}")
        print(f"  残差分量: 均值={np.mean(residual):.4f}, 标准差={np.std(residual):.4f}")
        print(f"  信噪比: {np.std(trend) / (np.std(residual) + 1e-8):.2f}")


def main() -> None:
    print_step_header(3, "时间序列分解")

    # 加载特征
    print("\n加载筛选后的特征...")
    filtered_normal = load_pickle(FILE_PATHS["filtered_normal_features"])
    filtered_anomaly = load_pickle(FILE_PATHS["filtered_anomaly_features"])

    # 创建分解器
    method = DECOMPOSE_CONFIG["method"]
    print(f"\n使用分解方法: {method.upper()}")
    decomposer = TimeSeriesDecomposer(method=method)

    # 分解正常单元
    print("\n分解正常单元特征...")
    imputed_normal: dict[str, np.ndarray] = {}
    normal_trend_components: dict[str, np.ndarray] = {}
    normal_residual_components: dict[str, np.ndarray] = {}
    for unit_id, features in filtered_normal.items():
        n_features = features.shape[1]
        imputed_unit = np.zeros_like(features)
        trend_unit = np.zeros_like(features)
        residual_unit = np.zeros_like(features)

        for feat_idx in range(n_features):
            time_series = features[:, feat_idx]
            trend, residual = decomposer.decompose(time_series)
            trend_unit[:, feat_idx] = trend
            residual_unit[:, feat_idx] = residual
            imputed_unit[:, feat_idx] = trend + residual

        imputed_normal[unit_id] = imputed_unit
        normal_trend_components[unit_id] = trend_unit
        normal_residual_components[unit_id] = residual_unit
    print(f"  ✓ 完成 {len(imputed_normal)} 个正常单元")

    # 分解异常单元
    print("\n分解异常单元特征...")
    imputed_anomaly: dict[str, np.ndarray] = {}
    anomaly_trend_components: dict[str, np.ndarray] = {}
    anomaly_residual_components: dict[str, np.ndarray] = {}
    for unit_id, features in filtered_anomaly.items():
        n_features = features.shape[1]
        imputed_unit = np.zeros_like(features)
        trend_unit = np.zeros_like(features)
        residual_unit = np.zeros_like(features)

        for feat_idx in range(n_features):
            time_series = features[:, feat_idx]
            trend, residual = decomposer.decompose(time_series)
            trend_unit[:, feat_idx] = trend
            residual_unit[:, feat_idx] = residual
            imputed_unit[:, feat_idx] = trend + residual

        imputed_anomaly[unit_id] = imputed_unit
        anomaly_trend_components[unit_id] = trend_unit
        anomaly_residual_components[unit_id] = residual_unit
    print(f"  ✓ 完成 {len(imputed_anomaly)} 个异常单元")

    # 可视化多个案例
    print("\n生成可视化...")
    normal_examples_dir = FILE_PATHS["decompose_normal_examples_dir"]
    anomaly_examples_dir = FILE_PATHS["decompose_anomaly_examples_dir"]
    os.makedirs(normal_examples_dir, exist_ok=True)
    os.makedirs(anomaly_examples_dir, exist_ok=True)

    if filtered_normal:
        first_unit_id = list(filtered_normal.keys())[0]
        decomposer.visualize_decomposition(
            filtered_normal[first_unit_id][:, 0],
            trend=normal_trend_components[first_unit_id][:, 0],
            residual=normal_residual_components[first_unit_id][:, 0],
            save_path=FILE_PATHS["decompose_visualization"],
            title_suffix=f" - 正常单元 {first_unit_id} 特征1",
        )

    def generate_examples(
        unit_ids: list[str],
        feature_store: dict[str, np.ndarray],
        trend_store: dict[str, np.ndarray],
        residual_store: dict[str, np.ndarray],
        save_dir: str,
        category: str,
    ) -> None:
        sample_units = unit_ids[: min(3, len(unit_ids))]
        for idx, unit_id in enumerate(sample_units, start=1):
            features = feature_store[unit_id]
            trend_unit = trend_store[unit_id]
            residual_unit = residual_store[unit_id]
            feature_count = min(2, features.shape[1])
            for feat_idx in range(feature_count):
                save_path = os.path.join(
                    save_dir,
                    f"{category.lower()}_unit_{unit_id}_feature_{feat_idx + 1}.png",
                )
                decomposer.visualize_decomposition(
                    features[:, feat_idx],
                    trend=trend_unit[:, feat_idx],
                    residual=residual_unit[:, feat_idx],
                    save_path=save_path,
                    title_suffix=f" - {category}案例{idx} 特征{feat_idx + 1}",
                )

    generate_examples(
        list(filtered_normal.keys()),
        filtered_normal,
        normal_trend_components,
        normal_residual_components,
        normal_examples_dir,
        "正常",
    )
    generate_examples(
        list(filtered_anomaly.keys()),
        filtered_anomaly,
        anomaly_trend_components,
        anomaly_residual_components,
        anomaly_examples_dir,
        "异常",
    )

    # 保存结果
    print("\n保存分解结果...")
    save_pickle(imputed_normal, FILE_PATHS["imputed_normal"])
    save_pickle(imputed_anomaly, FILE_PATHS["imputed_anomaly"])
    save_pickle(normal_trend_components, FILE_PATHS["normal_trend_components"])
    save_pickle(normal_residual_components, FILE_PATHS["normal_residual_components"])
    save_pickle(anomaly_trend_components, FILE_PATHS["anomaly_trend_components"])
    save_pickle(anomaly_residual_components, FILE_PATHS["anomaly_residual_components"])

    print_completion("时间序列分解")


if __name__ == "__main__":
    main()
