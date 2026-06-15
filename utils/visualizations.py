import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64


def plot_voltage_current(df: pd.DataFrame) -> BytesIO:
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    ax1.plot(df["timestamp"], df["voltage"], "b-", label="电压(V)", linewidth=0.8)
    ax2.plot(df["timestamp"], df["current"], "r-", label="电流(A)", linewidth=0.8, alpha=0.7)
    ax1.set_xlabel("时间")
    ax1.set_ylabel("电压 (V)", color="b")
    ax2.set_ylabel("电流 (A)", color="r")
    ax1.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def plot_fit_comparison(
    t: np.ndarray,
    v_measured: np.ndarray,
    v_predicted: np.ndarray,
) -> BytesIO:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(t, v_measured, "b.-", label="实验测量", markersize=3, linewidth=0.8)
    axes[0].plot(t, v_predicted, "r-", label="模型预测", linewidth=1.2)
    axes[0].set_ylabel("电压 (V)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    residual = v_measured - v_predicted
    axes[1].plot(t, residual, "g-", linewidth=0.8)
    axes[1].axhline(y=0, color="k", linestyle="--", linewidth=0.5)
    axes[1].set_ylabel("残差 (V)")
    axes[1].set_xlabel("时间 (s)")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def plot_residuals(residuals: np.ndarray) -> BytesIO:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(residuals, bins=50, density=True, alpha=0.7, color="steelblue", edgecolor="black")
    ax.set_xlabel("残差 (V)")
    ax.set_ylabel("概率密度")
    ax.set_title(f"残差分布 (mean={np.mean(residuals):.4f}, std={np.std(residuals):.4f})")
    ax.grid(True, alpha=0.3)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf
