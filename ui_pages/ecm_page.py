import streamlit as st
import pandas as pd
import numpy as np
from utils.plot_config import setup_chinese_font
setup_chinese_font()
import matplotlib.pyplot as plt
from core import ECM
from utils.preprocessing import detect_hppc_pulses


def render():
    st.subheader("🔧 等效电路模型参数辨识")
    dm = st.session_state.data_manager

    if len(dm.packs) == 0:
        st.info("请先在数据管理页面导入数据")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        sel_pack = st.selectbox("选择Pack", dm.list_packs(), key="ecm_pack")
    with col2:
        sel_mod = st.selectbox("选择模组", dm.list_modules(sel_pack), key="ecm_mod")
    with col3:
        model_order = st.selectbox("模型阶数", ["一阶RC", "二阶RC"], key="ecm_order")

    order = 1 if model_order == "一阶RC" else 2
    df = dm.get_module_data(sel_pack, sel_mod, clean=True)

    if df is None or len(df) < 10:
        st.warning("数据不足")
        return

    pulses = detect_hppc_pulses(df, current_threshold=0.3)
    st.info(f"检测到 {len(pulses)} 个HPPC脉冲特征")

    if not pulses:
        return

    pulse_idx = st.slider("选择脉冲序号", 0, max(0, len(pulses) - 1), 0, key="ecm_pulse")

    if st.button("开始参数辨识", key="ecm_run"):
        with st.spinner("参数辨识中..."):
            ecm = ECM(order=order)
            result = ecm.identify(df, pulse_idx)
            fit_data = ecm.get_fit_plot_data()
            st.session_state["ecm_result"] = result
            st.session_state["ecm_fit_data"] = fit_data
            st.session_state["ecm_instance"] = ecm

    if "ecm_result" in st.session_state:
        result = st.session_state["ecm_result"]
        fit_data = st.session_state["ecm_fit_data"]

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### 辨识参数及95%置信区间")
            param_df = pd.DataFrame({
                "参数": list(result["params"].keys()),
                "辨识值": [f"{v:.6f}" for v in result["params"].values()],
                "95%置信区间±": [f"{v:.6f}" for v in result["ci_95"].values()],
            })
            st.dataframe(param_df, width="stretch", hide_index=True)
            st.metric("拟合优度 R²", f"{result['r_squared']:.6f}")

        with col2:
            if fit_data is not None:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(fit_data["t_sec"], fit_data["v_measured"], "b.-", label="实验测量", markersize=3, linewidth=0.6)
                ax.plot(fit_data["t_sec"], fit_data["v_predicted"], "r-", label="模型预测", linewidth=1.2)
                ax.set_xlabel("时间 (s)")
                ax.set_ylabel("电压 (V)")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

        if fit_data is not None:
            fig, ax = plt.subplots(figsize=(12, 3))
            ax.hist(fit_data["residuals"], bins=50, density=True, alpha=0.7, color="steelblue", edgecolor="black")
            ax.set_title(f"残差分布  mean={np.mean(fit_data['residuals']):.5f}, std={np.std(fit_data['residuals']):.5f}")
            ax.set_xlabel("残差 (V)")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

    st.markdown("---")
    st.subheader("参数随SOC变化")

    soc_options = st.multiselect(
        "选择SOC点(%)",
        list(range(0, 101, 10)),
        default=[10, 30, 50, 70, 90],
        key="ecm_soc_points",
    )

    if st.button("辨识多SOC点参数", key="ecm_soc_run"):
        with st.spinner("多SOC点参数辨识中..."):
            ecm = ECM(order=order)
            soc_df = ecm.identify_vs_soc(df, soc_options)
            if len(soc_df) > 0:
                st.dataframe(soc_df, width="stretch", hide_index=True)
                param_cols = [c for c in soc_df.columns if c != "SOC(%)"]
                n_cols = len(param_cols)
                fig, axes = plt.subplots((n_cols + 1) // 2, 2, figsize=(12, 3 * ((n_cols + 1) // 2)))
                axes = axes.flatten() if n_cols > 1 else [axes]
                for i, col in enumerate(param_cols):
                    axes[i].plot(soc_df["SOC(%)"], soc_df[col], "o-")
                    axes[i].set_xlabel("SOC (%)")
                    axes[i].set_ylabel(col)
                    axes[i].grid(True, alpha=0.3)
                for j in range(i + 1, len(axes)):
                    axes[j].axis("off")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("未能辨识足够SOC点参数")
