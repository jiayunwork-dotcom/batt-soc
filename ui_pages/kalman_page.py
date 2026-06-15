import streamlit as st
import pandas as pd
import numpy as np
from utils.plot_config import setup_chinese_font
setup_chinese_font()
import matplotlib.pyplot as plt
from core import EKF, UKF, OCVSOCCalibrator


def render():
    st.subheader("🎯 卡尔曼滤波SOC估计")
    dm = st.session_state.data_manager

    if len(dm.packs) == 0:
        st.info("请先在数据管理页面导入数据")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        sel_pack = st.selectbox("选择Pack", dm.list_packs(), key="kf_pack")
    with col2:
        sel_mod = st.selectbox("选择模组", dm.list_modules(sel_pack), key="kf_mod")
    with col3:
        kf_type = st.selectbox("滤波算法", ["扩展卡尔曼滤波(EKF)", "无迹卡尔曼滤波(UKF)"], key="kf_type")

    df = dm.get_module_data(sel_pack, sel_mod, clean=True)
    q0 = dm.get_pack_capacity(sel_pack)

    if df is None or len(df) < 10:
        st.warning("数据不足")
        return

    with st.expander("参数配置 (过程噪声Q / 观测噪声R)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            q_soc = st.number_input("Q矩阵 - SOC过程噪声方差", value=1e-4, format="%.1e", step=1e-5)
            q_rc = st.number_input("Q矩阵 - Vrc过程噪声方差", value=1e-6, format="%.1e", step=1e-7)
        with col2:
            r_v = st.number_input("R矩阵 - 电压观测噪声方差", value=1e-3, format="%.1e", step=1e-4)

        r0 = st.number_input("欧姆内阻 R0 (Ω)", value=0.01, format="%.5f", step=0.001)
        r1 = st.number_input("极化内阻 R1 (Ω)", value=0.005, format="%.5f", step=0.001)
        c1 = st.number_input("极化电容 C1 (F)", value=1000.0, format="%.1f", step=100.0)
        init_soc = st.slider("初始SOC (%)", 0.0, 100.0, 100.0, 1.0)

    if "ocv_calibrator" not in st.session_state or st.session_state["ocv_calibrator"].ocv_mean is None:
        st.warning("未检测到OCV标定数据，使用默认OCV曲线(3.0~4.2V线性)")
        cal = OCVSOCCalibrator()
        cal.ocv_mean = np.linspace(3.0, 4.2, 101)
        cal._build_interpolator()
        ocv_func = cal.get_ocv
    else:
        ocv_func = st.session_state["ocv_calibrator"].get_ocv

    run_btn = st.button("运行SOC估计", key="kf_run")

    if run_btn or "kf_result_df" in st.session_state:
        if run_btn:
            with st.spinner("SOC估计中..."):
                params = {"R0": r0, "R1": r1, "C1": c1}
                if kf_type.startswith("扩展"):
                    kf = EKF(ocv_func, q0)
                    kf.reset(initial_soc=init_soc, params=params)
                    kf.Q_cov = np.diag([q_soc, q_rc])
                    kf.R_cov = np.array([[r_v]])
                    result_df = kf.run(df)
                    st.session_state["kf_result_df"] = result_df
                    st.session_state["kf_history"] = kf.history
                    st.session_state["kf_type_name"] = "EKF"
                else:
                    kf = UKF(ocv_func, q0)
                    kf.reset(initial_soc=init_soc, params=params)
                    kf.Q_cov = np.diag([q_soc, q_rc])
                    kf.R_cov = np.array([[r_v]])
                    result_df = kf.run(df)
                    st.session_state["kf_result_df"] = result_df
                    st.session_state["kf_history"] = kf.history
                    st.session_state["kf_type_name"] = "UKF"

        result_df = st.session_state["kf_result_df"]
        kf_type_name = st.session_state["kf_type_name"]

        col1, col2, col3, col4 = st.columns(4)
        soc_col = f"soc_{kf_type_name.lower()}"
        with col1:
            st.metric(f"{kf_type_name} SOC均值(%)", f"{result_df[soc_col].mean():.2f}")
        with col2:
            st.metric(f"{kf_type_name} SOC最小值(%)", f"{result_df[soc_col].min():.2f}")
        with col3:
            st.metric(f"{kf_type_name} SOC最大值(%)", f"{result_df[soc_col].max():.2f}")
        with col4:
            if "soc_ah" in result_df.columns:
                diff = (result_df[soc_col] - result_df["soc_ah"]).abs()
                st.metric(f"与AH积分平均偏差(%)", f"{diff.mean():.3f}")

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        axes[0].plot(result_df["timestamp"], result_df[soc_col], "r-", label=f"{kf_type_name}估计", linewidth=1.2)
        if "soc_ah" in result_df.columns:
            axes[0].plot(result_df["timestamp"], result_df["soc_ah"], "b--", label="安时积分法", linewidth=0.8, alpha=0.7)
        axes[0].set_ylabel("SOC (%)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(result_df["timestamp"], result_df["current"], "g-", linewidth=0.8)
        axes[1].set_ylabel("电流 (A)")
        axes[1].set_xlabel("时间")
        axes[1].grid(True, alpha=0.3)
        fig.autofmt_xdate()
        st.pyplot(fig)

        if "kf_history" in st.session_state and "k_gain" in st.session_state["kf_history"]:
            hist = st.session_state["kf_history"]
            if hist.get("k_gain"):
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                gains = np.array(hist["k_gain"])
                axes[0].plot(gains[:, 0], linewidth=0.8)
                axes[0].set_title(f"{kf_type_name}卡尔曼增益 - SOC分量")
                axes[0].grid(True, alpha=0.3)
                if gains.shape[1] > 1:
                    axes[1].plot(gains[:, 1], linewidth=0.8)
                axes[1].set_title(f"{kf_type_name}卡尔曼增益 - Vrc分量")
                axes[1].grid(True, alpha=0.3)
                st.pyplot(fig)

            if hist.get("cov"):
                covs = np.array(hist["cov"])
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(covs[:, 0], linewidth=0.8, label="P[0,0] (SOC方差)")
                ax.plot(covs[:, 1], linewidth=0.8, label="P[1,1] (Vrc方差)")
                ax.set_title("协方差收敛过程")
                ax.set_yscale("log")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
