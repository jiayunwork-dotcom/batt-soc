import streamlit as st
import pandas as pd
import numpy as np
from utils.plot_config import setup_chinese_font
setup_chinese_font()
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from core import EKF, UKF, OCVSOCCalibrator


def _compute_stats(result_df, soc_col, kf_label):
    stats = {}
    soc = result_df[soc_col]
    stats["max"] = soc.max()
    stats["min"] = soc.min()
    stats["mean"] = soc.mean()
    stats["final"] = soc.iloc[-1]
    if "soc_ah" in result_df.columns:
        diff = (soc - result_df["soc_ah"]).abs()
        stats["max_dev"] = diff.max()
        stats["rmse"] = np.sqrt(((soc - result_df["soc_ah"]) ** 2).mean())
    else:
        stats["max_dev"] = None
        stats["rmse"] = None
    return stats


def _run_kf(kf_cls, ocv_func, q0, df, init_soc, params, q_soc, q_rc, r_v):
    kf = kf_cls(ocv_func, q0)
    kf.reset(initial_soc=init_soc, params=params)
    kf.Q_cov = np.diag([q_soc, q_rc])
    kf.R_cov = np.array([[r_v]])
    result_df = kf.run(df)
    return result_df, kf


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

    compare_mode = st.toggle("🔀 对比模式", value=False, help="开启后同时运行EKF和UKF进行对比")

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
                if compare_mode:
                    ekf_df, ekf_inst = _run_kf(EKF, ocv_func, q0, df, init_soc, params, q_soc, q_rc, r_v)
                    ukf_df, ukf_inst = _run_kf(UKF, ocv_func, q0, df, init_soc, params, q_soc, q_rc, r_v)
                    st.session_state["kf_result_df"] = ekf_df
                    st.session_state["kf_result_df_ukf"] = ukf_df
                    st.session_state["kf_history_ekf"] = ekf_inst.history
                    st.session_state["kf_history_ukf"] = ukf_inst.history
                    st.session_state["kf_elapsed_ekf"] = ekf_inst.elapsed_time
                    st.session_state["kf_elapsed_ukf"] = ukf_inst.elapsed_time
                    st.session_state["kf_type_name"] = "EKF"
                    st.session_state["kf_compare"] = True
                else:
                    if kf_type.startswith("扩展"):
                        result_df, kf_inst = _run_kf(EKF, ocv_func, q0, df, init_soc, params, q_soc, q_rc, r_v)
                        st.session_state["kf_result_df"] = result_df
                        st.session_state["kf_history"] = kf_inst.history
                        st.session_state["kf_type_name"] = "EKF"
                        st.session_state["kf_elapsed"] = kf_inst.elapsed_time
                    else:
                        result_df, kf_inst = _run_kf(UKF, ocv_func, q0, df, init_soc, params, q_soc, q_rc, r_v)
                        st.session_state["kf_result_df"] = result_df
                        st.session_state["kf_history"] = kf_inst.history
                        st.session_state["kf_type_name"] = "UKF"
                        st.session_state["kf_elapsed"] = kf_inst.elapsed_time
                    st.session_state["kf_compare"] = False

        result_df = st.session_state["kf_result_df"]
        kf_type_name = st.session_state["kf_type_name"]
        is_compare = st.session_state.get("kf_compare", False)

        if is_compare:
            ukf_df = st.session_state["kf_result_df_ukf"]
            ekf_stats = _compute_stats(result_df, "soc_ekf", "EKF")
            ukf_stats = _compute_stats(ukf_df, "soc_ukf", "UKF")

            st.markdown("#### 📊 EKF 统计摘要")
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            with c1:
                st.metric("最大SOC (%)", f"{ekf_stats['max']:.2f}")
            with c2:
                st.metric("最小SOC (%)", f"{ekf_stats['min']:.2f}")
            with c3:
                st.metric("均值SOC (%)", f"{ekf_stats['mean']:.2f}")
            with c4:
                st.metric("终末SOC (%)", f"{ekf_stats['final']:.2f}")
            with c5:
                st.metric("最大偏差 (%)", f"{ekf_stats['max_dev']:.3f}" if ekf_stats["max_dev"] is not None else "N/A")
            with c6:
                st.metric("RMSE (%)", f"{ekf_stats['rmse']:.3f}" if ekf_stats["rmse"] is not None else "N/A")

            st.markdown("#### 📊 UKF 统计摘要")
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            with c1:
                st.metric("最大SOC (%)", f"{ukf_stats['max']:.2f}")
            with c2:
                st.metric("最小SOC (%)", f"{ukf_stats['min']:.2f}")
            with c3:
                st.metric("均值SOC (%)", f"{ukf_stats['mean']:.2f}")
            with c4:
                st.metric("终末SOC (%)", f"{ukf_stats['final']:.2f}")
            with c5:
                st.metric("最大偏差 (%)", f"{ukf_stats['max_dev']:.3f}" if ukf_stats["max_dev"] is not None else "N/A")
            with c6:
                st.metric("RMSE (%)", f"{ukf_stats['rmse']:.3f}" if ukf_stats["rmse"] is not None else "N/A")

            fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
            axes[0].plot(result_df["timestamp"], result_df["soc_ekf"], "r-", label="EKF估计", linewidth=1.2)
            axes[0].plot(ukf_df["timestamp"], ukf_df["soc_ukf"], "m-", label="UKF估计", linewidth=1.2)
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

            st.markdown("#### 📋 算法对比表格")
            ekf_elapsed = st.session_state.get("kf_elapsed_ekf", 0)
            ukf_elapsed = st.session_state.get("kf_elapsed_ukf", 0)
            compare_data = {
                "指标": ["RMSE (%)", "最大偏差 (%)", "计算耗时 (s)"],
                "EKF": [
                    f"{ekf_stats['rmse']:.4f}" if ekf_stats["rmse"] is not None else "N/A",
                    f"{ekf_stats['max_dev']:.4f}" if ekf_stats["max_dev"] is not None else "N/A",
                    f"{ekf_elapsed:.4f}",
                ],
                "UKF": [
                    f"{ukf_stats['rmse']:.4f}" if ukf_stats["rmse"] is not None else "N/A",
                    f"{ukf_stats['max_dev']:.4f}" if ukf_stats["max_dev"] is not None else "N/A",
                    f"{ukf_elapsed:.4f}",
                ],
            }
            st.table(pd.DataFrame(compare_data))

            ekf_hist = st.session_state.get("kf_history_ekf", {})
            ukf_hist = st.session_state.get("kf_history_ukf", {})

            if ekf_hist.get("k_gain") or ukf_hist.get("k_gain"):
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                if ekf_hist.get("k_gain"):
                    gains = np.array(ekf_hist["k_gain"])
                    axes[0].plot(gains[:, 0], "r-", linewidth=0.8, label="EKF")
                if ukf_hist.get("k_gain"):
                    gains_u = np.array(ukf_hist["k_gain"])
                    axes[0].plot(gains_u[:, 0], "m-", linewidth=0.8, label="UKF")
                axes[0].set_title("卡尔曼增益 - SOC分量")
                axes[0].set_xlabel("时间步")
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)

                if ekf_hist.get("k_gain"):
                    gains = np.array(ekf_hist["k_gain"])
                    if gains.shape[1] > 1:
                        axes[1].plot(gains[:, 1], "r-", linewidth=0.8, label="EKF")
                if ukf_hist.get("k_gain"):
                    gains_u = np.array(ukf_hist["k_gain"])
                    if gains_u.shape[1] > 1:
                        axes[1].plot(gains_u[:, 1], "m-", linewidth=0.8, label="UKF")
                axes[1].set_title("卡尔曼增益 - Vrc分量")
                axes[1].set_xlabel("时间步")
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                st.pyplot(fig)

            if ekf_hist.get("cov") or ukf_hist.get("cov"):
                fig, ax = plt.subplots(figsize=(10, 3))
                if ekf_hist.get("cov"):
                    covs = np.array(ekf_hist["cov"])
                    ax.plot(covs[:, 0], "r-", linewidth=0.8, label="EKF P[0,0]")
                    ax.plot(covs[:, 1], "r--", linewidth=0.8, label="EKF P[1,1]")
                if ukf_hist.get("cov"):
                    covs_u = np.array(ukf_hist["cov"])
                    ax.plot(covs_u[:, 0], "m-", linewidth=0.8, label="UKF P[0,0]")
                    ax.plot(covs_u[:, 1], "m--", linewidth=0.8, label="UKF P[1,1]")
                ax.set_title("协方差收敛过程")
                ax.set_yscale("log")
                ax.set_xlabel("时间步")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

        else:
            soc_col = f"soc_{kf_type_name.lower()}"
            stats = _compute_stats(result_df, soc_col, kf_type_name)

            st.markdown("#### 📊 统计摘要")
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            with c1:
                st.metric("最大SOC (%)", f"{stats['max']:.2f}")
            with c2:
                st.metric("最小SOC (%)", f"{stats['min']:.2f}")
            with c3:
                st.metric("均值SOC (%)", f"{stats['mean']:.2f}")
            with c4:
                st.metric("终末SOC (%)", f"{stats['final']:.2f}")
            with c5:
                st.metric("最大偏差 (%)", f"{stats['max_dev']:.3f}" if stats["max_dev"] is not None else "N/A")
            with c6:
                st.metric("RMSE (%)", f"{stats['rmse']:.3f}" if stats["rmse"] is not None else "N/A")

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

            hist = st.session_state.get("kf_history", {})
            if hist.get("k_gain"):
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                gains = np.array(hist["k_gain"])
                axes[0].plot(gains[:, 0], linewidth=0.8)
                axes[0].set_title(f"{kf_type_name} 卡尔曼增益 - SOC分量")
                axes[0].set_xlabel("时间步")
                axes[0].grid(True, alpha=0.3)
                if gains.shape[1] > 1:
                    axes[1].plot(gains[:, 1], linewidth=0.8)
                axes[1].set_title(f"{kf_type_name} 卡尔曼增益 - Vrc分量")
                axes[1].set_xlabel("时间步")
                axes[1].grid(True, alpha=0.3)
                st.pyplot(fig)

            if hist.get("cov"):
                covs = np.array(hist["cov"])
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(covs[:, 0], linewidth=0.8, label="P[0,0] (SOC方差)")
                ax.plot(covs[:, 1], linewidth=0.8, label="P[1,1] (Vrc方差)")
                ax.set_title("协方差收敛过程")
                ax.set_yscale("log")
                ax.set_xlabel("时间步")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

    st.markdown("---")
    st.markdown("#### 🔬 参数灵敏度分析")

    sens_btn = st.button("灵敏度分析", key="kf_sensitivity")

    if sens_btn or "kf_sens_results" in st.session_state:
        if sens_btn:
            with st.spinner("正在运行灵敏度分析（多参数扰动）..."):
                params = {"R0": r0, "R1": r1, "C1": c1}
                perturbations = [0.5, 0.8, 1.0, 1.2, 1.5]
                q_perturb_labels = ["Q×0.5", "Q×0.8", "基准", "Q×1.2", "Q×1.5"]
                r_perturb_labels = ["R×0.5", "R×0.8", "基准", "R×1.2", "R×1.5"]

                q_sens_results = []
                for pf, label in zip(perturbations, q_perturb_labels):
                    kf_inst = EKF(ocv_func, q0)
                    kf_inst.reset(initial_soc=init_soc, params=params)
                    kf_inst.Q_cov = np.diag([q_soc * pf, q_rc * pf])
                    kf_inst.R_cov = np.array([[r_v]])
                    res = kf_inst.run(df)
                    q_sens_results.append({"label": label, "factor": pf, "soc": res["soc_ekf"].values, "rmse": _compute_stats(res, "soc_ekf", "EKF").get("rmse")})

                r_sens_results = []
                for pf, label in zip(perturbations, r_perturb_labels):
                    kf_inst = EKF(ocv_func, q0)
                    kf_inst.reset(initial_soc=init_soc, params=params)
                    kf_inst.Q_cov = np.diag([q_soc, q_rc])
                    kf_inst.R_cov = np.array([[r_v * pf]])
                    res = kf_inst.run(df)
                    r_sens_results.append({"label": label, "factor": pf, "soc": res["soc_ekf"].values, "rmse": _compute_stats(res, "soc_ekf", "EKF").get("rmse")})

                st.session_state["kf_sens_results"] = {"q": q_sens_results, "r": r_sens_results}

        sens = st.session_state["kf_sens_results"]
        ts = df["timestamp"].values

        q_results = sens["q"]
        q_socs = np.array([r["soc"] for r in q_results])
        q_max = q_socs.max(axis=0)
        q_min = q_socs.min(axis=0)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].fill_between(range(len(q_max)), q_min, q_max, alpha=0.25, color="red", label="Q扰动范围")
        for r in q_results:
            lw = 1.5 if r["factor"] == 1.0 else 0.7
            ls = "-" if r["factor"] == 1.0 else "--"
            axes[0].plot(r["soc"], linewidth=lw, linestyle=ls, label=r["label"])
        axes[0].set_title("Q矩阵参数扰动对SOC估计的影响")
        axes[0].set_xlabel("时间步")
        axes[0].set_ylabel("SOC (%)")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        r_results = sens["r"]
        r_socs = np.array([r["soc"] for r in r_results])
        r_max = r_socs.max(axis=0)
        r_min = r_socs.min(axis=0)

        axes[1].fill_between(range(len(r_max)), r_min, r_max, alpha=0.25, color="blue", label="R扰动范围")
        for r in r_results:
            lw = 1.5 if r["factor"] == 1.0 else 0.7
            ls = "-" if r["factor"] == 1.0 else "--"
            axes[1].plot(r["soc"], linewidth=lw, linestyle=ls, label=r["label"])
        axes[1].set_title("R矩阵参数扰动对SOC估计的影响")
        axes[1].set_xlabel("时间步")
        axes[1].set_ylabel("SOC (%)")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("##### 灵敏度分析RMSE汇总")
        q_rmse_data = {"扰动水平": [r["label"] for r in q_results], "Q扰动RMSE (%)": [f"{r['rmse']:.4f}" if r["rmse"] is not None else "N/A" for r in q_results]}
        r_rmse_data = {"扰动水平": [r["label"] for r in r_results], "R扰动RMSE (%)": [f"{r['rmse']:.4f}" if r["rmse"] is not None else "N/A" for r in r_results]}
        merged = {k: v + r_rmse_data.get(k, []) for k, v in q_rmse_data.items()}
        merged["R扰动RMSE (%)"] = [""] * len(q_results) + [f"{r['rmse']:.4f}" if r["rmse"] is not None else "N/A" for r in r_results]
        merged["扰动水平"] = [r["label"] for r in q_results] + [r["label"] for r in r_results]
        merged["扰动参数"] = ["Q"] * len(q_results) + ["R"] * len(r_results)
        st.table(pd.DataFrame({"扰动参数": merged["扰动参数"], "扰动水平": merged["扰动水平"], "RMSE (%)": [f"{r['rmse']:.4f}" if r["rmse"] is not None else "N/A" for r in q_results] + [f"{r['rmse']:.4f}" if r["rmse"] is not None else "N/A" for r in r_results]}))
