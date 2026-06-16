import streamlit as st
import pandas as pd
import numpy as np
from utils.plot_config import setup_chinese_font
setup_chinese_font()
import matplotlib.pyplot as plt
from core.thermal_model import LumpedThermalModel, ThermalSensitivityAnalyzer, ThermalSafetyAnalyzer, ThermalInconsistencyAnalyzer


def _get_soc_curve(dm, sel_pack, sel_mod, df):
    if "kf_result_df" in st.session_state:
        kf_df = st.session_state["kf_result_df"]
        if "soc_ekf" in kf_df.columns:
            return kf_df["soc_ekf"].values
        if "soc_ukf" in kf_df.columns:
            return kf_df["soc_ukf"].values
    if "soc_ah" in df.columns:
        return df["soc_ah"].values
    return np.full(len(df), 50.0)


def _prepare_time_series(df):
    t0 = df["timestamp"].iloc[0]
    time_s = (df["timestamp"] - t0).dt.total_seconds().values.astype(float)
    time_s = np.clip(time_s, 0, None)
    dt = np.diff(time_s, prepend=0.0)
    dt[0] = dt[1] if len(dt) > 1 else 1.0
    for i in range(1, len(time_s)):
        if time_s[i] <= time_s[i - 1]:
            time_s[i] = time_s[i - 1] + 0.1
    return time_s


def _render_parameter_config():
    with st.expander("⚙️ 热模型参数配置", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            mass = st.number_input("电池质量 m (kg)", value=3.5, min_value=0.1, step=0.1, key="th_mass")
            cp = st.number_input("比热容 Cp (J/(kg·K))", value=1100.0, min_value=100.0, step=50.0, key="th_cp")
        with col2:
            cooling_mode = st.selectbox("散热模式", ["自然对流", "强制风冷"], key="th_cooling")
            h_default = 25.0 if cooling_mode == "强制风冷" else 8.0
            h_conv = st.number_input("对流换热系数 h (W/(m²·K))", value=h_default, min_value=0.5, step=0.5, key="th_h_conv")
        with col3:
            area = st.number_input("散热面积 A (m²)", value=0.04, min_value=0.001, step=0.005, format="%.3f", key="th_area")
            dentropy = st.number_input("熵变系数 dOCV/dT (mV/K)", value=0.1, min_value=-1.0, max_value=2.0, step=0.05, key="th_dentropy")

        col4, col5, col6 = st.columns(3)
        with col4:
            t_initial = st.number_input("初始温度 (°C)", value=25.0, step=0.5, key="th_t_init")
        with col5:
            t_amb_mode = st.selectbox("环境温度输入方式", ["恒定值", "时间序列"], key="th_tamb_mode")
        with col6:
            if t_amb_mode == "恒定值":
                t_amb_val = st.number_input("环境温度 (°C)", value=25.0, step=0.5, key="th_tamb_val")
            else:
                t_amb_val = st.number_input("环境温度起始值 (°C)", value=25.0, step=0.5, key="th_tamb_start")
                st.caption("时间序列将使用BMS温度数据减去ΔT模拟")

        col7, col8 = st.columns(2)
        with col7:
            r0 = st.number_input("欧姆内阻 R0 (Ω)", value=0.01, min_value=0.0, step=0.001, format="%.4f", key="th_r0")
        with col8:
            r1 = st.number_input("极化内阻 R1 (Ω)", value=0.005, min_value=0.0, step=0.001, format="%.4f", key="th_r1")

        c1 = st.number_input("极化电容 C1 (F)", value=1000.0, min_value=1.0, step=100.0, key="th_c1")

        params = {
            "mass_kg": mass,
            "cp_j_kgk": cp,
            "h_conv_w_m2k": h_conv,
            "surface_area_m2": area,
            "r0_ohm": r0,
            "r1_ohm": r1,
            "c1_f": c1,
            "dentropy_mv_k": dentropy,
        }
        ambient_config = {
            "mode": t_amb_mode,
            "value": t_amb_val,
        }
        return params, t_initial, ambient_config


def _render_coupling_analysis(dm, sel_pack, sel_mod, df, params, t_initial, ambient_config):
    st.markdown("---")
    st.subheader("🔗 热-电耦合分析")

    time_s = _prepare_time_series(df)
    current = df["current"].values.astype(float)
    soc = _get_soc_curve(dm, sel_pack, sel_mod, df)

    if len(time_s) != len(current):
        min_len = min(len(time_s), len(current), len(soc))
        time_s = time_s[:min_len]
        current = current[:min_len]
        soc = soc[:min_len]

    t_amb_const = None
    t_amb_series = None
    if ambient_config["mode"] == "恒定值":
        t_amb_const = ambient_config["value"]
    else:
        if "temperature" in df.columns:
            t_amb_series = df["temperature"].values.astype(float) - 2.0
        else:
            t_amb_const = ambient_config["value"]

    model = LumpedThermalModel(params)

    if st.button("运行热仿真", key="th_run_sim"):
        with st.spinner("热仿真计算中..."):
            sim_result = model.simulate(
                time_s=time_s,
                current=current,
                soc=soc,
                t_initial=t_initial,
                t_amb_const=t_amb_const,
                t_amb_series=t_amb_series,
            )
            st.session_state["th_sim_result"] = sim_result
            st.session_state["th_sim_time_s"] = time_s
            st.session_state["th_sim_current"] = current
            st.session_state["th_sim_soc"] = soc

    if "th_sim_result" not in st.session_state:
        st.info("请点击"运行热仿真"开始计算")
        return

    sim_result = st.session_state["th_sim_result"]
    sim_time_s = st.session_state["th_sim_time_s"]
    sim_current = st.session_state["th_sim_current"]

    temp_sim = sim_result["temperature"]
    t_amb = sim_result["t_ambient"]

    rmse_val = None
    max_dev = None
    temp_meas = None
    if "temperature" in df.columns:
        temp_meas = df["temperature"].values.astype(float)
        min_len = min(len(temp_sim), len(temp_meas))
        temp_sim_aligned = temp_sim[:min_len]
        temp_meas_aligned = temp_meas[:min_len]
        residual = temp_meas_aligned - temp_sim_aligned
        rmse_val = float(np.sqrt(np.mean(residual ** 2)))
        max_dev = float(np.max(np.abs(residual)))
    else:
        residual = None

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    with mc1:
        st.metric("仿真最高温度 (°C)", f"{np.max(temp_sim):.2f}")
    with mc2:
        st.metric("仿真终末温度 (°C)", f"{temp_sim[-1]:.2f}")
    with mc3:
        st.metric("温升幅度 (°C)", f"{temp_sim[-1] - temp_sim[0]:.2f}")
    with mc4:
        st.metric("温度RMSE (°C)", f"{rmse_val:.3f}" if rmse_val is not None else "N/A")
    with mc5:
        st.metric("最大温度偏差 (°C)", f"{max_dev:.3f}" if max_dev is not None else "N/A")

    timestamps = df["timestamp"].values
    min_len = min(len(timestamps), len(temp_sim))
    timestamps_plot = timestamps[:min_len]
    temp_sim_plot = temp_sim[:min_len]

    n_subplots = 3 if temp_meas is not None else 2
    fig, axes = plt.subplots(n_subplots, 1, figsize=(12, 3 + 2 * n_subplots), sharex=True)

    if temp_meas is not None:
        temp_meas_plot = temp_meas[:min_len]
        axes[0].plot(timestamps_plot, temp_meas_plot, "b-", label="实测温度(BMS)", linewidth=1.0, alpha=0.8)
        axes[0].plot(timestamps_plot, temp_sim_plot, "r-", label="仿真温度", linewidth=1.2)
        axes[0].plot(timestamps_plot, t_amb[:min_len], "g--", label="环境温度", linewidth=0.8, alpha=0.6)
        axes[0].set_ylabel("温度 (°C)")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        residual_plot = (temp_meas_plot - temp_sim_plot)
        axes[1].plot(timestamps_plot, residual_plot, "k-", linewidth=0.8)
        axes[1].axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        axes[1].set_ylabel("温度残差 (°C)")
        axes[1].grid(True, alpha=0.3)

        q_gen = sim_result["q_generation"][:min_len]
        q_dis = sim_result["q_dissipation"][:min_len]
        axes[2].plot(timestamps_plot, q_gen, "r-", label="总产热", linewidth=0.8)
        axes[2].plot(timestamps_plot, q_dis, "b-", label="散热量", linewidth=0.8)
        axes[2].set_ylabel("热流 (W)")
        axes[2].set_xlabel("时间")
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3)
    else:
        axes[0].plot(timestamps_plot, temp_sim_plot, "r-", label="仿真温度", linewidth=1.2)
        axes[0].plot(timestamps_plot, t_amb[:min_len], "g--", label="环境温度", linewidth=0.8, alpha=0.6)
        axes[0].set_ylabel("温度 (°C)")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        q_gen = sim_result["q_generation"][:min_len]
        q_dis = sim_result["q_dissipation"][:min_len]
        axes[1].plot(timestamps_plot, q_gen, "r-", label="总产热", linewidth=0.8)
        axes[1].plot(timestamps_plot, q_dis, "b-", label="散热量", linewidth=0.8)
        axes[1].set_ylabel("热流 (W)")
        axes[1].set_xlabel("时间")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

    fig.autofmt_xdate()
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("#### 产热分量分解")
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    q_ohm = sim_result["q_ohmic"][:min_len]
    q_pol = sim_result["q_polarization"][:min_len]
    q_ent = sim_result["q_entropy"][:min_len]
    ax2.stackplot(
        timestamps_plot,
        q_ohm, q_pol, q_ent,
        labels=["欧姆热 (I²R₀)", "极化热 (I·Vrc)", "熵变热 (I·T·dOCV/dT)"],
        colors=["#e74c3c", "#f39c12", "#3498db"],
        alpha=0.7,
    )
    ax2.set_ylabel("热流 (W)")
    ax2.set_xlabel("时间")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3)
    fig2.autofmt_xdate()
    plt.tight_layout()
    st.pyplot(fig2)


def _render_sensitivity_analysis(params, t_initial, ambient_config, time_s, current, soc):
    st.markdown("---")
    st.subheader("📊 参数灵敏度分析")

    sens_btn = st.button("运行灵敏度分析", key="th_sens_run")

    if sens_btn or "th_sens_result" in st.session_state:
        if sens_btn:
            with st.spinner("正在扫描参数扰动..."):
                model = LumpedThermalModel(params)
                t_amb_const = ambient_config["value"] if ambient_config["mode"] == "恒定值" else None
                analyzer = ThermalSensitivityAnalyzer(model)
                sens_result = analyzer.analyze(
                    time_s=time_s,
                    current=current,
                    soc=soc,
                    t_initial=t_initial,
                    t_amb_const=t_amb_const,
                )
                st.session_state["th_sens_result"] = sens_result

        sens_result = st.session_state["th_sens_result"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        colors = ["#e74c3c", "#2980b9", "#27ae60"]

        for idx, (pname, pdata) in enumerate(sens_result.items()):
            ax = axes[idx]
            upper = pdata["t_envelope_upper"]
            lower = pdata["t_envelope_lower"]
            x = np.arange(len(upper))

            ax.fill_between(x, lower, upper, alpha=0.2, color=colors[idx], label="±30%包络")

            for pert in pdata["perturbations"]:
                lw = 1.5 if abs(pert["factor"] - 1.0) < 0.01 else 0.6
                ls = "-" if abs(pert["factor"] - 1.0) < 0.01 else "--"
                ax.plot(pert["temperature"], linewidth=lw, linestyle=ls, alpha=0.7)

            ax.set_title(f"{pdata['label']} 灵敏度")
            ax.set_xlabel("时间步")
            ax.set_ylabel("温度 (°C)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("##### 灵敏度汇总表")
        table_data = []
        for pname, pdata in sens_result.items():
            for pert in pdata["perturbations"]:
                table_data.append({
                    "参数": pdata["label"],
                    "扰动系数": f"{pert['factor']:.2f}",
                    "最高温度 (°C)": f"{pert['t_max']:.2f}",
                    "终末温度 (°C)": f"{pert['t_final']:.2f}",
                })
        st.table(pd.DataFrame(table_data))


def _render_safety_boundary(time_s, temperature):
    st.markdown("---")
    st.subheader("🛡️ 安全边界分析")

    col1, col2 = st.columns([1, 3])
    with col1:
        threshold = st.number_input("温度安全阈值 (°C)", value=45.0, min_value=30.0, max_value=80.0, step=1.0, key="th_threshold")
        safety_btn = st.button("运行安全分析", key="th_safety_run")

    if safety_btn or "th_safety_result" in st.session_state:
        if safety_btn:
            with st.spinner("安全边界分析中..."):
                safety_analyzer = ThermalSafetyAnalyzer(temp_threshold=threshold)
                safety_result = safety_analyzer.analyze(time_s, temperature)
                st.session_state["th_safety_result"] = safety_result
                st.session_state["th_safety_threshold"] = threshold

        safety_result = st.session_state["th_safety_result"]
        threshold = st.session_state["th_safety_threshold"]

        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.metric("最高温度 (°C)", f"{safety_result['max_temperature']:.2f}")
        with mc2:
            st.metric("超温风险", "⚠️ 是" if safety_result["has_risk"] else "✅ 否")
        with mc3:
            st.metric("超温时段数", len(safety_result["over_periods"]))

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(time_s, temperature, "r-", label="仿真温度", linewidth=1.2)
        ax.axhline(y=threshold, color="red", linestyle="--", linewidth=1.5, label=f"安全阈值 ({threshold}°C)")

        if safety_result["has_risk"]:
            over_mask = safety_result["over_temperature_mask"]
            min_len = min(len(time_s), len(over_mask))
            ax.fill_between(
                time_s[:min_len],
                temperature[:min_len],
                threshold,
                where=over_mask[:min_len],
                color="red",
                alpha=0.3,
                label="超温区域",
            )

        ax.set_ylabel("温度 (°C)")
        ax.set_xlabel("时间 (s)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

        if safety_result["over_periods"]:
            st.markdown("##### ⚠️ 超温时段详情")
            over_df = pd.DataFrame(safety_result["over_periods"])
            display_cols = {
                "start_time_s": "起始时间 (s)",
                "end_time_s": "结束时间 (s)",
                "duration_s": "持续时间 (s)",
                "peak_temperature": "峰值温度 (°C)",
            }
            rename = {k: v for k, v in display_cols.items() if k in over_df.columns}
            st.table(over_df.rename(columns=rename)[list(rename.values())])
        else:
            st.success("仿真全程未超过安全阈值 ✅")


def _render_inconsistency_analysis(dm, sel_pack, params, t_initial, ambient_config):
    st.markdown("---")
    st.subheader("🌡️ 热分布不一致性分析")

    modules = dm.list_modules(sel_pack)
    if len(modules) < 2:
        st.warning("该Pack下模组数量不足2个，无法进行不一致性分析")
        return

    dispersion_threshold = st.number_input(
        "温度离散度阈值 (°C)", value=3.0, min_value=0.5, max_value=10.0, step=0.5, key="th_disp_threshold"
    )

    inconsistency_btn = st.button("运行不一致性分析", key="th_incon_run")

    if inconsistency_btn or "th_incon_result" in st.session_state:
        if inconsistency_btn:
            with st.spinner("多模组热仿真计算中..."):
                module_temps = {}
                module_time_s = None
                for mod_id in modules:
                    mod_df = dm.get_module_data(sel_pack, mod_id, clean=True)
                    if mod_df is None or len(mod_df) < 10:
                        continue
                    time_s = _prepare_time_series(mod_df)
                    current = mod_df["current"].values.astype(float)
                    soc_vals = mod_df["soc_ah"].values if "soc_ah" in mod_df.columns else np.full(len(mod_df), 50.0)

                    min_len = min(len(time_s), len(current), len(soc_vals))
                    time_s = time_s[:min_len]
                    current = current[:min_len]
                    soc_vals = soc_vals[:min_len]

                    t_amb_const = ambient_config["value"] if ambient_config["mode"] == "恒定值" else None
                    t_amb_series = None
                    if ambient_config["mode"] == "时间序列" and "temperature" in mod_df.columns:
                        t_amb_series = mod_df["temperature"].values.astype(float)[:min_len] - 2.0

                    model = LumpedThermalModel(params)
                    sim = model.simulate(
                        time_s=time_s,
                        current=current,
                        soc=soc_vals,
                        t_initial=t_initial,
                        t_amb_const=t_amb_const,
                        t_amb_series=t_amb_series,
                    )
                    module_temps[mod_id] = sim["temperature"]
                    if module_time_s is None:
                        module_time_s = time_s

                if len(module_temps) < 2:
                    st.warning("有效模组数据不足2个")
                    return

                min_len = min(len(v) for v in module_temps.values())
                for mid in module_temps:
                    module_temps[mid] = module_temps[mid][:min_len]
                common_time_s = module_time_s[:min_len]

                incon_analyzer = ThermalInconsistencyAnalyzer(dispersion_threshold=dispersion_threshold)
                incon_result = incon_analyzer.analyze(module_temps, common_time_s)
                st.session_state["th_incon_result"] = incon_result
                st.session_state["th_incon_module_temps"] = module_temps
                st.session_state["th_incon_time_s"] = common_time_s

        if "th_incon_result" not in st.session_state:
            return

        incon_result = st.session_state["th_incon_result"]
        module_temps = st.session_state["th_incon_module_temps"]
        common_time_s = st.session_state["th_incon_time_s"]

        score = incon_result["score"]
        if score >= 80:
            score_color = "🟢"
        elif score >= 50:
            score_color = "🟡"
        else:
            score_color = "🔴"

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("热不一致性评分", f"{score_color} {score:.1f}分")
        with mc2:
            st.metric("最大温度极差 (°C)", f"{incon_result['max_dispersion']:.2f}")
        with mc3:
            st.metric("平均温度极差 (°C)", f"{incon_result['mean_dispersion']:.2f}")
        with mc4:
            st.metric("预警时段数", len(incon_result["warning_periods"]))

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

        cmap = plt.cm.get_cmap("tab10", len(module_temps))
        for i, (mid, temps) in enumerate(module_temps.items()):
            axes[0].plot(common_time_s, temps, linewidth=1.0, label=f"模组 {mid}", color=cmap(i))
        axes[0].set_ylabel("温度 (°C)")
        axes[0].legend(fontsize=7, ncol=min(len(module_temps), 4))
        axes[0].grid(True, alpha=0.3)

        dispersion = incon_result["dispersion_series"]
        axes[1].plot(common_time_s, dispersion, "r-", linewidth=0.8, label="温度极差")
        axes[1].axhline(y=dispersion_threshold, color="red", linestyle="--", linewidth=1.0, label=f"预警阈值 ({dispersion_threshold}°C)")

        over_mask = dispersion > dispersion_threshold
        min_len = min(len(common_time_s), len(over_mask))
        axes[1].fill_between(
            common_time_s[:min_len],
            dispersion[:min_len],
            dispersion_threshold,
            where=over_mask[:min_len],
            color="red",
            alpha=0.2,
            label="超阈值区域",
        )
        axes[1].set_ylabel("温度极差 (°C)")
        axes[1].set_xlabel("时间 (s)")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        if incon_result["warning_periods"]:
            st.markdown("##### ⚠️ 温度离散度超阈值时段")
            warn_df = pd.DataFrame(incon_result["warning_periods"])
            display_cols = {
                "start_time_s": "起始时间 (s)",
                "end_time_s": "结束时间 (s)",
                "duration_s": "持续时间 (s)",
                "peak_dispersion": "峰值极差 (°C)",
            }
            rename = {k: v for k, v in display_cols.items() if k in warn_df.columns}
            st.table(warn_df.rename(columns=rename)[list(rename.values())])


def render():
    st.subheader("🌡️ 电池热管理仿真与分析")
    dm = st.session_state.data_manager

    if len(dm.packs) == 0:
        st.info("请先在数据管理页面导入BMS数据")
        return

    col1, col2 = st.columns(2)
    with col1:
        sel_pack = st.selectbox("选择Pack", dm.list_packs(), key="th_pack")
    with col2:
        modules = dm.list_modules(sel_pack)
        sel_mod = st.selectbox("选择模组", modules, key="th_mod")

    df = dm.get_module_data(sel_pack, sel_mod, clean=True)
    if df is None or len(df) < 10:
        st.warning("数据不足，请选择有效模组")
        return

    params, t_initial, ambient_config = _render_parameter_config()
    _render_coupling_analysis(dm, sel_pack, sel_mod, df, params, t_initial, ambient_config)

    if "th_sim_result" in st.session_state:
        sim_time_s = st.session_state["th_sim_time_s"]
        sim_result = st.session_state["th_sim_result"]
        sim_current = st.session_state["th_sim_current"]
        sim_soc = st.session_state["th_sim_soc"]
        _render_sensitivity_analysis(params, t_initial, ambient_config, sim_time_s, sim_current, sim_soc)
        _render_safety_boundary(sim_time_s, sim_result["temperature"])

    _render_inconsistency_analysis(dm, sel_pack, params, t_initial, ambient_config)
