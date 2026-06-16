import streamlit as st
import pandas as pd
import numpy as np
from utils.plot_config import setup_chinese_font
setup_chinese_font()
import matplotlib.pyplot as plt
from core import BatteryDiagnosis
from datetime import datetime


def _get_data_from_manager():
    dm = st.session_state.data_manager
    cycle_df = None
    cycle_charge_data = []
    packs = dm.list_packs()
    if packs:
        for pack_id in packs:
            modules = dm.list_modules(pack_id)
            for mod_id in modules:
                df = dm.get_module_data(pack_id, mod_id)
                if df is not None and len(df) > 100:
                    if "soc_ah" in df.columns and "voltage" in df.columns:
                        q0 = dm.get_pack_capacity(pack_id)
                        df_sorted = df.sort_values("timestamp").reset_index(drop=True)
                        capacity = (100 - df_sorted["soc_ah"].values) / 100.0 * q0
                        df_cycle = pd.DataFrame({
                            "timestamp": df_sorted["timestamp"].values,
                            "voltage": df_sorted["voltage"].values,
                            "capacity": capacity,
                            "current": df_sorted["current"].values,
                            "temperature": df_sorted["temperature"].values if "temperature" in df_sorted.columns else 25.0,
                        })
                        cycle_charge_data.append(df_cycle)
    return cycle_df, cycle_charge_data


def render():
    st.subheader("🔍 电池故障诊断与SOH在线估计")

    if "diagnosis" not in st.session_state:
        st.session_state.diagnosis = BatteryDiagnosis()
    diag: BatteryDiagnosis = st.session_state.diagnosis

    if "diag_cycle_df" not in st.session_state:
        st.session_state.diag_cycle_df = None
    if "diag_charge_data" not in st.session_state:
        st.session_state.diag_charge_data = None
    if "diag_dqdv_results" not in st.session_state:
        st.session_state.diag_dqdv_results = {}
    if "diag_impedance_df" not in st.session_state:
        st.session_state.diag_impedance_df = None
    if "diag_soh_models" not in st.session_state:
        st.session_state.diag_soh_models = {}
    if "diag_fault_events" not in st.session_state:
        st.session_state.diag_fault_events = []
    if "diag_report" not in st.session_state:
        st.session_state.diag_report = None

    data_tab, dqdv_tab, impedance_tab, soh_tab, fault_tab, report_tab = st.tabs([
        "📋 数据准备", "📈 容量增量分析(dQ/dV)", "⚡ 内阻在线估计", "🎯 SOH多模型估计", "⚠️ 故障预警", "📑 诊断报告"
    ])

    with data_tab:
        st.markdown("#### 数据来源配置")
        data_source = st.radio("数据来源", ["使用模拟数据(50循环)", "从BMS数据管理模块获取"], horizontal=True, key="diag_ds")
        if data_source == "使用模拟数据(50循环)":
            n_cycles = st.slider("模拟循环数", 10, 100, 50, key="diag_n_cycles")
            q0 = st.number_input("额定容量(Ah)", 10.0, 500.0, 100.0, key="diag_q0")
            r0_0 = st.number_input("初始内阻R0(Ω)", 0.001, 0.1, 0.01, step=0.001, format="%.4f", key="diag_r0_0")
            diag.q0_nominal = float(q0)
            diag.r0_initial = float(r0_0)
            if st.button("生成模拟退化数据", key="diag_gen"):
                with st.spinner("正在生成模拟数据..."):
                    cycle_df, cycle_charge_data = BatteryDiagnosis.generate_degradation_data(n_cycles, q0, r0_0)
                    st.session_state.diag_cycle_df = cycle_df
                    st.session_state.diag_charge_data = cycle_charge_data
                    st.session_state.diag_dqdv_results = {}
                    st.session_state.diag_impedance_df = None
                    st.session_state.diag_soh_models = {}
                    st.session_state.diag_fault_events = []
                    st.session_state.diag_report = None
                    st.success(f"已生成 {n_cycles} 个循环的模拟数据")
        else:
            if st.button("从BMS数据管理模块加载", key="diag_load_bms"):
                cycle_df, cycle_charge_data = _get_data_from_manager()
                if cycle_charge_data:
                    st.session_state.diag_charge_data = cycle_charge_data
                    st.success(f"已加载 {len(cycle_charge_data)} 组充放电数据")
                else:
                    st.warning("BMS数据管理模块中未找到有效数据，请使用模拟数据")
                if cycle_df is not None:
                    st.session_state.diag_cycle_df = cycle_df

        if st.session_state.diag_cycle_df is not None:
            st.markdown("#### 循环统计数据")
            st.dataframe(st.session_state.diag_cycle_df.head(10), width="stretch")
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            cycle_df = st.session_state.diag_cycle_df
            axes[0].plot(cycle_df["循环次数"], cycle_df["放电容量(Ah)"], "b.-")
            axes[0].set_xlabel("循环次数"); axes[0].set_ylabel("容量(Ah)"); axes[0].grid(True, alpha=0.3); axes[0].set_title("容量衰减")
            if "R0(Ω)" in cycle_df.columns:
                axes[1].plot(cycle_df["循环次数"], cycle_df["R0(Ω)"], "r.-")
                axes[1].set_xlabel("循环次数"); axes[1].set_ylabel("R0(Ω)"); axes[1].grid(True, alpha=0.3); axes[1].set_title("内阻增长")
            if "R1(Ω)" in cycle_df.columns:
                axes[2].plot(cycle_df["循环次数"], cycle_df["R1(Ω)"], "g.-")
                axes[2].set_xlabel("循环次数"); axes[2].set_ylabel("R1(Ω)"); axes[2].grid(True, alpha=0.3); axes[2].set_title("极化内阻")
            plt.tight_layout()
            st.pyplot(fig)

        if st.session_state.diag_charge_data is not None:
            st.markdown(f"#### 充放电数据 ({len(st.session_state.diag_charge_data)} 个循环)")
            sel_cycle = st.selectbox("选择查看循环", range(1, len(st.session_state.diag_charge_data) + 1), key="diag_sel_cycle")
            df_show = st.session_state.diag_charge_data[sel_cycle - 1]
            st.dataframe(df_show.head(), width="stretch")

    with dqdv_tab:
        st.markdown("#### 容量增量分析 (dQ/dV)")
        col1, col2 = st.columns(2)
        with col1:
            sg_window = st.slider("Savitzky-Golay滤波窗口长度", 5, 51, 11, 2, key="diag_sg_w")
            sg_poly = st.slider("多项式阶数", 1, 5, 3, key="diag_sg_p")
        with col2:
            if st.session_state.diag_charge_data is not None:
                cycle_options = list(range(1, len(st.session_state.diag_charge_data) + 1))
                sel_cycles = st.multiselect("选择用于分析的循环", cycle_options,
                    default=cycle_options[::max(1, len(cycle_options) // 10)], key="diag_sel_cycles")
            else:
                sel_cycles = []
                st.info("请先在【数据准备】页加载数据")

        if st.button("计算dQ/dV曲线", key="diag_calc_dqdv") and st.session_state.diag_charge_data is not None and sel_cycles:
            with st.spinner("正在计算dQ/dV..."):
                selected_data = [st.session_state.diag_charge_data[i - 1] for i in sel_cycles]
                dqdv_results = {}
                for idx, df in zip(sel_cycles, selected_data):
                    if "voltage" in df.columns and "capacity" in df.columns:
                        res = diag.compute_dqdv(df["voltage"].values, df["capacity"].values, sg_window, sg_poly)
                        if "error" not in res:
                            dqdv_results[f"循环{idx}"] = res
                st.session_state.diag_dqdv_results = dqdv_results
                diag.dqdv_results = dqdv_results
                st.success(f"已完成 {len(dqdv_results)} 个循环的dQ/dV分析")

        if st.session_state.diag_dqdv_results:
            results = st.session_state.diag_dqdv_results
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
            all_peaks_v = []
            all_peaks_h = []
            for (name, res), color in zip(results.items(), colors):
                ax.plot(res["voltage"], res["dqdv"], color=color, label=name, linewidth=1.0, alpha=0.8)
                if len(res["peak_voltage"]) > 0:
                    ax.scatter(res["peak_voltage"], res["peak_height"], color=color, marker="v", s=40, zorder=5)
                    for pv, ph in zip(res["peak_voltage"], res["peak_height"]):
                        all_peaks_v.append(pv)
                        all_peaks_h.append(ph)
                        ax.annotate(f"{pv:.2f}V", (pv, ph), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=7)
            ax.set_xlabel("电压 (V)")
            ax.set_ylabel("dQ/dV (Ah/V)")
            ax.set_title("多循环dQ/dV曲线对比(峰值已标注)")
            ax.legend(fontsize=8, loc="best")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            peak_rows = []
            for name, res in results.items():
                if len(res["peak_voltage"]) > 0:
                    for i, (pv, ph) in enumerate(zip(res["peak_voltage"], res["peak_height"])):
                        peak_rows.append({"循环": name, f"峰值#{i+1}_电压(V)": round(pv, 3), f"峰值#{i+1}_高度": round(ph, 2)})
            if peak_rows:
                st.markdown("#### 峰值检测结果")
                st.dataframe(pd.DataFrame(peak_rows), width="stretch", hide_index=True)

            if len(results) >= 2:
                st.markdown("#### 主峰漂移趋势")
                first_key = list(results.keys())[0]
                last_key = list(results.keys())[-1]
                fig2, ax2 = plt.subplots(figsize=(10, 4))
                cycle_nums = []
                main_peaks = []
                for name, res in results.items():
                    try:
                        cyc = int(name.replace("循环", ""))
                    except Exception:
                        cyc = 0
                    if len(res["peak_height"]) > 0:
                        main_idx = np.argmax(res["peak_height"])
                        cycle_nums.append(cyc)
                        main_peaks.append(res["peak_voltage"][main_idx])
                if cycle_nums:
                    ax2.plot(cycle_nums, main_peaks, "o-")
                    ax2.set_xlabel("循环次数"); ax2.set_ylabel("主峰电压 (V)")
                    ax2.set_title("主峰电压随循环漂移趋势")
                    ax2.grid(True, alpha=0.3)
                    st.pyplot(fig2)

    with impedance_tab:
        st.markdown("#### 内阻在线估计")
        col1, col2, col3 = st.columns(3)
        with col1:
            current_thr = st.number_input("电流阶跃阈值(A)", 0.1, 50.0, 1.0, 0.1, key="diag_cur_thr")
        with col2:
            min_dur = st.number_input("最小脉冲时长(s)", 0.1, 10.0, 0.5, 0.1, key="diag_min_dur")
        with col3:
            max_dur = st.number_input("最大脉冲时长(s)", 1.0, 60.0, 5.0, 0.5, key="diag_max_dur")

        if st.button("识别脉冲并估算内阻", key="diag_calc_imp"):
            with st.spinner("正在处理脉冲数据..."):
                if st.session_state.diag_charge_data is not None:
                    imp_df = diag.process_all_pulses(
                        st.session_state.diag_charge_data,
                        current_threshold=float(current_thr),
                        min_duration=float(min_dur),
                        max_duration=float(max_dur),
                    )
                    if len(imp_df) == 0 and st.session_state.diag_cycle_df is not None:
                        cycle_df = st.session_state.diag_cycle_df
                        if "R0(Ω)" in cycle_df.columns and "R1(Ω)" in cycle_df.columns:
                            imp_rows = []
                            for i in range(len(cycle_df)):
                                imp_rows.append({
                                    "循环次数": int(cycle_df["循环次数"].iloc[i]),
                                    "脉冲编号": 1,
                                    "R0(Ω)": cycle_df["R0(Ω)"].iloc[i],
                                    "R1(Ω)": cycle_df["R1(Ω)"].iloc[i],
                                    "tau(s)": cycle_df["tau(s)"].iloc[i] if "tau(s)" in cycle_df.columns else np.nan,
                                    "脉冲持续时间(s)": 2.0,
                                    "脉冲电流(A)": 10.0,
                                    "异常": "",
                                })
                            imp_df = pd.DataFrame(imp_rows)
                    st.session_state.diag_impedance_df = imp_df
                    if len(imp_df) > 0:
                        st.success(f"识别到 {len(imp_df)} 个有效脉冲")
                    else:
                        st.warning("未识别到有效脉冲，请调整阈值或检查数据")
                else:
                    st.warning("请先加载数据")

        if st.session_state.diag_impedance_df is not None and len(st.session_state.diag_impedance_df) > 0:
            imp_df = st.session_state.diag_impedance_df
            st.markdown("#### 脉冲识别与拟合结果")
            st.dataframe(imp_df.round(5), width="stretch", hide_index=True)

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            r0_by_cycle = imp_df.groupby("循环次数")["R0(Ω)"].mean()
            r1_by_cycle = imp_df.groupby("循环次数")["R1(Ω)"].mean()
            cycles_plt = r0_by_cycle.index.values

            r0_init = r0_by_cycle.iloc[0] if len(r0_by_cycle) > 0 else 1.0
            r0_mask = r0_by_cycle.values > r0_init * 1.2
            axes[0].plot(cycles_plt, r0_by_cycle.values, "b.-", label="R0", markersize=5)
            if r0_mask.any():
                axes[0].scatter(cycles_plt[r0_mask], r0_by_cycle.values[r0_mask], color="red", s=80, zorder=5, label=f"异常(>{20}%初始值)", edgecolor="k")
            axes[0].axhline(y=r0_init * 1.2, color="orange", linestyle="--", alpha=0.7, label="阈值(120%初始)")
            axes[0].set_xlabel("循环次数"); axes[0].set_ylabel("R0 (Ω)"); axes[0].set_title("瞬时内阻R0趋势")
            axes[0].legend(); axes[0].grid(True, alpha=0.3)

            r1_valid = ~np.isnan(r1_by_cycle.values)
            if r1_valid.sum() > 0:
                r1_init = r1_by_cycle.values[r1_valid][0]
                r1_mask = r1_by_cycle.values > r1_init * 1.2
                axes[1].plot(cycles_plt[r1_valid], r1_by_cycle.values[r1_valid], "g.-", label="R1", markersize=5)
                if r1_mask.any():
                    axes[1].scatter(cycles_plt[r1_mask], r1_by_cycle.values[r1_mask], color="red", s=80, zorder=5, label=f"异常(>{20}%初始值)", edgecolor="k")
                axes[1].axhline(y=r1_init * 1.2, color="orange", linestyle="--", alpha=0.7, label="阈值(120%初始)")
            axes[1].set_xlabel("循环次数"); axes[1].set_ylabel("R1 (Ω)"); axes[1].set_title("极化内阻R1趋势")
            axes[1].legend(); axes[1].grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

    with soh_tab:
        st.markdown("#### SOH多模型估计")
        if st.button("拟合全部SOH模型", key="diag_fit_soh"):
            with st.spinner("正在训练SOH估计模型..."):
                if st.session_state.diag_cycle_df is not None:
                    cycle_df = st.session_state.diag_cycle_df.copy()
                    if st.session_state.diag_impedance_df is not None and "R0(Ω)" not in cycle_df.columns:
                        imp_agg = st.session_state.diag_impedance_df.groupby("循环次数")[["R0(Ω)", "R1(Ω)", "tau(s)"]].mean().reset_index()
                        cycle_df = cycle_df.merge(imp_agg, on="循环次数", how="left")
                    dqdv_res = st.session_state.diag_dqdv_results if st.session_state.diag_dqdv_results else {}
                    models = diag.fit_all_soh_models(cycle_df, dqdv_res)
                    st.session_state.diag_soh_models = models
                    diag.soh_models = models
                    st.success(f"已拟合 {len(models)} 个SOH模型")
                else:
                    st.warning("请先准备循环数据")

        if st.session_state.diag_soh_models:
            models = st.session_state.diag_soh_models
            metric_rows = []
            for name, m in models.items():
                if "rmse" in m:
                    row = {"模型": name, "RMSE(%)": round(m["rmse"], 3), "MAE(%)": round(m["mae"], 3)}
                    if m.get("rmse_test") is not None:
                        row["测试集RMSE(%)"] = round(m["rmse_test"], 3)
                        row["测试集MAE(%)"] = round(m["mae_test"], 3)
                    metric_rows.append(row)
            if metric_rows:
                st.markdown("#### 模型性能指标")
                st.dataframe(pd.DataFrame(metric_rows), width="stretch", hide_index=True)

            model_names = list(models.keys())
            st.markdown("**选择显示的模型**")
            checkbox_cols = st.columns(len(model_names))
            selected_models = []
            model_latest_soh = {}
            for i, name in enumerate(model_names):
                m = models[name]
                latest_soh = float(m["soh_pred"][-1])
                model_latest_soh[name] = latest_soh
                with checkbox_cols[i]:
                    if st.checkbox(f"{name} ({latest_soh:.1f}%)", value=True, key=f"diag_soh_cb_{name}"):
                        selected_models.append(name)

            fig, ax = plt.subplots(figsize=(10, 6))
            soh_true_plotted = False
            colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
            color_map = dict(zip(model_names, colors))

            for name in selected_models:
                m = models[name]
                color = color_map[name]
                latest_soh = model_latest_soh[name]
                if "cycles_all" in m:
                    cyc = m["cycles_all"]
                    pred = m["soh_pred"]
                    ax.plot(cyc, pred, label=f"{name} | 最新: {latest_soh:.1f}% | RMSE={m.get('rmse', 0):.2f}%", linewidth=2, color=color)
                    if "soh_true" in m and not soh_true_plotted:
                        ax.scatter(cyc, m["soh_true"], color="black", alpha=0.4, s=20, label="真实SOH", zorder=2)
                        soh_true_plotted = True
                elif "cycles" in m:
                    cyc = m["cycles"]
                    pred = m["soh_pred"]
                    ax.plot(cyc, pred, label=f"{name} | 最新: {latest_soh:.1f}% | RMSE={m.get('rmse', 0):.2f}%", linewidth=2, color=color)
                    if "soh_true" in m and not soh_true_plotted:
                        ax.scatter(cyc, m["soh_true"], color="black", alpha=0.4, s=20, label="真实SOH", zorder=2)
                        soh_true_plotted = True
            ax.axhline(y=80, color="red", linestyle="--", label="EOL=80% SOH", linewidth=1)
            ax.set_xlabel("循环次数")
            ax.set_ylabel("SOH (%)")
            ax.set_title("多模型SOH估计结果对比")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            if "特征回归模型" in models:
                feat_model = models["特征回归模型"]
                if "coef" in feat_model:
                    st.markdown("#### Ridge回归特征系数")
                    coef_df = pd.DataFrame([{"特征": k, "系数": round(v, 6)} for k, v in feat_model["coef"].items()])
                    coef_df = pd.concat([coef_df, pd.DataFrame([{"特征": "截距", "系数": round(feat_model["intercept"], 4)}])], ignore_index=True)
                    st.dataframe(coef_df, width="stretch", hide_index=True)

    with fault_tab:
        st.markdown("#### 故障预警规则引擎")
        st.markdown("**阈值配置**")
        col1, col2 = st.columns(2)
        with col1:
            thr_r = st.slider("内阻突增阈值(%)", 5, 50, 15, key="diag_thr_r") / 100.0
            thr_cap = st.slider("容量跳变阈值(%)", 1, 20, 3, key="diag_thr_cap") / 100.0
            thr_temp = st.slider("充电温升阈值(°C)", 3, 15, 8, key="diag_thr_temp")
        with col2:
            thr_peak = st.slider("dQ/dV峰值消失阈值(首循环%)", 10, 80, 30, key="diag_thr_peak") / 100.0
            thr_soh = st.slider("SOH一致性标准差阈值(%)", 1, 20, 5, key="diag_thr_soh")

        thresholds = {
            "r_sudden_increase": float(thr_r),
            "capacity_jump": float(thr_cap),
            "peak_disappear_ratio": float(thr_peak),
            "soh_consistency_std": float(thr_soh),
            "charge_temp_rise": float(thr_temp),
        }

        if st.button("运行故障检测", key="diag_run_fault"):
            with st.spinner("正在执行故障规则检测..."):
                if st.session_state.diag_cycle_df is not None:
                    cycle_df = st.session_state.diag_cycle_df.copy()
                    if st.session_state.diag_impedance_df is not None and "R0(Ω)" not in cycle_df.columns:
                        imp_agg = st.session_state.diag_impedance_df.groupby("循环次数")[["R0(Ω)", "R1(Ω)", "tau(s)"]].mean().reset_index()
                        cycle_df = cycle_df.merge(imp_agg, on="循环次数", how="left")
                    events = diag.run_fault_detection(
                        cycle_df,
                        st.session_state.diag_dqdv_results if st.session_state.diag_dqdv_results else {},
                        st.session_state.diag_soh_models if st.session_state.diag_soh_models else {},
                        st.session_state.diag_charge_data if st.session_state.diag_charge_data else None,
                        thresholds,
                    )
                    st.session_state.diag_fault_events = events
                    diag.fault_events = events
                    st.success(f"检测完成，共触发 {len(events)} 条故障事件")
                else:
                    st.warning("请先准备数据")

        if st.session_state.diag_fault_events:
            events = st.session_state.diag_fault_events
            col_header, col_btn = st.columns([3, 1])
            with col_header:
                st.markdown(f"#### 故障日志 (共 {len(events)} 条)")
            with col_btn:
                event_df_raw = pd.DataFrame(events)
                csv_export_df = event_df_raw.copy()
                if "时间" in csv_export_df.columns:
                    csv_export_df["时间"] = csv_export_df["时间"].apply(
                        lambda x: x.strftime("%Y-%m-%d %H:%M:%S") if hasattr(x, "strftime") else str(x)
                    )
                csv_data = csv_export_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                st.download_button(
                    label="📥 导出CSV",
                    data=csv_data,
                    file_name=f"fault_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    disabled=False,
                    key="diag_export_csv",
                )
            level_color = {"警告": "🟡", "严重": "🟠", "危险": "🔴"}
            event_df = pd.DataFrame(events)
            if "严重等级" in event_df.columns:
                event_df["严重等级"] = event_df["严重等级"].apply(lambda x: f"{level_color.get(x, '⚪')} {x}")
            event_df["时间"] = event_df["时间"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S") if hasattr(x, "strftime") else str(x))
            st.dataframe(event_df, width="stretch", hide_index=True)

            level_counts = pd.Series([ev.get("严重等级", "未知") for ev in events]).value_counts()
            fig, ax = plt.subplots(figsize=(6, 4))
            colors_map = {"警告": "gold", "严重": "orange", "危险": "red"}
            bar_colors = [colors_map.get(i, "gray") for i in level_counts.index]
            ax.bar(level_counts.index, level_counts.values, color=bar_colors)
            ax.set_title("故障事件严重等级分布")
            ax.set_ylabel("数量")
            st.pyplot(fig)
        else:
            col_header, col_btn = st.columns([3, 1])
            with col_header:
                st.info("暂无故障事件，请运行故障检测")
            with col_btn:
                if st.button("📥 导出CSV", key="diag_export_csv_disabled", disabled=True, help="无故障事件可导出"):
                    pass

    with report_tab:
        st.markdown("#### 诊断报告")
        if st.button("生成诊断报告", key="diag_gen_report"):
            with st.spinner("正在生成诊断报告..."):
                if st.session_state.diag_cycle_df is not None:
                    report = diag.generate_diagnosis_report(
                        st.session_state.diag_cycle_df,
                        st.session_state.diag_soh_models if st.session_state.diag_soh_models else {},
                        st.session_state.diag_fault_events if st.session_state.diag_fault_events else [],
                    )
                    st.session_state.diag_report = report
                else:
                    st.warning("请先准备数据并执行分析")

        if st.session_state.diag_report:
            report = st.session_state.diag_report

            score = report["健康评分"]
            if score >= 80:
                score_color = "green"
                score_text = "优秀"
            elif score >= 60:
                score_color = "orange"
                score_text = "一般"
            else:
                score_color = "red"
                score_text = "较差"

            with st.expander("🏥 健康诊断摘要", expanded=True):
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("健康评分", f"{score:.1f}/100", score_text, delta_color="inverse" if score < 60 else "normal")
                with col_b:
                    soh_val = report.get("SOH均值(%)")
                    if soh_val is not None:
                        st.metric("当前SOH均值", f"{soh_val:.1f}%")
                with col_c:
                    rc = report.get("剩余寿命预估(循环)")
                    if rc is not None:
                        st.metric("剩余寿命预估", f"{rc:.0f} 循环")

                st.markdown("#### 各模型SOH估计")
                model_sohs = report.get("各模型SOH(%)", {})
                if model_sohs:
                    cols = st.columns(len(model_sohs))
                    for i, (mname, mval) in enumerate(model_sohs.items()):
                        with cols[i]:
                            st.metric(mname, f"{mval:.1f}%")

                st.markdown("#### 故障统计")
                st.metric("触发故障事件数", report.get("故障事件数", 0))

                st.markdown("#### 维护建议")
                for i, s in enumerate(report.get("维护建议", [])):
                    st.info(f"💡 建议{i+1}: {s}")

            with st.expander("📉 SOH衰减速率趋势分析", expanded=True):
                decay_analysis = report.get("SOH衰减速率分析")
                if decay_analysis is None:
                    st.info("请先拟合容量衰减模型以进行衰减速率分析")
                elif "error" in decay_analysis:
                    st.warning(decay_analysis["error"])
                else:
                    window_centers = decay_analysis["window_centers"]
                    decay_rates = decay_analysis["decay_rates"]
                    accelerating = decay_analysis["accelerating"]
                    window_size = decay_analysis["window_size"]

                    st.markdown(f"**分析说明**: 使用最近10个循环的SOH数据，以{window_size}个循环为滑动窗口计算衰减速率，窗口内做线性回归取斜率作为衰减速率。")

                    fig, ax = plt.subplots(figsize=(10, 5))
                    rates_arr = np.array(decay_rates)
                    centers_arr = np.array(window_centers)
                    accel_arr = np.array(accelerating)

                    normal_mask = ~accel_arr
                    accel_mask = accel_arr

                    if normal_mask.any():
                        ax.plot(centers_arr[normal_mask], rates_arr[normal_mask], "bo-", label="正常衰减速率", linewidth=2, markersize=8)
                    if accel_mask.any():
                        ax.plot(centers_arr[accel_mask], rates_arr[accel_mask], "ro-", label="加速衰减(>20%)", linewidth=2, markersize=8)

                    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
                    ax.set_xlabel("循环次数(窗口中心)")
                    ax.set_ylabel("SOH衰减速率 (%/循环)")
                    ax.set_title(f"滑动窗口SOH衰减速率趋势 (窗口大小={window_size})")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)

                    if any(accelerating):
                        accel_cycles = [centers_arr[i] for i in range(len(accelerating)) if accelerating[i]]
                        st.error(f"⚠️ 检测到衰减加速窗口: 循环 {', '.join([f'{int(c)}' for c in accel_cycles])}，建议加强监控并缩短维护周期")
                    else:
                        st.success("✅ 衰减速率稳定，未检测到明显加速")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("平均衰减速率", f"{np.mean(rates_arr):.4f} %/循环")
                    with col2:
                        st.metric("最小衰减速率", f"{np.min(rates_arr):.4f} %/循环")
                    with col3:
                        st.metric("最大衰减速率", f"{np.max(rates_arr):.4f} %/循环")

            with st.expander("📊 趋势数据汇总", expanded=False):
                if st.session_state.diag_cycle_df is not None:
                    cdf = st.session_state.diag_cycle_df
                    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                    axes[0, 0].plot(cdf["循环次数"], cdf["放电容量(Ah)"], "b-")
                    axes[0, 0].set_xlabel("循环"); axes[0, 0].set_ylabel("容量(Ah)")
                    axes[0, 0].set_title("容量衰减"); axes[0, 0].grid(True, alpha=0.3)

                    if "R0(Ω)" in cdf.columns:
                        axes[0, 1].plot(cdf["循环次数"], cdf["R0(Ω)"], "r-")
                        axes[0, 1].set_xlabel("循环"); axes[0, 1].set_ylabel("R0(Ω)")
                        axes[0, 1].set_title("内阻R0增长"); axes[0, 1].grid(True, alpha=0.3)

                    if st.session_state.diag_dqdv_results:
                        res = st.session_state.diag_dqdv_results
                        keys = list(res.keys())
                        if len(keys) >= 3:
                            show_keys = [keys[0], keys[len(keys) // 2], keys[-1]]
                            for name in show_keys:
                                if name in res:
                                    r = res[name]
                                    axes[1, 0].plot(r["voltage"], r["dqdv"], label=name, linewidth=1)
                            axes[1, 0].set_xlabel("电压(V)"); axes[1, 0].set_ylabel("dQ/dV")
                            axes[1, 0].set_title("dQ/dV演化"); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

                    if st.session_state.diag_soh_models:
                        for name, m in st.session_state.diag_soh_models.items():
                            if "cycles_all" in m:
                                axes[1, 1].plot(m["cycles_all"], m["soh_pred"], label=name, linewidth=1.5)
                            elif "cycles" in m:
                                axes[1, 1].plot(m["cycles"], m["soh_pred"], label=name, linewidth=1.5)
                        axes[1, 1].axhline(y=80, color="red", linestyle="--")
                        axes[1, 1].set_xlabel("循环"); axes[1, 1].set_ylabel("SOH(%)")
                        axes[1, 1].set_title("SOH多模型估计"); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
