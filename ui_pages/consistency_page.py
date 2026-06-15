import streamlit as st
import pandas as pd
import numpy as np
from utils.plot_config import setup_chinese_font
setup_chinese_font()
import matplotlib.pyplot as plt
from core import ConsistencyAnalyzer


def render():
    st.subheader("⚠️ 一致性分析与预警")
    dm = st.session_state.data_manager

    with st.expander("预警阈值配置", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.thresholds["soc_dispersion"] = st.slider(
                "SOC离散度阈值(%)", 1.0, 20.0, st.session_state.thresholds["soc_dispersion"], 0.5
            )
            st.session_state.thresholds["internal_resistance_ratio"] = st.slider(
                "内阻超标倍数(×均值)", 1.1, 3.0, st.session_state.thresholds["internal_resistance_ratio"], 0.1
            )
        with col2:
            st.session_state.thresholds["temperature_diff"] = st.slider(
                "温差阈值(°C)", 2.0, 20.0, st.session_state.thresholds["temperature_diff"], 1.0
            )
            st.session_state.thresholds["capacity_decay_ratio"] = st.slider(
                "容量衰减加速系数", 1.1, 5.0, st.session_state.thresholds["capacity_decay_ratio"], 0.1
            )

    if len(dm.packs) == 0:
        st.info("请先在数据管理页面导入数据")
        return

    sel_pack = st.selectbox("选择Pack", dm.list_packs(), key="ca_pack")
    modules = dm.list_modules(sel_pack)

    if st.button("执行一致性分析", key="ca_run"):
        with st.spinner("分析中..."):
            aligned = dm.get_aligned_pack_data(sel_pack)
            analyzer = ConsistencyAnalyzer(st.session_state.thresholds)
            result = analyzer.analyze_pack(aligned, modules)
            st.session_state["ca_result"] = result
            st.session_state["ca_aligned"] = aligned
            st.session_state.alerts = analyzer.alerts + st.session_state.alerts

    if "ca_result" not in st.session_state:
        return

    result = st.session_state["ca_result"]
    aligned = st.session_state["ca_aligned"]

    col1, col2, col3, col4 = st.columns(4)
    metrics = result.get("metrics", {})
    with col1:
        v_disp = np.array(metrics.get("voltage_dispersion", [0]))
        st.metric("电压离散度-均值(V)", f"{v_disp.mean():.4f}")
    with col2:
        t_disp = np.array(metrics.get("temperature_dispersion", [0]))
        st.metric("温差均值(°C)", f"{t_disp.mean():.2f}")
    with col3:
        st.metric("分析模组数量", len(metrics.get("modules", [])))
    with col4:
        st.metric("告警数量", len(result.get("alerts", [])))

    st.markdown("---")
    st.subheader("Pack内各模组分布(箱线图)")

    v_cols = [f"{m}_voltage" for m in modules if f"{m}_voltage" in aligned.columns]
    t_cols = [f"{m}_temperature" for m in modules if f"{m}_temperature" in aligned.columns]

    if v_cols:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        v_data = [aligned[c].dropna().values for c in v_cols]
        labels = [c.replace("_voltage", "") for c in v_cols]
        bp1 = axes[0].boxplot(v_data, labels=labels, patch_artist=True)
        for patch in bp1["boxes"]:
            patch.set_facecolor("lightblue")
        axes[0].set_title("模组电压分布")
        axes[0].set_ylabel("电压 (V)")
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].grid(True, alpha=0.3)

        if t_cols:
            t_data = [aligned[c].dropna().values for c in t_cols]
            bp2 = axes[1].boxplot(t_data, labels=labels, patch_artist=True)
            for patch in bp2["boxes"]:
                patch.set_facecolor("lightgreen")
            axes[1].set_title("模组温度分布")
            axes[1].set_ylabel("温度 (°C)")
            axes[1].tick_params(axis="x", rotation=45)
            axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("离散度时间趋势")
    ts = pd.to_datetime(metrics.get("timestamps", []))
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(ts, metrics.get("voltage_dispersion", []), "b-", linewidth=0.8)
    axes[0].set_ylabel("电压离散度 (V)")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(ts, metrics.get("temperature_dispersion", []), "r-", linewidth=0.8)
    axes[1].axhline(y=st.session_state.thresholds["temperature_diff"], color="red", linestyle="--", label="阈值")
    axes[1].set_ylabel("温差 (°C)")
    axes[1].set_xlabel("时间")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.autofmt_xdate()
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("🔥 短板识别")
    shortboards = result.get("shortboards", [])
    if shortboards:
        for sb in shortboards:
            st.error(f"**{sb['类型']}**: 模组 {sb['模组']} - {sb['值']} {sb['单位']}")

    st.markdown("---")
    st.subheader("📋 告警记录")

    alerts_df = None
    if "ca_result" in st.session_state and "alerts" in st.session_state.ca_result:
        new_alerts = st.session_state.ca_result["alerts"]
        if new_alerts:
            st.session_state.alerts = new_alerts + st.session_state.alerts

    if st.session_state.alerts:
        alerts_df = pd.DataFrame(st.session_state.alerts)
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_level = st.multiselect("按严重级别筛选", ["high", "medium", "low"], default=["high", "medium", "low"], key="ca_alert_filter")
        if filter_level:
            alerts_df = alerts_df[alerts_df["严重级别"].isin(filter_level)]
        st.dataframe(alerts_df, width="stretch")
    else:
        st.info("暂无告警记录")

    r_est = result.get("r_estimates", {})
    if r_est:
        st.markdown("---")
        st.subheader("内阻估计对比")
        r_df = pd.DataFrame([{"模组": m, "估计内阻(Ω)": f"{v:.6f}"} for m, v in r_est.items()])
        st.dataframe(r_df, width="stretch", hide_index=True)
        fig, ax = plt.subplots(figsize=(10, 4))
        mod_names = list(r_est.keys())
        r_vals = list(r_est.values())
        colors = ["red" if v > np.mean(r_vals) * st.session_state.thresholds["internal_resistance_ratio"] else "steelblue" for v in r_vals]
        ax.bar(mod_names, r_vals, color=colors)
        ax.axhline(y=np.mean(r_vals), color="black", linestyle="--", label=f"均值={np.mean(r_vals):.6f}")
        ax.axhline(y=np.mean(r_vals) * st.session_state.thresholds["internal_resistance_ratio"], color="red", linestyle="--", label="阈值")
        ax.set_ylabel("内阻 (Ω)")
        ax.legend()
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)
