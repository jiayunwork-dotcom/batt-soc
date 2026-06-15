import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from utils.plot_config import setup_chinese_font
setup_chinese_font()
import matplotlib.pyplot as plt


def _generate_sample_data(n_points: int = 2000) -> pd.DataFrame:
    module_names = ["M001", "M002", "M003", "M004"]
    n_per_mod = n_points // len(module_names)
    t = pd.date_range(start="2024-01-01", periods=n_per_mod, freq="10s")
    soc_base = np.linspace(100, 10, n_per_mod)
    current_base = np.where(soc_base > 70, 8.0, np.where(soc_base > 30, 5.0, 2.0))
    pulse_mask = (np.arange(n_per_mod) % 60) < 6
    current_base = np.where(pulse_mask, 25.0, current_base)
    temperature_base = 25 + 3 * np.sin(np.arange(n_per_mod) / 100)

    all_dfs = []
    for i, mod_id in enumerate(module_names):
        soc_mod = soc_base - i * 3.0
        voltage_offset = 0.02 * i
        r_mod = 0.004 + i * 0.001
        ocv_base = 3.0 + 0.012 * soc_mod + 0.07 * np.sin(soc_mod / 8)
        voltage = ocv_base - current_base * r_mod + 0.015 * np.sin(np.arange(n_per_mod) / 15 + i)
        voltage += voltage_offset + np.random.randn(n_per_mod) * 0.008
        temp = temperature_base + i * 2.5 + np.random.randn(n_per_mod) * 0.3
        mod_df = pd.DataFrame({
            "timestamp": t,
            "voltage": voltage,
            "current": current_base * (1 + i * 0.15),
            "temperature": temp,
            "module_id": mod_id,
        })
        all_dfs.append(mod_df)
    df = pd.concat(all_dfs, ignore_index=True)
    return df


def render():
    st.subheader("📊 BMS数据管理")
    dm = st.session_state.data_manager

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader("上传BMS数据CSV文件", type=["csv"])
    with col2:
        pack_id_input = st.text_input("Pack编号", value="PACK001")
        if st.button("生成示例数据", key="gen_sample"):
            sample_df = _generate_sample_data(2000)
            buf = BytesIO()
            sample_df.to_csv(buf, index=False)
            buf.seek(0)
            result = dm.import_csv(buf, pack_id_input)
            st.success(f"已导入示例数据: {len(result['modules'])} 个模组")

    if uploaded_file is not None:
        try:
            result = dm.import_csv(uploaded_file, pack_id_input)
            st.success(f"成功导入数据: {pack_id_input}")
            if result["warnings"]:
                with st.expander(f"⚠️ 数据校验警告 ({len(result['warnings'])})", expanded=False):
                    for w in result["warnings"]:
                        st.warning(w)
        except Exception as e:
            st.error(f"导入失败: {e}")

    st.markdown("---")

    if len(dm.packs) == 0:
        st.info("暂无数据，请上传CSV文件或生成示例数据")
        return

    overview = dm.get_pack_overview()
    st.dataframe(overview, width="stretch")

    st.markdown("---")
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        sel_pack = st.selectbox("选择Pack", dm.list_packs(), key="dm_pack_sel")
    with col_sel2:
        sel_mod = st.selectbox("选择模组", dm.list_modules(sel_pack), key="dm_mod_sel")

    df = dm.get_module_data(sel_pack, sel_mod, clean=False)
    if df is not None and len(df) > 0:
        st.markdown(f"### 模组 {sel_mod} 数据概览")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("电压均值(V)", f"{df['voltage'].mean():.3f}")
        with col2:
            st.metric("电流均值(A)", f"{df['current'].mean():.3f}")
        with col3:
            st.metric("温度均值(°C)", f"{df['temperature'].mean():.2f}")
        with col4:
            st.metric("异常记录数", int(df["anomaly_flag"].fillna("").str.len().gt(0).sum()))

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        axes[0].plot(df["timestamp"], df["voltage"], "b-", linewidth=0.6)
        axes[0].set_ylabel("电压 (V)")
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(df["timestamp"], df["current"], "r-", linewidth=0.6)
        axes[1].set_ylabel("电流 (A)")
        axes[1].grid(True, alpha=0.3)
        axes[2].plot(df["timestamp"], df["temperature"], "g-", linewidth=0.6)
        axes[2].set_ylabel("温度 (°C)")
        axes[2].grid(True, alpha=0.3)
        fig.autofmt_xdate()
        st.pyplot(fig)

        with st.expander("查看原始数据", expanded=False):
            st.dataframe(df, width="stretch")
