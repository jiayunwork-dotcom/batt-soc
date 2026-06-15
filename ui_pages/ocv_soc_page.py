import streamlit as st
import pandas as pd
import numpy as np
from utils.plot_config import setup_chinese_font
setup_chinese_font()
import matplotlib.pyplot as plt
from core import OCVSOCCalibrator
from io import BytesIO


def render():
    st.subheader("📈 OCV-SOC曲线标定")
    dm = st.session_state.data_manager

    if "ocv_calibrator" not in st.session_state:
        st.session_state["ocv_calibrator"] = OCVSOCCalibrator()

    cal: OCVSOCCalibrator = st.session_state["ocv_calibrator"]

    method = st.radio("标定方法", ["低倍率充放电法", "增量电压法(dV/dQ)"], horizontal=True)

    if len(dm.packs) == 0:
        st.info("请先在数据管理页面导入数据")
        return

    col1, col2 = st.columns(2)
    with col1:
        sel_pack = st.selectbox("选择Pack", dm.list_packs(), key="ocv_pack")
    with col2:
        sel_mod = st.selectbox("选择模组", dm.list_modules(sel_pack), key="ocv_mod")

    df = dm.get_module_data(sel_pack, sel_mod, clean=True)
    q0 = dm.get_pack_capacity(sel_pack)
    st.caption(f"额定容量基准 Q0 = {q0:.3f} Ah")

    if df is None or len(df) < 10:
        st.warning("数据不足")
        return

    if method == "低倍率充放电法":
        col1, col2 = st.columns(2)
        with col1:
            min_soc_c = st.slider("充电SOC范围(%)", 0, 100, (0, 100), key="ocv_c_charge")
        with col2:
            min_soc_d = st.slider("放电SOC范围(%)", 0, 100, (0, 100), key="ocv_c_discharge")

        df_sorted = df.sort_values("timestamp").reset_index(drop=True)
        if "soc_ah" in df_sorted.columns:
            df_charge = df_sorted[
                (df_sorted["current"] > 0)
                & (df_sorted["soc_ah"] >= min_soc_c[0])
                & (df_sorted["soc_ah"] <= min_soc_c[1])
            ]
            df_discharge = df_sorted[
                (df_sorted["current"] < 0)
                & (df_sorted["soc_ah"] >= min_soc_d[0])
                & (df_sorted["soc_ah"] <= min_soc_d[1])
            ]
            if st.button("执行OCV标定", key="ocv_calibrate"):
                with st.spinner("标定中..."):
                    result = cal.calibrate_low_rate(df_charge, df_discharge, q0)
                    st.success("标定完成")

    else:
        if st.button("执行dV/dQ标定", key="ocv_dvdq"):
            with st.spinner("dV/dQ分析中..."):
                result = cal.calibrate_dvdq(df, q0)
                if "dvdq_peaks" in result and result["dvdq_peaks"]:
                    st.write(f"检测到 {len(result['dvdq_peaks'])} 个相变平台峰值")
                    for p in result["dvdq_peaks"]:
                        st.caption(f"SOC={p['soc']:.1f}%, dV/dQ={p['dvdq']:.4f}")

    if cal.ocv_mean is not None:
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(10, 5))
            if cal.ocv_charge is not None:
                ax.plot(cal.soc_points, cal.ocv_charge, "b--", label="充电OCV", alpha=0.6)
            if cal.ocv_discharge is not None:
                ax.plot(cal.soc_points, cal.ocv_discharge, "r--", label="放电OCV", alpha=0.6)
            ax.plot(cal.soc_points, cal.ocv_mean, "k-", label="平均OCV", linewidth=2)
            ax.set_xlabel("SOC (%)")
            ax.set_ylabel("OCV (V)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        with col2:
            st.dataframe(cal.get_lookup_table(), width="stretch", height=300)

        st.markdown("---")
        st.subheader("多项式拟合")
        poly_order = st.slider("多项式阶数", 5, 8, 5, key="ocv_poly_order")
        if st.button("执行多项式拟合", key="ocv_fit_poly"):
            poly_res = cal.fit_polynomial(poly_order)
            st.success(f"拟合完成: R² = {poly_res['r_squared']:.6f}")
            v_poly = np.polyval(poly_res["coeffs"], cal.soc_points / 100.0)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(cal.soc_points, cal.ocv_mean, "k.-", label="原始OCV", markersize=3, linewidth=0.6)
            ax.plot(cal.soc_points, v_poly, "r-", label=f"{poly_order}阶多项式拟合", linewidth=1.5)
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        st.markdown("---")
        st.subheader("手动编辑异常点")
        col1, col2 = st.columns(2)
        with col1:
            edit_soc = st.number_input("编辑SOC点(%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
        with col2:
            edit_ocv = st.number_input("新OCV值(V)", min_value=2.5, max_value=4.2, value=3.7, step=0.01)
        if st.button("应用编辑", key="ocv_edit"):
            cal.edit_point(edit_soc, edit_ocv)
            st.success(f"已更新 SOC={edit_soc:.0f}% 处OCV为{edit_ocv}V")
            st.rerun()
