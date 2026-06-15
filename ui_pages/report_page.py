import streamlit as st
import pandas as pd
import numpy as np
from utils.plot_config import setup_chinese_font
setup_chinese_font()
import matplotlib.pyplot as plt
from core import ReportGenerator
from io import BytesIO


def render():
    st.subheader("📑 报告导出")
    dm = st.session_state.data_manager

    if len(dm.packs) == 0:
        st.info("请先在数据管理页面导入数据")
        return

    sel_pack = st.selectbox("选择Pack生成报告", dm.list_packs(), key="rp_pack")
    modules = dm.list_modules(sel_pack)

    st.markdown("### 报告内容预览")

    pack_info = {
        "module_count": len(modules),
        "capacity": dm.get_pack_capacity(sel_pack),
        "total_records": sum(len(dm.packs[sel_pack]["modules"].get(m, [])) for m in modules),
        "time_span": "-",
        "cycles": "N/A",
    }

    total_ts = []
    for m in modules:
        md = dm.get_module_data(sel_pack, m)
        if md is not None and len(md) > 0:
            total_ts.extend(md["timestamp"].tolist())
    if total_ts:
        pack_info["time_span"] = str(max(total_ts) - min(total_ts))

    module_params_list = []
    for m in modules:
        md = dm.get_module_data(sel_pack, m)
        if md is not None and len(md) > 0:
            module_params_list.append({
                "模组编号": m,
                "电压均值(V)": f"{md['voltage'].mean():.3f}",
                "电压标准差(V)": f"{md['voltage'].std():.4f}",
                "电流均值(A)": f"{md['current'].mean():.3f}",
                "温度均值(°C)": f"{md['temperature'].mean():.2f}",
                "记录数": len(md),
            })
    module_params_df = pd.DataFrame(module_params_list)

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Pack概览**")
        for k, v in pack_info.items():
            st.write(f"- {k}: {v}")
    with col2:
        st.write("**模组参数对比**")
        st.dataframe(module_params_df, width="stretch", hide_index=True)

    soc_result = None
    if "kf_result_df" in st.session_state:
        kf_type_name = st.session_state["kf_type_name"]
        soc_col = f"soc_{kf_type_name.lower()}"
        result_df = st.session_state["kf_result_df"]
        soc_result = {
            "soc_mean": float(result_df[soc_col].mean()),
            "soc_min": float(result_df[soc_col].min()),
            "soc_max": float(result_df[soc_col].max()),
            "method": kf_type_name,
        }
        st.markdown(f"**SOC估计结果摘要** ({kf_type_name})")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("SOC均值(%)", f"{soc_result['soc_mean']:.2f}")
        with col_s2:
            st.metric("SOC最小值(%)", f"{soc_result['soc_min']:.2f}")
        with col_s3:
            st.metric("SOC最大值(%)", f"{soc_result['soc_max']:.2f}")

    life_result = st.session_state.get("lp_models")
    if life_result:
        st.markdown("**寿命预测模型**")
        life_rows = []
        for name, m in life_result.items():
            if "params" in m:
                life_rows.append({
                    "模型": name,
                    "R²": f"{m['r_squared']:.4f}",
                    "EOL循环次数": f"{m['eol_cycles']:.0f}",
                    "剩余循环次数": f"{m['remaining_cycles']:.0f}",
                })
        st.dataframe(pd.DataFrame(life_rows), width="stretch", hide_index=True)

    consistency_result = st.session_state.get("ca_result")
    if consistency_result:
        st.markdown("**短板模组**")
        for sb in consistency_result.get("shortboards", []):
            st.write(f"- {sb['类型']}: 模组 {sb['模组']} = {sb['值']} {sb['单位']}")

    alerts_df = pd.DataFrame(st.session_state.alerts) if st.session_state.alerts else pd.DataFrame()
    if len(alerts_df) > 0:
        st.markdown(f"**告警统计**: 共 {len(alerts_df)} 条告警")

    if st.button("📄 生成PDF报告", key="rp_gen", type="primary"):
        with st.spinner("正在生成PDF报告..."):
            try:
                soc_plot_buf = None
                life_plot_buf = None

                if "kf_result_df" in st.session_state:
                    kf_type_name = st.session_state["kf_type_name"]
                    soc_col = f"soc_{kf_type_name.lower()}"
                    result_df = st.session_state["kf_result_df"]
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(result_df["timestamp"], result_df[soc_col], "r-", label=kf_type_name, linewidth=1)
                    if "soc_ah" in result_df.columns:
                        ax.plot(result_df["timestamp"], result_df["soc_ah"], "b--", label="AH积分", linewidth=0.8, alpha=0.7)
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    soc_plot_buf = BytesIO()
                    fig.savefig(soc_plot_buf, format="png", dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    soc_plot_buf.seek(0)
                    if soc_result:
                        soc_result["plot"] = soc_plot_buf

                if life_result:
                    lp = st.session_state.get("lp_instance")
                    cycles = st.session_state.get("lp_cycles", np.linspace(0, 1000, 100))
                    plot_cycles = np.linspace(0, max(cycles.max() * 1.5, 1500), 300)
                    fig, ax = plt.subplots(figsize=(10, 5))
                    for name, m in life_result.items():
                        if "func" in m:
                            y_pred = lp.predict_at_cycles(name, plot_cycles)
                            ax.plot(plot_cycles, y_pred, label=name, linewidth=1)
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    life_plot_buf = BytesIO()
                    fig.savefig(life_plot_buf, format="png", dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    life_plot_buf.seek(0)
                    if isinstance(life_result, dict):
                        life_result = dict(life_result)
                        life_result["plot"] = life_plot_buf

                rg = ReportGenerator()
                pdf_buf = rg.generate_pdf(
                    pack_id=sel_pack,
                    pack_info=pack_info,
                    module_params=module_params_df,
                    soc_result=soc_result,
                    life_result=life_result,
                    consistency_result=consistency_result,
                    alerts=alerts_df,
                )
                st.success("✅ 报告生成成功!")
                st.download_button(
                    label="⬇️ 下载PDF报告",
                    data=pdf_buf.getvalue(),
                    file_name=f"Battery_Health_Report_{sel_pack}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                )
            except Exception as e:
                st.error(f"报告生成失败: {e}")
                import traceback
                st.code(traceback.format_exc())
