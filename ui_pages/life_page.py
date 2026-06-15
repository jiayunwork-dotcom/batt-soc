import streamlit as st
import pandas as pd
import numpy as np
from utils.plot_config import setup_chinese_font
setup_chinese_font()
import matplotlib.pyplot as plt
from core import LifePrediction
from io import BytesIO


def _generate_sample_cycle_data():
    np.random.seed(42)
    n = 50
    cycles = np.linspace(0, 1000, n)
    q0 = 100.0
    k = 0.005
    capacity = q0 - k * cycles + np.random.randn(n) * 0.5
    return pd.DataFrame({"循环次数": cycles, "放电容量(Ah)": capacity})


def render():
    st.subheader("🔮 循环寿命预测")

    tab1, tab2, tab3 = st.tabs(["容量衰减建模", "温度加速因子", "DOD影响分析"])

    with tab1:
        data_source = st.radio("数据来源", ["上传循环数据CSV", "使用示例数据"], horizontal=True, key="lp_ds")
        cycle_df = None
        if data_source == "上传循环数据CSV":
            uploaded = st.file_uploader("上传循环数据(列: 循环次数, 放电容量)", type=["csv"])
            if uploaded is not None:
                cycle_df = pd.read_csv(uploaded)
        else:
            cycle_df = _generate_sample_cycle_data()

        if cycle_df is not None:
            st.dataframe(cycle_df.head(10), width="stretch")

            eol_ratio = st.slider("EOL容量比例(%)", 60, 90, 80, key="lp_eol") / 100.0
            q0 = cycle_df["放电容量(Ah)"].max()

            if st.button("拟合寿命模型", key="lp_fit"):
                with st.spinner("模型拟合中..."):
                    lp = LifePrediction(q0=q0, eol_ratio=eol_ratio)
                    models = lp.fit_all(cycle_df["循环次数"].values, cycle_df["放电容量(Ah)"].values)
                    st.session_state["lp_instance"] = lp
                    st.session_state["lp_models"] = models
                    st.session_state["lp_cycles"] = cycle_df["循环次数"].values

        if "lp_models" in st.session_state:
            models = st.session_state["lp_models"]
            lp = st.session_state["lp_instance"]

            res_rows = []
            for name, m in models.items():
                if "params" in m:
                    res_rows.append({
                        "模型": name,
                        "参数": ", ".join([f"{p:.4e}" for p in m["params"]]),
                        "R²": f"{m['r_squared']:.4f}",
                        "EOL循环次数": f"{m['eol_cycles']:.0f}",
                        "剩余循环次数": f"{m['remaining_cycles']:.0f}",
                    })
            st.dataframe(pd.DataFrame(res_rows), width="stretch", hide_index=True)

            plot_cycles = np.linspace(0, max(st.session_state["lp_cycles"].max() * 2, 2000), 500)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(cycle_df["循环次数"], cycle_df["放电容量(Ah)"], alpha=0.5, s=20, label="实测数据", zorder=5)
            for name, m in models.items():
                if "func" in m:
                    y_pred = lp.predict_at_cycles(name, plot_cycles)
                    ax.plot(plot_cycles, y_pred, label=f"{name} (R²={m.get('r_squared', 0):.3f})", linewidth=1.5)
            ax.axhline(y=lp.eol, color="red", linestyle="--", label=f"EOL={eol_ratio*100:.0f}% Q0", linewidth=1)
            ax.set_xlabel("循环次数")
            ax.set_ylabel("放电容量 (Ah)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

    with tab2:
        st.markdown("#### 不同温度下的衰减数据")
        st.caption("基于Arrhenius方程拟合活化能Ea")

        temps = st.multiselect("选择温度(°C)", [-10, 0, 10, 25, 35, 45, 55], default=[25, 45], key="lp_temps")

        if temps:
            temp_data = {}
            cols = st.columns(len(temps))
            for i, t in enumerate(temps):
                with cols[i]:
                    st.write(f"温度: {t}°C")
                    n = st.number_input(f"数据点数-{t}", min_value=3, max_value=200, value=20, key=f"lp_n_{t}")
                    if st.button(f"生成示例-{t}°C", key=f"lp_gen_{t}"):
                        np.random.seed(abs(t) + 100)
                        cyc = np.linspace(0, 800, n)
                        k_temp = 0.002 * np.exp(-0.3 * (t - 25) / 25)
                        cap = 100.0 - k_temp * cyc + np.random.randn(n) * 0.3
                        temp_data[str(t)] = (cyc, cap)
                        st.session_state[f"lp_temp_data_{t}"] = (cyc, cap)

            if st.button("拟合Arrhenius方程", key="lp_arrh"):
                all_temp_data = {}
                for t in temps:
                    key = f"lp_temp_data_{t}"
                    if key in st.session_state:
                        all_temp_data[str(t)] = st.session_state[key]
                if len(all_temp_data) >= 2:
                    lp = LifePrediction()
                    arrh_res = lp.fit_arrhenius(all_temp_data)
                    if arrh_res:
                        st.metric("活化能 Ea (eV)", f"{arrh_res.get('activation_energy_eV', 0):.4f}")
                        temp_curve = arrh_res.get("temp_curve")
                        if temp_curve is not None and len(temp_curve) > 0:
                            fig, ax = plt.subplots(figsize=(10, 4))
                            ax.plot(temp_curve["温度(°C)"], temp_curve["相对衰减速率"], "b-", linewidth=1.5)
                            ax.set_xlabel("温度 (°C)")
                            ax.set_ylabel("相对衰减速率 (归一化)")
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                else:
                    st.warning("请至少生成2个温度的数据")

    with tab3:
        st.markdown("#### 不同放电深度(DOD)下的容量衰减")
        dod_list = st.multiselect("选择DOD(%)", [20, 40, 60, 80, 100], default=[20, 40, 60, 80, 100], key="lp_dod_list")

        if dod_list and st.button("生成DOD对比示例数据并拟合", key="lp_dod_btn"):
            lp = LifePrediction(q0=100.0)
            dod_data = {}
            for dod in dod_list:
                np.random.seed(dod)
                n = 30
                cyc = np.linspace(0, 1500, n)
                k_dod = 0.001 * (dod / 50) ** 1.2
                cap = 100.0 - k_dod * cyc + np.random.randn(n) * 0.4
                dod_data[str(dod)] = (cyc, cap)
            result = lp.fit_dod(dod_data)

            fig, ax = plt.subplots(figsize=(10, 5))
            plot_cycles = np.linspace(0, 2000, 200)
            for dod_str, res in result.items():
                if "线性模型" in res and "params" in res["线性模型"]:
                    y = lp.model_linear(plot_cycles, *res["线性模型"]["params"])
                    ax.plot(plot_cycles, y, label=f"DOD={dod_str}%", linewidth=1.5)
            ax.set_xlabel("循环次数")
            ax.set_ylabel("放电容量 (Ah)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
