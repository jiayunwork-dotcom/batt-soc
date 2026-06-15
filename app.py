import streamlit as st
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(
    page_title="电池储能系统SOC估计与寿命预测平台",
    page_icon="🔋",
    layout="wide",
)

if "data_manager" not in st.session_state:
    from core import DataManager
    st.session_state.data_manager = DataManager()

if "alerts" not in st.session_state:
    st.session_state.alerts = []

if "thresholds" not in st.session_state:
    st.session_state.thresholds = {
        "soc_dispersion": 5.0,
        "internal_resistance_ratio": 1.5,
        "temperature_diff": 8.0,
        "capacity_decay_ratio": 2.0,
    }

st.title("🔋 电池储能系统SOC估计与循环寿命预测平台")
st.markdown("---")

st.sidebar.title("导航")
page = st.sidebar.radio(
    "选择功能模块",
    [
        "🏠 首页",
        "📊 BMS数据管理",
        "🔧 等效电路模型参数辨识",
        "📈 OCV-SOC曲线标定",
        "🎯 卡尔曼滤波SOC估计",
        "🔮 循环寿命预测",
        "⚠️ 一致性分析与预警",
        "📑 报告导出",
    ],
)

if page == "🏠 首页":
    st.subheader("平台概览")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("已接入Pack数量", len(st.session_state.data_manager.packs))
    with col2:
        total_modules = sum(len(p["modules"]) for p in st.session_state.data_manager.packs.values())
        st.metric("模组总数", total_modules)
    with col3:
        st.metric("告警数量", len(st.session_state.alerts))

    st.markdown("### 功能说明")
    st.info(
        """
        本平台面向储能电站运维人员，提供电池组健康状态监测和寿命评估功能：
        
        - **BMS数据管理**：导入、校验、存储BMS时序数据
        - **参数辨识**：基于HPPC脉冲辨识一阶/二阶RC等效电路模型参数
        - **OCV-SOC标定**：低倍率充放电法或增量电压法标定OCV-SOC曲线
        - **SOC估计**：EKF/UKF卡尔曼滤波算法估计电池SOC
        - **寿命预测**：多模型容量衰减建模与剩余寿命预测
        - **一致性分析**：Pack内模组一致性评估与智能预警
        - **报告导出**：一键生成PDF健康评估报告
        """
    )

elif page == "📊 BMS数据管理":
    from ui_pages import data_management_page
    data_management_page.render()

elif page == "🔧 等效电路模型参数辨识":
    from ui_pages import ecm_page
    ecm_page.render()

elif page == "📈 OCV-SOC曲线标定":
    from ui_pages import ocv_soc_page
    ocv_soc_page.render()

elif page == "🎯 卡尔曼滤波SOC估计":
    from ui_pages import kalman_page
    kalman_page.render()

elif page == "🔮 循环寿命预测":
    from ui_pages import life_page
    life_page.render()

elif page == "⚠️ 一致性分析与预警":
    from ui_pages import consistency_page
    consistency_page.render()

elif page == "📑 报告导出":
    from ui_pages import report_page
    report_page.render()
