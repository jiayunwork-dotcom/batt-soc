import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from typing import Optional


class ReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.styles.add(ParagraphStyle(name="ChineseTitle", parent=self.styles["Heading1"], fontSize=18, leading=22, alignment=1))
        self.styles.add(ParagraphStyle(name="ChineseH2", parent=self.styles["Heading2"], fontSize=14, leading=18))
        self.styles.add(ParagraphStyle(name="ChineseBody", parent=self.styles["BodyText"], fontSize=10, leading=14))

    def generate_pdf(
        self,
        pack_id: str,
        pack_info: dict,
        module_params: pd.DataFrame,
        soc_result: Optional[dict],
        life_result: Optional[dict],
        consistency_result: Optional[dict],
        alerts: pd.DataFrame,
    ) -> BytesIO:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=20 * mm, leftMargin=20 * mm, topMargin=20 * mm, bottomMargin=20 * mm)
        story = []

        story.append(Paragraph(f"电池健康评估报告", self.styles["ChineseTitle"]))
        story.append(Paragraph(f"Pack编号: {pack_id}", self.styles["ChineseBody"]))
        story.append(Paragraph(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", self.styles["ChineseBody"]))
        story.append(Spacer(1, 10 * mm))

        story.append(Paragraph("1. Pack概览", self.styles["ChineseH2"]))
        overview_data = [
            ["项目", "数值"],
            ["模组数量", str(pack_info.get("module_count", "-"))],
            ["额定容量(Ah)", f"{pack_info.get('capacity', 0):.3f}"],
            ["数据记录总数", str(pack_info.get("total_records", "-"))],
            ["数据时间跨度", str(pack_info.get("time_span", "-"))],
            ["总循环次数(估计)", str(pack_info.get("cycles", "-"))],
        ]
        t = Table(overview_data, colWidths=[80 * mm, 80 * mm])
        t.setStyle(self._table_style())
        story.append(t)
        story.append(Spacer(1, 8 * mm))

        story.append(Paragraph("2. 各模组参数对比", self.styles["ChineseH2"]))
        if module_params is not None and len(module_params) > 0:
            t = Table([module_params.columns.tolist()] + module_params.values.tolist(), repeatRows=1)
            t.setStyle(self._table_style())
            story.append(t)
        else:
            story.append(Paragraph("暂无参数数据", self.styles["ChineseBody"]))
        story.append(Spacer(1, 8 * mm))

        if soc_result is not None:
            story.append(Paragraph("3. SOC估计结果摘要", self.styles["ChineseH2"]))
            soc_data = [
                ["指标", "数值"],
                ["SOC均值(%)", f"{soc_result.get('soc_mean', 0):.2f}"],
                ["SOC最小值(%)", f"{soc_result.get('soc_min', 0):.2f}"],
                ["SOC最大值(%)", f"{soc_result.get('soc_max', 0):.2f}"],
                ["估计方法", soc_result.get("method", "-")],
            ]
            t = Table(soc_data, colWidths=[80 * mm, 80 * mm])
            t.setStyle(self._table_style())
            story.append(t)
            if "plot" in soc_result:
                story.append(Spacer(1, 5 * mm))
                story.append(Image(soc_result["plot"], width=160 * mm, height=80 * mm))
            story.append(Spacer(1, 8 * mm))

        if life_result is not None:
            story.append(Paragraph("4. 寿命预测结论", self.styles["ChineseH2"]))
            for model_name, res in life_result.items():
                if "params" in res:
                    story.append(Paragraph(f"{model_name}:", self.styles["ChineseBody"]))
                    life_data = [
                        ["指标", "数值"],
                        ["拟合优度R²", f"{res.get('r_squared', 0):.4f}"],
                        ["预计总循环次数(EOL=80%)", f"{res.get('eol_cycles', 0):.0f}"],
                        ["预计剩余循环次数", f"{res.get('remaining_cycles', 0):.0f}"],
                    ]
                    t = Table(life_data, colWidths=[80 * mm, 80 * mm])
                    t.setStyle(self._table_style())
                    story.append(t)
                    story.append(Spacer(1, 3 * mm))
            if "plot" in life_result:
                story.append(Image(life_result["plot"], width=160 * mm, height=80 * mm))
            story.append(Spacer(1, 8 * mm))

        if consistency_result is not None:
            story.append(Paragraph("5. 一致性评估", self.styles["ChineseH2"]))
            if "shortboards" in consistency_result:
                story.append(Paragraph("短板模组列表:", self.styles["ChineseBody"]))
                if consistency_result["shortboards"]:
                    sb_data = [["类型", "模组", "值", "单位"]]
                    for sb in consistency_result["shortboards"]:
                        sb_data.append([sb.get("类型", ""), sb.get("模组", ""), str(sb.get("值", "")), sb.get("单位", "")])
                    t = Table(sb_data, repeatRows=1)
                    t.setStyle(self._table_style())
                    story.append(t)
            story.append(Spacer(1, 5 * mm))

        story.append(Paragraph("6. 告警统计", self.styles["ChineseH2"]))
        if alerts is not None and len(alerts) > 0:
            t = Table([alerts.columns.tolist()] + alerts.values.tolist(), repeatRows=1)
            t.setStyle(self._table_style())
            story.append(t)
        else:
            story.append(Paragraph("当前无告警记录", self.styles["ChineseBody"]))

        doc.build(story)
        buffer.seek(0)
        return buffer

    def _table_style(self) -> TableStyle:
        return TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])

    def plot_to_buffer(self, fig) -> BytesIO:
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf
