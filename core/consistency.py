import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional


class ConsistencyAnalyzer:
    def __init__(self, thresholds: Optional[dict] = None):
        self.thresholds = thresholds or {
            "soc_dispersion": 5.0,
            "internal_resistance_ratio": 1.5,
            "temperature_diff": 8.0,
            "capacity_decay_ratio": 2.0,
        }
        self.alerts: list[dict] = []

    def analyze_pack(self, aligned_df: pd.DataFrame, module_ids: list[str]) -> dict:
        if aligned_df is None or len(aligned_df) < 2:
            return {}

        soc_cols = [f"{m}_soc_ah" for m in module_ids if f"{m}_voltage" in aligned_df.columns]
        v_cols = [f"{m}_voltage" for m in module_ids if f"{m}_voltage" in aligned_df.columns]
        t_cols = [f"{m}_temperature" for m in module_ids if f"{m}_temperature" in aligned_df.columns]

        result = {"time_series": {}}
        valid_modules = []

        for m in module_ids:
            if f"{m}_voltage" in aligned_df.columns:
                valid_modules.append(m)

        if v_cols:
            v_data = aligned_df[v_cols].values.astype(float)
            r_est = self._estimate_internal_resistance(aligned_df, valid_modules)
        else:
            r_est = {}

        result["metrics"] = {
            "modules": valid_modules,
            "soc_dispersion": [],
            "voltage_dispersion": [],
            "temperature_dispersion": [],
            "timestamps": aligned_df["timestamp"].astype(str).tolist(),
        }

        for idx in range(len(aligned_df)):
            soc_vals = []
            v_vals = []
            t_vals = []
            for m in valid_modules:
                if f"{m}_voltage" in aligned_df.columns:
                    v = aligned_df[f"{m}_voltage"].iloc[idx]
                    if not np.isnan(v):
                        v_vals.append(v)
                if f"{m}_temperature" in aligned_df.columns:
                    t = aligned_df[f"{m}_temperature"].iloc[idx]
                    if not np.isnan(t):
                        t_vals.append(t)
            if v_vals:
                result["metrics"]["voltage_dispersion"].append(float(np.std(v_vals)))
            else:
                result["metrics"]["voltage_dispersion"].append(0.0)
            if t_vals:
                result["metrics"]["temperature_dispersion"].append(float(np.max(t_vals) - np.min(t_vals)))
            else:
                result["metrics"]["temperature_dispersion"].append(0.0)
            result["metrics"]["soc_dispersion"].append(0.0)

        result["shortboards"] = self._detect_shortboards(aligned_df, valid_modules, r_est)
        self._check_alerts(aligned_df, valid_modules, result["metrics"], r_est)
        result["alerts"] = self.alerts[-10:] if self.alerts else []
        result["r_estimates"] = r_est
        return result

    def _estimate_internal_resistance(self, aligned_df: pd.DataFrame, module_ids: list[str]) -> dict:
        r_est = {}
        for m in module_ids:
            v_col = f"{m}_voltage"
            i_col = f"{m}_current"
            if v_col not in aligned_df.columns or i_col not in aligned_df.columns:
                continue
            v = aligned_df[v_col].values.astype(float)
            i = aligned_df[i_col].values.astype(float)
            valid = ~np.isnan(v) & ~np.isnan(i)
            if valid.sum() < 10:
                continue
            i_std = np.std(i[valid])
            if i_std < 0.01:
                continue
            try:
                coeffs = np.polyfit(i[valid], v[valid], 1)
                r_est[m] = abs(coeffs[0])
            except Exception:
                continue
        return r_est

    def _detect_shortboards(self, aligned_df: pd.DataFrame, module_ids: list[str], r_est: dict) -> list[dict]:
        shortboards = []
        v_means = {}
        t_means = {}
        for m in module_ids:
            if f"{m}_voltage" in aligned_df.columns:
                v_means[m] = np.nanmean(aligned_df[f"{m}_voltage"].values.astype(float))
            if f"{m}_temperature" in aligned_df.columns:
                t_means[m] = np.nanmean(aligned_df[f"{m}_temperature"].values.astype(float))
        if r_est:
            max_r_module = max(r_est, key=r_est.get)
            shortboards.append({"类型": "内阻最高", "模组": max_r_module, "值": round(r_est[max_r_module], 6), "单位": "Ω"})
        if v_means:
            min_v_module = min(v_means, key=v_means.get)
            shortboards.append({"类型": "电压最低", "模组": min_v_module, "值": round(v_means[min_v_module], 4), "单位": "V"})
        if t_means:
            max_t_module = max(t_means, key=t_means.get)
            shortboards.append({"类型": "温度最高", "模组": max_t_module, "值": round(t_means[max_t_module], 2), "单位": "°C"})
        return shortboards

    def _check_alerts(self, aligned_df: pd.DataFrame, module_ids: list[str], metrics: dict, r_est: dict):
        if not module_ids:
            return

        temp_disp = np.array(metrics["temperature_dispersion"])
        if (temp_disp > self.thresholds["temperature_diff"]).any():
            self._add_alert("温度差异过大", f"Pack内最大温差超过{self.thresholds['temperature_diff']}°C", "high")

        if r_est:
            r_values = list(r_est.values())
            r_mean = np.mean(r_values)
            for m, r_val in r_est.items():
                if r_val > r_mean * self.thresholds["internal_resistance_ratio"]:
                    self._add_alert("单体内阻过高", f"模组{m}内阻({r_val:.6f}Ω)超过均值{r_mean:.6f}Ω的{self.thresholds['internal_resistance_ratio']*100:.0f}%", "medium", m)

    def _add_alert(self, alert_type: str, message: str, level: str, module: Optional[str] = None):
        alert = {
            "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "类型": alert_type,
            "严重级别": level,
            "模组": module or "-",
            "消息": message,
        }
        self.alerts.append(alert)

    def get_module_boxplot_data(self, aligned_df: pd.DataFrame, module_ids: list[str]) -> dict:
        data = {"电压": {}, "温度": {}, "内阻": {}}
        for m in module_ids:
            if f"{m}_voltage" in aligned_df.columns:
                vals = aligned_df[f"{m}_voltage"].dropna().values.astype(float)
                if len(vals) > 0:
                    data["电压"][m] = [float(vals.min()), float(np.percentile(vals, 25)), float(np.median(vals)), float(np.percentile(vals, 75)), float(vals.max())]
            if f"{m}_temperature" in aligned_df.columns:
                vals = aligned_df[f"{m}_temperature"].dropna().values.astype(float)
                if len(vals) > 0:
                    data["温度"][m] = [float(vals.min()), float(np.percentile(vals, 25)), float(np.median(vals)), float(np.percentile(vals, 75)), float(vals.max())]
        return data

    def get_alerts(self) -> pd.DataFrame:
        return pd.DataFrame(self.alerts) if self.alerts else pd.DataFrame(columns=["时间", "类型", "严重级别", "模组", "消息"])
