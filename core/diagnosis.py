import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Optional, Callable
from datetime import datetime, timedelta


class BatteryDiagnosis:
    def __init__(self, q0_nominal: float = 100.0, r0_initial: float = 0.01):
        self.q0_nominal = q0_nominal
        self.r0_initial = r0_initial
        self.cycle_data: Optional[pd.DataFrame] = None
        self.dqdv_results: dict = {}
        self.impedance_results: list[dict] = []
        self.soh_models: dict = {}
        self.fault_events: list[dict] = []

    # ========== 模拟数据生成 ==========
    @staticmethod
    def generate_degradation_data(n_cycles: int = 50, q0: float = 100.0, r0_0: float = 0.01) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
        np.random.seed(42)
        cycles = np.arange(1, n_cycles + 1)
        k_cap = 0.5
        k_r = 0.000025
        k_r1 = 0.000012
        capacity = q0 - k_cap * np.sqrt(cycles) + np.random.randn(n_cycles) * 0.3
        r0 = r0_0 + k_r * cycles + np.random.randn(n_cycles) * 0.00005
        r1 = 0.005 + k_r1 * cycles + np.random.randn(n_cycles) * 0.00005

        cycle_df = pd.DataFrame({
            "循环次数": cycles,
            "放电容量(Ah)": capacity,
            "R0(Ω)": r0,
            "R1(Ω)": r1,
            "tau(s)": r1 * (1000 + 10 * cycles + np.random.randn(n_cycles) * 50),
            "充电时间(s)": 3600 + 50 * cycles + np.random.randn(n_cycles) * 30,
            "恒压段时长(s)": 600 + 30 * cycles + np.random.randn(n_cycles) * 20,
        })

        cycle_charge_data = []
        for i in range(n_cycles):
            n_points = 200
            v = np.linspace(3.0, 4.2, n_points)
            q_cap = capacity[i]
            q = np.linspace(0, q_cap, n_points)
            peak_positions = [3.4, 3.7, 4.0]
            peak_heights = [5.0 - 0.03 * i, 8.0 - 0.04 * i, 6.0 - 0.02 * i]
            dqdv = np.zeros(n_points)
            for pos, h in zip(peak_positions, peak_heights):
                sigma = 0.05
                dqdv += h * np.exp(-(v - pos) ** 2 / (2 * sigma ** 2))
            dqdv += np.random.randn(n_points) * 0.3
            q_cumulative = np.cumsum(dqdv) * (v[1] - v[0])
            q_cumulative = q_cap * q_cumulative / q_cumulative.max()
            ts_start = datetime(2024, 1, 1) + timedelta(days=i)
            timestamps = [ts_start + timedelta(seconds=j * 10) for j in range(n_points)]
            df_cycle = pd.DataFrame({
                "timestamp": timestamps,
                "voltage": v + np.random.randn(n_points) * 0.002,
                "capacity": q_cumulative,
                "dqdv_raw": dqdv,
                "current": np.ones(n_points) * (q_cap / (n_points * 10)) * 3600,
                "temperature": 25.0 + np.random.randn(n_points) * 0.5,
            })
            cycle_charge_data.append(df_cycle)

        return cycle_df, cycle_charge_data

    # ========== 1. dQ/dV 容量增量分析 ==========
    def compute_dqdv(
        self,
        voltage: np.ndarray,
        capacity: np.ndarray,
        window_length: int = 11,
        polyorder: int = 3,
    ) -> dict:
        voltage = np.asarray(voltage, dtype=float)
        capacity = np.asarray(capacity, dtype=float)
        valid = ~np.isnan(voltage) & ~np.isnan(capacity)
        voltage = voltage[valid]
        capacity = capacity[valid]
        if len(voltage) < window_length:
            return {"error": "数据点不足"}
        sort_idx = np.argsort(voltage)
        voltage = voltage[sort_idx]
        capacity = capacity[sort_idx]
        try:
            v_filtered = savgol_filter(voltage, window_length, polyorder)
            q_filtered = savgol_filter(capacity, window_length, polyorder)
        except Exception as e:
            return {"error": f"滤波失败: {e}"}
        dqdv = np.zeros_like(v_filtered)
        dqdv[1:-1] = (q_filtered[2:] - q_filtered[:-2]) / (v_filtered[2:] - v_filtered[:-2])
        dqdv[0] = dqdv[1]
        dqdv[-1] = dqdv[-2]
        try:
            dqdv_smooth = savgol_filter(dqdv, window_length, polyorder)
        except Exception:
            dqdv_smooth = dqdv
        peak_idx, peak_props = find_peaks(dqdv_smooth, height=np.max(dqdv_smooth) * 0.1, distance=5)
        result = {
            "voltage": v_filtered,
            "capacity": q_filtered,
            "dqdv": dqdv_smooth,
            "peak_voltage": v_filtered[peak_idx] if len(peak_idx) > 0 else np.array([]),
            "peak_height": peak_props["peak_heights"] if len(peak_idx) > 0 else np.array([]),
            "peak_indices": peak_idx,
        }
        return result

    def compute_dqdv_multi_cycle(self, cycle_data_list: list[pd.DataFrame], window_length: int = 11, polyorder: int = 3) -> dict:
        results = {}
        for i, df in enumerate(cycle_data_list):
            if "voltage" in df.columns and "capacity" in df.columns:
                res = self.compute_dqdv(df["voltage"].values, df["capacity"].values, window_length, polyorder)
                if "error" not in res:
                    results[f"循环{i+1}"] = res
        self.dqdv_results = results
        return results

    # ========== 2. 内阻在线估计 ==========
    def detect_pulse_segments(
        self,
        df: pd.DataFrame,
        current_threshold: float = 1.0,
        min_duration: float = 0.5,
        max_duration: float = 5.0,
    ) -> list[dict]:
        if len(df) < 3:
            return []
        df = df.sort_values("timestamp").reset_index(drop=True)
        current = df["current"].values.astype(float)
        voltage = df["voltage"].values.astype(float)
        timestamps = df["timestamp"].values
        dt = np.diff(timestamps).astype("timedelta64[ns]").astype(float) / 1e9
        dt = np.concatenate([[dt[0] if len(dt) > 0 else 1.0], dt])
        current_diff = np.abs(np.diff(current))
        step_indices = np.where(current_diff > current_threshold)[0]
        if len(step_indices) < 2:
            return []
        pulses = []
        for i in range(len(step_indices) - 1):
            start_idx = step_indices[i]
            end_idx = step_indices[i + 1]
            t_duration = np.sum(dt[start_idx:end_idx])
            if t_duration < min_duration or t_duration > max_duration:
                continue
            pulse_current = current[start_idx + 1:end_idx].mean()
            if abs(pulse_current) < current_threshold * 0.5:
                continue
            pulses.append({
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "start_time": timestamps[start_idx],
                "end_time": timestamps[end_idx],
                "duration": float(t_duration),
                "pulse_current": float(pulse_current),
            })
        return pulses

    def estimate_impedance_from_pulse(
        self,
        df: pd.DataFrame,
        pulse: dict,
    ) -> dict:
        df = df.sort_values("timestamp").reset_index(drop=True)
        start_idx = pulse["start_idx"]
        end_idx = pulse["end_idx"]
        current = df["current"].values.astype(float)
        voltage = df["voltage"].values.astype(float)
        timestamps = df["timestamp"].values
        current_step = current[start_idx + 1] - current[start_idx]
        if abs(current_step) < 1e-6:
            return {"error": "电流阶跃太小"}
        v_start = voltage[start_idx]
        v_end_pulse = voltage[end_idx]
        v_instant_drop = voltage[start_idx + 2] - voltage[start_idx] if start_idx + 2 < len(voltage) else voltage[start_idx + 1] - voltage[start_idx]
        r0 = abs(v_instant_drop / current_step)
        relax_start = end_idx
        relax_end = min(len(df), end_idx + 100)
        if relax_end - relax_start < 5:
            return {"R0": float(r0), "R1": np.nan, "tau": np.nan, "error": "松弛段数据不足"}
        t_relax = (timestamps[relax_start:relax_end] - timestamps[relax_start]).astype("timedelta64[ns]").astype(float) / 1e9
        v_relax = voltage[relax_start:relax_end]
        i_relax = current[relax_start:relax_end]
        v_steady = v_relax[-1] if len(v_relax) > 0 else voltage[relax_start]
        def rc_model(t, R1, tau):
            return v_steady + (v_relax[0] - v_steady) * np.exp(-t / tau) + R1 * i_relax[0] * (1 - np.exp(-t / tau))
        try:
            popt, _ = curve_fit(rc_model, t_relax, v_relax, p0=[0.005, 10.0], maxfev=10000)
            r1, tau = popt
        except Exception as e:
            return {"R0": float(r0), "R1": np.nan, "tau": np.nan, "error": f"RC拟合失败: {e}"}
        return {
            "R0": float(r0),
            "R1": float(max(0, r1)),
            "tau": float(max(0.1, tau)),
            "pulse_info": pulse,
            "v_steady": float(v_steady),
        }

    def process_all_pulses(self, df_list: list[pd.DataFrame], **pulse_kwargs) -> pd.DataFrame:
        all_results = []
        for cycle_idx, df in enumerate(df_list):
            pulses = self.detect_pulse_segments(df, **pulse_kwargs)
            for p_idx, pulse in enumerate(pulses):
                res = self.estimate_impedance_from_pulse(df, pulse)
                if "error" not in res or "R0" in res:
                    row = {
                        "循环次数": cycle_idx + 1,
                        "脉冲编号": p_idx + 1,
                        "R0(Ω)": res.get("R0", np.nan),
                        "R1(Ω)": res.get("R1", np.nan),
                        "tau(s)": res.get("tau", np.nan),
                        "脉冲持续时间(s)": pulse.get("duration", np.nan),
                        "脉冲电流(A)": pulse.get("pulse_current", np.nan),
                        "异常": res.get("error", ""),
                    }
                    all_results.append(row)
        self.impedance_results = all_results
        return pd.DataFrame(all_results)

    # ========== 3. SOH 多模型估计 ==========
    @staticmethod
    def model_capacity_sqrt(N: np.ndarray, Q0: float, k: float) -> np.ndarray:
        return Q0 - k * np.sqrt(N)

    @staticmethod
    def model_resistance_linear(N: np.ndarray, R0: float, a: float) -> np.ndarray:
        return R0 + a * N

    def fit_soh_capacity_model(self, cycles: np.ndarray, capacity: np.ndarray) -> dict:
        cycles = np.asarray(cycles, dtype=float)
        capacity = np.asarray(capacity, dtype=float)
        valid = ~np.isnan(cycles) & ~np.isnan(capacity)
        cycles = cycles[valid]
        capacity = capacity[valid]
        if len(cycles) < 3:
            return {"error": "数据点不足"}
        q0_init = capacity.max()
        try:
            popt, _ = curve_fit(self.model_capacity_sqrt, cycles, capacity, p0=[q0_init, 0.1], maxfev=10000)
            Q0_fit, k_fit = popt
            soh_pred = (self.model_capacity_sqrt(cycles, Q0_fit, k_fit) / self.q0_nominal) * 100
            soh_true = (capacity / self.q0_nominal) * 100
            rmse = np.sqrt(mean_squared_error(soh_true, soh_pred))
            mae = mean_absolute_error(soh_true, soh_pred)
            return {
                "type": "capacity_sqrt",
                "params": {"Q0": float(Q0_fit), "k": float(k_fit)},
                "cycles": cycles,
                "soh_pred": soh_pred,
                "soh_true": soh_true,
                "rmse": float(rmse),
                "mae": float(mae),
                "func": lambda N: (self.model_capacity_sqrt(np.asarray(N, dtype=float), Q0_fit, k_fit) / self.q0_nominal) * 100,
            }
        except Exception as e:
            return {"error": f"拟合失败: {e}"}

    def fit_soh_resistance_model(self, cycles: np.ndarray, r0: np.ndarray) -> dict:
        cycles = np.asarray(cycles, dtype=float)
        r0 = np.asarray(r0, dtype=float)
        valid = ~np.isnan(cycles) & ~np.isnan(r0)
        cycles = cycles[valid]
        r0 = r0[valid]
        if len(cycles) < 3:
            return {"error": "数据点不足"}
        r0_init = r0[0]
        try:
            popt, _ = curve_fit(self.model_resistance_linear, cycles, r0, p0=[r0_init, 0.0001], maxfev=10000)
            R0_fit, a_fit = popt
            r0_pred = self.model_resistance_linear(cycles, R0_fit, a_fit)
            ratio = r0_pred / R0_fit
            soh_pred = 100 - (ratio - 1) * (20 / 0.5)
            soh_pred = np.clip(soh_pred, 0, 100)
            r0_eol = R0_fit * 1.5
            soh_true = 100 - (r0 / R0_fit - 1) * (20 / 0.5)
            soh_true = np.clip(soh_true, 0, 100)
            rmse = np.sqrt(mean_squared_error(soh_true, soh_pred))
            mae = mean_absolute_error(soh_true, soh_pred)
            return {
                "type": "resistance_linear",
                "params": {"R0": float(R0_fit), "a": float(a_fit), "R0_EOL": float(r0_eol)},
                "cycles": cycles,
                "soh_pred": soh_pred,
                "soh_true": soh_true,
                "rmse": float(rmse),
                "mae": float(mae),
                "func": lambda N: np.clip(100 - ((self.model_resistance_linear(np.asarray(N, dtype=float), R0_fit, a_fit) / R0_fit - 1) * (20 / 0.5)), 0, 100),
            }
        except Exception as e:
            return {"error": f"拟合失败: {e}"}

    def extract_features(self, cycle_df: pd.DataFrame, dqdv_results: dict) -> pd.DataFrame:
        features = []
        cycles = cycle_df["循环次数"].values
        for i, cyc in enumerate(cycles):
            row = {"循环次数": cyc}
            row["R0(Ω)"] = cycle_df["R0(Ω)"].iloc[i] if "R0(Ω)" in cycle_df.columns else np.nan
            row["充电时间(s)"] = cycle_df["充电时间(s)"].iloc[i] if "充电时间(s)" in cycle_df.columns else np.nan
            row["恒压段时长(s)"] = cycle_df["恒压段时长(s)"].iloc[i] if "恒压段时长(s)" in cycle_df.columns else np.nan
            key = f"循环{int(cyc)}"
            if key in dqdv_results:
                dqdv = dqdv_results[key]
                peaks_v = dqdv.get("peak_voltage", np.array([]))
                peaks_h = dqdv.get("peak_height", np.array([]))
                if len(peaks_v) > 0:
                    main_idx = np.argmax(peaks_h)
                    row["主峰电压(V)"] = float(peaks_v[main_idx])
                    row["主峰高度"] = float(peaks_h[main_idx])
                else:
                    row["主峰电压(V)"] = np.nan
                    row["主峰高度"] = np.nan
                row["峰值数量"] = len(peaks_v)
            else:
                row["主峰电压(V)"] = np.nan
                row["主峰高度"] = np.nan
                row["峰值数量"] = 0
            features.append(row)
        return pd.DataFrame(features)

    def fit_soh_feature_model(
        self,
        cycle_df: pd.DataFrame,
        dqdv_results: dict,
        capacity: np.ndarray,
        train_ratio: float = 0.7,
    ) -> dict:
        cycles = cycle_df["循环次数"].values
        features_df = self.extract_features(cycle_df, dqdv_results)
        feature_cols = [c for c in features_df.columns if c != "循环次数"]
        X = features_df[feature_cols].fillna(0).values.astype(float)
        soh_true = (np.asarray(capacity, dtype=float) / self.q0_nominal) * 100
        n_train = int(len(cycles) * train_ratio)
        if n_train < 5:
            return {"error": "训练数据不足"}
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = soh_true[:n_train], soh_true[n_train:]
        cycles_train, cycles_test = cycles[:n_train], cycles[n_train:]
        try:
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            y_pred_all = np.concatenate([y_pred_train, y_pred_test])
            rmse = np.sqrt(mean_squared_error(soh_true, y_pred_all))
            mae = mean_absolute_error(soh_true, y_pred_all)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test)) if len(y_test) > 0 else np.nan
            mae_test = mean_absolute_error(y_test, y_pred_test) if len(y_test) > 0 else np.nan
            return {
                "type": "feature_ridge",
                "model": model,
                "feature_cols": feature_cols,
                "cycles_all": cycles,
                "cycles_train": cycles_train,
                "cycles_test": cycles_test,
                "soh_pred": y_pred_all,
                "soh_pred_train": y_pred_train,
                "soh_pred_test": y_pred_test,
                "soh_true": soh_true,
                "rmse": float(rmse),
                "mae": float(mae),
                "rmse_test": float(rmse_test) if not np.isnan(rmse_test) else None,
                "mae_test": float(mae_test) if not np.isnan(mae_test) else None,
                "coef": dict(zip(feature_cols, model.coef_.tolist())),
                "intercept": float(model.intercept_),
            }
        except Exception as e:
            return {"error": f"特征回归失败: {e}"}

    def fit_all_soh_models(
        self,
        cycle_df: pd.DataFrame,
        dqdv_results: Optional[dict] = None,
    ) -> dict:
        cycles = cycle_df["循环次数"].values
        capacity = cycle_df["放电容量(Ah)"].values
        if dqdv_results is None:
            dqdv_results = {}
        self.soh_models = {}
        cap_model = self.fit_soh_capacity_model(cycles, capacity)
        if "error" not in cap_model:
            self.soh_models["容量衰减模型"] = cap_model
        if "R0(Ω)" in cycle_df.columns:
            r0 = cycle_df["R0(Ω)"].values
            res_model = self.fit_soh_resistance_model(cycles, r0)
            if "error" not in res_model:
                self.soh_models["内阻增长模型"] = res_model
        feat_model = self.fit_soh_feature_model(cycle_df, dqdv_results, capacity)
        if "error" not in feat_model:
            self.soh_models["特征回归模型"] = feat_model
        return self.soh_models

    # ========== 4. 故障预警规则引擎 ==========
    def run_fault_detection(
        self,
        cycle_df: pd.DataFrame,
        dqdv_results: dict,
        soh_models: dict,
        thresholds: Optional[dict] = None,
    ) -> list[dict]:
        if thresholds is None:
            thresholds = {
                "r_sudden_increase": 0.15,
                "capacity_jump": 0.03,
                "peak_disappear_ratio": 0.3,
                "soh_consistency_std": 5.0,
            }
        events = []
        cycles = cycle_df["循环次数"].values
        if "R0(Ω)" in cycle_df.columns:
            r0 = cycle_df["R0(Ω)"].values
            for i in range(1, len(r0)):
                if not np.isnan(r0[i]) and not np.isnan(r0[i - 1]) and r0[i - 1] > 0:
                    growth = (r0[i] - r0[i - 1]) / r0[i - 1]
                    if growth > thresholds["r_sudden_increase"]:
                        level = "严重" if growth > 0.3 else "警告"
                        events.append({
                            "时间": datetime.now(),
                            "循环次数": int(cycles[i]),
                            "触发规则": "内阻突增",
                            "严重等级": level,
                            "数据快照": f"R0={r0[i]:.5f}Ω, 前一循环R0={r0[i-1]:.5f}Ω, 增长={growth*100:.1f}%",
                        })
        capacity = cycle_df["放电容量(Ah)"].values
        for i in range(1, len(capacity)):
            if not np.isnan(capacity[i]) and not np.isnan(capacity[i - 1]) and capacity[i - 1] > 0:
                decay = (capacity[i - 1] - capacity[i]) / capacity[i - 1]
                if decay > thresholds["capacity_jump"]:
                    level = "严重" if decay > 0.06 else "警告"
                    events.append({
                        "时间": datetime.now(),
                        "循环次数": int(cycles[i]),
                        "触发规则": "容量跳变",
                        "严重等级": level,
                        "数据快照": f"当前容量={capacity[i]:.2f}Ah, 前一循环={capacity[i-1]:.2f}Ah, 衰减={decay*100:.1f}%",
                    })
        if dqdv_results:
            first_key = f"循环{int(cycles[0])}"
            first_peak_h = None
            if first_key in dqdv_results:
                ph = dqdv_results[first_key].get("peak_height", np.array([]))
                if len(ph) > 0:
                    first_peak_h = float(np.max(ph))
            if first_peak_h is not None and first_peak_h > 0:
                for i, cyc in enumerate(cycles):
                    key = f"循环{int(cyc)}"
                    if key in dqdv_results:
                        ph = dqdv_results[key].get("peak_height", np.array([]))
                        if len(ph) > 0:
                            cur_max = float(np.max(ph))
                            if cur_max < first_peak_h * thresholds["peak_disappear_ratio"]:
                                events.append({
                                    "时间": datetime.now(),
                                    "循环次数": int(cyc),
                                    "触发规则": "dQ/dV峰值消失",
                                    "严重等级": "危险",
                                    "数据快照": f"主峰高度={cur_max:.2f}, 首循环主峰={first_peak_h:.2f}, 比例={cur_max/first_peak_h*100:.1f}%",
                                })
                                break
        if len(soh_models) >= 2:
            soh_per_model = {}
            for name, m in soh_models.items():
                if "soh_pred" in m:
                    soh_per_model[name] = m["soh_pred"]
            if len(soh_per_model) >= 2:
                min_len = min(len(v) for v in soh_per_model.values())
                soh_matrix = np.array([v[:min_len] for v in soh_per_model.values()])
                soh_std = np.std(soh_matrix, axis=0)
                for i in range(min_len):
                    if soh_std[i] > thresholds["soh_consistency_std"]:
                        model_vals = ", ".join([f"{k}={soh_per_model[k][i]:.1f}%" for k in soh_per_model])
                        events.append({
                            "时间": datetime.now(),
                            "循环次数": int(cycles[i]),
                            "触发规则": "SOH一致性异常",
                            "严重等级": "警告",
                            "数据快照": f"标准差={soh_std[i]:.2f}%, {model_vals}",
                        })
        self.fault_events = events
        return events

    # ========== 5. 诊断报告 ==========
    def generate_diagnosis_report(
        self,
        cycle_df: pd.DataFrame,
        soh_models: dict,
        fault_events: list[dict],
    ) -> dict:
        soh_values = []
        for m in soh_models.values():
            if "soh_pred" in m:
                soh_values.append(m["soh_pred"][-1])
        soh_mean = float(np.mean(soh_values)) if soh_values else np.nan
        health_score = max(0, min(100, soh_mean if not np.isnan(soh_mean) else 0))
        level_penalty = {"警告": 3, "严重": 8, "危险": 15}
        fault_types = set()
        for ev in fault_events:
            rule = ev.get("触发规则", "未知")
            if rule not in fault_types:
                fault_types.add(rule)
                health_score -= level_penalty.get(ev["严重等级"], 5)
        max_penalty = 25
        if soh_mean is not None:
            health_score = max(health_score, soh_mean - max_penalty)
        health_score = max(0, min(100, health_score))
        remaining_cycles = np.nan
        if "容量衰减模型" in soh_models:
            m = soh_models["容量衰减模型"]
            func = m.get("func")
            if func is not None:
                try:
                    current_max_cycle = cycle_df["循环次数"].max()
                    current_soh = func(current_max_cycle) if hasattr(func, '__call__') else 100
                    if current_soh <= 80:
                        remaining_cycles = 0.0
                    else:
                        k_est = m.get("params", {}).get("k", 0.1)
                        q0_est = m.get("params", {}).get("Q0", 100.0)
                        soh_eol = 80.0
                        q_eol = soh_eol / 100.0 * self.q0_nominal
                        if k_est > 0:
                            N_eol_est = ((q0_est - q_eol) / k_est) ** 2
                            remaining_cycles = max(0, float(N_eol_est - current_max_cycle))
                        else:
                            remaining_cycles = np.nan
                        if np.isnan(remaining_cycles) or remaining_cycles > 1e6:
                            N_test = np.linspace(current_max_cycle, current_max_cycle + 100000, 100000)
                            soh_extrap = func(N_test)
                            if soh_extrap[-1] <= 80:
                                idx = np.argmin(np.abs(soh_extrap - 80))
                                remaining_cycles = float(N_test[idx] - current_max_cycle)
                            else:
                                remaining_cycles = float(N_test[-1] - current_max_cycle)
                except Exception:
                    remaining_cycles = np.nan
        suggestions = []
        fault_types = set(ev["触发规则"] for ev in fault_events)
        if "内阻突增" in fault_types:
            suggestions.append("检测到内阻异常增长，建议检查连接端子是否松动、冷却系统是否正常工作，考虑对电池进行均衡维护。")
        if "容量跳变" in fault_types:
            suggestions.append("检测到容量异常衰减，建议暂停大电流充放电，排查是否存在微短路或电解液泄漏，进行完整充放电校准。")
        if "dQ/dV峰值消失" in fault_types:
            suggestions.append("dQ/dV主峰严重衰减，表明电池内部活性物质损失严重，电池已接近寿命终点，建议尽快更换。")
        if "SOH一致性异常" in fault_types:
            suggestions.append("多模型SOH估计偏差较大，建议补充更多循环数据重新训练模型，并检查传感器测量精度。")
        if not suggestions:
            if health_score >= 80:
                suggestions.append("电池状态良好，建议继续按照当前工况运行，定期进行健康检查。")
            elif health_score >= 60:
                suggestions.append("电池存在一定程度老化，建议降低充放电倍率，加强温度监控，适当缩短维护周期。")
            else:
                suggestions.append("电池老化较为严重，建议评估更换需求，避免在高负荷工况下使用。")
        return {
            "健康评分": float(health_score),
            "SOH均值(%)": float(soh_mean) if not np.isnan(soh_mean) else None,
            "各模型SOH(%)": {k: float(m["soh_pred"][-1]) for k, m in soh_models.items() if "soh_pred" in m},
            "剩余寿命预估(循环)": float(remaining_cycles) if not np.isnan(remaining_cycles) else None,
            "故障事件数": len(fault_events),
            "维护建议": suggestions,
        }
