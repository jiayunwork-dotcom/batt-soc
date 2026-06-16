import numpy as np
import pandas as pd
from typing import Optional, Callable


DEFAULT_PARAMS = {
    "mass_kg": 3.5,
    "cp_j_kgk": 1100.0,
    "h_conv_w_m2k": 8.0,
    "surface_area_m2": 0.04,
    "r0_ohm": 0.01,
    "r1_ohm": 0.005,
    "c1_f": 1000.0,
    "dentropy_mv_k": 0.1,
}


class LumpedThermalModel:
    def __init__(self, params: Optional[dict] = None):
        self.params = {**DEFAULT_PARAMS, **(params or {})}

    def _compute_heat_generation(self, current: np.ndarray, soc: np.ndarray, temperature: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        R0 = self.params["r0_ohm"]
        R1 = self.params["r1_ohm"]
        C1 = self.params["c1_f"]
        dS = self.params["dentropy_mv_k"] / 1000.0

        q_ohmic = current ** 2 * R0

        tau1 = R1 * C1
        q_polarization = current ** 2 * R1

        q_entropy = current * temperature * dS

        return q_ohmic, q_polarization, q_entropy

    def _compute_heat_dissipation(self, temperature: np.ndarray, t_amb: np.ndarray) -> np.ndarray:
        h = self.params["h_conv_w_m2k"]
        A = self.params["surface_area_m2"]
        return h * A * (temperature - t_amb)

    def simulate(
        self,
        time_s: np.ndarray,
        current: np.ndarray,
        soc: np.ndarray,
        t_initial: float = 25.0,
        t_amb_const: Optional[float] = None,
        t_amb_series: Optional[np.ndarray] = None,
    ) -> dict:
        n = len(time_s)
        dt = np.diff(time_s, prepend=time_s[0])
        dt[0] = dt[1] if len(dt) > 1 else 1.0
        dt = np.clip(dt, 0.01, None)

        t_amb = np.full(n, t_amb_const if t_amb_const is not None else 25.0)
        if t_amb_series is not None:
            if len(t_amb_series) == n:
                t_amb = t_amb_series.astype(float)
            elif len(t_amb_series) > 0:
                from scipy.interpolate import interp1d
                idx_orig = np.linspace(0, 1, len(t_amb_series))
                idx_target = np.linspace(0, 1, n)
                interp_func = interp1d(idx_orig, t_amb_series, kind="linear", fill_value="extrapolate")
                t_amb = interp_func(idx_target)

        temperature = np.zeros(n)
        temperature[0] = t_initial

        q_ohmic_arr = np.zeros(n)
        q_polar_arr = np.zeros(n)
        q_entropy_arr = np.zeros(n)
        q_dissip_arr = np.zeros(n)

        mass = self.params["mass_kg"]
        cp = self.params["cp_j_kgk"]
        R1 = self.params["r1_ohm"]
        C1 = self.params["c1_f"]
        tau1 = R1 * C1

        v_rc = 0.0

        for k in range(n):
            I_k = current[k]
            soc_k = soc[k]
            T_k = temperature[k]
            dt_k = dt[k]

            alpha = np.exp(-dt_k / tau1) if tau1 > 0 else 0
            v_rc = alpha * v_rc + R1 * (1 - alpha) * I_k

            q_ohm = I_k ** 2 * self.params["r0_ohm"]
            q_pol = I_k * v_rc
            dS = self.params["dentropy_mv_k"] / 1000.0
            q_ent = I_k * T_k * dS

            q_gen = q_ohm + q_pol + q_ent

            h = self.params["h_conv_w_m2k"]
            A = self.params["surface_area_m2"]
            q_dis = h * A * (T_k - t_amb[k])

            q_ohmic_arr[k] = q_ohm
            q_polar_arr[k] = q_pol
            q_entropy_arr[k] = q_ent
            q_dissip_arr[k] = q_dis

            if k < n - 1:
                dT = (q_gen - q_dis) * dt_k / (mass * cp)
                temperature[k + 1] = T_k + dT

        return {
            "temperature": temperature,
            "t_ambient": t_amb,
            "q_ohmic": q_ohmic_arr,
            "q_polarization": q_polar_arr,
            "q_entropy": q_entropy_arr,
            "q_generation": q_ohmic_arr + q_polar_arr + q_entropy_arr,
            "q_dissipation": q_dissip_arr,
            "time_s": time_s,
        }


class ThermalSensitivityAnalyzer:
    def __init__(self, base_model: LumpedThermalModel):
        self.base_model = base_model

    def analyze(
        self,
        time_s: np.ndarray,
        current: np.ndarray,
        soc: np.ndarray,
        t_initial: float = 25.0,
        t_amb_const: Optional[float] = None,
        t_amb_series: Optional[np.ndarray] = None,
        perturbation_range: float = 0.3,
        n_points: int = 7,
    ) -> dict:
        param_names = ["cp_j_kgk", "h_conv_w_m2k", "mass_kg"]
        param_labels = ["比热容 Cp", "对流系数 h", "质量 m"]
        factors = np.linspace(1 - perturbation_range, 1 + perturbation_range, n_points)

        results = {}

        for pname, plabel in zip(param_names, param_labels):
            param_results = []
            for f in factors:
                perturbed_params = {**self.base_model.params}
                perturbed_params[pname] = self.base_model.params[pname] * f
                model = LumpedThermalModel(perturbed_params)
                sim = model.simulate(time_s, current, soc, t_initial, t_amb_const, t_amb_series)
                param_results.append({
                    "factor": f,
                    "temperature": sim["temperature"],
                    "t_max": float(np.max(sim["temperature"])),
                    "t_final": float(sim["temperature"][-1]),
                })
            all_temps = np.array([r["temperature"] for r in param_results])
            results[pname] = {
                "label": plabel,
                "perturbations": param_results,
                "t_envelope_upper": np.max(all_temps, axis=0),
                "t_envelope_lower": np.min(all_temps, axis=0),
                "factors": factors.tolist(),
            }

        return results

    def analyze_two_param(
        self,
        param1_name: str,
        param2_name: str,
        time_s: np.ndarray,
        current: np.ndarray,
        soc: np.ndarray,
        t_initial: float = 25.0,
        t_amb_const: Optional[float] = None,
        t_amb_series: Optional[np.ndarray] = None,
        perturbation_range: float = 0.3,
        n_levels: int = 5,
    ) -> dict:
        param_labels_map = {
            "cp_j_kgk": "比热容 Cp",
            "h_conv_w_m2k": "对流系数 h",
            "mass_kg": "质量 m",
            "r0_ohm": "欧姆内阻 R0",
            "r1_ohm": "极化内阻 R1",
            "c1_f": "极化电容 C1",
            "dentropy_mv_k": "熵变系数 dOCV/dT",
        }
        label1 = param_labels_map.get(param1_name, param1_name)
        label2 = param_labels_map.get(param2_name, param2_name)

        factors1 = np.linspace(1 - perturbation_range, 1 + perturbation_range, n_levels)
        factors2 = np.linspace(1 - perturbation_range, 1 + perturbation_range, n_levels)

        t_max_grid = np.zeros((n_levels, n_levels))
        t_final_grid = np.zeros((n_levels, n_levels))

        for i, f1 in enumerate(factors1):
            for j, f2 in enumerate(factors2):
                perturbed_params = {**self.base_model.params}
                perturbed_params[param1_name] = self.base_model.params[param1_name] * f1
                perturbed_params[param2_name] = self.base_model.params[param2_name] * f2
                model = LumpedThermalModel(perturbed_params)
                sim = model.simulate(time_s, current, soc, t_initial, t_amb_const, t_amb_series)
                t_max_grid[i, j] = float(np.max(sim["temperature"]))
                t_final_grid[i, j] = float(sim["temperature"][-1])

        base_val1 = self.base_model.params[param1_name]
        base_val2 = self.base_model.params[param2_name]
        param1_values = base_val1 * factors1
        param2_values = base_val2 * factors2

        return {
            "param1_name": param1_name,
            "param2_name": param2_name,
            "param1_label": label1,
            "param2_label": label2,
            "param1_factors": factors1.tolist(),
            "param2_factors": factors2.tolist(),
            "param1_values": param1_values.tolist(),
            "param2_values": param2_values.tolist(),
            "t_max_grid": t_max_grid.tolist(),
            "t_final_grid": t_final_grid.tolist(),
            "t_max_max": float(np.max(t_max_grid)),
            "t_max_min": float(np.min(t_max_grid)),
            "n_levels": n_levels,
        }


class ThermalSafetyAnalyzer:
    def __init__(self, temp_threshold: float = 45.0, rate_threshold: float = 0.5):
        self.temp_threshold = temp_threshold
        self.rate_threshold = rate_threshold

    def analyze(self, time_s: np.ndarray, temperature: np.ndarray) -> dict:
        n = len(temperature)
        dt = np.diff(time_s, prepend=time_s[0])
        dt[0] = dt[1] if len(dt) > 1 else 1.0
        dt = np.clip(dt, 0.01, None)

        temp_rate = np.zeros(n)
        for i in range(1, n):
            temp_rate[i] = (temperature[i] - temperature[i - 1]) / dt[i] * 60.0
        temp_rate[0] = temp_rate[1] if n > 1 else 0.0

        over_temp_mask = temperature > self.temp_threshold
        over_rate_mask = temp_rate > self.rate_threshold

        def _find_periods(mask, time_arr, data_arr, alert_type):
            periods = []
            in_period = False
            start_idx = 0
            for i in range(len(mask)):
                if mask[i] and not in_period:
                    in_period = True
                    start_idx = i
                elif not mask[i] and in_period:
                    in_period = False
                    segment = data_arr[start_idx:i]
                    periods.append({
                        "type": alert_type,
                        "start_idx": start_idx,
                        "end_idx": i - 1,
                        "start_time_s": float(time_arr[start_idx]),
                        "end_time_s": float(time_arr[i - 1]),
                        "peak_value": float(np.max(segment)),
                        "peak_idx": int(start_idx + np.argmax(segment)),
                        "duration_s": float(time_arr[i - 1] - time_arr[start_idx]),
                    })
            if in_period:
                segment = data_arr[start_idx:]
                periods.append({
                    "type": alert_type,
                    "start_idx": start_idx,
                    "end_idx": len(mask) - 1,
                    "start_time_s": float(time_arr[start_idx]),
                    "end_time_s": float(time_arr[-1]),
                    "peak_value": float(np.max(segment)),
                    "peak_idx": int(start_idx + np.argmax(segment)),
                    "duration_s": float(time_arr[-1] - time_arr[start_idx]),
                })
            return periods

        over_temp_periods = _find_periods(over_temp_mask, time_s, temperature, "超温预警")
        over_rate_periods = _find_periods(over_rate_mask, time_s, temp_rate, "温升速率预警")

        all_periods = sorted(over_temp_periods + over_rate_periods, key=lambda x: x["start_idx"])

        max_temp = float(np.max(temperature)) if len(temperature) > 0 else 0.0
        max_rate = float(np.max(temp_rate)) if len(temp_rate) > 0 else 0.0
        has_risk = bool(over_temp_mask.any() or over_rate_mask.any())

        return {
            "temp_threshold": self.temp_threshold,
            "rate_threshold": self.rate_threshold,
            "over_temp_periods": over_temp_periods,
            "over_rate_periods": over_rate_periods,
            "all_alert_periods": all_periods,
            "max_temperature": max_temp,
            "max_temp_rate": max_rate,
            "has_risk": has_risk,
            "over_temperature_mask": over_temp_mask,
            "over_rate_mask": over_rate_mask,
            "temp_rate": temp_rate,
        }


class ThermalRunawayAnalyzer:
    def __init__(
        self,
        runaway_threshold: float = 80.0,
        warning_threshold: float = 55.0,
        thermal_conductivity: float = 0.8,
        module_mass_kg: float = 3.5,
        module_cp_j_kgk: float = 1100.0,
        module_distance_m: float = 0.02,
    ):
        self.runaway_threshold = runaway_threshold
        self.warning_threshold = warning_threshold
        self.thermal_conductivity = thermal_conductivity
        self.module_mass = module_mass_kg
        self.module_cp = module_cp_j_kgk
        self.module_distance = module_distance

    def analyze(
        self,
        module_temperatures: dict[str, np.ndarray],
        time_s: np.ndarray,
    ) -> dict:
        module_ids = list(module_temperatures.keys())
        n_modules = len(module_ids)
        n_time = len(time_s)

        module_status = {}
        for mid in module_ids:
            temps = module_temperatures[mid]
            warning_idx = None
            runaway_idx = None
            for i in range(n_time):
                if warning_idx is None and temps[i] >= self.warning_threshold:
                    warning_idx = i
                if runaway_idx is None and temps[i] >= self.runaway_threshold:
                    runaway_idx = i
            module_status[mid] = {
                "temperatures": temps,
                "warning_time_s": float(time_s[warning_idx]) if warning_idx is not None else None,
                "runaway_time_s": float(time_s[runaway_idx]) if runaway_idx is not None else None,
                "warning_idx": warning_idx,
                "runaway_idx": runaway_idx,
                "has_warning": warning_idx is not None,
                "has_runaway": runaway_idx is not None,
            }

        propagation_times = {}
        for i, mid_i in enumerate(module_ids):
            for j, mid_j in enumerate(module_ids):
                if i == j:
                    continue
                key = f"{mid_i}->{mid_j}"
                t_runaway_i = module_status[mid_i]["runaway_idx"]
                if t_runaway_i is None:
                    propagation_times[key] = None
                    continue

                temp_i = module_status[mid_i]["temperatures"]
                temp_j = module_status[mid_j]["temperatures"]

                delta_T = temp_i[t_runaway_i] - temp_j[t_runaway_i]
                if delta_T <= 0:
                    propagation_times[key] = None
                    continue

                heat_flux = self.thermal_conductivity * delta_T / self.module_distance
                heat_capacity = self.module_mass * self.module_cp
                delta_T_needed = self.runaway_threshold - temp_j[t_runaway_i]
                if delta_T_needed <= 0:
                    propagation_times[key] = 0.0
                    continue

                t_propagate = heat_capacity * delta_T_needed / heat_flux
                propagation_times[key] = float(t_propagate)

        timeline_events = []
        for mid in module_ids:
            status = module_status[mid]
            if status["warning_idx"] is not None:
                timeline_events.append({
                    "module_id": mid,
                    "event": "预警",
                    "time_s": status["warning_time_s"],
                    "end_time_s": status["runaway_time_s"] if status["runaway_idx"] is not None else time_s[-1],
                })
            if status["runaway_idx"] is not None:
                timeline_events.append({
                    "module_id": mid,
                    "event": "热失控",
                    "time_s": status["runaway_time_s"],
                    "end_time_s": time_s[-1],
                })

        has_runaway_risk = any(s["has_runaway"] for s in module_status.values())

        return {
            "module_status": module_status,
            "propagation_times": propagation_times,
            "timeline_events": timeline_events,
            "has_runaway_risk": has_runaway_risk,
            "runaway_threshold": self.runaway_threshold,
            "warning_threshold": self.warning_threshold,
            "time_s": time_s,
            "module_ids": module_ids,
        }


class ThermalInconsistencyAnalyzer:
    def __init__(self, dispersion_threshold: float = 3.0):
        self.dispersion_threshold = dispersion_threshold

    def analyze(
        self,
        module_temperatures: dict[str, np.ndarray],
        time_s: np.ndarray,
    ) -> dict:
        module_ids = list(module_temperatures.keys())
        if len(module_ids) < 2:
            return {
                "score": 100.0,
                "max_dispersion": 0.0,
                "mean_dispersion": 0.0,
                "warning_periods": [],
                "peak_dispersion_idx": None,
                "dispersion_series": np.zeros(len(time_s)),
                "module_ids": module_ids,
            }

        temp_matrix = np.array([module_temperatures[mid] for mid in module_ids])

        dispersion = np.max(temp_matrix, axis=0) - np.min(temp_matrix, axis=0)

        mean_disp = float(np.mean(dispersion))
        max_disp = float(np.max(dispersion))
        peak_idx = int(np.argmax(dispersion))

        over_mask = dispersion > self.dispersion_threshold
        warning_periods = []
        in_period = False
        start_idx = 0

        for i in range(len(over_mask)):
            if over_mask[i] and not in_period:
                in_period = True
                start_idx = i
            elif not over_mask[i] and in_period:
                in_period = False
                segment = dispersion[start_idx:i]
                warning_periods.append({
                    "start_idx": start_idx,
                    "end_idx": i - 1,
                    "start_time_s": float(time_s[start_idx]),
                    "end_time_s": float(time_s[i - 1]),
                    "peak_dispersion": float(np.max(segment)),
                    "duration_s": float(time_s[i - 1] - time_s[start_idx]),
                })
        if in_period:
            segment = dispersion[start_idx:]
            warning_periods.append({
                "start_idx": start_idx,
                "end_idx": len(over_mask) - 1,
                "start_time_s": float(time_s[start_idx]),
                "end_time_s": float(time_s[-1]),
                "peak_dispersion": float(np.max(segment)),
                "duration_s": float(time_s[-1] - time_s[start_idx]),
            })

        over_ratio = float(np.sum(over_mask)) / len(over_mask) if len(over_mask) > 0 else 0.0
        norm_mean = min(mean_disp / (self.dispersion_threshold * 3), 1.0)
        norm_max = min(max_disp / (self.dispersion_threshold * 5), 1.0)
        score = max(0.0, 100.0 * (1.0 - 0.4 * norm_mean - 0.3 * norm_max - 0.3 * over_ratio))

        return {
            "score": round(score, 1),
            "max_dispersion": max_disp,
            "mean_dispersion": mean_disp,
            "warning_periods": warning_periods,
            "peak_dispersion_idx": peak_idx,
            "dispersion_series": dispersion,
            "module_ids": module_ids,
        }
