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


class ThermalSafetyAnalyzer:
    def __init__(self, temp_threshold: float = 45.0):
        self.temp_threshold = temp_threshold

    def analyze(self, time_s: np.ndarray, temperature: np.ndarray) -> dict:
        over_mask = temperature > self.temp_threshold
        over_periods = []
        in_period = False
        start_idx = 0

        for i in range(len(over_mask)):
            if over_mask[i] and not in_period:
                in_period = True
                start_idx = i
            elif not over_mask[i] and in_period:
                in_period = False
                segment = temperature[start_idx:i]
                over_periods.append({
                    "start_idx": start_idx,
                    "end_idx": i - 1,
                    "start_time_s": float(time_s[start_idx]),
                    "end_time_s": float(time_s[i - 1]),
                    "peak_temperature": float(np.max(segment)),
                    "peak_idx": int(start_idx + np.argmax(segment)),
                    "duration_s": float(time_s[i - 1] - time_s[start_idx]),
                })
        if in_period:
            segment = temperature[start_idx:]
            over_periods.append({
                "start_idx": start_idx,
                "end_idx": len(over_mask) - 1,
                "start_time_s": float(time_s[start_idx]),
                "end_time_s": float(time_s[-1]),
                "peak_temperature": float(np.max(segment)),
                "peak_idx": int(start_idx + np.argmax(segment)),
                "duration_s": float(time_s[-1] - time_s[start_idx]),
            })

        max_temp = float(np.max(temperature)) if len(temperature) > 0 else 0.0
        has_risk = bool(over_mask.any())

        return {
            "threshold": self.temp_threshold,
            "over_periods": over_periods,
            "max_temperature": max_temp,
            "has_risk": has_risk,
            "over_temperature_mask": over_mask,
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
