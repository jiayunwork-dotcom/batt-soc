import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from typing import Optional


class LifePrediction:
    def __init__(self, q0: float = 100.0, eol_ratio: float = 0.8):
        self.q0 = q0
        self.eol = q0 * eol_ratio
        self.models: dict = {}
        self.temp_data: dict = {}
        self.dod_data: dict = {}
        self.ea_activation: Optional[float] = None

    @staticmethod
    def model_linear(N: np.ndarray, Q0: float, k: float) -> np.ndarray:
        return Q0 - k * N

    @staticmethod
    def model_sqrt(N: np.ndarray, Q0: float, a: float) -> np.ndarray:
        return Q0 - a * np.sqrt(N)

    @staticmethod
    def model_exp(N: np.ndarray, Q0: float, b: float) -> np.ndarray:
        return Q0 * np.exp(-b * N)

    def fit_all(self, cycles: np.ndarray, capacity: np.ndarray) -> dict:
        cycles = np.asarray(cycles, dtype=float)
        capacity = np.asarray(capacity, dtype=float)
        valid = ~np.isnan(cycles) & ~np.isnan(capacity)
        cycles = cycles[valid]
        capacity = capacity[valid]
        if len(cycles) < 3:
            raise ValueError("数据点不足(至少需要3个)")

        self.models = {}
        q0_init = capacity.max()

        for name, func, p0 in [
            ("线性模型", self.model_linear, [q0_init, 0.001]),
            ("根号模型", self.model_sqrt, [q0_init, 0.1]),
            ("指数模型", self.model_exp, [q0_init, 1e-5]),
        ]:
            try:
                popt, pcov = curve_fit(func, cycles, capacity, p0=p0, maxfev=10000)
                y_pred = func(cycles, *popt)
                ss_res = np.sum((capacity - y_pred) ** 2)
                ss_tot = np.sum((capacity - capacity.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                eol_cycles = self._solve_eol(func, popt)
                self.models[name] = {
                    "params": popt.tolist(),
                    "r_squared": float(r2),
                    "eol_cycles": float(eol_cycles),
                    "remaining_cycles": float(max(0, eol_cycles - cycles.max())),
                    "func": func,
                }
            except Exception as e:
                self.models[name] = {"error": str(e)}
        return self.models

    def _solve_eol(self, func, params) -> float:
        N_try = np.logspace(0, 8, 10000)
        Q_pred = func(N_try, *params)
        idx = np.argmin(np.abs(Q_pred - self.eol))
        return float(N_try[idx])

    def fit_arrhenius(self, temp_data: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
        R = 8.314
        self.temp_data = {}
        k_values = []
        inv_T_values = []
        for temp_str, (cycles, capacity) in temp_data.items():
            temp_c = float(temp_str)
            try:
                res = self.fit_all(cycles, capacity)
                if "线性模型" in res and "params" in res["线性模型"]:
                    k = res["线性模型"]["params"][1]
                    T_abs = temp_c + 273.15
                    k_values.append(k)
                    inv_T_values.append(1.0 / T_abs)
                    self.temp_data[temp_str] = res
            except Exception:
                continue
        if len(k_values) >= 2:
            log_k = np.log(np.maximum(k_values, 1e-10))
            slope, intercept = np.polyfit(inv_T_values, log_k, 1)
            self.ea_activation = float(-slope * R)
            return {
                "activation_energy_eV": self.ea_activation / 96485.0,
                "activation_energy_J_mol": self.ea_activation,
                "temp_curve": self._generate_temp_curve(),
            }
        return {}

    def _generate_temp_curve(self) -> pd.DataFrame:
        if self.ea_activation is None:
            return pd.DataFrame()
        R = 8.314
        temps = np.linspace(-10, 60, 71)
        T_abs = temps + 273.15
        k_ref = 1.0
        T_ref = 298.15
        k_rel = k_ref * np.exp(-self.ea_activation / R * (1.0 / T_abs - 1.0 / T_ref))
        return pd.DataFrame({"温度(°C)": temps, "相对衰减速率": k_rel})

    def fit_dod(self, dod_data: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
        self.dod_data = {}
        for dod_str, (cycles, capacity) in dod_data.items():
            try:
                self.dod_data[dod_str] = self.fit_all(cycles, capacity)
            except Exception as e:
                self.dod_data[dod_str] = {"error": str(e)}
        return self.dod_data

    def predict_at_cycles(self, model_name: str, N: float | np.ndarray) -> np.ndarray:
        if model_name not in self.models or "func" not in self.models[model_name]:
            raise ValueError(f"模型 {model_name} 未拟合")
        func = self.models[model_name]["func"]
        params = self.models[model_name]["params"]
        return func(np.asarray(N, dtype=float), *params)

    def get_comparison_data(self, cycles: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame({"循环次数": cycles})
        for name, model in self.models.items():
            if "func" in model:
                df[name] = self.predict_at_cycles(name, cycles)
        return df
