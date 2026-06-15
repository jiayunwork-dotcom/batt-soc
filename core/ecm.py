import numpy as np
import pandas as pd
from scipy.optimize import least_squares, curve_fit
from scipy.stats import t
from typing import Optional, Callable
from utils.preprocessing import detect_hppc_pulses


class ECM:
    def __init__(self, order: int = 1):
        self.order = min(max(order, 1), 2)
        self.params: dict = {}
        self.param_ci: dict = {}
        self.fit_result: Optional[dict] = None

    def simulate_voltage(self, t: np.ndarray, current: np.ndarray, params: np.ndarray) -> np.ndarray:
        if self.order == 1:
            R0, R1, C1, OCV = params
            tau1 = R1 * C1
            dt = np.diff(t, prepend=t[0])
            dt[0] = dt[1] if len(dt) > 1 else 1.0
            v_rc = np.zeros_like(t, dtype=float)
            for i in range(1, len(t)):
                alpha = np.exp(-dt[i] / tau1) if tau1 > 0 else 0
                v_rc[i] = alpha * v_rc[i - 1] + R1 * (1 - alpha) * current[i]
            return OCV - current * R0 - v_rc
        else:
            R0, R1, C1, R2, C2, OCV = params
            tau1, tau2 = R1 * C1, R2 * C2
            dt = np.diff(t, prepend=t[0])
            dt[0] = dt[1] if len(dt) > 1 else 1.0
            v_rc1 = np.zeros_like(t, dtype=float)
            v_rc2 = np.zeros_like(t, dtype=float)
            for i in range(1, len(t)):
                a1 = np.exp(-dt[i] / tau1) if tau1 > 0 else 0
                a2 = np.exp(-dt[i] / tau2) if tau2 > 0 else 0
                v_rc1[i] = a1 * v_rc1[i - 1] + R1 * (1 - a1) * current[i]
                v_rc2[i] = a2 * v_rc2[i - 1] + R2 * (1 - a2) * current[i]
            return OCV - current * R0 - v_rc1 - v_rc2

    def _residuals(self, params: np.ndarray, t: np.ndarray, current: np.ndarray, v_meas: np.ndarray) -> np.ndarray:
        return self.simulate_voltage(t, current, params) - v_meas

    def identify(self, df: pd.DataFrame, pulse_idx: Optional[int] = None) -> dict:
        df = df.sort_values("timestamp").reset_index(drop=True)
        pulses = detect_hppc_pulses(df, current_threshold=0.3)
        if not pulses:
            raise ValueError("未检测到HPPC脉冲特征，请检查数据")

        if pulse_idx is None:
            pulse = pulses[0]
        else:
            pulse = pulses[min(pulse_idx, len(pulses) - 1)]

        start = max(0, pulse["start_idx"] - 10)
        end = min(len(df), pulse["end_idx"] + 50)
        segment = df.iloc[start:end].copy()
        t0 = segment["timestamp"].iloc[0]
        t_sec = (segment["timestamp"] - t0).dt.total_seconds().values
        current = segment["current"].values.astype(float)
        voltage = segment["voltage"].values.astype(float)

        if self.order == 1:
            x0 = np.array([0.01, 0.005, 1000.0, voltage[0]])
            bounds = ([0.0, 0.0, 1.0, 2.5], [0.5, 0.5, 1e6, 4.2])
            param_names = ["R0(Ω)", "R1(Ω)", "C1(F)", "OCV(V)"]
        else:
            x0 = np.array([0.01, 0.005, 1000.0, 0.003, 5000.0, voltage[0]])
            bounds = (
                [0.0, 0.0, 1.0, 0.0, 1.0, 2.5],
                [0.5, 0.5, 1e6, 0.5, 1e6, 4.2],
            )
            param_names = ["R0(Ω)", "R1(Ω)", "C1(F)", "R2(Ω)", "C2(F)", "OCV(V)"]

        result = least_squares(
            self._residuals,
            x0,
            args=(t_sec, current, voltage),
            bounds=bounds,
            method="trf",
            max_nfev=10000,
        )

        v_pred = self.simulate_voltage(t_sec, current, result.x)
        residuals = voltage - v_pred
        n = len(voltage)
        p = len(result.x)
        dof = max(n - p, 1)
        mse = np.sum(residuals**2) / dof
        try:
            J = result.jac
            cov = mse * np.linalg.inv(J.T @ J)
            std_err = np.sqrt(np.diag(cov))
            ci = std_err * t.ppf(0.975, dof)
        except Exception:
            ci = np.full_like(result.x, np.nan)

        self.params = {name: float(val) for name, val in zip(param_names, result.x)}
        self.param_ci = {name: float(val) for name, val in zip(param_names, ci)}
        self.fit_result = {
            "t_sec": t_sec,
            "current": current,
            "v_measured": voltage,
            "v_predicted": v_pred,
            "residuals": residuals,
            "r_squared": float(1 - np.sum(residuals**2) / np.sum((voltage - voltage.mean()) ** 2)),
            "pulse": pulse,
        }
        return {"params": self.params, "ci_95": self.param_ci, "r_squared": self.fit_result["r_squared"]}

    def identify_vs_soc(self, df: pd.DataFrame, soc_points: Optional[list[float]] = None) -> pd.DataFrame:
        if soc_points is None:
            soc_points = list(range(0, 101, 10))
        pulses = detect_hppc_pulses(df, current_threshold=0.3)
        if not pulses:
            raise ValueError("未检测到HPPC脉冲")

        results = []
        valid_pulses = [p for p in pulses if p["soc_start"] is not None]
        for soc_target in soc_points:
            if not valid_pulses:
                continue
            nearest = min(valid_pulses, key=lambda p: abs(p["soc_start"] - soc_target))
            try:
                self.identify(df, pulses.index(nearest))
                row = {"SOC(%)": soc_target}
                row.update(self.params)
                results.append(row)
            except Exception:
                continue

        return pd.DataFrame(results)

    def get_fit_plot_data(self) -> Optional[dict]:
        return self.fit_result
