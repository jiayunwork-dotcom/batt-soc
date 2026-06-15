import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from typing import Optional


class OCVSOCCalibrator:
    def __init__(self):
        self.soc_points: np.ndarray = np.linspace(0, 100, 101)
        self.ocv_charge: Optional[np.ndarray] = None
        self.ocv_discharge: Optional[np.ndarray] = None
        self.ocv_mean: Optional[np.ndarray] = None
        self.poly_order: int = 5
        self.poly_coeffs: Optional[np.ndarray] = None
        self.interpolator: Optional[Callable] = None

    def calibrate_low_rate(self, df_charge: pd.DataFrame, df_discharge: pd.DataFrame, capacity_ah: float) -> dict:
        self.ocv_charge = self._process_curve(df_charge, capacity_ah, "charge")
        self.ocv_discharge = self._process_curve(df_discharge, capacity_ah, "discharge")
        if self.ocv_charge is not None and self.ocv_discharge is not None:
            self.ocv_mean = (self.ocv_charge + self.ocv_discharge) / 2.0
        elif self.ocv_charge is not None:
            self.ocv_mean = self.ocv_charge
        else:
            self.ocv_mean = self.ocv_discharge

        self._build_interpolator()
        return self._summary()

    def _process_curve(self, df: pd.DataFrame, capacity_ah: float, mode: str) -> Optional[np.ndarray]:
        if df is None or len(df) < 10:
            return None
        df = df.sort_values("timestamp").reset_index(drop=True)
        dt = df["timestamp"].diff().dt.total_seconds().fillna(0).values
        current = df["current"].values.astype(float)
        voltage = df["voltage"].values.astype(float)
        dq = current * dt / 3600.0
        cum_q = np.cumsum(dq)
        if mode == "charge":
            soc = 100.0 * (cum_q - cum_q.min()) / (cum_q.max() - cum_q.min() + 1e-9)
        else:
            soc = 100.0 * (1.0 - (cum_q - cum_q.min()) / (cum_q.max() - cum_q.min() + 1e-9))
        order = np.argsort(soc)
        soc_sorted = soc[order]
        v_sorted = voltage[order]
        valid = ~np.isnan(soc_sorted) & ~np.isnan(v_sorted)
        soc_sorted = soc_sorted[valid]
        v_sorted = v_sorted[valid]
        if len(soc_sorted) < 2:
            return None
        interp = interp1d(soc_sorted, v_sorted, kind="linear", bounds_error=False, fill_value="extrapolate")
        return np.clip(interp(self.soc_points), 2.5, 4.2)

    def calibrate_dvdq(self, df: pd.DataFrame, capacity_ah: float) -> dict:
        df = df.sort_values("timestamp").reset_index(drop=True)
        voltage = df["voltage"].values.astype(float)
        dt = df["timestamp"].diff().dt.total_seconds().fillna(0).values
        current = df["current"].values.astype(float)
        dq = np.abs(current * dt / 3600.0)
        cum_q = np.cumsum(dq)
        soc = 100.0 * (1.0 - cum_q / (capacity_ah + 1e-9))
        dv = np.gradient(voltage)
        dq_grad = np.gradient(cum_q) + 1e-9
        dvdq = np.abs(dv / dq_grad)
        order = np.argsort(soc)
        soc_sorted = np.clip(soc[order], 0, 100)
        v_sorted = voltage[order]
        valid = ~np.isnan(soc_sorted) & ~np.isnan(v_sorted)
        interp = interp1d(soc_sorted[valid], v_sorted[valid], kind="linear", bounds_error=False, fill_value="extrapolate")
        self.ocv_mean = np.clip(interp(self.soc_points), 2.5, 4.2)
        self._build_interpolator()
        return {**self._summary(), "dvdq_peaks": self._find_peaks(soc, dvdq)}

    def _find_peaks(self, soc: np.ndarray, dvdq: np.ndarray) -> list[dict]:
        from scipy.signal import find_peaks
        peaks, props = find_peaks(dvdq, height=np.std(dvdq) * 0.5, distance=10)
        result = []
        for p in peaks:
            result.append({"soc": float(soc[p]), "dvdq": float(dvdq[p])})
        return result

    def fit_polynomial(self, order: int = 5) -> dict:
        self.poly_order = min(max(order, 5), 8)
        if self.ocv_mean is None:
            raise ValueError("请先进行OCV标定")
        coeffs = np.polyfit(self.soc_points / 100.0, self.ocv_mean, self.poly_order)
        self.poly_coeffs = coeffs
        v_pred = np.polyval(coeffs, self.soc_points / 100.0)
        ss_res = np.sum((self.ocv_mean - v_pred) ** 2)
        ss_tot = np.sum((self.ocv_mean - self.ocv_mean.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return {"order": self.poly_order, "coeffs": coeffs.tolist(), "r_squared": float(r2)}

    def _build_interpolator(self):
        if self.ocv_mean is not None:
            self.interpolator = interp1d(self.soc_points, self.ocv_mean, kind="linear", bounds_error=False, fill_value=(self.ocv_mean[0], self.ocv_mean[-1]))

    def get_ocv(self, soc: float | np.ndarray) -> float | np.ndarray:
        if self.interpolator is not None:
            result = self.interpolator(np.clip(soc, 0, 100))
            return result
        return 3.7 * np.ones_like(np.asarray(soc), dtype=float)

    def edit_point(self, soc_percent: float, new_ocv: float):
        idx = int(round(soc_percent))
        idx = min(max(idx, 0), 100)
        if self.ocv_mean is not None:
            self.ocv_mean[idx] = new_ocv
            self._build_interpolator()

    def get_lookup_table(self) -> pd.DataFrame:
        return pd.DataFrame({"SOC(%)": self.soc_points, "OCV(V)": self.ocv_mean if self.ocv_mean is not None else np.nan})

    def _summary(self) -> dict:
        if self.ocv_mean is None:
            return {}
        return {
            "ocv_min": float(self.ocv_mean.min()),
            "ocv_max": float(self.ocv_mean.max()),
            "num_points": 101,
        }
