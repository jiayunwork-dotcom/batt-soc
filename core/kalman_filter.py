import numpy as np
import pandas as pd
from typing import Optional, Callable


class BaseKalmanFilter:
    def __init__(self, ocv_func: Callable, capacity_ah: float):
        self.ocv_func = ocv_func
        self.Q = capacity_ah * 3600.0
        self.x: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        self.Q_cov: Optional[np.ndarray] = None
        self.R_cov: Optional[np.ndarray] = None
        self.history: dict = {}

    def reset(self, initial_soc: float = 100.0, params: Optional[dict] = None):
        self.params = params or {"R0": 0.01, "R1": 0.005, "C1": 1000.0}
        self.state_dim = 2
        self.x = np.array([initial_soc, 0.0])
        self.P = np.eye(self.state_dim) * 1e-2
        self.Q_cov = np.diag([1e-4, 1e-6])
        self.R_cov = np.array([[1e-3]])
        self.history = {"soc": [], "v_pred": [], "k_gain": [], "cov": [], "time": []}


class EKF(BaseKalmanFilter):
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("timestamp").reset_index(drop=True)
        if self.x is None:
            self.reset()
        R0 = self.params.get("R0", 0.01)
        R1 = self.params.get("R1", 0.005)
        C1 = self.params.get("C1", 1000.0)
        tau1 = R1 * C1
        n = len(df)
        soc_est = np.zeros(n)
        v_pred = np.zeros(n)
        v_meas = df["voltage"].values.astype(float)
        current = df["current"].values.astype(float)
        ts = df["timestamp"].values
        dt = np.diff(ts).astype("timedelta64[s]").astype(float)
        dt = np.concatenate([[dt[0] if len(dt) > 0 else 1.0], dt])

        x = self.x.copy()
        P = self.P.copy()
        for k in range(n):
            I = current[k]
            dt_k = max(dt[k], 0.1)
            alpha = np.exp(-dt_k / tau1) if tau1 > 0 else 0
            F = np.array([
                [1.0, 0.0],
                [0.0, alpha],
            ])
            B = np.array([
                [-100.0 * dt_k / self.Q],
                [R1 * (1 - alpha)],
            ])
            x_pred = F @ x + B.flatten() * I
            x_pred[0] = np.clip(x_pred[0], 0.0, 100.0)
            P_pred = F @ P @ F.T + self.Q_cov
            soc_k = x_pred[0]
            h = self.ocv_func(soc_k) - I * R0 - x_pred[1]
            dOCVdSOC = (self.ocv_func(soc_k + 0.01) - self.ocv_func(soc_k - 0.01)) / 0.02
            H = np.array([[dOCVdSOC, -1.0]])
            S = H @ P_pred @ H.T + self.R_cov
            K = (P_pred @ H.T @ np.linalg.inv(S))
            innovation = v_meas[k] - h
            x = x_pred + (K * innovation).flatten()
            x[0] = np.clip(x[0], 0.0, 100.0)
            P = (np.eye(self.state_dim) - K @ H) @ P_pred
            soc_est[k] = x[0]
            v_pred[k] = h
            self.history["k_gain"].append(K.flatten().tolist())
            self.history["cov"].append(np.diag(P).tolist())
        self.x = x
        self.P = P
        result = df.copy()
        result["soc_ekf"] = soc_est
        result["v_predicted"] = v_pred
        result["soc_ah"] = self._ah_integration(df)
        self.history["soc"] = soc_est.tolist()
        self.history["v_pred"] = v_pred.tolist()
        self.history["time"] = df["timestamp"].astype(str).tolist()
        return result

    def _ah_integration(self, df: pd.DataFrame) -> np.ndarray:
        dt = df["timestamp"].diff().dt.total_seconds().fillna(0).values
        current = df["current"].values.astype(float)
        dq = current * dt / 3600.0
        capacity_ah = self.Q / 3600.0
        return np.clip(100.0 - 100.0 * np.cumsum(dq) / capacity_ah, 0.0, 100.0)


class UKF(BaseKalmanFilter):
    def __init__(self, ocv_func: Callable, capacity_ah: float, alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0):
        super().__init__(ocv_func, capacity_ah)
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

    def _sigma_points(self, x: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(x)
        lambda_ = self.alpha**2 * (n + self.kappa) - n
        scale = np.sqrt(n + lambda_)
        L = np.linalg.cholesky(P + 1e-9 * np.eye(n))
        sigmas = np.zeros((2 * n + 1, n))
        sigmas[0] = x
        for i in range(n):
            sigmas[i + 1] = x + scale * L[i]
            sigmas[n + i + 1] = x - scale * L[i]
        Wm = np.zeros(2 * n + 1)
        Wc = np.zeros(2 * n + 1)
        Wm[0] = lambda_ / (n + lambda_)
        Wc[0] = lambda_ / (n + lambda_) + (1 - self.alpha**2 + self.beta)
        for i in range(1, 2 * n + 1):
            Wm[i] = 1.0 / (2 * (n + lambda_))
            Wc[i] = 1.0 / (2 * (n + lambda_))
        return sigmas, Wm, Wc

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("timestamp").reset_index(drop=True)
        if self.x is None:
            self.reset()
        R0 = self.params.get("R0", 0.01)
        R1 = self.params.get("R1", 0.005)
        C1 = self.params.get("C1", 1000.0)
        tau1 = R1 * C1
        n = len(df)
        soc_est = np.zeros(n)
        v_pred = np.zeros(n)
        v_meas = df["voltage"].values.astype(float)
        current = df["current"].values.astype(float)
        ts = df["timestamp"].values
        dt = np.diff(ts).astype("timedelta64[s]").astype(float)
        dt = np.concatenate([[dt[0] if len(dt) > 0 else 1.0], dt])

        x = self.x.copy()
        P = self.P.copy()
        for k in range(n):
            I = current[k]
            dt_k = max(dt[k], 0.1)
            alpha = np.exp(-dt_k / tau1) if tau1 > 0 else 0
            sigmas, Wm, Wc = self._sigma_points(x, P)
            sigma_pred = np.zeros_like(sigmas)
            for i, s in enumerate(sigmas):
                sigma_pred[i, 0] = s[0] - 100.0 * I * dt_k / self.Q
                sigma_pred[i, 1] = alpha * s[1] + R1 * (1 - alpha) * I
            x_pred = Wm @ sigma_pred
            x_pred[0] = np.clip(x_pred[0], 0.0, 100.0)
            P_pred = np.zeros_like(P)
            for i in range(len(sigmas)):
                diff = sigma_pred[i] - x_pred
                P_pred += Wc[i] * np.outer(diff, diff)
            P_pred += self.Q_cov
            z_sigma = np.zeros(len(sigmas))
            for i, s in enumerate(sigma_pred):
                z_sigma[i] = self.ocv_func(s[0]) - I * R0 - s[1]
            z_pred = Wm @ z_sigma
            Pzz = 0.0
            for i in range(len(sigmas)):
                Pzz += Wc[i] * (z_sigma[i] - z_pred) ** 2
            Pzz += self.R_cov[0, 0]
            Pxz = np.zeros((2, 1))
            for i in range(len(sigmas)):
                Pxz += Wc[i] * np.outer(sigma_pred[i] - x_pred, z_sigma[i] - z_pred)
            K = Pxz / Pzz
            innovation = v_meas[k] - z_pred
            x = x_pred + (K * innovation).flatten()
            x[0] = np.clip(x[0], 0.0, 100.0)
            P = P_pred - K @ Pxz.T
            soc_est[k] = x[0]
            v_pred[k] = z_pred
        self.x = x
        self.P = P
        result = df.copy()
        result["soc_ukf"] = soc_est
        result["v_predicted"] = v_pred
        result["soc_ah"] = self._ah_integration(df)
        self.history["soc"] = soc_est.tolist()
        self.history["v_pred"] = v_pred.tolist()
        return result

    def _ah_integration(self, df: pd.DataFrame) -> np.ndarray:
        dt = df["timestamp"].diff().dt.total_seconds().fillna(0).values
        current = df["current"].values.astype(float)
        dq = current * dt / 3600.0
        capacity_ah = self.Q / 3600.0
        return np.clip(100.0 - 100.0 * np.cumsum(dq) / capacity_ah, 0.0, 100.0)
