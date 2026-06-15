import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def align_timestamps(module_data: dict[str, pd.DataFrame], max_error: float = 1.0) -> pd.DataFrame:
    if not module_data:
        return pd.DataFrame()

    ref_key = next(iter(module_data))
    ref_df = module_data[ref_key].sort_values("timestamp").reset_index(drop=True)
    if len(ref_df) == 0:
        return pd.DataFrame()

    common_ts = ref_df["timestamp"].values
    result = pd.DataFrame({"timestamp": common_ts})
    ref_ts_sec = (common_ts - common_ts[0]).astype("timedelta64[s]").astype(float)

    for mod_id, df in module_data.items():
        if df is None or len(df) == 0:
            continue
        df = df.sort_values("timestamp").reset_index(drop=True)
        mod_ts_sec = (df["timestamp"].values - common_ts[0]).astype("timedelta64[s]").astype(float)

        for col in ["voltage", "current", "temperature"]:
            if col not in df.columns:
                continue
            valid = ~np.isnan(df[col].values.astype(float))
            if valid.sum() < 2:
                result[f"{mod_id}_{col}"] = np.nan
                continue
            try:
                interpolator = interp1d(
                    mod_ts_sec[valid],
                    df[col].values[valid].astype(float),
                    kind="nearest",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                result[f"{mod_id}_{col}"] = interpolator(ref_ts_sec)
            except Exception:
                result[f"{mod_id}_{col}"] = np.nan

    return result


def compute_soc_ah_integration(
    df: pd.DataFrame,
    capacity_ah: float,
    initial_soc: float = 100.0,
) -> np.ndarray:
    if len(df) == 0:
        return np.array([])
    df = df.sort_values("timestamp").reset_index(drop=True)
    dt_sec = df["timestamp"].diff().dt.total_seconds().fillna(0).values
    current = df["current"].values.astype(float)
    dq = current * dt_sec / 3600.0
    soc = initial_soc - 100.0 * np.cumsum(dq) / capacity_ah
    soc = np.clip(soc, 0.0, 100.0)
    return soc


def detect_hppc_pulses(
    df: pd.DataFrame,
    current_threshold: float = 0.5,
    min_pulse_duration: int = 5,
) -> list[dict]:
    if len(df) < 3:
        return []

    df = df.sort_values("timestamp").reset_index(drop=True)
    current = df["current"].values.astype(float)
    current_diff = np.abs(np.diff(current))

    step_indices = np.where(current_diff > current_threshold)[0]
    if len(step_indices) < 2:
        return []

    pulses = []
    for i in range(len(step_indices) - 1):
        start_idx = step_indices[i]
        end_idx = step_indices[i + 1]
        if end_idx - start_idx < min_pulse_duration:
            continue
        pulse_current = current[start_idx + 1 : end_idx].mean()
        if abs(pulse_current) < current_threshold:
            continue
        pulse = {
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
            "start_time": df["timestamp"].iloc[start_idx],
            "end_time": df["timestamp"].iloc[end_idx],
            "pulse_current": float(pulse_current),
            "soc_start": float(df["soc_ah"].iloc[start_idx]) if "soc_ah" in df.columns else None,
        }
        pulses.append(pulse)
    return pulses
