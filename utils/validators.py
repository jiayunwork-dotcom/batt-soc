import numpy as np
import pandas as pd


def validate_bms_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    warnings = []
    required_cols = ["timestamp", "voltage", "current", "temperature"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要字段: {missing}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().any():
        warnings.append(f"存在 {df['timestamp'].isna().sum()} 条无效时间戳，已丢弃")
        df = df.dropna(subset=["timestamp"])

    df["anomaly_flag"] = ""

    voltage_mask = (df["voltage"] < 2.5) | (df["voltage"] > 4.2)
    if voltage_mask.any():
        warnings.append(f"存在 {voltage_mask.sum()} 条电压超出范围(2.5V-4.2V)，已标记异常")
        df.loc[voltage_mask, "anomaly_flag"] += "电压异常;"

    temp_mask = (df["temperature"] < -20) | (df["temperature"] > 60)
    if temp_mask.any():
        warnings.append(f"存在 {temp_mask.sum()} 条温度超出范围(-20~60°C)，已标记异常")
        df.loc[temp_mask, "anomaly_flag"] += "温度异常;"

    df["current_sign"] = np.where(df["current"] > 0, "充电", np.where(df["current"] < 0, "放电", "静置"))
    return df.reset_index(drop=True), warnings


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    if len(df) < 2:
        return df

    dt = df["timestamp"].diff().dt.total_seconds()
    dv = df["voltage"].diff().abs()
    dv_dt = dv / dt.replace(0, np.nan)
    jump_mask = dv_dt > 0.5
    if jump_mask.any():
        df.loc[jump_mask, "anomaly_flag"] = df.loc[jump_mask, "anomaly_flag"].fillna("") + "通信异常;"

    sensor_fault_mask = df["temperature"] == -40.0
    if sensor_fault_mask.any():
        df = df[~sensor_fault_mask].reset_index(drop=True)

    return df
