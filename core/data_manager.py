import numpy as np
import pandas as pd
from io import BytesIO
from typing import Optional
from utils.validators import validate_bms_data, detect_anomalies
from utils.preprocessing import align_timestamps, compute_soc_ah_integration


class DataManager:
    def __init__(self):
        self.packs: dict[str, dict] = {}

    def import_csv(self, file_content: BytesIO, pack_id: str) -> dict:
        try:
            df = pd.read_csv(file_content)
        except Exception as e:
            raise ValueError(f"CSV解析失败: {e}")

        required = ["timestamp", "voltage", "current", "temperature"]
        if "module_id" not in df.columns:
            df["module_id"] = "M001"
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"CSV缺少必要字段: {missing}")

        all_warnings = []
        stats = {}

        if pack_id not in self.packs:
            self.packs[pack_id] = {"modules": {}, "capacity_q0": None}

        for mod_id, mod_df in df.groupby("module_id"):
            mod_id_str = str(mod_id)
            validated, warnings = validate_bms_data(mod_df)
            all_warnings.extend([f"[{mod_id_str}] {w}" for w in warnings])
            validated = detect_anomalies(validated)

            if mod_id_str in self.packs[pack_id]["modules"]:
                existing = self.packs[pack_id]["modules"][mod_id_str]
                combined = pd.concat([existing, validated], ignore_index=True)
                combined = combined.sort_values("timestamp").drop_duplicates(
                    subset=["timestamp"], keep="last"
                ).reset_index(drop=True)
                self.packs[pack_id]["modules"][mod_id_str] = combined
            else:
                self.packs[pack_id]["modules"][mod_id_str] = validated

            mod_data = self.packs[pack_id]["modules"][mod_id_str]
            stats[mod_id_str] = {
                "记录总数": len(mod_data),
                "时间起始": str(mod_data["timestamp"].min()),
                "时间结束": str(mod_data["timestamp"].max()),
                "异常记录数": int(mod_data["anomaly_flag"].fillna("").str.len().gt(0).sum()),
            }

        self._detect_q0(pack_id)
        self._compute_soc_for_pack(pack_id)

        return {
            "pack_id": pack_id,
            "modules": stats,
            "warnings": all_warnings,
        }

    def _detect_q0(self, pack_id: str):
        if self.packs[pack_id]["capacity_q0"] is not None:
            return
        for mod_id, df in self.packs[pack_id]["modules"].items():
            df_sorted = df.sort_values("timestamp").reset_index(drop=True)
            if len(df_sorted) < 100:
                continue
            current = df_sorted["current"].values.astype(float)
            voltage = df_sorted["voltage"].values.astype(float)
            discharge_mask = current < -0.1
            if not discharge_mask.any():
                continue
            discharge_indices = np.where(discharge_mask)[0]
            if len(discharge_indices) < 10:
                continue
            groups = []
            current_group = [discharge_indices[0]]
            for i in range(1, len(discharge_indices)):
                if discharge_indices[i] - discharge_indices[i - 1] <= 5:
                    current_group.append(discharge_indices[i])
                else:
                    if len(current_group) >= 10:
                        groups.append(current_group)
                    current_group = [discharge_indices[i]]
            if len(current_group) >= 10:
                groups.append(current_group)
            for grp in groups:
                ts = df_sorted["timestamp"].iloc[grp]
                curr = current[grp]
                dt = ts.diff().dt.total_seconds().fillna(0).values
                capacity_ah = abs(np.sum(curr * dt) / 3600.0)
                v_range = voltage[grp].max() - voltage[grp].min()
                if capacity_ah > 1.0 and v_range > 0.5:
                    self.packs[pack_id]["capacity_q0"] = capacity_ah
                    return
        if self.packs[pack_id]["capacity_q0"] is None:
            self.packs[pack_id]["capacity_q0"] = 100.0

    def _compute_soc_for_pack(self, pack_id: str):
        q0 = self.packs[pack_id]["capacity_q0"] or 100.0
        for mod_id, df in self.packs[pack_id]["modules"].items():
            df = df.sort_values("timestamp").reset_index(drop=True)
            df["soc_ah"] = compute_soc_ah_integration(df, q0, initial_soc=100.0)
            self.packs[pack_id]["modules"][mod_id] = df

    def get_pack_overview(self) -> pd.DataFrame:
        rows = []
        for pack_id, pack_data in self.packs.items():
            for mod_id, df in pack_data["modules"].items():
                rows.append({
                    "Pack编号": pack_id,
                    "模组编号": mod_id,
                    "记录总数": len(df),
                    "起始时间": str(df["timestamp"].min()),
                    "最新时间": str(df["timestamp"].max()),
                    "时间跨度": str(df["timestamp"].max() - df["timestamp"].min()),
                    "异常记录数": int(df["anomaly_flag"].fillna("").str.len().gt(0).sum()),
                    "额定容量(Ah)": round(pack_data["capacity_q0"] or 0, 3),
                })
        return pd.DataFrame(rows)

    def get_module_data(self, pack_id: str, module_id: str, clean: bool = True) -> Optional[pd.DataFrame]:
        if pack_id not in self.packs:
            return None
        if module_id not in self.packs[pack_id]["modules"]:
            return None
        df = self.packs[pack_id]["modules"][module_id].copy()
        if clean:
            df = df[~df["anomaly_flag"].fillna("").str.contains("通信异常")].reset_index(drop=True)
        return df

    def get_aligned_pack_data(self, pack_id: str) -> Optional[pd.DataFrame]:
        if pack_id not in self.packs:
            return None
        module_data = {}
        for mod_id, df in self.packs[pack_id]["modules"].items():
            clean_df = df[~df["anomaly_flag"].fillna("").str.contains("通信异常")].reset_index(drop=True)
            module_data[mod_id] = clean_df
        return align_timestamps(module_data)

    def get_pack_capacity(self, pack_id: str) -> float:
        if pack_id not in self.packs:
            return 100.0
        return self.packs[pack_id]["capacity_q0"] or 100.0

    def list_modules(self, pack_id: str) -> list[str]:
        if pack_id not in self.packs:
            return []
        return sorted(self.packs[pack_id]["modules"].keys())

    def list_packs(self) -> list[str]:
        return sorted(self.packs.keys())
