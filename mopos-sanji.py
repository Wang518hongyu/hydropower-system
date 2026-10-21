# -*- coding: utf-8 -*-
"""
Kaidu River Cascade Reservoir Simulation + Constrained MOPSO
Pure hydropower model with full diagnostic output.

This version includes:
1. Ecological flow split into Qeco_required and Qeco_actual.
2. Qeco_actual = min(Qeco_required, Qturb + Q_nonpower)  (turbine flow counts).
3. Flood rule is actually validated (flood_rule_met).
4. Output adds Eco_Required_m3s / Eco_Actual_m3s, keeps Eco_Release_m3s for backward compatibility.
5. Adaptive FEASIBLE_TOL (1.1x min CV in archive, capped at 0.05).
6. Stagnation detection with Gaussian mutation of worst 30% particles.
7. Small random mutation (10%) applied after velocity/position update.
8. c1=2.0, c2=1.5 for more individual exploration.
9. storage_ratio upper bound relaxed to 0.80.
10. Dead water levels derived per SL 278-2002 (H_min/H_design >= 60%):
    Chahanwusu 80.0 m -> H_min/H_design = 78.1%
    Liushugou  76.0 m -> H_min/H_design = 76.3%
    Dashankou  53.0 m -> H_min/H_design = 73.8%
11. TWO-objective MOPSO: (RMSE_norm, PBIAS_norm). Constraint violation
    is handled as a HARD constraint via FEASIBLE_TOL, not as an objective.
    This avoids the degeneracy observed when CV was constant across the
    Pareto front.
12. normal_zone_power_fraction upper bound matched to Dashankou's
    max_power_fraction (0.90) to avoid search-space redundancy. Previously
    the bound was 0.95, which lay above the physical clip limit of 0.90,
    causing the optimizer to stagnate on a flat plateau.
"""

import numpy as np
import pandas as pd
import gc
import ctypes
import traceback
import os
from copy import deepcopy
from datetime import datetime
from scipy import stats


def trim_memory():
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


# =========================================================
# 1. RESERVOIR CURVE MODEL
# =========================================================
class ReservoirCurveModel:
    def __init__(self, curve_path, reservoir_name=""):
        self.curve_path = curve_path
        self.reservoir_name = reservoir_name
        self.curve, self.Vmax, self.H_design = self._load_curve_flexible(curve_path)

    def _load_curve_flexible(self, path):
        print("\nLoading curve file: {}".format(path))
        excel_file = pd.ExcelFile(path)
        print("Available sheets: {}".format(excel_file.sheet_names))

        sheet_name = None
        for name in ['Curve', 'curve', 'Data', 'data', 'Sheet1']:
            if name in excel_file.sheet_names:
                sheet_name = name
                break
        if sheet_name is None:
            sheet_name = excel_file.sheet_names[0]

        df = pd.read_excel(path, sheet_name=sheet_name)
        print("Shape: {}, Columns: {}".format(df.shape, list(df.columns)))

        Vmax = None
        H_design = None
        for meta_sheet in ['Metadata', 'metadata', 'Info', 'info']:
            if meta_sheet in excel_file.sheet_names:
                df_meta = pd.read_excel(path, sheet_name=meta_sheet)
                if 'Parameter' in df_meta.columns and 'Value' in df_meta.columns:
                    for _, row in df_meta.iterrows():
                        param = str(row['Parameter']).lower()
                        if 'vmax' in param:
                            Vmax = float(row['Value'])
                        if 'h_design' in param or 'design' in param:
                            H_design = float(row['Value'])
                break

        level_col, volume_col, area_col = None, None, None
        for col in df.columns:
            col_lower = str(col).lower()
            if level_col is None and any(
                kw in col_lower for kw in
                ['level', 'elevation', 'stage', 'h_m', 'z_m', 'water', 'h']
            ):
                level_col = col
            if volume_col is None and any(
                kw in col_lower for kw in
                ['volume', 'storage', 'capacity', 'v_m3', 'v']
            ):
                volume_col = col
            if area_col is None and any(
                kw in col_lower for kw in
                ['area', 'surface', 'a']
            ):
                area_col = col

        if level_col is None or volume_col is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                raise ValueError(
                    "{}: cannot identify level/volume columns.".format(path)
                )
            col_means = {col: df[col].mean() for col in numeric_cols}
            sorted_cols = sorted(col_means.items(), key=lambda x: x[1])
            level_col = sorted_cols[0][0]
            volume_col = sorted_cols[-1][0]
            if len(numeric_cols) >= 3:
                area_col = sorted_cols[1][0]

        levels = df[level_col].values.astype(float)
        volumes = df[volume_col].values.astype(float)

        if area_col is not None and area_col in df.columns:
            areas = df[area_col].values.astype(float)
        else:
            areas = np.zeros_like(levels)
            if len(levels) > 1:
                dH = np.diff(levels)
                dV = np.diff(volumes)
                safe_dH = np.where(np.abs(dH) < 1e-12, np.nan, dH)
                areas[1:] = np.nan_to_num(dV / safe_dH, nan=0.0, posinf=0.0, neginf=0.0)
                areas[0] = areas[1] if len(areas) > 1 else 0.0

        valid_mask = ~(np.isnan(levels) | np.isnan(volumes))
        levels = levels[valid_mask]
        volumes = volumes[valid_mask]
        if len(areas) == len(valid_mask):
            areas = areas[valid_mask]

        sort_idx = np.argsort(levels)
        levels = levels[sort_idx]
        volumes = volumes[sort_idx]
        if len(areas) == len(sort_idx):
            areas = areas[sort_idx]

        _, unique_idx = np.unique(levels, return_index=True)
        unique_idx = np.sort(unique_idx)
        levels = levels[unique_idx]
        volumes = volumes[unique_idx]
        if len(areas) >= len(unique_idx):
            areas = areas[unique_idx]

        if Vmax is None:
            Vmax = float(np.nanmax(volumes))
        if H_design is None:
            H_design = float(np.nanmax(levels))

        curve = {"levels": levels, "A": areas, "V": volumes}
        print(
            "Curve loaded: {} points, Vmax={:.2f} m3, H_design={:.2f} m, "
            "H_min={:.2f} m".format(
                len(levels), Vmax, H_design, float(np.nanmin(levels))
            )
        )
        return curve, float(Vmax), float(H_design)


# =========================================================
# 2. CASCADE RESERVOIR OPERATION MODULE
# =========================================================
class CascadeReservoirOperation:
    def __init__(self, curve_models, configs, verbose=True):
        self.curve_models = curve_models
        self.configs = deepcopy(configs)
        self.n_reservoirs = len(curve_models)
        self.verbose = verbose

        self.reservoirs = []
        for i, (curve_model, config) in enumerate(zip(curve_models, self.configs)):
            self.reservoirs.append(self._init_reservoir(curve_model, config, i))

        self.routing_coeff = self.configs[-1].get("routing_coeff", 0.98)
        self.travel_time = self.configs[-1].get("travel_time", 0)
        self.interbasin_flow_ratio = self.configs[-1].get("interbasin_flow_ratio", 0.0)

        self.total_Vmax = sum(res['Vmax'] for res in self.reservoirs)
        self.total_Pmax = sum(res['Pmax'] for res in self.reservoirs)

        self.water_balance_stats = {}
        self.water_deficit = 0
        self.ecological_violation = 0
        self.flood_rule_violation = 0
        self.reset_simulation_state()

        # Print dead storage + SL 278-2002 dead level verification
        print("\n" + "=" * 70)
        print("DEAD STORAGE DIAGNOSTIC")
        print("=" * 70)
        print("  Criterion (SL 278-2002): H_min_at_dead / H_design_gross >= 0.60")
        print("-" * 70)
        for res in self.reservoirs:
            ratio = res['dead_volume'] / res['Vmax'] if res['Vmax'] > 0 else 0.0
            usable = res['Vmax'] - res['dead_volume']

            H_design_gross = res['curve_model'].H_design - res['tailwater']
            H_min_at_dead = res['dead_level'] - res['tailwater']
            head_ratio = H_min_at_dead / H_design_gross if H_design_gross > 0 else 0.0

            print("  {}:".format(res['name']))
            print("    dead_level={:.2f} m, tailwater={:.2f} m, H_design={:.2f} m".format(
                res['dead_level'], res['tailwater'], res['curve_model'].H_design))
            print("    V_dead={:.4e} m3, V_dead/Vmax={:.4f} ({:.2f}%), usable={:.4e} m3".format(
                res['dead_volume'], ratio, ratio * 100.0, usable))
            print("    H_min_at_dead={:.2f} m, H_design_gross={:.2f} m, "
                  "H_min/H_design={:.2f}%".format(
                      H_min_at_dead, H_design_gross, head_ratio * 100.0))

            if head_ratio < 0.60:
                print("    [WARNING] H_min/H_design < 60% (SL 278-2002)")
            elif head_ratio > 0.85:
                print("    [WARNING] H_min/H_design > 85% (dead level may be too high)")
            else:
                print("    [OK] Compliant with SL 278-2002")
        print("=" * 70 + "\n")

    def _init_reservoir(self, curve_model, config, idx):
        impoundment_start_date = config.get(
            "impoundment_start_date",
            config.get("start_date", config.get("commission_date", "1900-01-01"))
        )
        generation_start_date = config.get(
            "generation_start_date",
            config.get("start_date", config.get("commission_date", impoundment_start_date))
        )

        res = {
            'curve_model': curve_model,
            'config': deepcopy(config),
            'idx': idx,
            'name': config.get('name', 'Reservoir_{}'.format(idx + 1)),
            'rho': 1000.0,
            'g': 9.81,
            'eta': config.get("efficiency", 0.9),
            'Pmax': config["installed_capacity_kw"] / 1000.0,
            'dead_level': config.get("dead_water_level", 30.0),
            'spill_level': config.get("spill_level", curve_model.H_design),
            'Vmax': curve_model.Vmax,
            'initial_volume': config.get("initial_volume", 0.0),
            'storage_ratio': config.get("storage_ratio", 0.3),
            'tailwater': config.get("tailwater", 5.0),
            'integral_error': 0.0,
            'prev_error': 0.0,
            'impoundment_start_date': pd.Timestamp(impoundment_start_date),
            'generation_start_date': pd.Timestamp(generation_start_date),
            'min_eco_release': float(config.get("min_eco_release_m3s", 0.0)),
            'flood_trigger_inflow': float(config.get("flood_trigger_inflow_m3s", np.inf)),
        }

        res['monthly_min_eco_release'] = config.get("monthly_min_eco_release_m3s", None)

        res['dead_volume'] = np.interp(
            res['dead_level'], curve_model.curve["levels"], curve_model.curve["V"]
        )

        raw_initial_volume = float(config.get("initial_volume", 0.0))
        monthly_low_cfg = config.get("monthly_normal_low", None)
        start_month = res['impoundment_start_date'].month

        if monthly_low_cfg is not None:
            init_level_default = monthly_low_cfg.get(start_month, res['dead_level'])
        else:
            init_level_default = config.get("normal_low_default", res['dead_level'])

        if raw_initial_volume <= 0:
            res['initial_volume'] = float(
                np.interp(init_level_default, curve_model.curve["levels"], curve_model.curve["V"])
            )
        else:
            res['initial_volume'] = float(np.clip(raw_initial_volume, 0.0, res['Vmax']))

        res['initial_volume'] = max(res['initial_volume'], res['dead_volume'])

        res['monthly_normal_high'] = config.get("monthly_normal_high", None)
        res['default_normal_high'] = config.get("normal_high_default", res['spill_level'] - 1.0)
        res['monthly_normal_low'] = config.get("monthly_normal_low", None)
        res['default_normal_low'] = config.get("normal_low_default", res['dead_level'] + 2.0)

        res['monthly_flood_control'] = config.get("monthly_flood_control_level", None)
        res['flood_control_level_fixed'] = config.get("flood_control_level", res['spill_level'])

        res['active_drawdown_enabled'] = config.get("active_drawdown_enabled", True)
        res['active_drawdown_rate'] = config.get("active_drawdown_rate", 100.0)

        res['use_pid'] = config.get("use_pid", True)
        res['pid_Kp'] = config.get("pid_Kp", 0.15)
        res['pid_Ki'] = config.get("pid_Ki", 0.01)
        res['pid_Kd'] = config.get("pid_Kd", 0.05)
        res['pid_target_offset'] = config.get("pid_target_offset", 0.3)

        res['priority_order'] = ["flood", "eco", "power"]

        res['normal_zone_power_frac'] = config.get("normal_zone_power_fraction", 0.70)
        res['max_power_frac'] = config.get("max_power_fraction", 0.95)
        res['min_power_frac'] = config.get("min_power_fraction", 0.15)
        res['high_buffer'] = config.get("high_buffer", 0.5)
        res['interp_targets'] = config.get("interpolate_targets", True)

        return res

    def reset_simulation_state(self):
        for res in self.reservoirs:
            res['integral_error'] = 0.0
            res['prev_error'] = 0.0

        self.water_balance_stats = {
            res['name']: {
                'total_residual': 0.0, 'max_residual': 0.0, 'count': 0,
                'residuals': [], 'daily_residuals': []
            }
            for res in self.reservoirs
        }
        self.water_deficit = 0
        self.ecological_violation = 0
        self.flood_rule_violation = 0

    def _get_monthly_level(self, monthly_dict, default_val, month):
        if monthly_dict is not None:
            return monthly_dict.get(month, default_val)
        return default_val

    def _get_daily_level(self, date, monthly_dict, default_val):
        month = date.month
        day = date.day
        year = date.year

        first_day = pd.Timestamp(year, month, 1)
        if month == 12:
            next_first_day = pd.Timestamp(year + 1, 1, 1)
        else:
            next_first_day = pd.Timestamp(year, month + 1, 1)

        days_in_month = (next_first_day - first_day).days
        val_this = self._get_monthly_level(monthly_dict, default_val, month)
        next_month = month + 1 if month < 12 else 1
        val_next = self._get_monthly_level(monthly_dict, default_val, next_month)

        frac = (day - 1) / days_in_month
        return val_this + (val_next - val_this) * frac

    def _get_normal_high(self, res, date):
        if res['interp_targets']:
            return self._get_daily_level(date, res['monthly_normal_high'], res['default_normal_high'])
        return self._get_monthly_level(res['monthly_normal_high'], res['default_normal_high'], date.month)

    def _get_normal_low(self, res, date):
        if res['interp_targets']:
            return self._get_daily_level(date, res['monthly_normal_low'], res['default_normal_low'])
        return self._get_monthly_level(res['monthly_normal_low'], res['default_normal_low'], date.month)

    def _get_flood_control_level(self, res, date):
        if res['interp_targets']:
            return self._get_daily_level(date, res['monthly_flood_control'], res['flood_control_level_fixed'])
        if res['monthly_flood_control'] is not None:
            return res['monthly_flood_control'].get(date.month, res['flood_control_level_fixed'])
        return res['flood_control_level_fixed']

    def _storage_control(self, res, V):
        if res['Vmax'] <= 0:
            return 0.0
        storage_state = V / res['Vmax']
        factor = res['storage_ratio'] + (1.0 - res['storage_ratio']) * storage_state
        return np.clip(factor, 0.0, 1.0)

    def _get_eco_value(self, res, month):
        if res['monthly_min_eco_release'] is not None:
            return float(res['monthly_min_eco_release'].get(month, res['min_eco_release']))
        return float(res['min_eco_release'])

    # ---------------------------------------------------------
    # Three-stage dispatch
    # ---------------------------------------------------------
    def _simulate_inactive_reservoir(self, res, Qin, date, V):
        Qin = max(float(Qin), 0.0)
        H_current = np.interp(V, res['curve_model'].curve["V"], res['curve_model'].curve["levels"])
        return {
            'V': V, 'H': H_current, 'power_mw': 0.0,
            'Qturb': 0.0, 'Q_nonpower': Qin, 'Qflood': 0.0,
            'Qdrawdown': 0.0, 'Qspill': 0.0,
            'Qagr': 0.0, 'Qdom': 0.0, 'Qind': 0.0,
            'Qeco_required': 0.0,
            'Qeco_actual': 0.0,
            'Q_mandatory': 0.0, 'Qturb_additional': 0.0,
            'total_outflow': Qin,
            'storage_factor': 0.0, 'power_frac': 0.0, 'head': 0.0,
            'residual': 0.0, 'V_before': V, 'V_after': V,
            'is_operational': False, 'operation_stage': 'PRE_IMPOUNDMENT',
            'is_impounding': False, 'is_generating': False,
            'eco_min_required': 0.0,
            'eco_actual': 0.0,
            'eco_constraint_met': True, 'eco_supplement': 0.0,
            'flood_trigger_inflow': res['flood_trigger_inflow'],
            'forced_flood_triggered': False, 'forced_flood_required': 0.0,
            'forced_flood_supplement': 0.0, 'flood_rule_met': True,
            'original_total_outflow': Qin
        }

    def _simulate_single_reservoir(self, res, Qin, date, V, allow_generation=True):
        curve_model = res['curve_model']
        Qin = max(float(Qin), 0.0)
        month = date.month

        H_current = np.interp(V, curve_model.curve["V"], curve_model.curve["levels"])
        V_before = V

        # Flood split-flow
        trigger = res['flood_trigger_inflow']
        forced_triggered = False
        forced_flood_required = 0.0

        if np.isfinite(trigger) and Qin > trigger:
            forced_triggered = True
            forced_flood_required = Qin - trigger
            Qflood = forced_flood_required
            Qin_effective = trigger
        else:
            Qflood = 0.0
            Qin_effective = Qin

        forced_flood_supplement = Qflood

        # Ecological flow requirement
        Qeco_required = self._get_eco_value(res, month)
        Qagr = 0.0
        Qdom = 0.0
        Qind = 0.0
        Q_demand_total = Qeco_required

        H_high = self._get_normal_high(res, date)
        H_low = self._get_normal_low(res, date)

        # PID target power fraction
        if res['use_pid']:
            target_level = H_high - res['pid_target_offset']
            error = target_level - H_current
            P = res['pid_Kp'] * error
            res['integral_error'] += error
            I = res['pid_Ki'] * res['integral_error']
            D = res['pid_Kd'] * (error - res['prev_error'])
            res['prev_error'] = error
            u = P + I + D
            base_frac = res['normal_zone_power_frac']
            power_frac_unclamped = base_frac - u
            power_frac = max(res['min_power_frac'], min(power_frac_unclamped, res['max_power_frac']))
            if power_frac != power_frac_unclamped:
                res['integral_error'] -= error
        else:
            if H_current >= H_high - res['high_buffer']:
                ratio = (H_current - (H_high - res['high_buffer'])) / max(res['high_buffer'], 1e-6)
                ratio = max(0.0, min(ratio, 1.0))
                power_frac = res['normal_zone_power_frac'] + (res['max_power_frac'] - res['normal_zone_power_frac']) * ratio
            elif H_current >= H_low:
                power_frac = res['normal_zone_power_frac']
            else:
                power_frac = res['normal_zone_power_frac'] * 0.5

        storage_factor = self._storage_control(res, V)

        if H_current >= res['dead_level']:
            head = max(H_current - res['tailwater'], 0.1)
            Qmax_turb = res['Pmax'] * 1e6 / (res['rho'] * res['g'] * head * res['eta'])
        else:
            head = 0.1
            Qmax_turb = 0.0

        available_release = Qin_effective + max(V - res['dead_volume'], 0.0) / 86400.0

        if allow_generation and Qmax_turb > 0:
            Qturb_target = Q_demand_total + Qmax_turb * power_frac * storage_factor
            Qturb = min(Qturb_target, Qmax_turb, available_release)
            Qturb = max(Qturb, 0.0)
            Qturb_additional = max(Qturb - Q_demand_total, 0.0)
        else:
            Qturb = 0.0
            Qturb_additional = 0.0

        Q_nonpower = max(Q_demand_total - Qturb, 0.0)

        total_release_requested = Qturb + Q_nonpower
        if total_release_requested > available_release and total_release_requested > 0:
            scale = available_release / total_release_requested
            Qturb *= scale
            Q_nonpower *= scale
            Qturb_additional = max(Qturb - Q_demand_total, 0.0)

        # Actual ecological release: turbine flow also counts as eco release
        Qeco_actual = min(Qeco_required, Qturb + Q_nonpower)

        if allow_generation and Qturb > 0 and H_current >= res['dead_level']:
            head_eff = max(H_current - res['tailwater'], 0.1)
            power_mw = res['rho'] * res['g'] * Qturb * head_eff * res['eta'] / 1e6
            power_mw = min(power_mw, res['Pmax'])
        else:
            power_mw = 0.0

        V += (Qin_effective - Qturb - Q_nonpower) * 86400.0
        V = max(0.0, V)
        H_current = np.interp(V, curve_model.curve["V"], curve_model.curve["levels"])

        Qdrawdown = 0.0
        if res['active_drawdown_enabled'] and H_current > H_high + 0.5:
            V_high = np.interp(H_high, curve_model.curve["levels"], curve_model.curve["V"])
            excess = V - V_high
            if excess > 0:
                Qdrawdown = min(excess / 86400.0, res['active_drawdown_rate'])
                V -= Qdrawdown * 86400.0
                H_current = np.interp(V, curve_model.curve["V"], curve_model.curve["levels"])

        Qspill = 0.0
        if V > res['Vmax']:
            Qspill = (V - res['Vmax']) / 86400.0
            V = res['Vmax']

        total_outflow = Qturb + Q_nonpower + Qflood + Qdrawdown + Qspill
        V_after = V
        residual = V_after - V_before - (Qin - total_outflow) * 86400.0

        # Flood rule check
        flood_rule_met = True
        if forced_triggered:
            flood_rule_met = bool(Qflood >= forced_flood_required - 1e-8)

        return {
            'V': V, 'H': H_current, 'power_mw': power_mw,
            'Qturb': Qturb, 'Q_nonpower': Q_nonpower, 'Qflood': Qflood,
            'Qdrawdown': Qdrawdown, 'Qspill': Qspill,
            'Qagr': Qagr, 'Qdom': Qdom, 'Qind': Qind,
            'Qeco_required': Qeco_required,
            'Qeco_actual': Qeco_actual,
            'Q_mandatory': Q_demand_total, 'Qturb_additional': Qturb_additional,
            'total_outflow': total_outflow,
            'storage_factor': storage_factor, 'power_frac': power_frac,
            'head': (max(H_current - res['tailwater'], 0.1) if H_current >= res['dead_level'] else 0.0),
            'residual': residual, 'V_before': V_before, 'V_after': V_after,
            'is_operational': True,
            'operation_stage': ('GENERATION' if allow_generation else 'IMPOUNDMENT_ONLY'),
            'is_impounding': True, 'is_generating': bool(allow_generation),
            'eco_min_required': Qeco_required,
            'eco_actual': Qeco_actual,
            'eco_constraint_met': True, 'eco_supplement': 0.0,
            'flood_trigger_inflow': trigger,
            'forced_flood_triggered': forced_triggered,
            'forced_flood_required': forced_flood_required,
            'forced_flood_supplement': forced_flood_supplement,
            'flood_rule_met': flood_rule_met,
            'original_total_outflow': total_outflow
        }

    def _apply_hard_constraints(self, res, result, Qin, date):
        Qin = max(float(Qin), 0.0)
        dt = 86400.0

        # Ecological hard constraint using Qeco_actual
        eco_required = self._get_eco_value(res, date.month)
        eco_actual = max(float(result.get('Qeco_actual', 0.0)), 0.0)

        eco_deficit = max(eco_required - eco_actual, 0.0)

        available_volume = max(result['V'] - res['dead_volume'], 0.0)
        available_flow = available_volume / dt

        eco_supplement = min(eco_deficit, available_flow)

        if eco_supplement > 0:
            result['Qeco_actual'] += eco_supplement
            result['Q_nonpower'] += eco_supplement
            result['total_outflow'] += eco_supplement
            result['V'] -= eco_supplement * dt

        eco_constraint_met = bool(result['Qeco_actual'] >= eco_required - 1e-8)

        result['V'] = max(result['V'], 0.0)
        result['V_after'] = result['V']
        result['H'] = np.interp(
            result['V'], res['curve_model'].curve["V"], res['curve_model'].curve["levels"]
        )
        result['head'] = (
            max(result['H'] - res['tailwater'], 0.1)
            if result['H'] >= res['dead_level'] else 0.0
        )
        result['eco_min_required'] = eco_required
        result['eco_actual'] = result['Qeco_actual']
        result['eco_supplement'] = eco_supplement
        result['eco_constraint_met'] = eco_constraint_met
        result['residual'] = (
            result['V_after'] - result['V_before']
            - (Qin - result['total_outflow']) * dt
        )
        return result

    def _simulate_reservoir_day(self, res, Qin, date, V):
        date = pd.Timestamp(date)
        if date < res['impoundment_start_date']:
            return self._simulate_inactive_reservoir(res, Qin, date, V)

        allow_generation = (date >= res['generation_start_date'])
        result = self._simulate_single_reservoir(res, Qin, date, V, allow_generation=allow_generation)
        result = self._apply_hard_constraints(res, result, Qin, date)
        return result

    # =========================================================
    # FULL SIMULATION (with complete output)
    # =========================================================
    def simulate_cascade_full(self, df, output_dir="results"):
        if df.empty:
            raise ValueError("No data!")

        os.makedirs(output_dir, exist_ok=True)
        output_prefix = os.path.join(output_dir, "cascade_output")

        volumes = [res['initial_volume'] for res in self.reservoirs]
        self.reset_simulation_state()

        daily_data = []
        reservoir_daily_data = {res['name']: [] for res in self.reservoirs}
        water_balance_daily = []

        print("Running cascade simulation...")
        total_days = len(df)

        for idx, (_, row) in enumerate(df.iterrows()):
            Qin_original = max(float(row["Data"]), 0.0)
            date = pd.Timestamp(row["Date"])
            current_inflow = Qin_original
            results = []
            local_inflows = []

            for i, res in enumerate(self.reservoirs):
                local_inflow = current_inflow
                local_inflows.append(local_inflow)

                result = self._simulate_reservoir_day(res, current_inflow, date, volumes[i])
                results.append(result)
                volumes[i] = result['V']

                if result['is_operational']:
                    if result['H'] < res['dead_level'] - 1e-8:
                        self.water_deficit += 1
                    if not result['eco_constraint_met']:
                        self.ecological_violation += 1
                    if not result['flood_rule_met']:
                        self.flood_rule_violation += 1

                abs_residual = abs(result['residual'])
                self.water_balance_stats[res['name']]['total_residual'] += abs_residual
                self.water_balance_stats[res['name']]['max_residual'] = max(
                    self.water_balance_stats[res['name']]['max_residual'], abs_residual)
                self.water_balance_stats[res['name']]['count'] += 1
                self.water_balance_stats[res['name']]['residuals'].append(abs_residual)

                water_balance_daily.append({
                    'Date': date, 'Reservoir': res['name'],
                    'Is_Operational': result['is_operational'],
                    'V_before_m3': result['V_before'], 'V_after_m3': result['V_after'],
                    'Delta_V_m3': result['V_after'] - result['V_before'],
                    'Qin_m3': local_inflow * 86400.0,
                    'Qout_m3': result['total_outflow'] * 86400.0,
                    'Net_Flow_m3': (local_inflow - result['total_outflow']) * 86400.0,
                    'Residual_m3': result['residual'],
                    'Abs_Residual_m3': abs_residual,
                    'Relative_Residual_pct': abs_residual / res['Vmax'] * 100.0 if res['Vmax'] > 0 else 0.0,
                })

                if i < self.n_reservoirs - 1:
                    current_inflow = result['total_outflow'] * self.routing_coeff
                    if self.interbasin_flow_ratio > 0:
                        current_inflow += Qin_original * self.interbasin_flow_ratio

            total_volume = sum(r['V'] for r in results)
            total_power = sum(r['power_mw'] for r in results)
            storage_ratio = total_volume / self.total_Vmax if self.total_Vmax > 0 else 0.0
            total_outflow_system = results[-1]['total_outflow']
            water_utilization = total_outflow_system / Qin_original if Qin_original > 0 else 0.0
            total_residual = sum(r['residual'] for r in results)

            daily_record = {
                'Date': date, 'Year': date.year, 'Month': date.month, 'Day': date.day,
                'System_Inflow_m3s': Qin_original,
                'System_Outflow_m3s': total_outflow_system,
                'System_Water_Utilization': water_utilization,
                'System_Total_Volume_m3': total_volume,
                'System_Storage_Ratio': storage_ratio,
                'System_Total_Power_MW': total_power,
                'System_Water_Balance_Residual_m3': total_residual,
            }

            for i, (res, result) in enumerate(zip(self.reservoirs, results)):
                prefix = 'R{}_{}'.format(i + 1, res['name'])
                inflow_val = local_inflows[i]

                daily_record.update({
                    '{}_Is_Operational'.format(prefix): result['is_operational'],
                    '{}_Operation_Stage'.format(prefix): result.get('operation_stage', ''),
                    '{}_Is_Impounding'.format(prefix): result.get('is_impounding', False),
                    '{}_Is_Generating'.format(prefix): result.get('is_generating', False),
                    '{}_Inflow_m3s'.format(prefix): inflow_val,
                    '{}_Level_m'.format(prefix): result['H'],
                    '{}_Volume_m3'.format(prefix): result['V'],
                    '{}_Storage_Ratio'.format(prefix): result['V'] / res['Vmax'] if res['Vmax'] > 0 else 0.0,
                    '{}_Power_MW'.format(prefix): result['power_mw'],
                    '{}_Head_m'.format(prefix): result['head'],
                    '{}_Turbine_Flow_m3s'.format(prefix): result['Qturb'],
                    '{}_Mandatory_Release_m3s'.format(prefix): result['Q_mandatory'],
                    '{}_Additional_Power_Flow_m3s'.format(prefix): result['Qturb_additional'],
                    '{}_Flood_Release_m3s'.format(prefix): result['Qflood'],
                    '{}_Drawdown_Release_m3s'.format(prefix): result['Qdrawdown'],
                    '{}_Spill_m3s'.format(prefix): result['Qspill'],
                    '{}_Total_Outflow_m3s'.format(prefix): result['total_outflow'],
                    '{}_Agri_Release_m3s'.format(prefix): result['Qagr'],
                    '{}_Dom_Release_m3s'.format(prefix): result['Qdom'],
                    '{}_Eco_Required_m3s'.format(prefix): result['Qeco_required'],
                    '{}_Eco_Actual_m3s'.format(prefix): result['Qeco_actual'],
                    '{}_Eco_Release_m3s'.format(prefix): result['Qeco_actual'],
                    '{}_Ind_Release_m3s'.format(prefix): result['Qind'],
                    '{}_Eco_Min_Required_m3s'.format(prefix): result['eco_min_required'],
                    '{}_Eco_Constraint_Met'.format(prefix): result['eco_constraint_met'],
                    '{}_Eco_Supplement_m3s'.format(prefix): result['eco_supplement'],
                    '{}_Flood_Trigger_Inflow_m3s'.format(prefix): result['flood_trigger_inflow'],
                    '{}_Forced_Flood_Triggered'.format(prefix): result['forced_flood_triggered'],
                    '{}_Flood_Rule_Met'.format(prefix): result['flood_rule_met'],
                    '{}_Water_Balance_Residual_m3'.format(prefix): result['residual'],
                })

                reservoir_record = {
                    'Date': date, 'Year': date.year, 'Month': date.month, 'Day': date.day,
                    'Is_Operational': result['is_operational'],
                    'Operation_Stage': result.get('operation_stage', ''),
                    'Is_Impounding': result.get('is_impounding', False),
                    'Is_Generating': result.get('is_generating', False),
                    'Inflow_m3s': inflow_val,
                    'Level_m': result['H'], 'Volume_m3': result['V'],
                    'Storage_Ratio': result['V'] / res['Vmax'] if res['Vmax'] > 0 else 0.0,
                    'Power_MW': result['power_mw'], 'Head_m': result['head'],
                    'Turbine_Flow_m3s': result['Qturb'],
                    'Mandatory_Release_m3s': result['Q_mandatory'],
                    'Additional_Power_Flow_m3s': result['Qturb_additional'],
                    'Flood_Release_m3s': result['Qflood'],
                    'Drawdown_Release_m3s': result['Qdrawdown'],
                    'Spill_m3s': result['Qspill'],
                    'Total_Outflow_m3s': result['total_outflow'],
                    'Agri_Release_m3s': result['Qagr'],
                    'Dom_Release_m3s': result['Qdom'],
                    'Eco_Required_m3s': result['Qeco_required'],
                    'Eco_Actual_m3s': result['Qeco_actual'],
                    'Eco_Release_m3s': result['Qeco_actual'],
                    'Ind_Release_m3s': result['Qind'],
                    'Eco_Min_Required_m3s': result['eco_min_required'],
                    'Eco_Constraint_Met': result['eco_constraint_met'],
                    'Eco_Supplement_m3s': result['eco_supplement'],
                    'Flood_Trigger_Inflow_m3s': result['flood_trigger_inflow'],
                    'Forced_Flood_Triggered': result['forced_flood_triggered'],
                    'Flood_Rule_Met': result['flood_rule_met'],
                    'Water_Balance_Residual_m3': result['residual'],
                    'Storage_Factor': result['storage_factor'],
                    'Power_Fraction': result['power_frac'],
                }
                reservoir_daily_data[res['name']].append(reservoir_record)

            daily_data.append(daily_record)

            if (idx + 1) % 2000 == 0:
                print("  Progress: {}/{} days simulated".format(idx + 1, total_days))

        daily_df = pd.DataFrame(daily_data)
        daily_df.set_index('Date', inplace=True)
        daily_df.to_csv("{}_system_daily_results.csv".format(output_prefix), encoding="utf-8-sig")
        print("System daily results saved")

        reservoir_daily_dfs = {}
        for res_name, data_list in reservoir_daily_data.items():
            res_df = pd.DataFrame(data_list)
            res_df.set_index('Date', inplace=True)
            reservoir_daily_dfs[res_name] = res_df
            res_df.to_csv("{}_{}_daily_results.csv".format(output_prefix, res_name), encoding="utf-8-sig")
            print("{} daily results saved".format(res_name))

        wb_df = pd.DataFrame(water_balance_daily)
        wb_df.to_csv("{}_water_balance_daily.csv".format(output_prefix), index=False, encoding="utf-8-sig")
        print("Water balance daily data saved")

        self._generate_individual_monthly_stats(reservoir_daily_dfs, output_prefix)
        self._generate_individual_annual_stats(reservoir_daily_dfs, output_prefix)
        self._generate_system_monthly_stats(daily_df, output_prefix)
        self._generate_system_annual_stats(daily_df, output_prefix)
        self._generate_performance_metrics(daily_df, output_prefix)
        self._print_water_balance_diagnostics(output_prefix)
        self._save_constraint_diagnostics(reservoir_daily_dfs, output_prefix)

        return daily_df, reservoir_daily_dfs

    def _save_constraint_diagnostics(self, reservoir_daily_dfs, output_prefix):
        rows = []
        print("\n" + "=" * 70)
        print("ENGINEERING CONSTRAINT DIAGNOSTICS")
        print("=" * 70)

        for res in self.reservoirs:
            name = res['name']
            df = reservoir_daily_dfs[name].copy()
            active = df[df['Is_Operational'] == True].copy()
            if active.empty:
                continue

            eco_fail = (~active['Eco_Constraint_Met'].astype(bool)).sum()
            flood_fail = (~active['Flood_Rule_Met'].astype(bool)).sum()
            trigger_days = active['Forced_Flood_Triggered'].astype(bool).sum()

            eco_compliance = 100.0 * (1.0 - eco_fail / len(active))
            flood_compliance = 100.0 * (1.0 - flood_fail / len(active))

            print("\n{}:".format(name))
            print("  Active days: {}".format(len(active)))
            print("  Eco violation days: {}".format(eco_fail))
            print("  Eco compliance: {:.4f}%".format(eco_compliance))
            print("  Trigger days: {}".format(trigger_days))
            print("  Flood-rule violation days: {}".format(flood_fail))
            print("  Flood-rule compliance: {:.4f}%".format(flood_compliance))

            rows.append({
                'Reservoir': name,
                'Active_Days': len(active),
                'Eco_Violation_Days': int(eco_fail),
                'Eco_Compliance_pct': eco_compliance,
                'Flood_Trigger_Days': int(trigger_days),
                'Flood_Rule_Violation_Days': int(flood_fail),
                'Flood_Rule_Compliance_pct': flood_compliance,
            })

        pd.DataFrame(rows).to_csv(
            "{}_constraint_diagnostics.csv".format(output_prefix),
            index=False, encoding="utf-8-sig"
        )

    def _print_water_balance_diagnostics(self, output_prefix):
        print("\n" + "=" * 70)
        print("COMPREHENSIVE WATER BALANCE DIAGNOSTICS")
        print("=" * 70)

        wb_data = []
        for res_name, stats in self.water_balance_stats.items():
            residuals = np.array(stats['residuals'])
            mean_residual = np.mean(residuals) if len(residuals) > 0 else 0.0
            std_residual = np.std(residuals) if len(residuals) > 0 else 0.0
            max_residual = np.max(residuals) if len(residuals) > 0 else 0.0
            total_residual = np.sum(residuals) if len(residuals) > 0 else 0.0
            p50 = np.percentile(residuals, 50) if len(residuals) > 0 else 0.0
            p95 = np.percentile(residuals, 95) if len(residuals) > 0 else 0.0
            p99 = np.percentile(residuals, 99) if len(residuals) > 0 else 0.0

            res_vmax = next(r['Vmax'] for r in self.reservoirs if r['name'] == res_name)

            print("\n{} Reservoir:".format(res_name))
            print("  Days simulated: {}".format(stats['count']))
            print("  Mean |Residual|: {:.6f} m3".format(mean_residual))
            print("  Std |Residual|: {:.6f} m3".format(std_residual))
            print("  Max |Residual|: {:.6f} m3".format(max_residual))
            print("  Total |Residual|: {:.6f} m3".format(total_residual))
            print("  P50 |Residual|: {:.6f} m3".format(p50))
            print("  P95 |Residual|: {:.6f} m3".format(p95))
            print("  P99 |Residual|: {:.6f} m3".format(p99))
            print("  Reservoir Vmax: {:.2f} m3".format(res_vmax))

            if res_vmax > 0:
                print("  Relative mean residual: {:.8f}% of Vmax".format(mean_residual / res_vmax * 100))
                print("  Relative max residual: {:.8f}% of Vmax".format(max_residual / res_vmax * 100))

            wb_data.append({
                'Reservoir': res_name,
                'Days_Simulated': stats['count'],
                'Mean_Abs_Residual_m3': mean_residual,
                'Std_Abs_Residual_m3': std_residual,
                'Max_Abs_Residual_m3': max_residual,
                'Total_Abs_Residual_m3': total_residual,
                'P50_Residual_m3': p50,
                'P95_Residual_m3': p95,
                'P99_Residual_m3': p99,
                'Vmax_m3': res_vmax,
                'Relative_Mean_Residual_pct': mean_residual / res_vmax * 100 if res_vmax > 0 else np.nan,
                'Relative_Max_Residual_pct': max_residual / res_vmax * 100 if res_vmax > 0 else np.nan,
            })

        pd.DataFrame(wb_data).to_csv(
            "{}_water_balance_diagnostics.csv".format(output_prefix),
            index=False, encoding="utf-8-sig"
        )
        print("\nWater balance diagnostics saved")

    def _generate_individual_monthly_stats(self, reservoir_daily_dfs, output_prefix):
        for res_name, res_df in reservoir_daily_dfs.items():
            monthly_data = []
            for (year, month), group in res_df.groupby([res_df.index.year, res_df.index.month]):
                eco_short_days = int((~group['Eco_Constraint_Met'].astype(bool)).sum())
                record = {
                    'Year': year, 'Month': month,
                    'Total_Inflow_m3': group['Inflow_m3s'].sum() * 86400.0,
                    'Total_Outflow_m3': group['Total_Outflow_m3s'].sum() * 86400.0,
                    'Total_Power_MWh': group['Power_MW'].sum() * 24.0,
                    'Avg_Level_m': group['Level_m'].mean(),
                    'Max_Level_m': group['Level_m'].max(),
                    'Min_Level_m': group['Level_m'].min(),
                    'Avg_Volume_m3': group['Volume_m3'].mean(),
                    'Avg_Storage_Ratio': group['Storage_Ratio'].mean(),
                    'Avg_Head_m': group['Head_m'].mean(),
                    'Total_Turbine_Flow_m3': group['Turbine_Flow_m3s'].sum() * 86400.0,
                    'Total_Eco_Required_m3': group['Eco_Required_m3s'].sum() * 86400.0,
                    'Total_Eco_Actual_m3': group['Eco_Actual_m3s'].sum() * 86400.0,
                    'Total_Eco_Release_m3': group['Eco_Release_m3s'].sum() * 86400.0,
                    'Total_Flood_Release_m3': group['Flood_Release_m3s'].sum() * 86400.0,
                    'Total_Spill_m3': group['Spill_m3s'].sum() * 86400.0,
                    'Eco_Compliance_pct': group['Eco_Constraint_Met'].astype(float).mean() * 100.0,
                    'Eco_Short_Days': eco_short_days,
                    'Flood_Rule_Compliance_pct': group['Flood_Rule_Met'].astype(float).mean() * 100.0,
                    'Mean_WB_Residual_m3': group['Water_Balance_Residual_m3'].abs().mean(),
                    'Max_WB_Residual_m3': group['Water_Balance_Residual_m3'].abs().max(),
                }
                monthly_data.append(record)
            pd.DataFrame(monthly_data).to_csv(
                "{}_{}_monthly_stats.csv".format(output_prefix, res_name),
                index=False, encoding="utf-8-sig"
            )
            print("{} monthly statistics saved".format(res_name))

    def _generate_individual_annual_stats(self, reservoir_daily_dfs, output_prefix):
        res_pmax_dict = {res['name']: res['Pmax'] for res in self.reservoirs}
        for res_name, res_df in reservoir_daily_dfs.items():
            annual_data = []
            for year, group in res_df.groupby(res_df.index.year):
                pmax = max(res_pmax_dict.get(res_name, 1.0), 1e-12)
                eco_short_days = int((~group['Eco_Constraint_Met'].astype(bool)).sum())
                record = {
                    'Year': year,
                    'Pre_Impoundment_Days': int((group['Operation_Stage'] == 'PRE_IMPOUNDMENT').sum()),
                    'Impoundment_Only_Days': int((group['Operation_Stage'] == 'IMPOUNDMENT_ONLY').sum()),
                    'Generation_Days': int((group['Operation_Stage'] == 'GENERATION').sum()),
                    'Total_Inflow_m3': group['Inflow_m3s'].sum() * 86400.0,
                    'Total_Outflow_m3': group['Total_Outflow_m3s'].sum() * 86400.0,
                    'Total_Power_MWh': group['Power_MW'].sum() * 24.0,
                    'Avg_Level_m': group['Level_m'].mean(),
                    'Avg_Storage_Ratio': group['Storage_Ratio'].mean(),
                    'Total_Turbine_Flow_m3': group['Turbine_Flow_m3s'].sum() * 86400.0,
                    'Total_Eco_Required_m3': group['Eco_Required_m3s'].sum() * 86400.0,
                    'Total_Eco_Actual_m3': group['Eco_Actual_m3s'].sum() * 86400.0,
                    'Total_Eco_Release_m3': group['Eco_Release_m3s'].sum() * 86400.0,
                    'Eco_Compliance_pct': group['Eco_Constraint_Met'].astype(float).mean() * 100.0,
                    'Eco_Short_Days': eco_short_days,
                    'Flood_Rule_Compliance_pct': group['Flood_Rule_Met'].astype(float).mean() * 100.0,
                    'Utilization_Hours': group['Power_MW'].sum() * 24.0 / pmax,
                }
                annual_data.append(record)
            pd.DataFrame(annual_data).to_csv(
                "{}_{}_annual_stats.csv".format(output_prefix, res_name),
                index=False, encoding="utf-8-sig"
            )
            print("{} annual statistics saved".format(res_name))

    def _generate_system_monthly_stats(self, daily_df, output_prefix):
        monthly_data = []
        for (year, month), group in daily_df.groupby([daily_df.index.year, daily_df.index.month]):
            record = {
                'Year': year, 'Month': month,
                'System_Total_Inflow_m3': group['System_Inflow_m3s'].sum() * 86400.0,
                'System_Total_Outflow_m3': group['System_Outflow_m3s'].sum() * 86400.0,
                'System_Total_Power_MWh': group['System_Total_Power_MW'].sum() * 24.0,
                'System_Avg_Storage_m3': group['System_Total_Volume_m3'].mean(),
                'System_Avg_Storage_Ratio': group['System_Storage_Ratio'].mean(),
                'System_Avg_Water_Utilization': group['System_Water_Utilization'].mean(),
                'System_Mean_WB_Residual_m3': group['System_Water_Balance_Residual_m3'].abs().mean(),
            }
            monthly_data.append(record)
        pd.DataFrame(monthly_data).to_csv(
            "{}_system_monthly_stats.csv".format(output_prefix),
            index=False, encoding="utf-8-sig"
        )
        print("System monthly statistics saved")

    def _generate_system_annual_stats(self, daily_df, output_prefix):
        annual_data = []
        for year, group in daily_df.groupby(daily_df.index.year):
            record = {
                'Year': year,
                'System_Total_Inflow_m3': group['System_Inflow_m3s'].sum() * 86400.0,
                'System_Total_Power_MWh': group['System_Total_Power_MW'].sum() * 24.0,
                'System_Avg_Storage_m3': group['System_Total_Volume_m3'].mean(),
                'System_Utilization_Hours': group['System_Total_Power_MW'].sum() * 24.0 / max(self.total_Pmax, 1e-12),
            }
            annual_data.append(record)
        pd.DataFrame(annual_data).to_csv(
            "{}_system_annual_stats.csv".format(output_prefix),
            index=False, encoding="utf-8-sig"
        )
        print("System annual statistics saved")

    def _generate_performance_metrics(self, daily_df, output_prefix):
        metrics = {'Metric': [], 'Value': [], 'Unit': [], 'Description': []}

        def add_metric(name, value, unit, desc):
            metrics['Metric'].append(name)
            metrics['Value'].append(value)
            metrics['Unit'].append(unit)
            metrics['Description'].append(desc)

        add_metric('Total_Installed_Capacity', self.total_Pmax, 'MW', 'Total installed capacity')
        add_metric('Total_Storage_Capacity', self.total_Vmax, 'm3', 'Total storage capacity')

        total_power_mwh = daily_df['System_Total_Power_MW'].sum() * 24.0
        add_metric('Total_Power_Generation', total_power_mwh, 'MWh', 'Total power generation')

        avg_utilization = (
            daily_df['System_Total_Power_MW'].sum() * 24.0
            / max(self.total_Pmax, 1e-12) / max(len(daily_df), 1) * 365.0
        )
        add_metric('Average_Annual_Utilization_Hours', avg_utilization, 'hours/year',
                   'Average annual utilization hours')

        for i, res in enumerate(self.reservoirs):
            prefix = 'R{}_{}'.format(i + 1, res['name'])
            res_power_mwh = daily_df['{}_Power_MW'.format(prefix)].sum() * 24.0
            res_util = res_power_mwh / max(res['Pmax'], 1e-12) / max(len(daily_df), 1) * 365.0
            add_metric('{}_Power_Generation'.format(prefix), res_power_mwh, 'MWh',
                       'Power generation of {}'.format(res['name']))
            add_metric('{}_Utilization_Hours'.format(prefix), res_util, 'hours/year',
                       'Annual utilization hours of {}'.format(res['name']))

        pd.DataFrame(metrics).to_csv(
            "{}_performance_metrics.csv".format(output_prefix),
            index=False, encoding="utf-8-sig"
        )
        print("Performance metrics saved")

    # =========================================================
    # Lightweight simulation for MOPSO
    # =========================================================
    def simulate_cascade_monthly_gen(self, df):
        if df.empty:
            return pd.Series(dtype=float), {}

        volumes = [res['initial_volume'] for res in self.reservoirs]
        self.reset_simulation_state()

        monthly_energy = {}
        eco_violation_by_res = {res['name']: 0 for res in self.reservoirs}
        flood_violation_by_res = {res['name']: 0 for res in self.reservoirs}
        flood_trigger_days = {res['name']: 0 for res in self.reservoirs}
        water_deficit_by_res = {res['name']: 0 for res in self.reservoirs}

        for _, row in df.iterrows():
            Qin_original = max(float(row["Data"]), 0.0)
            date = pd.Timestamp(row["Date"])
            month = date.month
            current_inflow = Qin_original

            for i, res in enumerate(self.reservoirs):
                local_inflow = current_inflow
                result = self._simulate_reservoir_day(res, local_inflow, date, volumes[i])
                volumes[i] = result['V']

                if result['is_operational']:
                    if result['H'] < res['dead_level'] - 1e-8:
                        water_deficit_by_res[res['name']] += 1
                    if not result['eco_constraint_met']:
                        eco_violation_by_res[res['name']] += 1
                    if result['forced_flood_triggered']:
                        flood_trigger_days[res['name']] += 1
                    if not result['flood_rule_met']:
                        flood_violation_by_res[res['name']] += 1

                if i < self.n_reservoirs - 1:
                    current_inflow = result['total_outflow'] * self.routing_coeff
                    if self.interbasin_flow_ratio > 0:
                        current_inflow += Qin_original * self.interbasin_flow_ratio

                if i == self.n_reservoirs - 1:
                    daily_energy_mwh = result['power_mw'] * 24.0
                    key = (date.year, month)
                    monthly_energy[key] = monthly_energy.get(key, 0.0) + daily_energy_mwh

        if not monthly_energy:
            return pd.Series(dtype=float), {}

        dates = [pd.Timestamp(year=y, month=m, day=1) for (y, m) in monthly_energy.keys()]
        values = [v / 10.0 for v in monthly_energy.values()]
        monthly_series = pd.Series(values, index=dates, name='gen').sort_index()

        diagnostics = {
            'water_deficit_by_res': water_deficit_by_res,
            'eco_violation_by_res': eco_violation_by_res,
            'flood_violation_by_res': flood_violation_by_res,
            'flood_trigger_days': flood_trigger_days,
            'water_deficit_total': int(sum(water_deficit_by_res.values())),
            'eco_violation_total': int(sum(eco_violation_by_res.values())),
            'flood_violation_total': int(sum(flood_violation_by_res.values())),
        }
        return monthly_series, diagnostics

    def update_calibration_params(self, params, reservoir_idx=2):
        res = self.reservoirs[reservoir_idx]
        res['normal_zone_power_frac'] = params.get(
            "normal_zone_power_fraction", res['normal_zone_power_frac']
        )
        res['storage_ratio'] = params.get("storage_ratio", res['storage_ratio'])

    def compute_downstream_accounting(self, df, downstream_monthly_m3s, start_date=None, end_date=None):
        if df.empty:
            return {}, pd.DataFrame()

        volumes = [res['initial_volume'] for res in self.reservoirs]
        self.reset_simulation_state()

        sum_outflow = {res['name']: {m: 0.0 for m in range(1, 13)} for res in self.reservoirs}
        cnt = {res['name']: {m: 0 for m in range(1, 13)} for res in self.reservoirs}

        for _, row in df.iterrows():
            Qin_original = max(float(row["Data"]), 0.0)
            date = pd.Timestamp(row["Date"])
            month = date.month
            if start_date is not None and date < start_date:
                continue
            if end_date is not None and date > end_date:
                continue

            current_inflow = Qin_original
            for i, res in enumerate(self.reservoirs):
                result = self._simulate_reservoir_day(res, current_inflow, date, volumes[i])
                volumes[i] = result['V']
                sum_outflow[res['name']][month] += result['total_outflow']
                cnt[res['name']][month] += 1
                if i < self.n_reservoirs - 1:
                    current_inflow = result['total_outflow'] * self.routing_coeff

        reservoir_outflow = {}
        for res in self.reservoirs:
            name = res['name']
            monthly_means = {}
            for m in range(1, 13):
                monthly_means[m] = sum_outflow[name][m] / cnt[name][m] if cnt[name][m] > 0 else 0.0
            reservoir_outflow[name] = monthly_means

        dashankou_outflow = reservoir_outflow['Dashankou']
        rows = []
        for m in range(1, 13):
            outflow = dashankou_outflow[m]
            demand = downstream_monthly_m3s.get(m, 0.0)
            remaining = outflow - demand
            rows.append({
                'Month': m,
                'Dashankou_Outflow_m3s': outflow,
                'Downstream_Demand_m3s': demand,
                'Remaining_Flow_m3s': remaining,
            })
        return reservoir_outflow, pd.DataFrame(rows)


# =========================================================
# 3. DATA LOADING
# =========================================================
def load_inflow_data(path):
    df = pd.read_excel(path)
    required = {'Year', 'Month', 'Day', 'Data'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("{} missing fields: {}".format(path, sorted(missing)))
    df["Date"] = pd.to_datetime(df[["Year", "Month", "Day"]])
    df['Data'] = pd.to_numeric(df['Data'], errors='coerce')
    df = df.dropna(subset=['Date', 'Data']).sort_values('Date').reset_index(drop=True)
    return df


def load_actual_power(path):
    df = pd.read_excel(path)
    if 'time' not in df.columns or 'power' not in df.columns:
        raise ValueError("{} must contain columns: time and power.".format(path))
    df['time'] = pd.to_datetime(df['time'])
    df['time'] = df['time'].dt.to_period('M').dt.to_timestamp()
    df['power'] = pd.to_numeric(df['power'], errors='coerce')
    df = df.dropna(subset=['time', 'power']).sort_values('time')
    df.set_index('time', inplace=True)
    return df['power']


# =========================================================
# 4. METRICS
# =========================================================
def compute_all_metrics(sim, obs):
    sim = np.asarray(sim, dtype=float)
    obs = np.asarray(obs, dtype=float)
    mask = ~(np.isnan(sim) | np.isnan(obs))
    sim = sim[mask]; obs = obs[mask]
    if len(sim) < 2:
        return None

    rmse = np.sqrt(np.mean((sim - obs) ** 2))
    den_nse = np.sum((obs - obs.mean()) ** 2)
    nse = 1.0 - np.sum((sim - obs) ** 2) / den_nse if den_nse > 1e-12 else np.nan
    if np.std(sim) > 1e-12 and np.std(obs) > 1e-12:
        r = np.corrcoef(sim, obs)[0, 1]
    else:
        r = np.nan
    alpha = np.std(sim) / np.std(obs) if np.std(obs) > 1e-12 else np.nan
    beta = np.mean(sim) / np.mean(obs) if abs(np.mean(obs)) > 1e-12 else np.nan
    if np.isfinite(r) and np.isfinite(alpha) and np.isfinite(beta):
        kge = 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    else:
        kge = np.nan
    r2 = r ** 2 if np.isfinite(r) else np.nan
    pbias = 100.0 * np.sum(obs - sim) / np.sum(obs) if abs(np.sum(obs)) > 1e-12 else np.nan
    mae = np.mean(np.abs(sim - obs))

    return {
        'RMSE': float(rmse),
        'NSE': float(nse) if np.isfinite(nse) else np.nan,
        'KGE': float(kge) if np.isfinite(kge) else np.nan,
        'R2': float(r2) if np.isfinite(r2) else np.nan,
        'PBIAS': float(pbias) if np.isfinite(pbias) else np.nan,
        'MAE': float(mae)
    }


# =========================================================
# 5. MOPSO EVALUATION  (TWO OBJECTIVES)
# =========================================================
def evaluate_mopso_detailed(op_template, calib_inflow_df, actual_all, params):
    op = deepcopy(op_template)
    op.update_calibration_params(params, reservoir_idx=2)
    op.reset_simulation_state()

    sim_monthly, diagnostics = op.simulate_cascade_monthly_gen(calib_inflow_df)

    calib_start = pd.Timestamp('2010-01-01')
    calib_end = pd.Timestamp('2014-12-01')

    actual_calib = actual_all.loc[
        (actual_all.index >= calib_start) & (actual_all.index <= calib_end)
    ]
    common = actual_calib.index.intersection(sim_monthly.index)
    if len(common) < 12:
        return None

    sim = sim_monthly.loc[common].values.astype(float)
    obs = actual_calib.loc[common].values.astype(float)

    valid = np.isfinite(sim) & np.isfinite(obs)
    sim = sim[valid]; obs = obs[valid]
    if len(sim) < 12:
        return None

    rmse = np.sqrt(np.mean((sim - obs) ** 2))
    rmse_norm = rmse / max(np.std(obs), 1e-6)
    pbias = 100.0 * np.sum(obs - sim) / np.sum(obs) if abs(np.sum(obs)) > 1e-12 else 0.0
    pbias_norm = abs(pbias) / 100.0

    water_deficit = diagnostics['water_deficit_total']
    ecological_violation = diagnostics['eco_violation_total']
    flood_violation = diagnostics['flood_violation_total']

    n_days = max(len(calib_inflow_df), 1)
    eco_norm = ecological_violation / n_days
    flood_norm = flood_violation / n_days
    deficit_norm = water_deficit / n_days

    constraint_violation = eco_norm * 1.0 + flood_norm * 0.5 + deficit_norm * 0.3

    metrics = compute_all_metrics(sim, obs)

    components = {
        'RMSE': rmse, 'RMSE_norm': rmse_norm,
        'PBIAS': pbias, 'PBIAS_norm': pbias_norm,
        'NSE': (metrics['NSE'] if metrics is not None else np.nan),
        'KGE': (metrics['KGE'] if metrics is not None else np.nan),
        'R2': (metrics['R2'] if metrics is not None else np.nan),
        'Water_Deficit_days': water_deficit,
        'Eco_Violation_days': ecological_violation,
        'Flood_Rule_Violation_days': flood_violation,
        'Eco_Violation_Chahanwusu': diagnostics['eco_violation_by_res']['Chahanwusu'],
        'Eco_Violation_Liushugou': diagnostics['eco_violation_by_res']['Liushugou'],
        'Eco_Violation_Dashankou': diagnostics['eco_violation_by_res']['Dashankou'],
        'Flood_Trigger_Days_Chahanwusu': diagnostics['flood_trigger_days']['Chahanwusu'],
        'Flood_Trigger_Days_Liushugou': diagnostics['flood_trigger_days']['Liushugou'],
        'Flood_Trigger_Days_Dashankou': diagnostics['flood_trigger_days']['Dashankou'],
        'Constraint_Violation': constraint_violation,
    }

    # TWO objectives: RMSE_norm and PBIAS_norm
    # Constraint violation is a HARD constraint handled by FEASIBLE_TOL,
    # not an objective.
    return {
        'objectives': np.array([rmse_norm, pbias_norm], dtype=float),
        'constraint_violation': float(constraint_violation),
        'components': components
    }


# =========================================================
# 6. MOPSO  (TWO OBJECTIVES)
# =========================================================
class MultiObjectivePSO:
    FEASIBLE_TOL = 0.01

    def __init__(self, bounds, pop_size=40, max_iter=50, archive_size=80,
                 w_start=0.9, w_end=0.4, c1=2.0, c2=1.5,
                 output_dir="results", seed=42):
        self.bounds = bounds
        self.param_keys = list(bounds.keys())
        self.dim = len(self.param_keys)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.archive_size = archive_size
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.rng = np.random.default_rng(seed)
        self.low = np.array([bounds[k][0] for k in self.param_keys], dtype=float)
        self.high = np.array([bounds[k][1] for k in self.param_keys], dtype=float)

        self.pop = self.rng.uniform(self.low, self.high, size=(pop_size, self.dim))
        span = self.high - self.low
        self.vel = self.rng.uniform(-0.1 * span, 0.1 * span, size=(pop_size, self.dim))

        self.pbest_pos = self.pop.copy()
        self.pbest_obj = [None for _ in range(pop_size)]
        self.pbest_cv = np.full(pop_size, np.inf)

        self.archive = []
        self.history = []
        self.detailed_history = []

    def _decode(self, vector):
        return {k: float(v) for k, v in zip(self.param_keys, vector)}

    @classmethod
    def _update_feasible_tol(cls, archive, tol_cap=0.05, tol_floor=1e-6):
        if not archive:
            return
        cvs = [x['constraint_violation'] for x in archive]
        min_cv = float(np.min(cvs))
        new_tol = min(min_cv * 1.1, tol_cap)
        cls.FEASIBLE_TOL = max(new_tol, tol_floor)

    @classmethod
    def _dominates(cls, obj_a, cv_a, obj_b, cv_b):
        feasible_a = (cv_a <= cls.FEASIBLE_TOL)
        feasible_b = (cv_b <= cls.FEASIBLE_TOL)
        if feasible_a and not feasible_b:
            return True
        if not feasible_a and feasible_b:
            return False
        if not feasible_a and not feasible_b:
            return cv_a < cv_b
        return bool(np.all(obj_a <= obj_b) and np.any(obj_a < obj_b))

    @staticmethod
    def _crowding_distance(archive):
        n = len(archive)
        if n == 0:
            return np.array([])
        if n <= 2:
            return np.full(n, np.inf)
        F = np.vstack([x['objectives'] for x in archive])
        distance = np.zeros(n, dtype=float)
        for m in range(F.shape[1]):
            order = np.argsort(F[:, m])
            distance[order[0]] = np.inf
            distance[order[-1]] = np.inf
            fmin = F[order[0], m]
            fmax = F[order[-1], m]
            if fmax - fmin <= 1e-12:
                continue
            for j in range(1, n - 1):
                if np.isinf(distance[order[j]]):
                    continue
                distance[order[j]] += (F[order[j + 1], m] - F[order[j - 1], m]) / (fmax - fmin)
        return distance

    def _update_archive(self, candidates):
        pool = self.archive + candidates
        feasible_pool = [x for x in pool if x['constraint_violation'] <= self.FEASIBLE_TOL]
        if feasible_pool:
            pool = feasible_pool
        else:
            min_cv = min(x['constraint_violation'] for x in pool)
            pool = [x for x in pool if x['constraint_violation'] == min_cv]

        nondominated = []
        for i, cand in enumerate(pool):
            dominated = False
            for j, other in enumerate(pool):
                if i == j:
                    continue
                if self._dominates(other['objectives'], other['constraint_violation'],
                                   cand['objectives'], cand['constraint_violation']):
                    dominated = True
                    break
            if not dominated:
                nondominated.append(cand)

        unique = []
        for cand in nondominated:
            is_dup = False
            for u in unique:
                if (np.allclose(cand['objectives'], u['objectives'], rtol=1e-8, atol=1e-10)
                        and np.allclose(cand['position'], u['position'], rtol=1e-8, atol=1e-10)):
                    is_dup = True
                    break
            if not is_dup:
                unique.append(cand)

        while len(unique) > self.archive_size:
            cd = self._crowding_distance(unique)
            finite_idx = np.where(np.isfinite(cd))[0]
            if len(finite_idx) == 0:
                remove_idx = int(self.rng.integers(0, len(unique)))
            else:
                remove_idx = int(finite_idx[np.argmin(cd[finite_idx])])
            unique.pop(remove_idx)

        self.archive = unique

    def _select_leader(self):
        if not self.archive:
            return self.pop[self.rng.integers(0, self.pop_size)].copy()
        cd = self._crowding_distance(self.archive)
        weights = cd.copy()
        finite = np.isfinite(weights)
        max_finite = np.max(weights[finite]) if np.any(finite) else 1.0
        weights[~finite] = max_finite * 2.0 + 1.0
        weights = np.maximum(weights, 1e-12)
        weights = weights / np.sum(weights)
        idx = self.rng.choice(len(self.archive), p=weights)
        return self.archive[idx]['position'].copy()

    def _clip_and_reflect(self, pos, vel):
        for d in range(self.dim):
            low = self.low[d]; high = self.high[d]
            if pos[d] < low:
                pos[d] = low; vel[d] *= -0.5
            elif pos[d] > high:
                pos[d] = high; vel[d] *= -0.5
        return pos, vel

    def optimize(self, eval_func, verbose=True):
        self.history = []
        self.detailed_history = []

        stagnation_counter = 0
        last_best_key = None

        for it in range(self.max_iter):
            if self.max_iter > 1:
                w = self.w_start - (self.w_start - self.w_end) * it / (self.max_iter - 1)
            else:
                w = self.w_end

            iteration_candidates = []
            for i in range(self.pop_size):
                params = self._decode(self.pop[i])
                try:
                    result = eval_func(params)
                except Exception:
                    print("\n[WARNING] Iteration={}, Particle={} failed:".format(it + 1, i + 1))
                    traceback.print_exc()
                    result = None

                if result is None:
                    objectives = np.array([1e9, 1e9], dtype=float)
                    cv = 1e12; components = {}
                else:
                    objectives = np.asarray(result['objectives'], dtype=float)
                    cv = float(result['constraint_violation'])
                    components = result['components']

                candidate = {
                    'position': self.pop[i].copy(),
                    'objectives': objectives.copy(),
                    'constraint_violation': cv,
                    'params': params.copy(),
                    'components': deepcopy(components)
                }
                iteration_candidates.append(candidate)

                record = {
                    'Iteration': it + 1, 'Particle': i + 1,
                    **params,
                    'Objective_RMSE_norm': objectives[0],
                    'Objective_PBIAS_norm': objectives[1],
                    'Constraint_Violation': cv,
                    **components
                }
                self.detailed_history.append(record)

                if self.pbest_obj[i] is None:
                    replace = True
                else:
                    current_dom = self._dominates(objectives, cv, self.pbest_obj[i], self.pbest_cv[i])
                    old_dom = self._dominates(self.pbest_obj[i], self.pbest_cv[i], objectives, cv)
                    if current_dom:
                        replace = True
                    elif old_dom:
                        replace = False
                    else:
                        replace = bool(self.rng.random() < 0.5)

                if replace:
                    self.pbest_obj[i] = objectives.copy()
                    self.pbest_cv[i] = cv
                    self.pbest_pos[i] = self.pop[i].copy()

                gc.collect(); trim_memory()

            self._update_archive(iteration_candidates)

            # Adaptive feasibility tolerance
            self._update_feasible_tol(self.archive)

            feasible_count = sum(
                x['constraint_violation'] <= self.FEASIBLE_TOL
                for x in iteration_candidates
            )

            hist = {
                'Iteration': it + 1,
                'Archive_Size': len(self.archive),
                'Feasible_Particles': feasible_count,
                'FEASIBLE_TOL': self.FEASIBLE_TOL,
            }
            if self.archive:
                F = np.vstack([x['objectives'] for x in self.archive])
                cv_vals = [x['constraint_violation'] for x in self.archive]
                hist['Best_RMSE_norm'] = float(np.min(F[:, 0]))
                hist['Best_PBIAS_norm'] = float(np.min(F[:, 1]))
                hist['Best_CV'] = float(np.min(cv_vals))
            self.history.append(hist)

            # --- Stagnation detection ---
            if self.archive:
                F_cur = np.vstack([x['objectives'] for x in self.archive])
                cv_cur = [x['constraint_violation'] for x in self.archive]
                cur_key = (
                    round(float(F_cur[:, 0].min()), 8),
                    round(float(F_cur[:, 1].min()), 8),
                    round(float(np.min(cv_cur)), 8),
                )
                if last_best_key is not None and cur_key == last_best_key:
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0
                last_best_key = cur_key

            if stagnation_counter >= 5:
                n_mutate = max(1, int(0.3 * self.pop_size))
                obj_scores = [x['objectives'][0] for x in iteration_candidates]
                worst_idx = np.argsort(obj_scores)[-n_mutate:]
                for w_i in worst_idx:
                    sigma = 0.15 * (self.high - self.low)
                    self.pop[w_i] = np.clip(
                        self.pop[w_i] + self.rng.normal(0, sigma),
                        self.low, self.high
                    )
                stagnation_counter = 0
                if verbose:
                    print("  [Stagnation] Mutation applied to {} particles".format(n_mutate))

            # --- Velocity / position update with small random mutation ---
            mutation_prob = 0.1
            for i in range(self.pop_size):
                leader = self._select_leader()
                r1 = self.rng.random(self.dim)
                r2 = self.rng.random(self.dim)
                self.vel[i] = (w * self.vel[i]
                               + self.c1 * r1 * (self.pbest_pos[i] - self.pop[i])
                               + self.c2 * r2 * (leader - self.pop[i]))
                self.pop[i] = self.pop[i] + self.vel[i]
                self.pop[i], self.vel[i] = self._clip_and_reflect(self.pop[i], self.vel[i])

                mask = self.rng.random(self.dim) < mutation_prob
                if np.any(mask):
                    self.pop[i] = np.where(
                        mask,
                        self.rng.uniform(self.low, self.high),
                        self.pop[i]
                    )

            if verbose:
                if self.archive:
                    F = np.vstack([x['objectives'] for x in self.archive])
                    cv_vals = [x['constraint_violation'] for x in self.archive]
                    print("Iter {:3d}/{} | Archive={:3d} | Feasible={:3d}/{} | TOL={:.5f} | "
                          "min RMSE={:.6f} | min PBIAS={:.6f} | min CV={:.4f}".format(
                              it + 1, self.max_iter, len(self.archive), feasible_count,
                              self.pop_size, self.FEASIBLE_TOL,
                              F[:, 0].min(), F[:, 1].min(), min(cv_vals)))
                else:
                    print("Iter {:3d}/{} | Archive empty".format(it + 1, self.max_iter))

        pd.DataFrame(self.detailed_history).to_csv(
            os.path.join(self.output_dir, "mopso_detailed_history.csv"),
            index=False, encoding="utf-8-sig")
        pd.DataFrame(self.history).to_csv(
            os.path.join(self.output_dir, "mopso_convergence.csv"),
            index=False, encoding="utf-8-sig")
        self._save_pareto_front()
        return self.archive

    def _save_pareto_front(self):
        rows = []
        for i, x in enumerate(self.archive, start=1):
            rows.append({
                'Solution': i,
                **x['params'],
                'Objective_RMSE_norm': x['objectives'][0],
                'Objective_PBIAS_norm': x['objectives'][1],
                'Constraint_Violation': x['constraint_violation'],
                **x['components']
            })
        pd.DataFrame(rows).to_csv(
            os.path.join(self.output_dir, "mopso_pareto_front.csv"),
            index=False, encoding="utf-8-sig")


# =========================================================
# 7. TOPSIS
# =========================================================
def select_topsis_solution(archive):
    tol = MultiObjectivePSO.FEASIBLE_TOL
    feasible = [x for x in archive if x['constraint_violation'] <= tol]
    if not feasible:
        feasible = sorted(archive, key=lambda x: x['constraint_violation'])[:10]
        print("[TOPSIS] No solution under tol={}, fallback to top-10 lowest CV".format(tol))

    F = np.vstack([x['objectives'] for x in feasible])
    fmin = np.min(F, axis=0); fmax = np.max(F, axis=0)
    denom = np.where(fmax - fmin > 1e-12, fmax - fmin, 1.0)
    Z = (F - fmin) / denom
    ideal = np.zeros(Z.shape[1]); anti_ideal = np.ones(Z.shape[1])
    d_pos = np.linalg.norm(Z - ideal, axis=1)
    d_neg = np.linalg.norm(Z - anti_ideal, axis=1)
    score = d_neg / (d_pos + d_neg + 1e-12)
    best_idx = int(np.argmax(score))
    best = deepcopy(feasible[best_idx])
    best['TOPSIS_Score'] = float(score[best_idx])
    return best


# =========================================================
# 8. MAIN
# =========================================================
if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR = "results_kaidu_{}".format(timestamp)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("KAIDU CASCADE - PURE HYDROPOWER + FULL DIAGNOSTIC OUTPUT")
    print("pop_size=40, max_iter=50, TWO-objective MOPSO")
    print("Objectives: RMSE_norm, PBIAS_norm")
    print("Constraint violation handled as HARD constraint (FEASIBLE_TOL)")
    print("Dead levels derived per SL 278-2002 (H_min/H_design >= 60%)")
    print("normal_zone_power_fraction upper bound = 0.90 (= max_power_fraction)")
    print("Results will be saved to: {}/".format(RESULTS_DIR))
    print("=" * 70)

    CURVE_PATHS = [
        "chahanwusu/inverse_ch.xlsx",
        "liushugou/inverse_liushugou.xlsx",
        "dashankou/inverse_dashankou.xlsx"
    ]
    RESERVOIR_NAMES = ["Chahanwusu", "Liushugou", "Dashankou"]
    INFLOW_PATH = "hebin.xlsx"
    ACTUAL_PATH = "dianliang.xlsx"

    bounds = {
        "normal_zone_power_fraction": (0.20, 0.90),   # matched to Dashankou max_power_frac
        "storage_ratio": (0.05, 0.80)
    }

    eco_monthly_up = {
        1: 8.0, 2: 8.0, 3: 12.0,
        4: 18.0, 5: 22.57, 6: 22.57,
        7: 22.57, 8: 22.57, 9: 18.0,
        10: 12.0, 11: 8.0, 12: 8.0
    }
    eco_monthly_down = {
        1: 4.0, 2: 4.0, 3: 6.0,
        4: 9.0, 5: 11.20, 6: 11.20,
        7: 11.20, 8: 11.20, 9: 9.0,
        10: 6.0, 11: 4.0, 12: 4.0
    }

    downstream_demand_monthly = {
        1: 10.0, 2: 10.0, 3: 18.0,
        4: 92.0, 5: 142.0, 6: 166.0,
        7: 186.0, 8: 166.0, 9: 116.0,
        10: 70.0, 11: 25.0, 12: 12.0
    }

    configs = [
        {
            "name": "Chahanwusu",
            "impoundment_start_date": "2007-01-01",
            "generation_start_date": "2007-01-01",
            "min_eco_release_m3s": 22.57,
            "monthly_min_eco_release_m3s": eco_monthly_up,
            "flood_trigger_inflow_m3s": 220.0,
            "installed_capacity_kw": 330000,
            "tailwater": 5, "efficiency": 0.90, "initial_volume": 0,
            # Dead level derived per SL 278-2002:
            #   H_min/H_design_gross = (80-5)/(101-5) = 78.1%  >= 60%
            "dead_water_level": 80.0,
            "spill_level": 97.5,
            "monthly_normal_high": {1: 96.5, 2: 96.5, 3: 96.5, 4: 96.0, 5: 95.5, 6: 96.0,
                                    7: 96.0, 8: 96.0, 9: 95.5, 10: 96.0, 11: 96.5, 12: 96.5},
            "monthly_normal_low": {1: 92.0, 2: 92.0, 3: 92.0, 4: 91.5, 5: 91.0, 6: 91.0,
                                   7: 91.5, 8: 91.5, 9: 91.5, 10: 92.0, 11: 92.0, 12: 92.0},
            "monthly_flood_control_level": {1: 96.0, 2: 96.0, 3: 96.0, 4: 95.5, 5: 95.0, 6: 95.0,
                                            7: 95.0, 8: 95.0, 9: 95.0, 10: 95.5, 11: 96.0, 12: 96.0},
            "active_drawdown_enabled": True, "active_drawdown_rate": 150.0,
            "use_pid": True, "pid_Kp": 0.12, "pid_Ki": 0.008,
            "pid_Kd": 0.04, "pid_target_offset": 0.5,
            "max_power_fraction": 0.95, "min_power_fraction": 0.15, "high_buffer": 0.5,
            "storage_ratio": 0.25, "interpolate_targets": True
        },
        {
            "name": "Liushugou",
            "impoundment_start_date": "2013-01-01",
            "generation_start_date": "2014-01-01",
            "min_eco_release_m3s": 22.57,
            "monthly_min_eco_release_m3s": eco_monthly_up,
            "flood_trigger_inflow_m3s": 220.0,
            "installed_capacity_kw": 180000,
            "tailwater": 5, "efficiency": 0.90, "initial_volume": 0,
            # Dead level derived per SL 278-2002:
            #   H_min/H_design_gross = (76-5)/(98-5) = 76.3%  >= 60%
            "dead_water_level": 76.0,
            "spill_level": 93.5,
            "monthly_normal_high": {1: 92.5, 2: 92.5, 3: 92.5, 4: 92.0, 5: 91.5, 6: 92.0,
                                    7: 92.0, 8: 92.0, 9: 91.5, 10: 92.0, 11: 92.5, 12: 92.5},
            "monthly_normal_low": {1: 89.5, 2: 89.5, 3: 89.5, 4: 89.0, 5: 88.5, 6: 88.5,
                                   7: 89.0, 8: 89.0, 9: 89.0, 10: 89.5, 11: 89.5, 12: 89.5},
            "monthly_flood_control_level": {1: 92.0, 2: 92.0, 3: 92.0, 4: 91.5, 5: 91.0, 6: 91.0,
                                            7: 91.0, 8: 91.0, 9: 91.0, 10: 91.5, 11: 92.0, 12: 92.0},
            "active_drawdown_enabled": True, "active_drawdown_rate": 120.0,
            "use_pid": True, "pid_Kp": 0.13, "pid_Ki": 0.009,
            "pid_Kd": 0.045, "pid_target_offset": 0.4,
            "max_power_fraction": 0.95, "min_power_fraction": 0.15, "high_buffer": 0.5,
            "storage_ratio": 0.28, "interpolate_targets": True
        },
        {
            "name": "Dashankou",
            "impoundment_start_date": "2008-01-01",
            "generation_start_date": "2008-01-01",
            "min_eco_release_m3s": 11.20,
            "monthly_min_eco_release_m3s": eco_monthly_down,
            "flood_trigger_inflow_m3s": 160.0,
            "installed_capacity_kw": 80000,
            "tailwater": 5, "efficiency": 0.90, "initial_volume": 0,
            # Dead level derived per SL 278-2002:
            #   H_min/H_design_gross = (53-5)/(70-5) = 73.8%  >= 60%
            "dead_water_level": 53.0,
            "spill_level": 68.0,
            "monthly_normal_high": {1: 67.5, 2: 67.5, 3: 67.5, 4: 67.0, 5: 66.5, 6: 67.0,
                                    7: 67.0, 8: 67.0, 9: 66.5, 10: 67.0, 11: 67.5, 12: 67.5},
            "monthly_normal_low": {1: 63.0, 2: 63.0, 3: 63.0, 4: 62.5, 5: 62.0, 6: 62.0,
                                   7: 62.5, 8: 62.5, 9: 62.5, 10: 63.0, 11: 63.0, 12: 63.0},
            "monthly_flood_control_level": {1: 67.0, 2: 67.0, 3: 67.0, 4: 67.0, 5: 66.5, 6: 66.0,
                                            7: 66.0, 8: 66.0, 9: 66.5, 10: 67.0, 11: 67.0, 12: 67.0},
            "active_drawdown_enabled": True, "active_drawdown_rate": 100.0,
            "use_pid": True, "pid_Kp": 0.15, "pid_Ki": 0.01,
            "pid_Kd": 0.05, "pid_target_offset": 0.6,
            "max_power_fraction": 0.90, "min_power_fraction": 0.20, "high_buffer": 1.0,
            "storage_ratio": 0.30, "interpolate_targets": True,
            "routing_coeff": 0.98, "travel_time": 0, "interbasin_flow_ratio": 0.0
        }
    ]

    print("Loading reservoir curves...")
    curve_models = [ReservoirCurveModel(CURVE_PATHS[i], RESERVOIR_NAMES[i]) for i in range(3)]

    print("\nLoading inflow and actual power data...")
    inflow_df = load_inflow_data(INFLOW_PATH)
    actual_all = load_actual_power(ACTUAL_PATH)
    print("Data loaded successfully.")

    warmup_start_dt = pd.Timestamp("2008-01-01")
    calib_end_dt = pd.Timestamp("2014-12-31")
    calib_inflow_df = inflow_df[
        (inflow_df["Date"] >= warmup_start_dt) & (inflow_df["Date"] <= calib_end_dt)
    ].copy()

    cascade_op = CascadeReservoirOperation(curve_models, configs, verbose=True)

    def fitness_mopso(params):
        return evaluate_mopso_detailed(cascade_op, calib_inflow_df, actual_all, params)

    print("\n" + "=" * 70)
    print("Starting MOPSO (pop_size=40, max_iter=50, 2 objectives)")
    print("=" * 70)

    mopso = MultiObjectivePSO(
        bounds=bounds, pop_size=40, max_iter=50, archive_size=80,
        w_start=0.9, w_end=0.4, c1=2.0, c2=1.5,
        output_dir=RESULTS_DIR, seed=42
    )

    pareto_archive = mopso.optimize(fitness_mopso, verbose=True)
    if not pareto_archive:
        raise RuntimeError("MOPSO failed: Pareto archive is empty.")

    print("\nPareto solutions: {}".format(len(pareto_archive)))

    best_solution = select_topsis_solution(pareto_archive)
    best_params = best_solution['params']

    print("\n" + "=" * 70)
    print("MOPSO + TOPSIS BEST SOLUTION")
    print("=" * 70)
    print("\nBest parameters:")
    for key, value in best_params.items():
        print("  {}: {:.6f}".format(key, value))

    print("\nObjectives (2-objective MOPSO):")
    print("  RMSE_norm: {:.6f}".format(best_solution['objectives'][0]))
    print("  abs(PBIAS)/100: {:.6f}".format(best_solution['objectives'][1]))
    print("  Constraint violation (hard constraint): {:.6f}".format(
        best_solution['constraint_violation']))
    print("  TOPSIS score: {:.6f}".format(best_solution['TOPSIS_Score']))

    print("\nConstraint diagnostics of best solution:")
    comp = best_solution['components']
    print("  Eco violation days: {}".format(comp.get('Eco_Violation_days', 'N/A')))
    print("  Flood violation days: {}".format(comp.get('Flood_Rule_Violation_days', 'N/A')))
    print("  Water deficit days: {}".format(comp.get('Water_Deficit_days', 'N/A')))

    # =====================================================
    # Full simulation with best params
    # =====================================================
    print("\n" + "=" * 70)
    print("FULL CASCADE SIMULATION WITH BEST PARAMS")
    print("=" * 70)

    final_op = CascadeReservoirOperation(curve_models, configs, verbose=True)
    final_op.update_calibration_params(best_params, reservoir_idx=2)
    final_op.reset_simulation_state()

    result_full, reservoir_daily_dfs = final_op.simulate_cascade_full(
        inflow_df, output_dir=RESULTS_DIR
    )

    # =====================================================
    # Validation metrics
    # =====================================================
    downstream_power = result_full['R3_Dashankou_Power_MW']
    monthly_sim = downstream_power.resample('ME').sum() * 24.0 / 10.0
    monthly_sim.index = monthly_sim.index.to_period('M').to_timestamp()

    cal_sim = monthly_sim.loc['2010-01':'2014-12']
    cal_obs = actual_all.loc['2010-01':'2014-12']
    cal_common = cal_sim.index.intersection(cal_obs.index)
    cal_metrics = compute_all_metrics(cal_sim.loc[cal_common].values,
                                       cal_obs.loc[cal_common].values)

    val_sim = monthly_sim.loc['2015-01':'2019-12']
    val_obs = actual_all.loc['2015-01':'2019-12']
    val_common = val_sim.index.intersection(val_obs.index)
    val_metrics = compute_all_metrics(val_sim.loc[val_common].values,
                                       val_obs.loc[val_common].values)

    print("\n" + "=" * 60)
    print("VALIDATION METRICS (Dashankou Reservoir)")
    print("=" * 60)
    print("\nCalibration Period (2010-2014):")
    if cal_metrics is not None:
        for k, v in cal_metrics.items():
            print("  {}: {:.4f}".format(k, v))
    print("\nValidation Period (2015-2019):")
    if val_metrics is not None:
        for k, v in val_metrics.items():
            print("  {}: {:.4f}".format(k, v))

    if cal_metrics is not None and val_metrics is not None:
        pd.DataFrame({
            'Metric': list(cal_metrics.keys()),
            'Calibration': list(cal_metrics.values()),
            'Validation': [val_metrics.get(k, np.nan) for k in cal_metrics.keys()],
        }).to_csv(os.path.join(RESULTS_DIR, "validation_metrics.csv"),
                   index=False, encoding="utf-8-sig")

    comparison = pd.DataFrame({
        "Year_Month": monthly_sim.index.strftime("%Y-%m"),
        "Simulated_10^4kWh": monthly_sim.values,
        "Actual_10^4kWh": actual_all.reindex(monthly_sim.index).values
    })
    comparison["Error_10^4kWh"] = comparison["Simulated_10^4kWh"] - comparison["Actual_10^4kWh"]
    comparison.to_excel(os.path.join(RESULTS_DIR, "cascade_monthly_comparison.xlsx"), index=False)

    # =====================================================
    # Downstream water accounting
    # =====================================================
    print("\n" + "=" * 70)
    print("DOWNSTREAM WATER ACCOUNTING")
    print("=" * 70)

    final_op2 = CascadeReservoirOperation(curve_models, configs, verbose=False)
    final_op2.update_calibration_params(best_params, reservoir_idx=2)
    reservoir_outflow, accounting_df = final_op2.compute_downstream_accounting(
        inflow_df, downstream_demand_monthly,
        start_date=pd.Timestamp("2010-01-01"),
        end_date=pd.Timestamp("2019-12-31")
    )

    print("\nMean monthly outflow per reservoir (m3/s):")
    for name, mm in reservoir_outflow.items():
        print("  {}:".format(name))
        for m in range(1, 13):
            print("    Month {:>2d}: {:.2f}".format(m, mm[m]))

    print("\nDashankou outflow vs downstream demand (m3/s):")
    print(accounting_df.to_string(index=False))

    accounting_df.to_csv(os.path.join(RESULTS_DIR, "downstream_water_accounting.csv"),
                          index=False, encoding="utf-8-sig")

    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    annual_outflow_volume = sum(accounting_df['Dashankou_Outflow_m3s'].iloc[m-1] * 86400 * days_per_month[m-1]
                                for m in range(1, 13))
    annual_demand_volume = sum(accounting_df['Downstream_Demand_m3s'].iloc[m-1] * 86400 * days_per_month[m-1]
                                for m in range(1, 13))
    annual_remaining_volume = sum(accounting_df['Remaining_Flow_m3s'].iloc[m-1] * 86400 * days_per_month[m-1]
                                   for m in range(1, 13))

    print("\nAnnual summary (based on mean monthly):")
    print("  Dashankou total outflow:   {:.4e} m3/year".format(annual_outflow_volume))
    print("  Downstream demand:         {:.4e} m3/year".format(annual_demand_volume))
    print("  Remaining to Bosten Lake:  {:.4e} m3/year".format(annual_remaining_volume))

    # =====================================================
    # Save best params / TOPSIS solution
    # =====================================================
    pd.DataFrame([{
        **best_params,
        'Objective_RMSE_norm': best_solution['objectives'][0],
        'Objective_PBIAS_norm': best_solution['objectives'][1],
        'Constraint_Violation': best_solution['constraint_violation'],
        'TOPSIS_Score': best_solution['TOPSIS_Score'],
        'FEASIBLE_TOL_Final': MultiObjectivePSO.FEASIBLE_TOL,
        **best_solution['components']
    }]).to_csv(os.path.join(RESULTS_DIR, "mopso_topsis_best_solution.csv"),
               index=False, encoding="utf-8-sig")

    pd.DataFrame({
        'Parameter': list(best_params.keys()),
        'Value': list(best_params.values())
    }).to_csv(os.path.join(RESULTS_DIR, "best_params.csv"),
               index=False, encoding="utf-8-sig")

    print("\nAll results saved to: {}/".format(RESULTS_DIR))
    print("\nGenerated files:")
    for f in sorted(os.listdir(RESULTS_DIR)):
        print("  - {}".format(f))