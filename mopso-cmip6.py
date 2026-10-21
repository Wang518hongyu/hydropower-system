# -*- coding: utf-8 -*-
"""
Kaidu Cascade Reservoir - MOPSO Calibration + Historical + Future Scenarios
==========================================================================
- Core dispatch: pure-hydropower model (2-objective version)
- Calibration: 2-objective MOPSO + TOPSIS
  (RMSE_norm, PBIAS_norm) with constraint violation as HARD constraint
- Historical simulation: 2008-01-01 -> 2016-12-31
- Future scenarios: SSP126 / SSP370 / SSP585 (inherit storage at 2016-12-31)
- Dead levels: 80/76/53 m per SL 278-2002
- Adaptive FEASIBLE_TOL + stagnation mutation + 10% random mutation
- Legacy columns (Agri/Dom/Ind/Eco release) + new diagnostics
"""

import os
import gc
import ctypes
import traceback
import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime
from scipy import stats


# =========================================================
# 0. UTILITIES
# =========================================================
def trim_memory():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def clean_memory():
    gc.collect()
    trim_memory()


def ensure(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_df(df, path):
    if df is None or len(df) == 0:
        return
    df.to_csv(path, index=False, encoding="utf-8-sig")


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
        sheet_name = None
        for name in ['Curve', 'curve', 'Data', 'data', 'Sheet1']:
            if name in excel_file.sheet_names:
                sheet_name = name
                break
        if sheet_name is None:
            sheet_name = excel_file.sheet_names[0]
        df = pd.read_excel(path, sheet_name=sheet_name)

        Vmax, H_design = None, None
        for meta_sheet in ['Metadata', 'metadata', 'Info', 'info']:
            if meta_sheet in excel_file.sheet_names:
                df_meta = pd.read_excel(path, sheet_name=meta_sheet)
                if 'Parameter' in df_meta.columns and 'Value' in df_meta.columns:
                    for _, row in df_meta.iterrows():
                        p = str(row['Parameter']).lower()
                        if 'vmax' in p:
                            Vmax = float(row['Value'])
                        if 'h_design' in p or 'design' in p:
                            H_design = float(row['Value'])
                break

        level_col = None
        volume_col = None
        area_col = None
        for col in df.columns:
            cl = str(col).lower()
            if level_col is None and any(k in cl for k in
                    ['level', 'elevation', 'stage', 'h_m', 'z_m', 'water', 'h']):
                level_col = col
            if volume_col is None and any(k in cl for k in
                    ['volume', 'storage', 'capacity', 'v_m3', 'v']):
                volume_col = col
            if area_col is None and any(k in cl for k in
                    ['area', 'surface', 'a']):
                area_col = col

        if level_col is None or volume_col is None:
            nums = df.select_dtypes(include=[np.number]).columns.tolist()
            col_means = {c: df[c].mean() for c in nums}
            sorted_cols = sorted(col_means.items(), key=lambda x: x[1])
            level_col = sorted_cols[0][0]
            volume_col = sorted_cols[-1][0]
            if len(nums) >= 3:
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
                areas[1:] = np.nan_to_num(dV / safe_dH, nan=0.0,
                                          posinf=0.0, neginf=0.0)
                areas[0] = areas[1] if len(areas) > 1 else 0.0

        mask = ~(np.isnan(levels) | np.isnan(volumes))
        levels = levels[mask]
        volumes = volumes[mask]
        if len(areas) == len(mask):
            areas = areas[mask]

        idx = np.argsort(levels)
        levels = levels[idx]
        volumes = volumes[idx]
        if len(areas) == len(idx):
            areas = areas[idx]

        _, uniq = np.unique(levels, return_index=True)
        uniq = np.sort(uniq)
        levels = levels[uniq]
        volumes = volumes[uniq]
        if len(areas) >= len(uniq):
            areas = areas[uniq]

        if Vmax is None:
            Vmax = float(np.nanmax(volumes))
        if H_design is None:
            H_design = float(np.nanmax(levels))

        curve = {"levels": levels, "A": areas, "V": volumes}
        print("Curve loaded: {} pts, Vmax={:.2f} m3, H_design={:.2f} m, "
              "H_min={:.2f} m".format(
                  len(levels), Vmax, H_design, float(np.nanmin(levels))))
        return curve, float(Vmax), float(H_design)


# =========================================================
# 2. CASCADE RESERVOIR OPERATION (pure hydro + future support)
# =========================================================
class CascadeReservoirOperation:
    def __init__(self, curve_models, configs, verbose=True):
        self.curve_models = curve_models
        self.configs = deepcopy(configs)
        self.n_reservoirs = len(curve_models)
        self.verbose = verbose

        self.reservoirs = []
        for i, (cm, cfg) in enumerate(zip(curve_models, self.configs)):
            self.reservoirs.append(self._init_reservoir(cm, cfg, i))

        self.routing_coeff = self.configs[-1].get("routing_coeff", 0.98)
        self.travel_time = self.configs[-1].get("travel_time", 0)
        self.interbasin_flow_ratio = self.configs[-1].get(
            "interbasin_flow_ratio", 0.0)

        self.total_Vmax = sum(r['Vmax'] for r in self.reservoirs)
        self.total_Pmax = sum(r['Pmax'] for r in self.reservoirs)

        self.final_volumes = None
        self.water_balance_stats = {}
        self.water_deficit = 0
        self.ecological_violation = 0
        self.flood_rule_violation = 0
        self._reset_water_balance_stats()

        if verbose:
            self._print_dead_storage_diagnostic()

    def _print_dead_storage_diagnostic(self):
        print("\n" + "=" * 70)
        print("DEAD STORAGE DIAGNOSTIC")
        print("=" * 70)
        print("  Criterion (SL 278-2002): H_min_at_dead / H_design_gross >= 0.60")
        print("-" * 70)
        for res in self.reservoirs:
            ratio = res['dead_volume'] / res['Vmax'] if res['Vmax'] > 0 else 0.0
            usable = res['Vmax'] - res['dead_volume']
            Hg = res['curve_model'].H_design - res['tailwater']
            Hm = res['dead_level'] - res['tailwater']
            hr = Hm / Hg if Hg > 0 else 0.0
            print("  {}:".format(res['name']))
            print("    dead_level={:.2f} m, tailwater={:.2f} m, H_design={:.2f} m"
                  .format(res['dead_level'], res['tailwater'],
                          res['curve_model'].H_design))
            print("    V_dead={:.4e} m3, V_dead/Vmax={:.4f} ({:.2f}%), "
                  "usable={:.4e} m3".format(
                      res['dead_volume'], ratio, ratio * 100.0, usable))
            print("    H_min_at_dead={:.2f} m, H_design_gross={:.2f} m, "
                  "H_min/H_design={:.2f}%".format(Hm, Hg, hr * 100.0))
            if hr < 0.60:
                print("    [WARNING] H_min/H_design < 60% (SL 278-2002)")
            elif hr > 0.85:
                print("    [WARNING] H_min/H_design > 85%")
            else:
                print("    [OK] Compliant with SL 278-2002")
        print("=" * 70 + "\n")

    def _reset_water_balance_stats(self):
        self.water_balance_stats = {
            r['name']: {'total_residual': 0.0, 'max_residual': 0.0,
                        'count': 0, 'residuals': []}
            for r in self.reservoirs
        }
        self.water_deficit = 0
        self.ecological_violation = 0
        self.flood_rule_violation = 0

    def _init_reservoir(self, cm, cfg, idx):
        impound = cfg.get("impoundment_start_date",
                         cfg.get("start_date",
                                 cfg.get("commission_date", "1900-01-01")))
        gen = cfg.get("generation_start_date",
                      cfg.get("start_date",
                              cfg.get("commission_date", impound)))

        res = {
            'curve_model': cm, 'config': deepcopy(cfg), 'idx': idx,
            'name': cfg.get('name', 'Reservoir_%d' % (idx + 1)),
            'rho': 1000.0, 'g': 9.81,
            'eta': cfg.get("efficiency", 0.9),
            'Pmax': cfg["installed_capacity_kw"] / 1000.0,
            'dead_level': cfg.get("dead_water_level", 30.0),
            'spill_level': cfg.get("spill_level", cm.H_design),
            'Vmax': cm.Vmax,
            'initial_volume': cfg.get("initial_volume", 0.0),
            'storage_ratio': cfg.get("storage_ratio", 0.3),
            'tailwater': cfg.get("tailwater", 5.0),
            'integral_error': 0.0, 'prev_error': 0.0,
            'impoundment_start_date': pd.Timestamp(impound),
            'generation_start_date': pd.Timestamp(gen),
            'min_eco_release': float(cfg.get("min_eco_release_m3s", 0.0)),
            'flood_trigger_inflow': float(
                cfg.get("flood_trigger_inflow_m3s", np.inf)),
            'agri_ratio': float(cfg.get("agri_ratio", 0.0)),
            'dom_ratio': float(cfg.get("domestic_ratio", 0.0)),
            'eco_ratio': float(cfg.get("eco_ratio", 0.0)),
            'ind_ratio': float(cfg.get("industry_ratio", 0.0)),
        }
        res['monthly_min_eco_release'] = cfg.get(
            "monthly_min_eco_release_m3s", None)
        res['dead_volume'] = float(np.interp(res['dead_level'],
                                             cm.curve["levels"], cm.curve["V"]))

        raw = float(cfg.get("initial_volume", 0.0))
        ml = cfg.get("monthly_normal_low", None)
        sm = res['impoundment_start_date'].month
        init_default = ml.get(sm, res['dead_level']) if ml else \
            cfg.get("normal_low_default", res['dead_level'])
        if raw <= 0:
            res['initial_volume'] = float(np.interp(
                init_default, cm.curve["levels"], cm.curve["V"]))
        else:
            res['initial_volume'] = float(np.clip(raw, 0.0, res['Vmax']))
        res['initial_volume'] = max(res['initial_volume'], res['dead_volume'])

        res['monthly_normal_high'] = cfg.get("monthly_normal_high", None)
        res['default_normal_high'] = cfg.get("normal_high_default",
                                             res['spill_level'] - 1.0)
        res['monthly_normal_low'] = cfg.get("monthly_normal_low", None)
        res['default_normal_low'] = cfg.get("normal_low_default",
                                            res['dead_level'] + 2.0)
        res['monthly_flood_control'] = cfg.get(
            "monthly_flood_control_level", None)
        res['flood_control_level_fixed'] = cfg.get("flood_control_level",
                                                   res['spill_level'])
        res['active_drawdown_enabled'] = cfg.get(
            "active_drawdown_enabled", True)
        res['active_drawdown_rate'] = cfg.get("active_drawdown_rate", 100.0)
        res['use_pid'] = cfg.get("use_pid", True)
        res['pid_Kp'] = cfg.get("pid_Kp", 0.15)
        res['pid_Ki'] = cfg.get("pid_Ki", 0.01)
        res['pid_Kd'] = cfg.get("pid_Kd", 0.05)
        res['pid_target_offset'] = cfg.get("pid_target_offset", 0.3)
        res['normal_zone_power_frac'] = cfg.get(
            "normal_zone_power_fraction", 0.70)
        res['max_power_frac'] = cfg.get("max_power_fraction", 0.95)
        res['min_power_frac'] = cfg.get("min_power_fraction", 0.15)
        res['high_buffer'] = cfg.get("high_buffer", 0.5)
        res['interp_targets'] = cfg.get("interpolate_targets", True)
        return res

    # ----- Future-scenario storage management -----
    def set_initial_volumes(self, volumes):
        if len(volumes) != self.n_reservoirs:
            raise ValueError("Initial volume count mismatch")
        for res, V in zip(self.reservoirs, volumes):
            V = float(np.clip(V, 0.0, res['Vmax']))
            res['initial_volume'] = V
            if self.verbose:
                print("  {}: initial storage = {:.2f} m3 ({:.2f}% of Vmax)"
                      .format(res['name'], V, V / res['Vmax'] * 100))

    def get_final_volumes(self):
        if self.final_volumes is None:
            raise RuntimeError("No simulation completed yet.")
        return list(self.final_volumes)

    def save_initial_storage(self, volumes, path):
        rows = []
        for res, V in zip(self.reservoirs, volumes):
            rows.append({
                'Reservoir': res['name'],
                'Initial_volume_m3': V,
                'Storage_ratio': V / res['Vmax'] if res['Vmax'] > 0 else 0.0,
            })
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")

    def update_calibration_params(self, params, reservoir_idx=2):
        res = self.reservoirs[reservoir_idx]
        res['normal_zone_power_frac'] = params.get(
            "normal_zone_power_fraction", res['normal_zone_power_frac'])
        res['storage_ratio'] = params.get("storage_ratio", res['storage_ratio'])

    def reset_controller_state(self):
        for res in self.reservoirs:
            res['integral_error'] = 0.0
            res['prev_error'] = 0.0

    def reset_simulation_state(self):
        for res in self.reservoirs:
            cfg_vol = res['config'].get("initial_volume", 0.0)
            if cfg_vol > 0:
                res['initial_volume'] = float(np.clip(cfg_vol, 0.0, res['Vmax']))
            res['integral_error'] = 0.0
            res['prev_error'] = 0.0
        self._reset_water_balance_stats()

    # ----- Level helpers -----
    def _get_monthly_level(self, d, default, month):
        return d.get(month, default) if d is not None else default

    def _get_daily_level(self, date, d, default):
        m, day, y = date.month, date.day, date.year
        first = pd.Timestamp(y, m, 1)
        nxt = pd.Timestamp(y + 1, 1, 1) if m == 12 else pd.Timestamp(y, m + 1, 1)
        days = (nxt - first).days
        v1 = self._get_monthly_level(d, default, m)
        nm = m + 1 if m < 12 else 1
        v2 = self._get_monthly_level(d, default, nm)
        return v1 + (v2 - v1) * (day - 1) / days

    def _get_normal_high(self, res, date):
        if res['interp_targets']:
            return self._get_daily_level(date, res['monthly_normal_high'],
                                         res['default_normal_high'])
        return self._get_monthly_level(res['monthly_normal_high'],
                                       res['default_normal_high'], date.month)

    def _get_normal_low(self, res, date):
        if res['interp_targets']:
            return self._get_daily_level(date, res['monthly_normal_low'],
                                         res['default_normal_low'])
        return self._get_monthly_level(res['monthly_normal_low'],
                                       res['default_normal_low'], date.month)

    def _storage_control(self, res, V):
        if res['Vmax'] <= 0:
            return 0.0
        s = V / res['Vmax']
        return float(np.clip(
            res['storage_ratio'] + (1.0 - res['storage_ratio']) * s, 0.0, 1.0))

    def _get_eco_value(self, res, month):
        if res['monthly_min_eco_release'] is not None:
            return float(res['monthly_min_eco_release'].get(
                month, res['min_eco_release']))
        return float(res['min_eco_release'])

    # ----- Inactive / single-reservoir dispatch -----
    def _simulate_inactive(self, res, Qin, date, V):
        Qin = max(float(Qin), 0.0)
        H = np.interp(V, res['curve_model'].curve["V"],
                      res['curve_model'].curve["levels"])
        return {
            'V': V, 'H': H, 'power_mw': 0.0,
            'Qturb': 0.0, 'Q_nonpower': Qin, 'Qflood': 0.0,
            'Qdrawdown': 0.0, 'Qspill': 0.0,
            'Qagr': 0.0, 'Qdom': 0.0, 'Qind': 0.0,
            'Qeco_required': 0.0, 'Qeco_actual': 0.0,
            'Q_mandatory': 0.0, 'Qturb_additional': 0.0,
            'total_outflow': Qin, 'storage_factor': 0.0,
            'power_frac': 0.0, 'head': 0.0,
            'residual': 0.0, 'V_before': V, 'V_after': V,
            'is_operational': False, 'operation_stage': 'PRE_IMPOUNDMENT',
            'is_impounding': False, 'is_generating': False,
            'eco_min_required': 0.0, 'eco_actual': 0.0,
            'eco_constraint_met': True, 'eco_supplement': 0.0,
            'flood_trigger_inflow': res['flood_trigger_inflow'],
            'forced_flood_triggered': False, 'forced_flood_required': 0.0,
            'forced_flood_supplement': 0.0, 'flood_rule_met': True,
            'original_total_outflow': Qin,
        }

    def _simulate_single(self, res, Qin, date, V, allow_gen=True):
        cm = res['curve_model']
        Qin = max(float(Qin), 0.0)
        month = date.month
        H = np.interp(V, cm.curve["V"], cm.curve["levels"])
        V_before = V

        # ---- Flood split-flow + rule validation ----
        trigger = res['flood_trigger_inflow']
        forced = False
        flood_req = 0.0
        if np.isfinite(trigger) and Qin > trigger:
            forced = True
            flood_req = Qin - trigger
            Qflood = flood_req
            Qin_eff = trigger
        else:
            Qflood = 0.0
            Qin_eff = Qin

        # ---- Ecological flow requirement (split) ----
        Qeco_required = self._get_eco_value(res, month)
        H_high = self._get_normal_high(res, date)
        H_low = self._get_normal_low(res, date)

        if res['use_pid']:
            target = H_high - res['pid_target_offset']
            err = target - H
            P = res['pid_Kp'] * err
            res['integral_error'] += err
            I = res['pid_Ki'] * res['integral_error']
            D = res['pid_Kd'] * (err - res['prev_error'])
            res['prev_error'] = err
            u = P + I + D
            base = res['normal_zone_power_frac']
            pf_un = base - u
            pf = max(res['min_power_frac'], min(pf_un, res['max_power_frac']))
            if pf != pf_un:
                res['integral_error'] -= err
        else:
            if H >= H_high - res['high_buffer']:
                r = (H - (H_high - res['high_buffer'])) / max(
                    res['high_buffer'], 1e-6)
                r = float(np.clip(r, 0.0, 1.0))
                pf = res['normal_zone_power_frac'] + \
                    (res['max_power_frac'] - res['normal_zone_power_frac']) * r
            elif H >= H_low:
                pf = res['normal_zone_power_frac']
            else:
                pf = res['normal_zone_power_frac'] * 0.5

        sf = self._storage_control(res, V)
        if H >= res['dead_level']:
            head = max(H - res['tailwater'], 0.1)
            Qmax = res['Pmax'] * 1e6 / (
                res['rho'] * res['g'] * head * res['eta'])
        else:
            head = 0.1
            Qmax = 0.0

        avail = Qin_eff + max(V - res['dead_volume'], 0.0) / 86400.0

        if allow_gen and Qmax > 0:
            Qt_tar = Qeco_required + Qmax * pf * sf
            Qt = float(np.clip(min(Qt_tar, Qmax, avail), 0.0, None))
            Qt_add = max(Qt - Qeco_required, 0.0)
        else:
            Qt = 0.0
            Qt_add = 0.0

        Qn = max(Qeco_required - Qt, 0.0)
        total_req = Qt + Qn
        if total_req > avail and total_req > 0:
            sc = avail / total_req
            Qt *= sc
            Qn *= sc
            Qt_add = max(Qt - Qeco_required, 0.0)

        # ---- Actual eco release (turbine flow counts) ----
        Qeco_actual = min(Qeco_required, Qt + Qn)

        if allow_gen and Qt > 0 and H >= res['dead_level']:
            h_eff = max(H - res['tailwater'], 0.1)
            pmw = res['rho'] * res['g'] * Qt * h_eff * res['eta'] / 1e6
            pmw = min(pmw, res['Pmax'])
        else:
            pmw = 0.0

        V += (Qin_eff - Qt - Qn) * 86400.0
        V = max(0.0, V)
        H = np.interp(V, cm.curve["V"], cm.curve["levels"])

        Qdd = 0.0
        if res['active_drawdown_enabled'] and H > H_high + 0.5:
            Vh = np.interp(H_high, cm.curve["levels"], cm.curve["V"])
            exc = V - Vh
            if exc > 0:
                Qdd = min(exc / 86400.0, res['active_drawdown_rate'])
                V -= Qdd * 86400.0
                H = np.interp(V, cm.curve["V"], cm.curve["levels"])

        Qsp = 0.0
        if V > res['Vmax']:
            Qsp = (V - res['Vmax']) / 86400.0
            V = res['Vmax']

        total_out = Qt + Qn + Qflood + Qdd + Qsp
        residual = V - V_before - (Qin - total_out) * 86400.0

        flood_met = True
        if forced:
            flood_met = bool(Qflood >= flood_req - 1e-8)

        return {
            'V': V, 'H': H, 'power_mw': pmw,
            'Qturb': Qt, 'Q_nonpower': Qn, 'Qflood': Qflood,
            'Qdrawdown': Qdd, 'Qspill': Qsp,
            'Qagr': 0.0, 'Qdom': 0.0, 'Qind': 0.0,
            'Qeco_required': Qeco_required,
            'Qeco_actual': Qeco_actual,
            'Q_mandatory': Qeco_required, 'Qturb_additional': Qt_add,
            'total_outflow': total_out,
            'storage_factor': sf, 'power_frac': pf,
            'head': max(H - res['tailwater'], 0.1) if H >= res['dead_level'] else 0.0,
            'residual': residual, 'V_before': V_before, 'V_after': V,
            'is_operational': True,
            'operation_stage': 'GENERATION' if allow_gen else 'IMPOUNDMENT_ONLY',
            'is_impounding': True, 'is_generating': bool(allow_gen),
            'eco_min_required': Qeco_required,
            'eco_actual': Qeco_actual,
            'eco_constraint_met': True, 'eco_supplement': 0.0,
            'flood_trigger_inflow': trigger,
            'forced_flood_triggered': forced,
            'forced_flood_required': flood_req,
            'forced_flood_supplement': Qflood,
            'flood_rule_met': flood_met,
            'original_total_outflow': total_out,
        }

    def _apply_hard(self, res, result, Qin, date):
        dt = 86400.0
        eco_req = self._get_eco_value(res, date.month)
        eco_act = max(float(result.get('Qeco_actual', 0.0)), 0.0)
        deficit = max(eco_req - eco_act, 0.0)
        avail_v = max(result['V'] - res['dead_volume'], 0.0)
        supp = min(deficit, avail_v / dt)
        if supp > 0:
            result['Qeco_actual'] += supp
            result['Q_nonpower'] += supp
            result['total_outflow'] += supp
            result['V'] -= supp * dt
        result['eco_constraint_met'] = bool(result['Qeco_actual'] >= eco_req - 1e-8)
        result['V'] = max(result['V'], 0.0)
        result['V_after'] = result['V']
        result['H'] = np.interp(result['V'], res['curve_model'].curve["V"],
                                res['curve_model'].curve["levels"])
        result['head'] = (max(result['H'] - res['tailwater'], 0.1)
                          if result['H'] >= res['dead_level'] else 0.0)
        result['eco_min_required'] = eco_req
        result['eco_actual'] = result['Qeco_actual']
        result['eco_supplement'] = supp
        result['residual'] = (result['V_after'] - result['V_before']
                              - (Qin - result['total_outflow']) * dt)
        return result

    def _simulate_day(self, res, Qin, date, V):
        date = pd.Timestamp(date)
        if date < res['impoundment_start_date']:
            return self._simulate_inactive(res, Qin, date, V)
        allow = date >= res['generation_start_date']
        r = self._simulate_single(res, Qin, date, V, allow_gen=allow)
        return self._apply_hard(res, r, Qin, date)

    # =====================================================
    # FULL SIMULATION
    # =====================================================
    def simulate_cascade_full(self, df, output_dir="results",
                              save_outputs=True, label="sim"):
        if df.empty:
            raise ValueError("No data.")
        ensure(output_dir)
        prefix = os.path.join(output_dir, "cascade_output_{}".format(label))

        volumes = [r['initial_volume'] for r in self.reservoirs]
        self.reset_controller_state()
        self._reset_water_balance_stats()

        daily_data = []
        res_daily = {r['name']: [] for r in self.reservoirs}
        water_balance_daily = []

        print("\nRunning cascade simulation [{}]...".format(label))
        total_days = len(df)

        for idx, (_, row) in enumerate(df.iterrows()):
            Qin = max(float(row["Data"]), 0.0)
            date = pd.Timestamp(row["Date"])
            cur_in = Qin
            results = []
            local_ins = []

            for i, res in enumerate(self.reservoirs):
                local_ins.append(cur_in)
                r = self._simulate_day(res, cur_in, date, volumes[i])
                results.append(r)
                volumes[i] = r['V']

                if r['is_operational']:
                    if r['H'] < res['dead_level'] - 1e-8:
                        self.water_deficit += 1
                    if not r['eco_constraint_met']:
                        self.ecological_violation += 1
                    if not r['flood_rule_met']:
                        self.flood_rule_violation += 1

                abs_r = abs(r['residual'])
                st = self.water_balance_stats[res['name']]
                st['total_residual'] += abs_r
                st['max_residual'] = max(st['max_residual'], abs_r)
                st['count'] += 1
                st['residuals'].append(abs_r)

                water_balance_daily.append({
                    'Date': date, 'Reservoir': res['name'],
                    'Is_Operational': r['is_operational'],
                    'V_before_m3': r['V_before'], 'V_after_m3': r['V_after'],
                    'Delta_V_m3': r['V_after'] - r['V_before'],
                    'Qin_m3': local_ins[i] * 86400.0,
                    'Qout_m3': r['total_outflow'] * 86400.0,
                    'Net_Flow_m3': (local_ins[i] - r['total_outflow']) * 86400.0,
                    'Residual_m3': r['residual'],
                    'Abs_Residual_m3': abs_r,
                    'Relative_Residual_pct':
                        abs_r / res['Vmax'] * 100.0 if res['Vmax'] > 0 else 0.0,
                })

                if i < self.n_reservoirs - 1:
                    cur_in = r['total_outflow'] * self.routing_coeff
                    if self.interbasin_flow_ratio > 0:
                        cur_in += Qin * self.interbasin_flow_ratio

            total_v = sum(r['V'] for r in results)
            total_p = sum(r['power_mw'] for r in results)
            sys_sr = total_v / self.total_Vmax if self.total_Vmax > 0 else 0.0
            sys_out = results[-1]['total_outflow']
            wu = sys_out / Qin if Qin > 0 else 0.0
            total_res = sum(r['residual'] for r in results)

            rec = {
                'Date': date, 'Year': date.year,
                'Month': date.month, 'Day': date.day,
                'System_Inflow_m3s': Qin,
                'System_Outflow_m3s': sys_out,
                'System_Water_Utilization': wu,
                'System_Total_Volume_m3': total_v,
                'System_Storage_Ratio': sys_sr,
                'System_Total_Power_MW': total_p,
                'System_Water_Balance_Residual_m3': total_res,
            }

            for i, (res, r) in enumerate(zip(self.reservoirs, results)):
                pf = 'R%d_%s' % (i + 1, res['name'])
                in_val = local_ins[i]

                rec.update({
                    '%s_Is_Operational' % pf: r['is_operational'],
                    '%s_Operation_Stage' % pf: r.get('operation_stage', ''),
                    '%s_Is_Impounding' % pf: r.get('is_impounding', False),
                    '%s_Is_Generating' % pf: r.get('is_generating', False),
                    '%s_Inflow_m3s' % pf: in_val,
                    '%s_Level_m' % pf: r['H'],
                    '%s_Volume_m3' % pf: r['V'],
                    '%s_Storage_Ratio' % pf: (r['V'] / res['Vmax']
                                              if res['Vmax'] > 0 else 0.0),
                    '%s_Power_MW' % pf: r['power_mw'],
                    '%s_Head_m' % pf: r['head'],
                    '%s_Turbine_Flow_m3s' % pf: r['Qturb'],
                    '%s_Mandatory_Release_m3s' % pf: r['Q_mandatory'],
                    '%s_Additional_Power_Flow_m3s' % pf: r['Qturb_additional'],
                    '%s_Flood_Release_m3s' % pf: r['Qflood'],
                    '%s_Drawdown_Release_m3s' % pf: r['Qdrawdown'],
                    '%s_Spill_m3s' % pf: r['Qspill'],
                    '%s_Total_Outflow_m3s' % pf: r['total_outflow'],
                    '%s_Agri_Release_m3s' % pf: r['Qagr'],
                    '%s_Dom_Release_m3s' % pf: r['Qdom'],
                    '%s_Eco_Required_m3s' % pf: r['Qeco_required'],
                    '%s_Eco_Actual_m3s' % pf: r['Qeco_actual'],
                    '%s_Eco_Release_m3s' % pf: r['Qeco_actual'],
                    '%s_Ind_Release_m3s' % pf: r['Qind'],
                    '%s_Eco_Min_Required_m3s' % pf: r['eco_min_required'],
                    '%s_Eco_Constraint_Met' % pf: r['eco_constraint_met'],
                    '%s_Eco_Supplement_m3s' % pf: r['eco_supplement'],
                    '%s_Flood_Trigger_Inflow_m3s' % pf: r['flood_trigger_inflow'],
                    '%s_Forced_Flood_Triggered' % pf: r['forced_flood_triggered'],
                    '%s_Flood_Rule_Met' % pf: r['flood_rule_met'],
                    '%s_Water_Balance_Residual_m3' % pf: r['residual'],
                })

                res_daily[res['name']].append({
                    'Date': date, 'Year': date.year,
                    'Month': date.month, 'Day': date.day,
                    'Is_Operational': r['is_operational'],
                    'Operation_Stage': r.get('operation_stage', ''),
                    'Is_Impounding': r.get('is_impounding', False),
                    'Is_Generating': r.get('is_generating', False),
                    'Inflow_m3s': in_val,
                    'Level_m': r['H'], 'Volume_m3': r['V'],
                    'Storage_Ratio': r['V'] / res['Vmax'] if res['Vmax'] > 0 else 0.0,
                    'Power_MW': r['power_mw'], 'Head_m': r['head'],
                    'Turbine_Flow_m3s': r['Qturb'],
                    'Mandatory_Release_m3s': r['Q_mandatory'],
                    'Additional_Power_Flow_m3s': r['Qturb_additional'],
                    'Flood_Release_m3s': r['Qflood'],
                    'Drawdown_Release_m3s': r['Qdrawdown'],
                    'Spill_m3s': r['Qspill'],
                    'Total_Outflow_m3s': r['total_outflow'],
                    'Agri_Release_m3s': r['Qagr'],
                    'Dom_Release_m3s': r['Qdom'],
                    'Eco_Required_m3s': r['Qeco_required'],
                    'Eco_Actual_m3s': r['Qeco_actual'],
                    'Eco_Release_m3s': r['Qeco_actual'],
                    'Ind_Release_m3s': r['Qind'],
                    'Eco_Min_Required_m3s': r['eco_min_required'],
                    'Eco_Constraint_Met': r['eco_constraint_met'],
                    'Eco_Supplement_m3s': r['eco_supplement'],
                    'Flood_Trigger_Inflow_m3s': r['flood_trigger_inflow'],
                    'Forced_Flood_Triggered': r['forced_flood_triggered'],
                    'Flood_Rule_Met': r['flood_rule_met'],
                    'Water_Balance_Residual_m3': r['residual'],
                    'Storage_Factor': r['storage_factor'],
                    'Power_Fraction': r['power_frac'],
                })

            daily_data.append(rec)

            if (idx + 1) % 2000 == 0:
                print("  %d/%d days" % (idx + 1, total_days))

        daily_df = pd.DataFrame(daily_data).set_index('Date')
        res_dfs = {}
        for name, lst in res_daily.items():
            rdf = pd.DataFrame(lst).set_index('Date')
            res_dfs[name] = rdf

        self.final_volumes = list(volumes)

        if save_outputs:
            ensure(output_dir)
            daily_df.to_csv(prefix + "_system_daily_results.csv",
                            encoding="utf-8-sig")
            for name, rdf in res_dfs.items():
                rdf.to_csv(prefix + "_%s_daily_results.csv" % name,
                           encoding="utf-8-sig")
            pd.DataFrame(water_balance_daily).to_csv(
                prefix + "_water_balance_daily.csv",
                index=False, encoding="utf-8-sig")
            self._generate_individual_monthly_stats(res_dfs, prefix)
            self._generate_individual_annual_stats(res_dfs, prefix)
            self._generate_system_monthly_stats(daily_df, prefix)
            self._generate_system_annual_stats(daily_df, prefix)
            self._generate_performance_metrics(daily_df, prefix)
            self._print_water_balance_diagnostics(prefix)
            self._save_constraint_diagnostics(res_dfs, prefix)

        return daily_df, res_dfs

    def simulate_cascade_monthly_gen(self, df):
        if df.empty:
            return pd.Series(dtype=float), {}
        volumes = [r['initial_volume'] for r in self.reservoirs]
        self.reset_controller_state()
        self._reset_water_balance_stats()

        monthly = {}
        eco_v = {r['name']: 0 for r in self.reservoirs}
        flood_v = {r['name']: 0 for r in self.reservoirs}
        flood_t = {r['name']: 0 for r in self.reservoirs}
        deficit = {r['name']: 0 for r in self.reservoirs}

        for _, row in df.iterrows():
            Qin = max(float(row["Data"]), 0.0)
            date = pd.Timestamp(row["Date"])
            month = date.month
            cur = Qin

            for i, res in enumerate(self.reservoirs):
                r = self._simulate_day(res, cur, date, volumes[i])
                volumes[i] = r['V']
                if r['is_operational']:
                    if r['H'] < res['dead_level'] - 1e-8:
                        deficit[res['name']] += 1
                    if not r['eco_constraint_met']:
                        eco_v[res['name']] += 1
                    if r['forced_flood_triggered']:
                        flood_t[res['name']] += 1
                    if not r['flood_rule_met']:
                        flood_v[res['name']] += 1
                if i < self.n_reservoirs - 1:
                    cur = r['total_outflow'] * self.routing_coeff
                    if self.interbasin_flow_ratio > 0:
                        cur += Qin * self.interbasin_flow_ratio
                if i == self.n_reservoirs - 1:
                    monthly[(date.year, month)] = \
                        monthly.get((date.year, month), 0.0) + r['power_mw'] * 24.0

        if not monthly:
            return pd.Series(dtype=float), {}
        dates = [pd.Timestamp(y, m, 1) for (y, m) in monthly.keys()]
        vals = [v / 10.0 for v in monthly.values()]
        s = pd.Series(vals, index=dates, name='gen').sort_index()

        diag = {
            'water_deficit_by_res': deficit,
            'eco_violation_by_res': eco_v,
            'flood_violation_by_res': flood_v,
            'flood_trigger_days': flood_t,
            'water_deficit_total': int(sum(deficit.values())),
            'eco_violation_total': int(sum(eco_v.values())),
            'flood_violation_total': int(sum(flood_v.values())),
        }
        return s, diag

    # ----- Output helpers (same as main model) -----
    def _save_constraint_diagnostics(self, res_dfs, prefix):
        rows = []
        print("\n" + "=" * 70)
        print("ENGINEERING CONSTRAINT DIAGNOSTICS")
        print("=" * 70)
        for res in self.reservoirs:
            name = res['name']
            df = res_dfs[name].copy()
            active = df[df['Is_Operational'] == True].copy()
            if active.empty:
                continue
            eco_fail = int((~active['Eco_Constraint_Met'].astype(bool)).sum())
            flood_fail = int((~active['Flood_Rule_Met'].astype(bool)).sum())
            trigger_days = int(active['Forced_Flood_Triggered'].astype(bool).sum())
            eco_cmp = 100.0 * (1.0 - eco_fail / len(active))
            flood_cmp = 100.0 * (1.0 - flood_fail / len(active))
            print("\n{}:".format(name))
            print("  Active days: {}".format(len(active)))
            print("  Eco violation days: {}".format(eco_fail))
            print("  Eco compliance: {:.4f}%".format(eco_cmp))
            print("  Trigger days: {}".format(trigger_days))
            print("  Flood-rule violation days: {}".format(flood_fail))
            print("  Flood-rule compliance: {:.4f}%".format(flood_cmp))
            rows.append({
                'Reservoir': name,
                'Active_Days': len(active),
                'Eco_Violation_Days': eco_fail,
                'Eco_Compliance_pct': eco_cmp,
                'Flood_Trigger_Days': trigger_days,
                'Flood_Rule_Violation_Days': flood_fail,
                'Flood_Rule_Compliance_pct': flood_cmp,
            })
        pd.DataFrame(rows).to_csv(
            prefix + "_constraint_diagnostics.csv",
            index=False, encoding="utf-8-sig")

    def _print_water_balance_diagnostics(self, prefix):
        print("\n" + "=" * 70)
        print("WATER BALANCE DIAGNOSTICS")
        print("=" * 70)
        rows = []
        for res_name, st in self.water_balance_stats.items():
            r = np.array(st['residuals']) if st['residuals'] else np.array([0.0])
            mean_r = float(np.mean(r))
            std_r = float(np.std(r))
            max_r = float(np.max(r))
            total_r = float(np.sum(r))
            p50 = float(np.percentile(r, 50))
            p95 = float(np.percentile(r, 95))
            p99 = float(np.percentile(r, 99))
            vmax = next(x['Vmax'] for x in self.reservoirs
                        if x['name'] == res_name)
            print("\n{}:".format(res_name))
            print("  Days: {}".format(st['count']))
            print("  Mean |res|: {:.6f} m3".format(mean_r))
            print("  Max  |res|: {:.6f} m3".format(max_r))
            print("  Rel. mean: {:.8f}% of Vmax".format(mean_r / vmax * 100))
            rows.append({
                'Reservoir': res_name,
                'Days_Simulated': st['count'],
                'Mean_Abs_Residual_m3': mean_r,
                'Std_Abs_Residual_m3': std_r,
                'Max_Abs_Residual_m3': max_r,
                'Total_Abs_Residual_m3': total_r,
                'P50_Residual_m3': p50,
                'P95_Residual_m3': p95,
                'P99_Residual_m3': p99,
                'Vmax_m3': vmax,
                'Relative_Mean_Residual_pct': mean_r / vmax * 100 if vmax > 0 else np.nan,
                'Relative_Max_Residual_pct': max_r / vmax * 100 if vmax > 0 else np.nan,
            })
        pd.DataFrame(rows).to_csv(
            prefix + "_water_balance_diagnostics.csv",
            index=False, encoding="utf-8-sig")

    def _generate_individual_monthly_stats(self, res_dfs, prefix):
        for name, df in res_dfs.items():
            rows = []
            for (y, m), g in df.groupby([df.index.year, df.index.month]):
                eco_short = int((~g['Eco_Constraint_Met'].astype(bool)).sum())
                rows.append({
                    'Year': y, 'Month': m,
                    'Total_Inflow_m3': g['Inflow_m3s'].sum() * 86400.0,
                    'Total_Outflow_m3': g['Total_Outflow_m3s'].sum() * 86400.0,
                    'Total_Power_MWh': g['Power_MW'].sum() * 24.0,
                    'Avg_Level_m': g['Level_m'].mean(),
                    'Max_Level_m': g['Level_m'].max(),
                    'Min_Level_m': g['Level_m'].min(),
                    'Avg_Volume_m3': g['Volume_m3'].mean(),
                    'Avg_Storage_Ratio': g['Storage_Ratio'].mean(),
                    'Avg_Head_m': g['Head_m'].mean(),
                    'Total_Turbine_Flow_m3': g['Turbine_Flow_m3s'].sum() * 86400.0,
                    'Total_Eco_Required_m3': g['Eco_Required_m3s'].sum() * 86400.0,
                    'Total_Eco_Actual_m3': g['Eco_Actual_m3s'].sum() * 86400.0,
                    'Total_Eco_Release_m3': g['Eco_Release_m3s'].sum() * 86400.0,
                    'Total_Flood_Release_m3': g['Flood_Release_m3s'].sum() * 86400.0,
                    'Total_Spill_m3': g['Spill_m3s'].sum() * 86400.0,
                    'Eco_Compliance_pct':
                        g['Eco_Constraint_Met'].astype(float).mean() * 100.0,
                    'Eco_Short_Days': eco_short,
                    'Flood_Rule_Compliance_pct':
                        g['Flood_Rule_Met'].astype(float).mean() * 100.0,
                    'Mean_WB_Residual_m3':
                        g['Water_Balance_Residual_m3'].abs().mean(),
                    'Max_WB_Residual_m3':
                        g['Water_Balance_Residual_m3'].abs().max(),
                })
            pd.DataFrame(rows).to_csv(
                prefix + "_%s_monthly_stats.csv" % name,
                index=False, encoding="utf-8-sig")

    def _generate_individual_annual_stats(self, res_dfs, prefix):
        pmax = {r['name']: r['Pmax'] for r in self.reservoirs}
        for name, df in res_dfs.items():
            rows = []
            for y, g in df.groupby(df.index.year):
                p = max(pmax.get(name, 1.0), 1e-12)
                eco_short = int((~g['Eco_Constraint_Met'].astype(bool)).sum())
                rows.append({
                    'Year': y,
                    'Pre_Impoundment_Days':
                        int((g['Operation_Stage'] == 'PRE_IMPOUNDMENT').sum()),
                    'Impoundment_Only_Days':
                        int((g['Operation_Stage'] == 'IMPOUNDMENT_ONLY').sum()),
                    'Generation_Days':
                        int((g['Operation_Stage'] == 'GENERATION').sum()),
                    'Total_Inflow_m3': g['Inflow_m3s'].sum() * 86400.0,
                    'Total_Outflow_m3': g['Total_Outflow_m3s'].sum() * 86400.0,
                    'Total_Power_MWh': g['Power_MW'].sum() * 24.0,
                    'Avg_Level_m': g['Level_m'].mean(),
                    'Avg_Storage_Ratio': g['Storage_Ratio'].mean(),
                    'Total_Turbine_Flow_m3': g['Turbine_Flow_m3s'].sum() * 86400.0,
                    'Total_Eco_Required_m3': g['Eco_Required_m3s'].sum() * 86400.0,
                    'Total_Eco_Actual_m3': g['Eco_Actual_m3s'].sum() * 86400.0,
                    'Total_Eco_Release_m3': g['Eco_Release_m3s'].sum() * 86400.0,
                    'Eco_Compliance_pct':
                        g['Eco_Constraint_Met'].astype(float).mean() * 100.0,
                    'Eco_Short_Days': eco_short,
                    'Flood_Rule_Compliance_pct':
                        g['Flood_Rule_Met'].astype(float).mean() * 100.0,
                    'Utilization_Hours': g['Power_MW'].sum() * 24.0 / p,
                })
            pd.DataFrame(rows).to_csv(
                prefix + "_%s_annual_stats.csv" % name,
                index=False, encoding="utf-8-sig")

    def _generate_system_monthly_stats(self, daily_df, prefix):
        rows = []
        for (y, m), g in daily_df.groupby([daily_df.index.year,
                                           daily_df.index.month]):
            rows.append({
                'Year': y, 'Month': m,
                'System_Total_Inflow_m3': g['System_Inflow_m3s'].sum() * 86400.0,
                'System_Total_Outflow_m3': g['System_Outflow_m3s'].sum() * 86400.0,
                'System_Total_Power_MWh': g['System_Total_Power_MW'].sum() * 24.0,
                'System_Avg_Storage_m3': g['System_Total_Volume_m3'].mean(),
                'System_Avg_Storage_Ratio': g['System_Storage_Ratio'].mean(),
                'System_Avg_Water_Utilization':
                    g['System_Water_Utilization'].mean(),
                'System_Mean_WB_Residual_m3':
                    g['System_Water_Balance_Residual_m3'].abs().mean(),
            })
        pd.DataFrame(rows).to_csv(
            prefix + "_system_monthly_stats.csv",
            index=False, encoding="utf-8-sig")

    def _generate_system_annual_stats(self, daily_df, prefix):
        rows = []
        for y, g in daily_df.groupby(daily_df.index.year):
            rows.append({
                'Year': y,
                'System_Total_Inflow_m3': g['System_Inflow_m3s'].sum() * 86400.0,
                'System_Total_Power_MWh': g['System_Total_Power_MW'].sum() * 24.0,
                'System_Avg_Storage_m3': g['System_Total_Volume_m3'].mean(),
                'System_Utilization_Hours':
                    g['System_Total_Power_MW'].sum() * 24.0
                    / max(self.total_Pmax, 1e-12),
            })
        pd.DataFrame(rows).to_csv(
            prefix + "_system_annual_stats.csv",
            index=False, encoding="utf-8-sig")

    def _generate_performance_metrics(self, daily_df, prefix):
        m = {'Metric': [], 'Value': [], 'Unit': [], 'Description': []}

        def add(name, val, unit, desc):
            m['Metric'].append(name)
            m['Value'].append(val)
            m['Unit'].append(unit)
            m['Description'].append(desc)

        add('Total_Installed_Capacity', self.total_Pmax, 'MW',
            'Total installed capacity')
        add('Total_Storage_Capacity', self.total_Vmax, 'm3',
            'Total storage capacity')
        total_mwh = daily_df['System_Total_Power_MW'].sum() * 24.0
        add('Total_Power_Generation', total_mwh, 'MWh',
            'Total power generation')
        avg_util = (daily_df['System_Total_Power_MW'].sum() * 24.0
                    / max(self.total_Pmax, 1e-12) / max(len(daily_df), 1) * 365.0)
        add('Average_Annual_Utilization_Hours', avg_util, 'hours/year',
            'Average annual utilization hours')

        for i, res in enumerate(self.reservoirs):
            p = 'R{}_{}'.format(i + 1, res['name'])
            rp = daily_df['{}_Power_MW'.format(p)].sum() * 24.0
            ru = rp / max(res['Pmax'], 1e-12) / max(len(daily_df), 1) * 365.0
            add('{}_Power_Generation'.format(p), rp, 'MWh',
                'Power generation of {}'.format(res['name']))
            add('{}_Utilization_Hours'.format(p), ru, 'hours/year',
                'Annual utilization hours of {}'.format(res['name']))
        pd.DataFrame(m).to_csv(
            prefix + "_performance_metrics.csv",
            index=False, encoding="utf-8-sig")


# =========================================================
# 3. DATA LOADING
# =========================================================
def load_inflow_data(path):
    df = pd.read_excel(path)
    req = {'Year', 'Month', 'Day', 'Data'}
    if not req.issubset(df.columns):
        raise ValueError("{} missing: {}".format(path, sorted(req - set(df.columns))))
    df["Date"] = pd.to_datetime(df[["Year", "Month", "Day"]])
    df['Data'] = pd.to_numeric(df['Data'], errors='coerce')
    return df.dropna(subset=['Date', 'Data']).sort_values('Date').reset_index(drop=True)


def load_future_inflow_data(path):
    print("\nLoading future runoff: {}".format(path))
    if not os.path.exists(path):
        raise FileNotFoundError("Future file not found: {}".format(path))
    df = pd.read_csv(path)
    if "date" not in df.columns or "Q_total_m3s" not in df.columns:
        raise ValueError("CSV must have 'date' and 'Q_total_m3s' columns")
    df["Date"] = pd.to_datetime(df["date"])
    df["Data"] = df["Q_total_m3s"].astype(float)
    out = df[["Date", "Data"]].sort_values("Date").reset_index(drop=True)
    out = out[~out["Date"].duplicated(keep='first')]
    print("  Period: {} ~ {}".format(out['Date'].min(), out['Date'].max()))
    print("  Days: {}".format(len(out)))
    return out


def load_actual_power(path):
    df = pd.read_excel(path)
    if 'time' not in df.columns or 'power' not in df.columns:
        raise ValueError("{} must contain 'time' and 'power'.".format(path))
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
    m = ~(np.isnan(sim) | np.isnan(obs))
    sim, obs = sim[m], obs[m]
    if len(sim) < 2:
        return None

    rmse = float(np.sqrt(np.mean((sim - obs) ** 2)))
    dn = np.sum((obs - obs.mean()) ** 2)
    nse = float(1.0 - np.sum((sim - obs) ** 2) / dn) if dn > 1e-12 else np.nan
    r = (float(np.corrcoef(sim, obs)[0, 1])
         if np.std(sim) > 1e-12 and np.std(obs) > 1e-12 else np.nan)
    a = float(np.std(sim) / np.std(obs)) if np.std(obs) > 1e-12 else np.nan
    b = float(np.mean(sim) / np.mean(obs)) if abs(np.mean(obs)) > 1e-12 else np.nan
    kge = (1.0 - np.sqrt((r - 1.0) ** 2 + (a - 1.0) ** 2 + (b - 1.0) ** 2)
           if np.isfinite(r) and np.isfinite(a) and np.isfinite(b) else np.nan)
    r2 = r ** 2 if np.isfinite(r) else np.nan
    pb = 100.0 * np.sum(obs - sim) / np.sum(obs) if abs(np.sum(obs)) > 1e-12 else np.nan
    mae = float(np.mean(np.abs(sim - obs)))

    return {'RMSE': rmse,
            'NSE': nse if np.isfinite(nse) else np.nan,
            'KGE': kge if np.isfinite(kge) else np.nan,
            'R2': r2 if np.isfinite(r2) else np.nan,
            'PBIAS': pb if np.isfinite(pb) else np.nan,
            'MAE': mae}


# =========================================================
# 5. MOPSO EVALUATION (2 objectives)
# =========================================================
def evaluate_mopso_detailed(op_template, calib_inflow_df, actual_all, params):
    op = deepcopy(op_template)
    op.update_calibration_params(params, reservoir_idx=2)
    op.reset_controller_state()
    op._reset_water_balance_stats()

    sim_m, diag = op.simulate_cascade_monthly_gen(calib_inflow_df)

    c0 = pd.Timestamp('2010-01-01')
    c1 = pd.Timestamp('2014-12-31')
    act = actual_all.loc[(actual_all.index >= c0) & (actual_all.index <= c1)]
    common = act.index.intersection(sim_m.index)
    if len(common) < 12:
        return None

    sim = sim_m.loc[common].values.astype(float)
    obs = act.loc[common].values.astype(float)
    valid = np.isfinite(sim) & np.isfinite(obs)
    sim, obs = sim[valid], obs[valid]
    if len(sim) < 12:
        return None

    rmse = float(np.sqrt(np.mean((sim - obs) ** 2)))
    rmse_n = rmse / max(np.std(obs), 1e-6)
    pb = 100.0 * np.sum(obs - sim) / np.sum(obs) if abs(np.sum(obs)) > 1e-12 else 0.0
    pb_n = abs(pb) / 100.0

    n = max(len(calib_inflow_df), 1)
    eco_n = diag['eco_violation_total'] / n
    flood_n = diag['flood_violation_total'] / n
    deficit_n = diag['water_deficit_total'] / n
    cv = 1.0 * eco_n + 0.5 * flood_n + 0.3 * deficit_n

    metrics = compute_all_metrics(sim, obs)

    comp = {
        'RMSE': rmse, 'RMSE_norm': rmse_n,
        'PBIAS': pb, 'PBIAS_norm': pb_n,
        'NSE': (metrics['NSE'] if metrics else np.nan),
        'KGE': (metrics['KGE'] if metrics else np.nan),
        'R2': (metrics['R2'] if metrics else np.nan),
        'Water_Deficit_days': diag['water_deficit_total'],
        'Eco_Violation_days': diag['eco_violation_total'],
        'Flood_Rule_Violation_days': diag['flood_violation_total'],
        'Eco_Violation_Chahanwusu': diag['eco_violation_by_res']['Chahanwusu'],
        'Eco_Violation_Liushugou': diag['eco_violation_by_res']['Liushugou'],
        'Eco_Violation_Dashankou': diag['eco_violation_by_res']['Dashankou'],
        'Flood_Trigger_Days_Chahanwusu': diag['flood_trigger_days']['Chahanwusu'],
        'Flood_Trigger_Days_Liushugou': diag['flood_trigger_days']['Liushugou'],
        'Flood_Trigger_Days_Dashankou': diag['flood_trigger_days']['Dashankou'],
        'Constraint_Violation': cv,
    }

    # TWO objectives; CV is a HARD constraint
    return {
        'objectives': np.array([rmse_n, pb_n], dtype=float),
        'constraint_violation': float(cv),
        'components': comp,
    }


# =========================================================
# 6. MOPSO + TOPSIS (2 objectives)
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
        self.output_dir = ensure(output_dir)

        self.rng = np.random.default_rng(seed)
        self.low = np.array([bounds[k][0] for k in self.param_keys], dtype=float)
        self.high = np.array([bounds[k][1] for k in self.param_keys], dtype=float)

        self.pop = self.rng.uniform(self.low, self.high,
                                    size=(pop_size, self.dim))
        span = self.high - self.low
        self.vel = self.rng.uniform(-0.1 * span, 0.1 * span,
                                    size=(pop_size, self.dim))

        self.pbest_pos = self.pop.copy()
        self.pbest_obj = [None for _ in range(pop_size)]
        self.pbest_cv = np.full(pop_size, np.inf)

        self.archive = []
        self.history = []
        self.detailed_history = []

    def _decode(self, v):
        return {k: float(x) for k, x in zip(self.param_keys, v)}

    @classmethod
    def _update_feasible_tol(cls, archive, tol_cap=0.05, tol_floor=1e-6):
        if not archive:
            return
        cvs = [x['constraint_violation'] for x in archive]
        min_cv = float(np.min(cvs))
        new_tol = min(min_cv * 1.1, tol_cap)
        cls.FEASIBLE_TOL = max(new_tol, tol_floor)

    @classmethod
    def _dominates(cls, oa, ca, ob, cb):
        fa = (ca <= cls.FEASIBLE_TOL)
        fb = (cb <= cls.FEASIBLE_TOL)
        if fa and not fb:
            return True
        if not fa and fb:
            return False
        if not fa and not fb:
            return ca < cb
        return bool(np.all(oa <= ob) and np.any(oa < ob))

    @staticmethod
    def _crowding(archive):
        n = len(archive)
        if n == 0:
            return np.array([])
        if n <= 2:
            return np.full(n, np.inf)
        F = np.vstack([x['objectives'] for x in archive])
        d = np.zeros(n, dtype=float)
        for m in range(F.shape[1]):
            order = np.argsort(F[:, m])
            d[order[0]] = np.inf
            d[order[-1]] = np.inf
            fmin = F[order[0], m]
            fmax = F[order[-1], m]
            if fmax - fmin <= 1e-12:
                continue
            for j in range(1, n - 1):
                if np.isinf(d[order[j]]):
                    continue
                d[order[j]] += (F[order[j + 1], m] - F[order[j - 1], m]) / \
                    (fmax - fmin)
        return d

    def _update_archive(self, candidates):
        pool = self.archive + candidates
        feas = [x for x in pool if x['constraint_violation'] <= self.FEASIBLE_TOL]
        if feas:
            pool = feas
        else:
            mcv = min(x['constraint_violation'] for x in pool)
            pool = [x for x in pool if x['constraint_violation'] == mcv]

        nd = []
        for i, c in enumerate(pool):
            dom = False
            for j, o in enumerate(pool):
                if i == j:
                    continue
                if self._dominates(o['objectives'], o['constraint_violation'],
                                   c['objectives'], c['constraint_violation']):
                    dom = True
                    break
            if not dom:
                nd.append(c)

        uniq = []
        for c in nd:
            dup = False
            for u in uniq:
                if (np.allclose(c['objectives'], u['objectives'],
                                rtol=1e-8, atol=1e-10)
                        and np.allclose(c['position'], u['position'],
                                        rtol=1e-8, atol=1e-10)):
                    dup = True
                    break
            if not dup:
                uniq.append(c)

        while len(uniq) > self.archive_size:
            cd = self._crowding(uniq)
            fi = np.where(np.isfinite(cd))[0]
            if len(fi) == 0:
                ri = int(self.rng.integers(0, len(uniq)))
            else:
                ri = int(fi[np.argmin(cd[fi])])
            uniq.pop(ri)

        self.archive = uniq

    def _select_leader(self):
        if not self.archive:
            return self.pop[self.rng.integers(0, self.pop_size)].copy()
        cd = self._crowding(self.archive)
        w = cd.copy()
        fin = np.isfinite(w)
        mf = np.max(w[fin]) if np.any(fin) else 1.0
        w[~fin] = mf * 2.0 + 1.0
        w = np.maximum(w, 1e-12)
        w = w / np.sum(w)
        idx = self.rng.choice(len(self.archive), p=w)
        return self.archive[idx]['position'].copy()

    def _clip(self, pos, vel):
        for d in range(self.dim):
            lo, hi = self.low[d], self.high[d]
            if pos[d] < lo:
                pos[d] = lo
                vel[d] *= -0.5
            elif pos[d] > hi:
                pos[d] = hi
                vel[d] *= -0.5
        return pos, vel

    def optimize(self, eval_func, verbose=True):
        self.history = []
        self.detailed_history = []

        stagnation_counter = 0
        last_best_key = None

        for it in range(self.max_iter):
            w = (self.w_start - (self.w_start - self.w_end) * it /
                 (self.max_iter - 1)) if self.max_iter > 1 else self.w_end

            cands = []
            for i in range(self.pop_size):
                params = self._decode(self.pop[i])
                try:
                    res = eval_func(params)
                except Exception:
                    print("\nWarning: Iter=%d Particle=%d failed."
                          % (it + 1, i + 1))
                    traceback.print_exc()
                    res = None

                if res is None:
                    obj = np.array([1e9, 1e9], dtype=float)
                    cv = 1e12
                    comp = {}
                else:
                    obj = np.asarray(res['objectives'], dtype=float)
                    cv = float(res['constraint_violation'])
                    comp = res['components']

                c = {'position': self.pop[i].copy(),
                     'objectives': obj.copy(),
                     'constraint_violation': cv,
                     'params': params.copy(),
                     'components': deepcopy(comp)}
                cands.append(c)

                rec = {'Iteration': it + 1, 'Particle': i + 1,
                       **params,
                       'Objective_RMSE_norm': obj[0],
                       'Objective_PBIAS_norm': obj[1],
                       'Constraint_Violation': cv,
                       **comp}
                self.detailed_history.append(rec)

                if self.pbest_obj[i] is None:
                    rep = True
                else:
                    cur_dom = self._dominates(obj, cv,
                                              self.pbest_obj[i], self.pbest_cv[i])
                    old_dom = self._dominates(self.pbest_obj[i], self.pbest_cv[i],
                                              obj, cv)
                    if cur_dom:
                        rep = True
                    elif old_dom:
                        rep = False
                    else:
                        rep = bool(self.rng.random() < 0.5)

                if rep:
                    self.pbest_obj[i] = obj.copy()
                    self.pbest_cv[i] = cv
                    self.pbest_pos[i] = self.pop[i].copy()

                gc.collect()
                trim_memory()

            self._update_archive(cands)
            self._update_feasible_tol(self.archive)

            nf = sum(x['constraint_violation'] <= self.FEASIBLE_TOL
                     for x in cands)

            hist = {'Iteration': it + 1,
                    'Archive_Size': len(self.archive),
                    'Feasible_Particles': nf,
                    'FEASIBLE_TOL': self.FEASIBLE_TOL}
            if self.archive:
                F = np.vstack([x['objectives'] for x in self.archive])
                cv_vals = [x['constraint_violation'] for x in self.archive]
                hist['Best_RMSE_norm'] = float(np.min(F[:, 0]))
                hist['Best_PBIAS_norm'] = float(np.min(F[:, 1]))
                hist['Best_CV'] = float(np.min(cv_vals))
            self.history.append(hist)

            # Stagnation detection
            if self.archive:
                F_cur = np.vstack([x['objectives'] for x in self.archive])
                cv_cur = [x['constraint_violation'] for x in self.archive]
                cur_key = (round(float(F_cur[:, 0].min()), 8),
                           round(float(F_cur[:, 1].min()), 8),
                           round(float(np.min(cv_cur)), 8))
                if last_best_key is not None and cur_key == last_best_key:
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0
                last_best_key = cur_key

            if stagnation_counter >= 5:
                n_mut = max(1, int(0.3 * self.pop_size))
                obj_scores = [x['objectives'][0] for x in cands]
                worst = np.argsort(obj_scores)[-n_mut:]
                for wi in worst:
                    sigma = 0.15 * (self.high - self.low)
                    self.pop[wi] = np.clip(
                        self.pop[wi] + self.rng.normal(0, sigma),
                        self.low, self.high)
                stagnation_counter = 0
                if verbose:
                    print("  [Stagnation] Mutation on %d particles" % n_mut)

            mutation_prob = 0.1
            for i in range(self.pop_size):
                leader = self._select_leader()
                r1 = self.rng.random(self.dim)
                r2 = self.rng.random(self.dim)
                self.vel[i] = (w * self.vel[i]
                               + self.c1 * r1 * (self.pbest_pos[i] - self.pop[i])
                               + self.c2 * r2 * (leader - self.pop[i]))
                self.pop[i] = self.pop[i] + self.vel[i]
                self.pop[i], self.vel[i] = self._clip(self.pop[i], self.vel[i])

                mask = self.rng.random(self.dim) < mutation_prob
                if np.any(mask):
                    self.pop[i] = np.where(
                        mask,
                        self.rng.uniform(self.low, self.high),
                        self.pop[i])

            if verbose:
                if self.archive:
                    F = np.vstack([x['objectives'] for x in self.archive])
                    cv_vals = [x['constraint_violation'] for x in self.archive]
                    print("Iter %3d/%d | Arc=%3d | Feas=%3d/%d | TOL=%.5f | "
                          "minRMSE=%.6f minPB=%.6f minCV=%.4f"
                          % (it + 1, self.max_iter, len(self.archive), nf,
                             self.pop_size, self.FEASIBLE_TOL,
                             F[:, 0].min(), F[:, 1].min(), min(cv_vals)))
                else:
                    print("Iter %3d/%d | Archive empty"
                          % (it + 1, self.max_iter))

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
            rows.append({'Solution': i, **x['params'],
                         'Objective_RMSE_norm': x['objectives'][0],
                         'Objective_PBIAS_norm': x['objectives'][1],
                         'Constraint_Violation': x['constraint_violation'],
                         **x['components']})
        pd.DataFrame(rows).to_csv(
            os.path.join(self.output_dir, "mopso_pareto_front.csv"),
            index=False, encoding="utf-8-sig")


def select_topsis_solution(archive):
    tol = MultiObjectivePSO.FEASIBLE_TOL
    feas = [x for x in archive if x['constraint_violation'] <= tol]
    if not feas:
        feas = sorted(archive, key=lambda x: x['constraint_violation'])[:10]
        print("[TOPSIS] No feasible solution, fallback to top-10.")
    F = np.vstack([x['objectives'] for x in feas])
    fmin, fmax = np.min(F, axis=0), np.max(F, axis=0)
    den = np.where(fmax - fmin > 1e-12, fmax - fmin, 1.0)
    Z = (F - fmin) / den
    ideal = np.zeros(Z.shape[1])
    anti = np.ones(Z.shape[1])
    dp = np.linalg.norm(Z - ideal, axis=1)
    dn = np.linalg.norm(Z - anti, axis=1)
    score = dn / (dp + dn + 1e-12)
    bi = int(np.argmax(score))
    best = deepcopy(feas[bi])
    best['TOPSIS_Score'] = float(score[bi])
    return best


# =========================================================
# 7. MAIN
# =========================================================
if __name__ == "__main__":

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR = "results_kaidu_future_{}".format(ts)
    ensure(RESULTS_DIR)

    print("=" * 70)
    print("KAIDU - MOPSO Calibration + Historical + Future Scenarios")
    print("2-objective MOPSO | Dead levels per SL 278-2002 | Future CMIP6")
    print("Output: {}/".format(RESULTS_DIR))
    print("=" * 70)

    # ---------------- Paths ----------------
    CURVE_PATHS = [
        "chahanwusu/inverse_ch.xlsx",
        "liushugou/inverse_liushugou.xlsx",
        "dashankou/inverse_dashankou.xlsx",
    ]
    RESERVOIR_NAMES = ["Chahanwusu", "Liushugou", "Dashankou"]
    INFLOW_PATH = "hebin.xlsx"
    ACTUAL_PATH = "dianliang.xlsx"
    FUTURE_FILES = {
        "SSP126": "daily_runoff_ssp126_2017_2070_m3s.csv",
        "SSP370": "daily_runoff_ssp370_2017_2070_m3s.csv",
        "SSP585": "daily_runoff_ssp585_2017_2070_m3s.csv",
    }

    # ---------------- Periods ----------------
    CALIB_START = pd.Timestamp("2010-01-01")
    CALIB_END = pd.Timestamp("2014-12-31")
    HIST_START = pd.Timestamp("2008-01-01")
    HIST_END = pd.Timestamp("2016-12-31")
    FUTURE_START = pd.Timestamp("2017-01-01")

    # ---------------- Bounds (aligned with main model) ----------------
    bounds = {
        "normal_zone_power_fraction": (0.20, 0.90),
        "storage_ratio": (0.05, 0.80),
    }

    # ---------------- Eco flow ----------------
    eco_up = {1: 8.0, 2: 8.0, 3: 12.0, 4: 18.0,
              5: 22.57, 6: 22.57, 7: 22.57, 8: 22.57,
              9: 18.0, 10: 12.0, 11: 8.0, 12: 8.0}
    eco_down = {1: 4.0, 2: 4.0, 3: 6.0, 4: 9.0,
                5: 11.20, 6: 11.20, 7: 11.20, 8: 11.20,
                9: 9.0, 10: 6.0, 11: 4.0, 12: 4.0}

    # ---------------- Reservoir configs (dead levels 80/76/53) ----------------
    configs = [
        {"name": "Chahanwusu",
         "impoundment_start_date": "2007-01-01",
         "generation_start_date": "2007-01-01",
         "min_eco_release_m3s": 22.57,
         "monthly_min_eco_release_m3s": eco_up,
         "flood_trigger_inflow_m3s": 220.0,
         "installed_capacity_kw": 330000,
         "tailwater": 5, "efficiency": 0.90, "initial_volume": 0,
         # SL 278-2002: (80-5)/(101-5) = 78.1%
         "dead_water_level": 80.0, "spill_level": 97.5,
         "monthly_normal_high": {1: 96.5, 2: 96.5, 3: 96.5, 4: 96.0,
                                 5: 95.5, 6: 96.0, 7: 96.0, 8: 96.0,
                                 9: 95.5, 10: 96.0, 11: 96.5, 12: 96.5},
         "monthly_normal_low": {1: 92.0, 2: 92.0, 3: 92.0, 4: 91.5,
                                5: 91.0, 6: 91.0, 7: 91.5, 8: 91.5,
                                9: 91.5, 10: 92.0, 11: 92.0, 12: 92.0},
         "monthly_flood_control_level": {1: 96.0, 2: 96.0, 3: 96.0, 4: 95.5,
                                         5: 95.0, 6: 95.0, 7: 95.0, 8: 95.0,
                                         9: 95.0, 10: 95.5, 11: 96.0, 12: 96.0},
         "active_drawdown_enabled": True, "active_drawdown_rate": 150.0,
         "use_pid": True, "pid_Kp": 0.12, "pid_Ki": 0.008,
         "pid_Kd": 0.04, "pid_target_offset": 0.5,
         "max_power_fraction": 0.95, "min_power_fraction": 0.15,
         "high_buffer": 0.5, "storage_ratio": 0.25,
         "interpolate_targets": True},
        {"name": "Liushugou",
         "impoundment_start_date": "2013-01-01",
         "generation_start_date": "2014-01-01",
         "min_eco_release_m3s": 22.57,
         "monthly_min_eco_release_m3s": eco_up,
         "flood_trigger_inflow_m3s": 220.0,
         "installed_capacity_kw": 180000,
         "tailwater": 5, "efficiency": 0.90, "initial_volume": 0,
         # SL 278-2002: (76-5)/(98-5) = 76.3%
         "dead_water_level": 76.0, "spill_level": 93.5,
         "monthly_normal_high": {1: 92.5, 2: 92.5, 3: 92.5, 4: 92.0,
                                 5: 91.5, 6: 92.0, 7: 92.0, 8: 92.0,
                                 9: 91.5, 10: 92.0, 11: 92.5, 12: 92.5},
         "monthly_normal_low": {1: 89.5, 2: 89.5, 3: 89.5, 4: 89.0,
                                5: 88.5, 6: 88.5, 7: 89.0, 8: 89.0,
                                9: 89.0, 10: 89.5, 11: 89.5, 12: 89.5},
         "monthly_flood_control_level": {1: 92.0, 2: 92.0, 3: 92.0, 4: 91.5,
                                         5: 91.0, 6: 91.0, 7: 91.0, 8: 91.0,
                                         9: 91.0, 10: 91.5, 11: 92.0, 12: 92.0},
         "active_drawdown_enabled": True, "active_drawdown_rate": 120.0,
         "use_pid": True, "pid_Kp": 0.13, "pid_Ki": 0.009,
         "pid_Kd": 0.045, "pid_target_offset": 0.4,
         "max_power_fraction": 0.95, "min_power_fraction": 0.15,
         "high_buffer": 0.5, "storage_ratio": 0.28,
         "interpolate_targets": True},
        {"name": "Dashankou",
         "impoundment_start_date": "2008-01-01",
         "generation_start_date": "2008-01-01",
         "min_eco_release_m3s": 11.20,
         "monthly_min_eco_release_m3s": eco_down,
         "flood_trigger_inflow_m3s": 160.0,
         "installed_capacity_kw": 80000,
         "tailwater": 5, "efficiency": 0.90, "initial_volume": 0,
         # SL 278-2002: (53-5)/(70-5) = 73.8%
         "dead_water_level": 53.0, "spill_level": 68.0,
         "monthly_normal_high": {1: 67.5, 2: 67.5, 3: 67.5, 4: 67.0,
                                 5: 66.5, 6: 67.0, 7: 67.0, 8: 67.0,
                                 9: 66.5, 10: 67.0, 11: 67.5, 12: 67.5},
         "monthly_normal_low": {1: 63.0, 2: 63.0, 3: 63.0, 4: 62.5,
                                5: 62.0, 6: 62.0, 7: 62.5, 8: 62.5,
                                9: 62.5, 10: 63.0, 11: 63.0, 12: 63.0},
         "monthly_flood_control_level": {1: 67.0, 2: 67.0, 3: 67.0, 4: 67.0,
                                         5: 66.5, 6: 66.0, 7: 66.0, 8: 66.0,
                                         9: 66.5, 10: 67.0, 11: 67.0, 12: 67.0},
         "active_drawdown_enabled": True, "active_drawdown_rate": 100.0,
         "use_pid": True, "pid_Kp": 0.15, "pid_Ki": 0.01,
         "pid_Kd": 0.05, "pid_target_offset": 0.6,
         "max_power_fraction": 0.90, "min_power_fraction": 0.20,
         "high_buffer": 1.0, "storage_ratio": 0.30,
         "interpolate_targets": True,
         "routing_coeff": 0.98, "travel_time": 0,
         "interbasin_flow_ratio": 0.0},
    ]

    # =====================================================
    # STEP 1: Load curves + historical data
    # =====================================================
    print("=" * 70)
    print("STEP 1: Load curves and historical data")
    print("=" * 70)
    curve_models = [ReservoirCurveModel(p, n)
                    for p, n in zip(CURVE_PATHS, RESERVOIR_NAMES)]
    inflow_df = load_inflow_data(INFLOW_PATH)
    actual_all = load_actual_power(ACTUAL_PATH)
    print("Inflow: {} ~ {}".format(inflow_df['Date'].min(),
                                    inflow_df['Date'].max()))
    print("Actual: {} ~ {}".format(actual_all.index.min(),
                                    actual_all.index.max()))

    cascade_op = CascadeReservoirOperation(curve_models, configs, verbose=True)

    # =====================================================
    # STEP 2: MOPSO calibration (2010-2014)
    # =====================================================
    print("\n" + "=" * 70)
    print("STEP 2: MOPSO calibration (2010-2014)")
    print("=" * 70)
    calib_df = inflow_df[(inflow_df["Date"] >= CALIB_START)
                         & (inflow_df["Date"] <= CALIB_END)
                         ].copy().reset_index(drop=True)
    print("Calibration days: {}".format(len(calib_df)))

    def fitness_mopso(params):
        return evaluate_mopso_detailed(cascade_op, calib_df, actual_all, params)

    mopso = MultiObjectivePSO(
        bounds=bounds, pop_size=40, max_iter=50, archive_size=80,
        w_start=0.9, w_end=0.4, c1=2.0, c2=1.5,
        output_dir=RESULTS_DIR, seed=42
    )
    pareto_archive = mopso.optimize(fitness_mopso, verbose=True)
    if not pareto_archive:
        raise RuntimeError("MOPSO failed: empty Pareto archive.")
    print("\nPareto front size: {}".format(len(pareto_archive)))

    best_solution = select_topsis_solution(pareto_archive)
    best_params = best_solution['params']

    print("\n" + "=" * 70)
    print("MOPSO + TOPSIS BEST SOLUTION")
    print("=" * 70)
    print("Best parameters:")
    for k, v in best_params.items():
        print("  {}: {:.6f}".format(k, v))
    print("Objectives:")
    print("  RMSE_norm     : {:.6f}".format(best_solution['objectives'][0]))
    print("  |PBIAS|/100   : {:.6f}".format(best_solution['objectives'][1]))
    print("  CV (constraint): {:.6f}".format(best_solution['constraint_violation']))
    print("  TOPSIS score  : {:.6f}".format(best_solution['TOPSIS_Score']))
    print("Constraint diagnostics:")
    comp = best_solution['components']
    print("  Eco violation days  : {}".format(comp.get('Eco_Violation_days')))
    print("  Flood violation days: {}".format(
        comp.get('Flood_Rule_Violation_days')))
    print("  Water deficit days  : {}".format(comp.get('Water_Deficit_days')))

    pd.DataFrame([{
        **best_params,
        'Objective_RMSE_norm': best_solution['objectives'][0],
        'Objective_PBIAS_norm': best_solution['objectives'][1],
        'Constraint_Violation': best_solution['constraint_violation'],
        'TOPSIS_Score': best_solution['TOPSIS_Score'],
        'FEASIBLE_TOL_Final': MultiObjectivePSO.FEASIBLE_TOL,
        **best_solution['components'],
    }]).to_csv(os.path.join(RESULTS_DIR, "mopso_topsis_best_solution.csv"),
               index=False, encoding="utf-8-sig")

    pd.DataFrame({
        'Parameter': list(best_params.keys()),
        'Value': list(best_params.values()),
    }).to_csv(os.path.join(RESULTS_DIR, "best_params.csv"),
              index=False, encoding="utf-8-sig")

    cascade_op.update_calibration_params(best_params, reservoir_idx=2)

    # =====================================================
    # STEP 3: Historical full simulation (2008-2016)
    # =====================================================
    print("\n" + "=" * 70)
    print("STEP 3: Historical simulation ({} ~ {})".format(
        HIST_START.date(), HIST_END.date()))
    print("=" * 70)
    hist_df = inflow_df[(inflow_df["Date"] >= HIST_START)
                        & (inflow_df["Date"] <= HIST_END)
                        ].copy().reset_index(drop=True)
    print("Historical days: {}".format(len(hist_df)))

    cascade_op.reset_simulation_state()
    history_dir = ensure(os.path.join(RESULTS_DIR, "historical"))
    hist_daily, hist_res = cascade_op.simulate_cascade_full(
        hist_df, output_dir=history_dir, save_outputs=True, label="historical")

    final_storage = cascade_op.get_final_volumes()
    cascade_op.save_initial_storage(
        final_storage,
        os.path.join(RESULTS_DIR, "future_initial_storage.csv"))

    print("\nFinal storage at {} (used as initial for future):".format(
        HIST_END.date()))
    for r, V in zip(cascade_op.reservoirs, final_storage):
        print("  {}: {:.2f} m3 ({:.2f}% of Vmax)".format(
            r['name'], V, V / r['Vmax'] * 100))

    downstream_power = hist_daily['R3_Dashankou_Power_MW']
    monthly_sim = downstream_power.resample('ME').sum() * 24 / 10.0
    monthly_sim.index = monthly_sim.index.to_period('M').to_timestamp()

    cal_sim = monthly_sim.loc['2010-01':'2014-12']
    cal_obs = actual_all.loc['2010-01':'2014-12']
    cal_c = cal_sim.index.intersection(cal_obs.index)
    cal_m = compute_all_metrics(cal_sim.loc[cal_c].values,
                                cal_obs.loc[cal_c].values)

    val_sim = monthly_sim.loc['2015-01':'2016-12']
    val_obs = actual_all.loc['2015-01':'2016-12']
    val_c = val_sim.index.intersection(val_obs.index)
    val_m = compute_all_metrics(val_sim.loc[val_c].values,
                                val_obs.loc[val_c].values)

    print("\n" + "=" * 60)
    print("VALIDATION METRICS (Dashankou)")
    print("=" * 60)
    print("Calibration (2010-2014):")
    if cal_m:
        for k, v in cal_m.items():
            print("  {}: {:.4f}".format(k, v))
    print("Validation (2015-2016):")
    if val_m:
        for k, v in val_m.items():
            print("  {}: {:.4f}".format(k, v))

    if cal_m and val_m:
        pd.DataFrame({
            'Metric': list(cal_m.keys()),
            'Calibration': list(cal_m.values()),
            'Validation': [val_m.get(k, np.nan) for k in cal_m.keys()],
        }).to_csv(os.path.join(RESULTS_DIR, "validation_metrics.csv"),
                  index=False, encoding="utf-8-sig")

    comparison = pd.DataFrame({
        "Year_Month": monthly_sim.index.strftime("%Y-%m"),
        "Simulated_10^4kWh": monthly_sim.values,
        "Actual_10^4kWh": actual_all.reindex(monthly_sim.index).values,
    })
    comparison["Error_10^4kWh"] = (comparison["Simulated_10^4kWh"]
                                   - comparison["Actual_10^4kWh"])
    comparison.to_excel(
        os.path.join(RESULTS_DIR, "cascade_monthly_comparison.xlsx"),
        index=False)

    # =====================================================
    # STEP 4: Future CMIP6 scenarios
    # =====================================================
    print("\n" + "=" * 70)
    print("STEP 4: Future CMIP6 scenario simulations")
    print("  Future start cutoff: {}".format(FUTURE_START))
    print("=" * 70)

    scenario_summary = []

    for scen, fp in FUTURE_FILES.items():
        print("\n" + "#" * 70)
        print("# Scenario: {} | {}".format(scen, fp))
        print("#" * 70)
        if not os.path.exists(fp):
            print("[SKIP] File not found: {}".format(fp))
            continue

        try:
            fut_df = load_future_inflow_data(fp)
            fut_df = (fut_df[fut_df["Date"] >= FUTURE_START]
                      .sort_values("Date")
                      .drop_duplicates("Date")
                      .reset_index(drop=True))

            if fut_df.empty:
                print("[SKIP] {}: no data after {}".format(scen, FUTURE_START))
                continue

            print("  Trimmed: {} ~ {} ({} days)".format(
                fut_df['Date'].min().date(), fut_df['Date'].max().date(),
                len(fut_df)))

            cascade_op.update_calibration_params(best_params, reservoir_idx=2)
            pd.DataFrame({'Parameter': list(best_params.keys()),
                          'Value': list(best_params.values())}
                         ).to_csv(
                os.path.join(RESULTS_DIR, "params_{}.csv".format(scen)),
                index=False, encoding="utf-8-sig")

            cascade_op.set_initial_volumes(final_storage)
            cascade_op.reset_controller_state()

            scen_dir = ensure(os.path.join(RESULTS_DIR, scen))
            fut_daily, fut_res = cascade_op.simulate_cascade_full(
                fut_df, output_dir=scen_dir, save_outputs=True, label=scen)

            fut_final = cascade_op.get_final_volumes()
            cascade_op.save_initial_storage(
                fut_final, os.path.join(scen_dir, "final_storage.csv"))

            row = {
                'Scenario': scen,
                'Start': str(fut_df['Date'].min().date()),
                'End': str(fut_df['Date'].max().date()),
                'Days': len(fut_df),
                'Total_Power_MWh': float(
                    fut_daily['System_Total_Power_MW'].sum() * 24),
                'Avg_Storage_m3': float(
                    fut_daily['System_Total_Volume_m3'].mean()),
                'Avg_Storage_Ratio': float(
                    fut_daily['System_Storage_Ratio'].mean()),
            }
            for i, r in enumerate(cascade_op.reservoirs):
                row[r["name"] + '_End_Storage_m3'] = fut_final[i]
                row[r["name"] + '_End_Storage_Ratio'] = (
                    fut_final[i] / r['Vmax'] if r['Vmax'] > 0 else 0.0)
            scenario_summary.append(row)

            print("\n[{}] Completed.".format(scen))
            print("  Total power: {:.2f} MWh".format(row['Total_Power_MWh']))
            end_str = ", ".join([
                "{}={:.1f}%".format(
                    r['name'], fut_final[i] / r['Vmax'] * 100)
                for i, r in enumerate(cascade_op.reservoirs)
            ])
            print("  End storage ratio: " + end_str)

            del fut_df, fut_daily, fut_res, fut_final
            clean_memory()

        except Exception as e:
            print("[ERROR] Scenario {} failed: {}".format(scen, e))
            traceback.print_exc()
            continue

    if scenario_summary:
        df_sum = pd.DataFrame(scenario_summary)
        df_sum.to_csv(os.path.join(RESULTS_DIR, "scenario_summary.csv"),
                      index=False, encoding="utf-8-sig")
        print("\n" + "=" * 70)
        print("SCENARIO SUMMARY")
        print("=" * 70)
        print(df_sum.to_string(index=False))

    print("\n" + "=" * 70)
    print("ALL DONE. Results in: {}/".format(RESULTS_DIR))
    print("=" * 70)
    for root, dirs, files in os.walk(RESULTS_DIR):
        for f in sorted(files):
            print("  {}".format(
                os.path.relpath(os.path.join(root, f), RESULTS_DIR)))