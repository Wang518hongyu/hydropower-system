# -*- coding: utf-8 -*-
"""
Kaidu Cascade Reservoir - Pure Hydropower Model + 2-Objective MOPSO
+ Full Uncertainty Analysis (Optimized + Parallel + Fixed)

Modification log:
  [FIX-1] MultiObjectivePSO.__init__ resets FEASIBLE_TOL = 0.01
  [FIX-2] Bootstrap uses complete hydrological-year blocks
  [FIX-3] Bootstrap CI uses percentile (2.5% - 97.5%) intervals
  [FIX-4] U4b prediction band uses Bootstrap-calibrated parameters
  [OPT-5] N_JOBS_GLOBAL capped at 64
  [OPT-6] annual_inflow computed as physical volume (m^3)
  [FIX-7] U4b adds residual-based uncertainty expansion (PICP fix)
  [FIX-8] U6 restricts wet/normal/dry classification to observed years
"""

import os
import gc
import ctypes
import traceback
import multiprocessing
import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime
from scipy import stats

try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False
    print("[Parallel] joblib not installed, running serial. "
          "Install via: pip install joblib")


# =========================================================
# 0. PARALLEL CONFIG
# =========================================================

if _HAS_JOBLIB:
    _CPU = multiprocessing.cpu_count()
    N_JOBS_GLOBAL = max(1, min(_CPU - 4, 64))
else:
    N_JOBS_GLOBAL = 1

print("[Parallel] CPU count = %d, N_JOBS_GLOBAL = %d"
      % (multiprocessing.cpu_count(), N_JOBS_GLOBAL))


def _par(n_jobs, items, func):
    if n_jobs > 1 and _HAS_JOBLIB:
        return Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
            delayed(func)(it) for it in items)
    return [func(it) for it in items]


# =========================================================
# 0b. MODULE-LEVEL PARALLEL WORKERS
# =========================================================

def _safe_eval_mopso(eval_func, params):
    try:
        return eval_func(params)
    except Exception:
        traceback.print_exc()
        return None


def _run_one_mopso(args):
    (run_id, seed, MOPSOClass, bounds, output_dir,
     n_jobs_inner, evaluator) = args
    run_dir = ensure(os.path.join(output_dir, "run_%02d" % (run_id + 1)))
    try:
        opt = MOPSOClass(bounds, output_dir=run_dir,
                         seed=seed, n_jobs=n_jobs_inner)
        archive = opt.optimize(evaluator, verbose=False)
    except Exception:
        traceback.print_exc()
        return None
    if not archive:
        return None

    _save_archive(archive, run_dir)
    _save_opt_history(opt, run_dir)

    F = np.vstack([x['objectives'] for x in archive])
    pd.DataFrame(F, columns=['Obj_RMSE_norm', 'Obj_PBIAS_norm']
                 ).to_csv(
        os.path.join(run_dir, "pareto_front_for_plot.csv"),
        index=False, encoding="utf-8-sig")

    par_rows = []
    for i, x in enumerate(archive, start=1):
        r = {'Run_ID': run_id + 1, 'Solution': i}
        r.update(x.get('params', {}))
        par_rows.append(r)
    pd.DataFrame(par_rows).to_csv(
        os.path.join(run_dir, "pareto_params.csv"),
        index=False, encoding="utf-8-sig")

    best_row = None
    try:
        best_run = select_topsis_solution(archive)
        best_row = {'Run_ID': run_id + 1, 'Seed': seed}
        best_row.update(best_run.get('params', {}))
        best_row['TOPSIS_Obj_RMSE_norm'] = float(best_run['objectives'][0])
        best_row['TOPSIS_Obj_PBIAS_norm'] = float(best_run['objectives'][1])
        best_row['TOPSIS_CV'] = float(best_run['constraint_violation'])
        best_row['TOPSIS_Score'] = float(best_run.get('TOPSIS_Score', np.nan))
    except Exception:
        pass

    return {'run_id': run_id, 'seed': seed, 'F': F,
            'par_rows': par_rows, 'best_row': best_row}


def _eval_grid_point(args):
    (v1, v2, k1, k2, curve_models, configs,
     calib_df, actual_c) = args
    try:
        params = {k1: float(v1), k2: float(v2)}
        op = CascadeReservoirOperation(
            curve_models, configs, verbose=False)
        op.update_calibration_params(params, reservoir_idx=2)
        op.reset_simulation_state()
        sim_m, diag = op.simulate_cascade_monthly_gen(calib_df)
    except Exception:
        traceback.print_exc()
        return None

    common = actual_c.index.intersection(sim_m.index)
    if len(common) < 12:
        return None

    sim = sim_m.loc[common].values.astype(float)
    obs = actual_c.loc[common].values.astype(float)
    m = np.isfinite(sim) & np.isfinite(obs)
    sim = sim[m]; obs = obs[m]
    if len(sim) < 12:
        return None

    rmse = np.sqrt(np.mean((sim - obs) ** 2))
    rmse_n = rmse / max(np.std(obs), 1e-6)
    pb = 100.0 * np.sum(obs - sim) / np.sum(obs) \
        if abs(np.sum(obs)) > 1e-12 else 0.0
    pb_n = abs(pb) / 100.0

    n = max(len(calib_df), 1)
    cv = (diag['eco_violation_total'] / n) * 1.0 \
        + (diag['flood_violation_total'] / n) * 0.5 \
        + (diag['water_deficit_total'] / n) * 0.3

    nse = 1.0 - np.sum((sim - obs) ** 2) / \
        max(np.sum((obs - obs.mean()) ** 2), 1e-12)

    return {k1: float(v1), k2: float(v2),
            'RMSE_norm': float(rmse_n),
            'PBIAS_norm': float(pb_n),
            'CV': float(cv), 'NSE': float(nse)}


def _run_one_bootstrap(args):
    (rid, series, MOPSOClass, bounds, output_dir,
     n_jobs_inner, calib_evaluator, seed_base) = args
    rdir = ensure(os.path.join(output_dir, "replicate_%03d" % rid))
    series.to_csv(os.path.join(rdir, "actual_resampled.csv"),
                  encoding="utf-8-sig")

    def eval_rep(params, _s=series):
        return calib_evaluator(params, _s)

    try:
        opt = MOPSOClass(bounds, output_dir=rdir,
                         seed=seed_base + rid, n_jobs=n_jobs_inner)
        archive = opt.optimize(eval_rep, verbose=False)
    except Exception:
        traceback.print_exc()
        return None
    if not archive:
        return None

    _save_archive(archive, rdir)
    _save_opt_history(opt, rdir)

    F = np.vstack([x['objectives'] for x in archive])
    try:
        best = select_topsis_solution(archive)
    except Exception:
        return None

    row = {'Replicate': rid, 'F': F,
           'TOPSIS_Obj_RMSE': float(best['objectives'][0]),
           'TOPSIS_Obj_PBIAS': float(best['objectives'][1]),
           'TOPSIS_CV': float(best['constraint_violation'])}
    for k, v in best['params'].items():
        row['Best_' + k] = float(v)
    return row


def _run_one_ensemble_member(args):
    (i, params, curve_models, configs, inflow_df, output_dir) = args
    try:
        op = CascadeReservoirOperation(
            curve_models, configs, verbose=False)
        op.update_calibration_params(params, reservoir_idx=2)
        op.reset_simulation_state()
        result, _ = op.simulate_cascade_full(
            inflow_df, output_dir=output_dir, write_files=False)
        downstream = result['R3_Dashankou_Power_MW']
        monthly_sim = downstream.resample('ME').sum() * 24.0 / 10.0
        monthly_sim.index = \
            monthly_sim.index.to_period('M').to_timestamp()
        return monthly_sim
    except Exception:
        traceback.print_exc()
        return None


def _run_one_dead_level(args):
    (tag, dl, names, configs_base, curve_models,
     inflow_df, actual_all, best_params, output_dir) = args
    configs = deepcopy(configs_base)
    for i, name in enumerate(names):
        configs[i]["dead_water_level"] = dl[name]

    try:
        op = CascadeReservoirOperation(
            curve_models, configs, verbose=False)
        op.update_calibration_params(best_params, reservoir_idx=2)
        op.reset_simulation_state()
        result, _ = op.simulate_cascade_full(
            inflow_df, output_dir=output_dir, write_files=False)
    except Exception:
        traceback.print_exc()
        return None

    downstream = result['R3_Dashankou_Power_MW']
    monthly_sim = downstream.resample('ME').sum() * 24.0 / 10.0
    monthly_sim.index = monthly_sim.index.to_period('M').to_timestamp()

    val_sim = monthly_sim.loc['2015-01':'2019-12']
    val_obs = actual_all.loc['2015-01':'2019-12']
    common = val_sim.index.intersection(val_obs.index)
    mv = compute_all_metrics(val_sim.loc[common].values,
                             val_obs.loc[common].values) \
        if len(common) >= 2 else None

    cal_sim = monthly_sim.loc['2010-01':'2014-12']
    cal_obs = actual_all.loc['2010-01':'2014-12']
    cc = cal_sim.index.intersection(cal_obs.index)
    mc = compute_all_metrics(cal_sim.loc[cc].values,
                             cal_obs.loc[cc].values) \
        if len(cc) >= 2 else None

    return {'Scenario': tag,
            'Chahanwusu_dead': dl["Chahanwusu"],
            'Liushugou_dead': dl["Liushugou"],
            'Dashankou_dead': dl["Dashankou"],
            'Cal_NSE': mc['NSE'] if mc else np.nan,
            'Cal_KGE': mc['KGE'] if mc else np.nan,
            'Cal_PBIAS': mc['PBIAS'] if mc else np.nan,
            'Val_NSE': mv['NSE'] if mv else np.nan,
            'Val_KGE': mv['KGE'] if mv else np.nan,
            'Val_PBIAS': mv['PBIAS'] if mv else np.nan,
            'Eco_Violation_Total': op.ecological_violation,
            'Flood_Violation_Total': op.flood_rule_violation,
            'Water_Deficit_Total': op.water_deficit}


# =========================================================
# 1. UTILITIES
# =========================================================

def trim_memory():
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


def ensure(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_df(df, path):
    if df is None or len(df) == 0:
        return
    df.to_csv(path, index=False, encoding="utf-8-sig")


# =========================================================
# 2. RESERVOIR CURVE MODEL
# =========================================================

class ReservoirCurveModel:
    def __init__(self, curve_path, reservoir_name=""):
        self.curve_path = curve_path
        self.reservoir_name = reservoir_name
        self.curve, self.Vmax, self.H_design = \
            self._load_curve_flexible(curve_path)

    def _load_curve_flexible(self, path):
        print("\nLoading curve file: %s" % path)
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
                if 'Parameter' in df_meta.columns and \
                   'Value' in df_meta.columns:
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
                    ['level', 'elevation', 'stage', 'h_m', 'z_m',
                     'water', 'h']):
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
        print("Curve loaded: %d pts, Vmax=%.2f m3, H_design=%.2f m, "
              "H_min=%.2f m"
              % (len(levels), Vmax, H_design,
                 float(np.nanmin(levels))))
        return curve, float(Vmax), float(H_design)


# =========================================================
# 3. CASCADE RESERVOIR OPERATION
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

        self.water_balance_stats = {}
        self.water_deficit = 0
        self.ecological_violation = 0
        self.flood_rule_violation = 0
        self.reset_simulation_state()

        if verbose:
            print("\n" + "=" * 70)
            print("DEAD STORAGE DIAGNOSTIC")
            print("=" * 70)
            print("  Criterion (SL 278-2002): H_min_at_dead / "
                  "H_design_gross >= 0.60")
            print("-" * 70)
            for res in self.reservoirs:
                ratio = res['dead_volume'] / res['Vmax'] \
                    if res['Vmax'] > 0 else 0.0
                usable = res['Vmax'] - res['dead_volume']
                Hg = res['curve_model'].H_design - res['tailwater']
                Hm = res['dead_level'] - res['tailwater']
                hr = Hm / Hg if Hg > 0 else 0.0
                print("  %s:" % res['name'])
                print("    dead_level=%.2f m, tailwater=%.2f m, "
                      "H_design=%.2f m"
                      % (res['dead_level'], res['tailwater'],
                         res['curve_model'].H_design))
                print("    V_dead=%.4e m3, V_dead/Vmax=%.4f (%.2f%%), "
                      "usable=%.4e m3"
                      % (res['dead_volume'], ratio, ratio * 100.0,
                         usable))
                print("    H_min_at_dead=%.2f m, H_design_gross=%.2f m, "
                      "H_min/H_design=%.2f%%"
                      % (Hm, Hg, hr * 100.0))
                if hr < 0.60:
                    print("    [WARNING] H_min/H_design < 60% (SL 278-2002)")
                elif hr > 0.85:
                    print("    [WARNING] H_min/H_design > 85%")
                else:
                    print("    [OK] Compliant with SL 278-2002")
            print("=" * 70 + "\n")

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
        }
        res['monthly_min_eco_release'] = cfg.get(
            "monthly_min_eco_release_m3s", None)
        res['dead_volume'] = np.interp(res['dead_level'],
                                       cm.curve["levels"], cm.curve["V"])

        raw = float(cfg.get("initial_volume", 0.0))
        ml = cfg.get("monthly_normal_low", None)
        sm = res['impoundment_start_date'].month
        init_level_default = ml.get(sm, res['dead_level']) if ml else \
            cfg.get("normal_low_default", res['dead_level'])
        if raw <= 0:
            res['initial_volume'] = float(np.interp(
                init_level_default, cm.curve["levels"], cm.curve["V"]))
        else:
            res['initial_volume'] = float(np.clip(raw, 0.0, res['Vmax']))
        res['initial_volume'] = max(res['initial_volume'],
                                     res['dead_volume'])

        res['monthly_normal_high'] = cfg.get("monthly_normal_high", None)
        res['default_normal_high'] = cfg.get("normal_high_default",
                                             res['spill_level'] - 1.0)
        res['monthly_normal_low'] = cfg.get("monthly_normal_low", None)
        res['default_normal_low'] = cfg.get("normal_low_default",
                                            res['dead_level'] + 2.0)
        res['monthly_flood_control'] = cfg.get(
            "monthly_flood_control_level", None)
        res['flood_control_level_fixed'] = cfg.get(
            "flood_control_level", res['spill_level'])
        res['active_drawdown_enabled'] = cfg.get(
            "active_drawdown_enabled", True)
        res['active_drawdown_rate'] = cfg.get(
            "active_drawdown_rate", 100.0)
        res['use_pid'] = cfg.get("use_pid", True)
        res['pid_Kp'] = cfg.get("pid_Kp", 0.15)
        res['pid_Ki'] = cfg.get("pid_Ki", 0.01)
        res['pid_Kd'] = cfg.get("pid_Kd", 0.05)
        res['pid_target_offset'] = cfg.get("pid_target_offset", 0.3)
        res['priority_order'] = ["flood", "eco", "power"]
        res['normal_zone_power_frac'] = cfg.get(
            "normal_zone_power_fraction", 0.70)
        res['max_power_frac'] = cfg.get("max_power_fraction", 0.95)
        res['min_power_frac'] = cfg.get("min_power_fraction", 0.15)
        res['high_buffer'] = cfg.get("high_buffer", 0.5)
        res['interp_targets'] = cfg.get("interpolate_targets", True)
        return res

    def reset_simulation_state(self):
        for r in self.reservoirs:
            r['integral_error'] = 0.0
            r['prev_error'] = 0.0
        self.water_balance_stats = {
            r['name']: {'total_residual': 0.0, 'max_residual': 0.0,
                        'count': 0, 'residuals': []}
            for r in self.reservoirs}
        self.water_deficit = 0
        self.ecological_violation = 0
        self.flood_rule_violation = 0

    def _get_monthly_level(self, d, default, month):
        return d.get(month, default) if d is not None else default

    def _get_daily_level(self, date, d, default):
        month, day, year = date.month, date.day, date.year
        first = pd.Timestamp(year, month, 1)
        next_first = pd.Timestamp(year + 1, 1, 1) if month == 12 else \
            pd.Timestamp(year, month + 1, 1)
        days = (next_first - first).days
        v1 = self._get_monthly_level(d, default, month)
        nm = month + 1 if month < 12 else 1
        v2 = self._get_monthly_level(d, default, nm)
        frac = (day - 1) / days
        return v1 + (v2 - v1) * frac

    def _get_normal_high(self, res, date):
        if res['interp_targets']:
            return self._get_daily_level(date, res['monthly_normal_high'],
                                         res['default_normal_high'])
        return self._get_monthly_level(res['monthly_normal_high'],
                                       res['default_normal_high'],
                                       date.month)

    def _get_normal_low(self, res, date):
        if res['interp_targets']:
            return self._get_daily_level(date, res['monthly_normal_low'],
                                         res['default_normal_low'])
        return self._get_monthly_level(res['monthly_normal_low'],
                                       res['default_normal_low'],
                                       date.month)

    def _storage_control(self, res, V):
        if res['Vmax'] <= 0:
            return 0.0
        s = V / res['Vmax']
        f = res['storage_ratio'] + (1.0 - res['storage_ratio']) * s
        return np.clip(f, 0.0, 1.0)

    def _get_eco_value(self, res, month):
        if res['monthly_min_eco_release'] is not None:
            return float(res['monthly_min_eco_release'].get(
                month, res['min_eco_release']))
        return float(res['min_eco_release'])

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
            'is_operational': False,
            'operation_stage': 'PRE_IMPOUNDMENT',
            'is_impounding': False, 'is_generating': False,
            'eco_min_required': 0.0, 'eco_actual': 0.0,
            'eco_constraint_met': True, 'eco_supplement': 0.0,
            'flood_trigger_inflow': res['flood_trigger_inflow'],
            'forced_flood_triggered': False,
            'forced_flood_required': 0.0,
            'forced_flood_supplement': 0.0,
            'flood_rule_met': True,
            'original_total_outflow': Qin}

    def _simulate_single(self, res, Qin, date, V, allow_gen=True):
        cm = res['curve_model']
        Qin = max(float(Qin), 0.0)
        month = date.month
        H = np.interp(V, cm.curve["V"], cm.curve["levels"])
        V_before = V

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

        Qeco_required = self._get_eco_value(res, month)
        Q_demand = Qeco_required

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
            pf_unc = base - u
            pf = max(res['min_power_frac'],
                     min(pf_unc, res['max_power_frac']))
            if pf != pf_unc:
                res['integral_error'] -= err
        else:
            if H >= H_high - res['high_buffer']:
                ratio = (H - (H_high - res['high_buffer'])) / \
                    max(res['high_buffer'], 1e-6)
                ratio = max(0.0, min(ratio, 1.0))
                pf = res['normal_zone_power_frac'] + \
                    (res['max_power_frac'] -
                     res['normal_zone_power_frac']) * ratio
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
            Qt_tar = Q_demand + Qmax * pf * sf
            Qt = min(Qt_tar, Qmax, avail)
            Qt = max(Qt, 0.0)
            Qt_add = max(Qt - Q_demand, 0.0)
        else:
            Qt = 0.0
            Qt_add = 0.0

        Qn = max(Q_demand - Qt, 0.0)
        total_req = Qt + Qn
        if total_req > avail and total_req > 0:
            sc = avail / total_req
            Qt *= sc
            Qn *= sc
            Qt_add = max(Qt - Q_demand, 0.0)

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
        resid = V - V_before - (Qin - total_out) * 86400.0

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
            'Q_mandatory': Q_demand, 'Qturb_additional': Qt_add,
            'total_outflow': total_out,
            'storage_factor': sf, 'power_frac': pf,
            'head': (max(H - res['tailwater'], 0.1)
                     if H >= res['dead_level'] else 0.0),
            'residual': resid, 'V_before': V_before, 'V_after': V,
            'is_operational': True,
            'operation_stage': ('GENERATION' if allow_gen
                                else 'IMPOUNDMENT_ONLY'),
            'is_impounding': True, 'is_generating': bool(allow_gen),
            'eco_min_required': Qeco_required,
            'eco_actual': Qeco_actual,
            'eco_constraint_met': True, 'eco_supplement': 0.0,
            'flood_trigger_inflow': trigger,
            'forced_flood_triggered': forced,
            'forced_flood_required': flood_req,
            'forced_flood_supplement': Qflood,
            'flood_rule_met': flood_met,
            'original_total_outflow': total_out}

    def _apply_hard(self, res, result, Qin, date):
        Qin = max(float(Qin), 0.0)
        dt = 86400.0
        eco_req = self._get_eco_value(res, date.month)
        eco_act = max(float(result.get('Qeco_actual', 0.0)), 0.0)
        deficit = max(eco_req - eco_act, 0.0)
        avail_v = max(result['V'] - res['dead_volume'], 0.0)
        avail_f = avail_v / dt
        supp = min(deficit, avail_f)
        if supp > 0:
            result['Qeco_actual'] += supp
            result['Q_nonpower'] += supp
            result['total_outflow'] += supp
            result['V'] -= supp * dt
        met = bool(result['Qeco_actual'] >= eco_req - 1e-8)
        result['V'] = max(result['V'], 0.0)
        result['V_after'] = result['V']
        result['H'] = np.interp(result['V'],
                                res['curve_model'].curve["V"],
                                res['curve_model'].curve["levels"])
        result['head'] = (max(result['H'] - res['tailwater'], 0.1)
                          if result['H'] >= res['dead_level'] else 0.0)
        result['eco_min_required'] = eco_req
        result['eco_actual'] = result['Qeco_actual']
        result['eco_supplement'] = supp
        result['eco_constraint_met'] = met
        result['residual'] = (result['V_after'] - result['V_before']
                              - (Qin - result['total_outflow']) * dt)
        return result

    def _simulate_day(self, res, Qin, date, V):
        date = pd.Timestamp(date)
        if date < res['impoundment_start_date']:
            return self._simulate_inactive(res, Qin, date, V)
        allow = (date >= res['generation_start_date'])
        r = self._simulate_single(res, Qin, date, V, allow_gen=allow)
        return self._apply_hard(res, r, Qin, date)

    def simulate_cascade_full(self, df, output_dir="results",
                              write_files=True):
        if df.empty:
            raise ValueError("No data.")
        ensure(output_dir)
        prefix = os.path.join(output_dir, "cascade_output")

        volumes = [r['initial_volume'] for r in self.reservoirs]
        self.reset_simulation_state()

        daily_data = []
        res_daily = {r['name']: [] for r in self.reservoirs}
        wb_daily = []

        print("Running cascade simulation...")
        total_days = len(df)

        for idx, (_, row) in enumerate(df.iterrows()):
            Qin = max(float(row["Data"]), 0.0)
            date = pd.Timestamp(row["Date"])
            cur_in = Qin
            results = []
            local_ins = []

            for i, res in enumerate(self.reservoirs):
                local_ins.append(cur_in)
                result = self._simulate_day(res, cur_in, date, volumes[i])
                results.append(result)
                volumes[i] = result['V']

                if result['is_operational']:
                    if result['H'] < res['dead_level'] - 1e-8:
                        self.water_deficit += 1
                    if not result['eco_constraint_met']:
                        self.ecological_violation += 1
                    if not result['flood_rule_met']:
                        self.flood_rule_violation += 1

                abs_r = abs(result['residual'])
                st = self.water_balance_stats[res['name']]
                st['total_residual'] += abs_r
                st['max_residual'] = max(st['max_residual'], abs_r)
                st['count'] += 1
                st['residuals'].append(abs_r)

                if write_files:
                    wb_daily.append({
                        'Date': date, 'Reservoir': res['name'],
                        'Is_Operational': result['is_operational'],
                        'V_before_m3': result['V_before'],
                        'V_after_m3': result['V_after'],
                        'Qin_m3': cur_in * 86400.0,
                        'Qout_m3': result['total_outflow'] * 86400.0,
                        'Residual_m3': result['residual']})

                if i < self.n_reservoirs - 1:
                    cur_in = result['total_outflow'] * self.routing_coeff
                    if self.interbasin_flow_ratio > 0:
                        cur_in += Qin * self.interbasin_flow_ratio

            total_v = sum(r['V'] for r in results)
            total_p = sum(r['power_mw'] for r in results)
            sys_sr = total_v / self.total_Vmax \
                if self.total_Vmax > 0 else 0.0
            sys_out = results[-1]['total_outflow']
            wu = sys_out / Qin if Qin > 0 else 0.0
            total_res = sum(r['residual'] for r in results)

            rec = {'Date': date, 'Year': date.year,
                   'Month': date.month, 'Day': date.day,
                   'System_Inflow_m3s': Qin,
                   'System_Outflow_m3s': sys_out,
                   'System_Water_Utilization': wu,
                   'System_Total_Volume_m3': total_v,
                   'System_Storage_Ratio': sys_sr,
                   'System_Total_Power_MW': total_p,
                   'System_Water_Balance_Residual_m3': total_res}

            for i, (res, result) in enumerate(
                    zip(self.reservoirs, results)):
                pf = 'R%d_%s' % (i + 1, res['name'])
                in_val = local_ins[i]
                rec.update({
                    '%s_Is_Operational' % pf: result['is_operational'],
                    '%s_Operation_Stage' % pf:
                        result.get('operation_stage', ''),
                    '%s_Is_Impounding' % pf:
                        result.get('is_impounding', False),
                    '%s_Is_Generating' % pf:
                        result.get('is_generating', False),
                    '%s_Inflow_m3s' % pf: in_val,
                    '%s_Level_m' % pf: result['H'],
                    '%s_Volume_m3' % pf: result['V'],
                    '%s_Storage_Ratio' % pf:
                        (result['V'] / res['Vmax']
                         if res['Vmax'] > 0 else 0.0),
                    '%s_Power_MW' % pf: result['power_mw'],
                    '%s_Head_m' % pf: result['head'],
                    '%s_Turbine_Flow_m3s' % pf: result['Qturb'],
                    '%s_Mandatory_Release_m3s' % pf:
                        result['Q_mandatory'],
                    '%s_Additional_Power_Flow_m3s' % pf:
                        result['Qturb_additional'],
                    '%s_Flood_Release_m3s' % pf: result['Qflood'],
                    '%s_Drawdown_Release_m3s' % pf: result['Qdrawdown'],
                    '%s_Spill_m3s' % pf: result['Qspill'],
                    '%s_Total_Outflow_m3s' % pf: result['total_outflow'],
                    '%s_Agri_Release_m3s' % pf: result['Qagr'],
                    '%s_Dom_Release_m3s' % pf: result['Qdom'],
                    '%s_Eco_Required_m3s' % pf: result['Qeco_required'],
                    '%s_Eco_Actual_m3s' % pf: result['Qeco_actual'],
                    '%s_Eco_Release_m3s' % pf: result['Qeco_actual'],
                    '%s_Ind_Release_m3s' % pf: result['Qind'],
                    '%s_Eco_Min_Required_m3s' % pf:
                        result['eco_min_required'],
                    '%s_Eco_Constraint_Met' % pf:
                        result['eco_constraint_met'],
                    '%s_Eco_Supplement_m3s' % pf:
                        result['eco_supplement'],
                    '%s_Flood_Trigger_Inflow_m3s' % pf:
                        result['flood_trigger_inflow'],
                    '%s_Forced_Flood_Triggered' % pf:
                        result['forced_flood_triggered'],
                    '%s_Flood_Rule_Met' % pf: result['flood_rule_met'],
                    '%s_Water_Balance_Residual_m3' % pf:
                        result['residual']})

                res_daily[res['name']].append({
                    'Date': date, 'Year': date.year,
                    'Month': date.month, 'Day': date.day,
                    'Is_Operational': result['is_operational'],
                    'Operation_Stage': result.get('operation_stage', ''),
                    'Is_Impounding': result.get('is_impounding', False),
                    'Is_Generating': result.get('is_generating', False),
                    'Inflow_m3s': in_val,
                    'Level_m': result['H'], 'Volume_m3': result['V'],
                    'Storage_Ratio': (result['V'] / res['Vmax']
                                      if res['Vmax'] > 0 else 0.0),
                    'Power_MW': result['power_mw'],
                    'Head_m': result['head'],
                    'Turbine_Flow_m3s': result['Qturb'],
                    'Mandatory_Release_m3s': result['Q_mandatory'],
                    'Additional_Power_Flow_m3s':
                        result['Qturb_additional'],
                    'Flood_Release_m3s': result['Qflood'],
                    'Drawdown_Release_m3s': result['Qdrawdown'],
                    'Spill_m3s': result['Qspill'],
                    'Total_Outflow_m3s': result['total_outflow'],
                    'Eco_Required_m3s': result['Qeco_required'],
                    'Eco_Actual_m3s': result['Qeco_actual'],
                    'Eco_Release_m3s': result['Qeco_actual'],
                    'Eco_Min_Required_m3s': result['eco_min_required'],
                    'Eco_Constraint_Met': result['eco_constraint_met'],
                    'Eco_Supplement_m3s': result['eco_supplement'],
                    'Flood_Trigger_Inflow_m3s':
                        result['flood_trigger_inflow'],
                    'Forced_Flood_Triggered':
                        result['forced_flood_triggered'],
                    'Flood_Rule_Met': result['flood_rule_met'],
                    'Water_Balance_Residual_m3': result['residual'],
                    'Storage_Factor': result['storage_factor'],
                    'Power_Fraction': result['power_frac']})

            daily_data.append(rec)

            if (idx + 1) % 2000 == 0:
                print("  %d/%d days" % (idx + 1, total_days))

        daily_df = pd.DataFrame(daily_data)
        daily_df.set_index('Date', inplace=True)

        res_dfs = {}
        for name, lst in res_daily.items():
            r = pd.DataFrame(lst)
            r.set_index('Date', inplace=True)
            res_dfs[name] = r

        if write_files:
            daily_df.to_csv(
                "%s_system_daily_results.csv" % prefix,
                encoding="utf-8-sig")
            for name, r in res_dfs.items():
                r.to_csv(
                    "%s_%s_daily_results.csv" % (prefix, name),
                    encoding="utf-8-sig")
                print("%s daily results saved" % name)
            pd.DataFrame(wb_daily).to_csv(
                "%s_water_balance_daily.csv" % prefix,
                index=False, encoding="utf-8-sig")
            print("Water balance daily data saved")

        return daily_df, res_dfs

    def simulate_cascade_monthly_gen(self, df):
        if df.empty:
            return pd.Series(dtype=float), {}

        volumes = [r['initial_volume'] for r in self.reservoirs]
        self.reset_simulation_state()

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
                result = self._simulate_day(res, cur, date, volumes[i])
                volumes[i] = result['V']

                if result['is_operational']:
                    if result['H'] < res['dead_level'] - 1e-8:
                        deficit[res['name']] += 1
                    if not result['eco_constraint_met']:
                        eco_v[res['name']] += 1
                    if result['forced_flood_triggered']:
                        flood_t[res['name']] += 1
                    if not result['flood_rule_met']:
                        flood_v[res['name']] += 1

                if i < self.n_reservoirs - 1:
                    cur = result['total_outflow'] * self.routing_coeff
                    if self.interbasin_flow_ratio > 0:
                        cur += Qin * self.interbasin_flow_ratio

                if i == self.n_reservoirs - 1:
                    e = result['power_mw'] * 24.0
                    k = (date.year, month)
                    monthly[k] = monthly.get(k, 0.0) + e

        if not monthly:
            return pd.Series(dtype=float), {}

        dates = [pd.Timestamp(year=y, month=m, day=1)
                 for (y, m) in monthly.keys()]
        vals = [v / 10.0 for v in monthly.values()]
        s = pd.Series(vals, index=dates, name='gen').sort_index()

        diag = {'water_deficit_by_res': deficit,
                'eco_violation_by_res': eco_v,
                'flood_violation_by_res': flood_v,
                'flood_trigger_days': flood_t,
                'water_deficit_total': int(sum(deficit.values())),
                'eco_violation_total': int(sum(eco_v.values())),
                'flood_violation_total': int(sum(flood_v.values()))}
        return s, diag

    def update_calibration_params(self, params, reservoir_idx=2):
        res = self.reservoirs[reservoir_idx]
        if "normal_zone_power_fraction" in params:
            res['normal_zone_power_frac'] = \
                params["normal_zone_power_fraction"]
        if "storage_ratio" in params:
            res['storage_ratio'] = params["storage_ratio"]


# =========================================================
# 4. DATA LOADING
# =========================================================

def load_inflow_data(path):
    df = pd.read_excel(path)
    req = {'Year', 'Month', 'Day', 'Data'}
    if not req.issubset(df.columns):
        raise ValueError("%s missing fields: %s"
                         % (path, sorted(req - set(df.columns))))
    df["Date"] = pd.to_datetime(df[["Year", "Month", "Day"]])
    df['Data'] = pd.to_numeric(df['Data'], errors='coerce')
    return df.dropna(subset=['Date', 'Data']).sort_values(
        'Date').reset_index(drop=True)


def load_actual_power(path):
    df = pd.read_excel(path)
    if 'time' not in df.columns or 'power' not in df.columns:
        raise ValueError("%s must contain columns: time and power." % path)
    df['time'] = pd.to_datetime(df['time'])
    df['time'] = df['time'].dt.to_period('M').dt.to_timestamp()
    df['power'] = pd.to_numeric(df['power'], errors='coerce')
    df = df.dropna(subset=['time', 'power']).sort_values('time')
    df.set_index('time', inplace=True)
    return df['power']


# =========================================================
# 5. METRICS
# =========================================================

def compute_all_metrics(sim, obs):
    sim = np.asarray(sim, dtype=float)
    obs = np.asarray(obs, dtype=float)
    m = ~(np.isnan(sim) | np.isnan(obs))
    sim = sim[m]; obs = obs[m]
    if len(sim) < 2:
        return None

    rmse = np.sqrt(np.mean((sim - obs) ** 2))
    dn = np.sum((obs - obs.mean()) ** 2)
    nse = 1.0 - np.sum((sim - obs) ** 2) / dn if dn > 1e-12 else np.nan
    r = (np.corrcoef(sim, obs)[0, 1]
         if np.std(sim) > 1e-12 and np.std(obs) > 1e-12 else np.nan)
    a = np.std(sim) / np.std(obs) if np.std(obs) > 1e-12 else np.nan
    b = np.mean(sim) / np.mean(obs) if abs(np.mean(obs)) > 1e-12 else np.nan
    kge = (1.0 - np.sqrt((r - 1.0) ** 2 + (a - 1.0) ** 2
                         + (b - 1.0) ** 2)
           if np.isfinite(r) and np.isfinite(a) and np.isfinite(b)
           else np.nan)
    r2 = r ** 2 if np.isfinite(r) else np.nan
    pb = 100.0 * np.sum(obs - sim) / np.sum(obs) \
        if abs(np.sum(obs)) > 1e-12 else np.nan
    mae = np.mean(np.abs(sim - obs))

    return {'RMSE': float(rmse),
            'NSE': float(nse) if np.isfinite(nse) else np.nan,
            'KGE': float(kge) if np.isfinite(kge) else np.nan,
            'R2': float(r2) if np.isfinite(r2) else np.nan,
            'PBIAS': float(pb) if np.isfinite(pb) else np.nan,
            'MAE': float(mae)}


# =========================================================
# 6. MOPSO EVALUATION
# =========================================================

def evaluate_mopso_detailed(op_template, calib_df, actual_all, params):
    op = deepcopy(op_template)
    op.update_calibration_params(params, reservoir_idx=2)
    op.reset_simulation_state()

    sim_m, diag = op.simulate_cascade_monthly_gen(calib_df)

    c0 = pd.Timestamp('2010-01-01')
    c1 = pd.Timestamp('2014-12-01')
    actual_c = actual_all.loc[(actual_all.index >= c0)
                              & (actual_all.index <= c1)]
    common = actual_c.index.intersection(sim_m.index)
    if len(common) < 12:
        return None

    sim = sim_m.loc[common].values.astype(float)
    obs = actual_c.loc[common].values.astype(float)
    m = np.isfinite(sim) & np.isfinite(obs)
    sim = sim[m]; obs = obs[m]
    if len(sim) < 12:
        return None

    rmse = np.sqrt(np.mean((sim - obs) ** 2))
    rmse_n = rmse / max(np.std(obs), 1e-6)
    pb = 100.0 * np.sum(obs - sim) / np.sum(obs) \
        if abs(np.sum(obs)) > 1e-12 else 0.0
    pb_n = abs(pb) / 100.0

    wd = diag['water_deficit_total']
    ev = diag['eco_violation_total']
    fv = diag['flood_violation_total']
    n = max(len(calib_df), 1)
    cv = (ev / n) * 1.0 + (fv / n) * 0.5 + (wd / n) * 0.3

    metrics = compute_all_metrics(sim, obs)
    comp = {
        'RMSE': rmse, 'RMSE_norm': rmse_n,
        'PBIAS': pb, 'PBIAS_norm': pb_n,
        'NSE': (metrics['NSE'] if metrics else np.nan),
        'KGE': (metrics['KGE'] if metrics else np.nan),
        'R2': (metrics['R2'] if metrics else np.nan),
        'Water_Deficit_days': wd,
        'Eco_Violation_days': ev,
        'Flood_Rule_Violation_days': fv,
        'Eco_Violation_Chahanwusu':
            diag['eco_violation_by_res']['Chahanwusu'],
        'Eco_Violation_Liushugou':
            diag['eco_violation_by_res']['Liushugou'],
        'Eco_Violation_Dashankou':
            diag['eco_violation_by_res']['Dashankou'],
        'Flood_Trigger_Days_Chahanwusu':
            diag['flood_trigger_days']['Chahanwusu'],
        'Flood_Trigger_Days_Liushugou':
            diag['flood_trigger_days']['Liushugou'],
        'Flood_Trigger_Days_Dashankou':
            diag['flood_trigger_days']['Dashankou'],
        'Constraint_Violation': cv}

    return {'objectives': np.array([rmse_n, pb_n], dtype=float),
            'constraint_violation': float(cv),
            'components': comp}


# =========================================================
# 7. MOPSO
# =========================================================

class MultiObjectivePSO:
    FEASIBLE_TOL = 0.01

    def __init__(self, bounds, pop_size=40, max_iter=50, archive_size=80,
                 w_start=0.9, w_end=0.4, c1=2.0, c2=1.5,
                 output_dir="results", seed=42, n_jobs=1):

        type(self).FEASIBLE_TOL = 0.01

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
        self.n_jobs = int(n_jobs)
        ensure(output_dir)

        self.rng = np.random.default_rng(seed)
        self.low = np.array([bounds[k][0]
                             for k in self.param_keys], dtype=float)
        self.high = np.array([bounds[k][1]
                              for k in self.param_keys], dtype=float)

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
                d[order[j]] += (F[order[j + 1], m]
                                - F[order[j - 1], m]) / (fmax - fmin)
        return d

    def _update_archive(self, candidates):
        pool = self.archive + candidates
        feas = [x for x in pool
                if x['constraint_violation'] <= self.FEASIBLE_TOL]
        if feas:
            pool = feas
        else:
            mcv = min(x['constraint_violation'] for x in pool)
            pool = [x for x in pool
                    if x['constraint_violation'] == mcv]

        nd = []
        for i, c in enumerate(pool):
            dom = False
            for j, o in enumerate(pool):
                if i == j:
                    continue
                if self._dominates(
                        o['objectives'], o['constraint_violation'],
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
                pos[d] = lo; vel[d] *= -0.5
            elif pos[d] > hi:
                pos[d] = hi; vel[d] *= -0.5
        return pos, vel

    def optimize(self, eval_func, verbose=True):
        self.history = []
        self.detailed_history = []

        stagnation_counter = 0
        last_best_key = None

        for it in range(self.max_iter):
            w = (self.w_start - (self.w_start - self.w_end) * it /
                 (self.max_iter - 1)) \
                if self.max_iter > 1 else self.w_end

            params_list = [self._decode(self.pop[i])
                           for i in range(self.pop_size)]

            if self.n_jobs > 1 and _HAS_JOBLIB:
                results = Parallel(
                    n_jobs=self.n_jobs, backend='loky', verbose=0,
                )(delayed(_safe_eval_mopso)(eval_func, p)
                  for p in params_list)
            else:
                results = [_safe_eval_mopso(eval_func, p)
                           for p in params_list]

            cands = []
            for i, res in enumerate(results):
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
                     'params': params_list[i].copy(),
                     'components': deepcopy(comp)}
                cands.append(c)

                rec = {'Iteration': it + 1, 'Particle': i + 1,
                       **params_list[i],
                       'Objective_RMSE_norm': obj[0],
                       'Objective_PBIAS_norm': obj[1],
                       'Constraint_Violation': cv, **comp}
                self.detailed_history.append(rec)

                if self.pbest_obj[i] is None:
                    rep = True
                else:
                    cur_dom = self._dominates(
                        obj, cv, self.pbest_obj[i], self.pbest_cv[i])
                    old_dom = self._dominates(
                        self.pbest_obj[i], self.pbest_cv[i], obj, cv)
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
                cv_vals = [x['constraint_violation']
                           for x in self.archive]
                hist['Best_RMSE_norm'] = float(np.min(F[:, 0]))
                hist['Best_PBIAS_norm'] = float(np.min(F[:, 1]))
                hist['Best_CV'] = float(np.min(cv_vals))
            self.history.append(hist)

            if self.archive:
                F_cur = np.vstack([x['objectives']
                                   for x in self.archive])
                cv_cur = [x['constraint_violation']
                          for x in self.archive]
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
                    print("  [Stagnation] Mutation on %d particles"
                          % n_mut)

            mutation_prob = 0.1
            for i in range(self.pop_size):
                leader = self._select_leader()
                r1 = self.rng.random(self.dim)
                r2 = self.rng.random(self.dim)
                self.vel[i] = (
                    w * self.vel[i]
                    + self.c1 * r1 * (self.pbest_pos[i] - self.pop[i])
                    + self.c2 * r2 * (leader - self.pop[i]))
                self.pop[i] = self.pop[i] + self.vel[i]
                self.pop[i], self.vel[i] = self._clip(
                    self.pop[i], self.vel[i])

                mask = self.rng.random(self.dim) < mutation_prob
                if np.any(mask):
                    self.pop[i] = np.where(
                        mask,
                        self.rng.uniform(self.low, self.high),
                        self.pop[i])

            if verbose:
                if self.archive:
                    F = np.vstack([x['objectives']
                                   for x in self.archive])
                    cv_vals = [x['constraint_violation']
                               for x in self.archive]
                    print("Iter %3d/%d | Arc=%3d | Feas=%3d/%d | "
                          "TOL=%.5f | minRMSE=%.6f minPB=%.6f minCV=%.4f"
                          % (it + 1, self.max_iter, len(self.archive),
                             nf, self.pop_size, self.FEASIBLE_TOL,
                             F[:, 0].min(), F[:, 1].min(),
                             min(cv_vals)))
                else:
                    print("Iter %3d/%d | Archive empty"
                          % (it + 1, self.max_iter))

        save_df(pd.DataFrame(self.detailed_history),
                os.path.join(self.output_dir,
                             "mopso_detailed_history.csv"))
        save_df(pd.DataFrame(self.history),
                os.path.join(self.output_dir, "mopso_convergence.csv"))
        self._save_pareto_front()
        return self.archive

    def _save_pareto_front(self):
        rows = []
        for i, x in enumerate(self.archive, start=1):
            rows.append({'Solution': i, **x['params'],
                         'Objective_RMSE_norm': x['objectives'][0],
                         'Objective_PBIAS_norm': x['objectives'][1],
                         'Constraint_Violation':
                             x['constraint_violation'],
                         **x['components']})
        save_df(pd.DataFrame(rows),
                os.path.join(self.output_dir, "mopso_pareto_front.csv"))


# =========================================================
# 8. TOPSIS
# =========================================================

def select_topsis_solution(archive):
    tol = MultiObjectivePSO.FEASIBLE_TOL
    feas = [x for x in archive if x['constraint_violation'] <= tol]
    if not feas:
        feas = sorted(archive,
                      key=lambda x: x['constraint_violation'])[:10]
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
# 9. PARETO QUALITY METRICS
# =========================================================

def hypervolume_2d(front, ref_point):
    F = np.asarray(front, dtype=float)
    r = np.asarray(ref_point, dtype=float)
    F = F[np.all(F <= r, axis=1)]
    if len(F) == 0:
        return 0.0
    idx = np.argsort(F[:, 0])
    F = F[idx]
    hv = 0.0
    prev = r[1]
    for x, y in F:
        hv += (r[0] - x) * (prev - y)
        prev = y
    return float(hv)


def spacing_metric(front):
    F = np.asarray(front, dtype=float)
    n = len(F)
    if n < 2:
        return 0.0
    d = np.zeros(n)
    for i in range(n):
        dist = np.linalg.norm(F - F[i], axis=1)
        dist[i] = np.inf
        d[i] = dist.min()
    return float(np.sqrt(np.mean((d - d.mean()) ** 2)))


def igd_metric(front, ref_front):
    F = np.asarray(front, dtype=float)
    R = np.asarray(ref_front, dtype=float)
    if len(F) == 0 or len(R) == 0:
        return np.nan
    d = np.array([np.min(np.linalg.norm(F - r, axis=1)) for r in R])
    return float(d.mean())


def front_center_distance(front, ref_front):
    F = np.asarray(front, dtype=float).mean(axis=0)
    R = np.asarray(ref_front, dtype=float).mean(axis=0)
    return float(np.linalg.norm(F - R))


def merge_reference_front(all_fronts):
    F = np.vstack(all_fronts)
    keep = np.ones(len(F), dtype=bool)
    for i in range(len(F)):
        if not keep[i]:
            continue
        dom = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        if dom.any():
            keep[i] = False
    return F[keep]


def save_individual_fronts(all_F, output_dir):
    rows = []
    for rid, F in enumerate(all_F, start=1):
        for j, (x, y) in enumerate(F, start=1):
            rows.append({'Run_ID': rid, 'Point_ID': j,
                         'Obj_RMSE_norm': float(x),
                         'Obj_PBIAS_norm': float(y)})
    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, "multi_start_individual_fronts.csv"),
        index=False, encoding="utf-8-sig")


def save_pooled_front(ref_front, output_dir):
    pd.DataFrame(ref_front,
                 columns=['Obj_RMSE_norm', 'Obj_PBIAS_norm']
                 ).to_csv(
        os.path.join(output_dir, "multi_start_pooled_front.csv"),
        index=False, encoding="utf-8-sig")


# =========================================================
# 10. SHARED UTILITIES
# =========================================================

def _save_archive(archive, run_dir):
    if not archive:
        return
    F = np.vstack([x['objectives'] for x in archive])
    pd.DataFrame(F, columns=['Obj_RMSE_norm', 'Obj_PBIAS_norm']
                 ).to_csv(
        os.path.join(run_dir, "pareto_objectives.csv"),
        index=False, encoding="utf-8-sig")

    rows = []
    for i, x in enumerate(archive, start=1):
        r = {'Solution': i}
        r.update(x.get('params', {}))
        rows.append(r)
    pd.DataFrame(rows).to_csv(
        os.path.join(run_dir, "pareto_params.csv"),
        index=False, encoding="utf-8-sig")

    rows = []
    for i, x in enumerate(archive, start=1):
        r = {'Solution': i,
             'Obj_RMSE_norm': x['objectives'][0],
             'Obj_PBIAS_norm': x['objectives'][1],
             'Constraint_Violation':
                 x.get('constraint_violation', np.nan)}
        r.update(x.get('params', {}))
        r.update(x.get('components', {}))
        rows.append(r)
    pd.DataFrame(rows).to_csv(
        os.path.join(run_dir, "pareto_detail.csv"),
        index=False, encoding="utf-8-sig")


def _save_opt_history(opt, run_dir):
    if hasattr(opt, 'history') and opt.history:
        pd.DataFrame(opt.history).to_csv(
            os.path.join(run_dir, "convergence.csv"),
            index=False, encoding="utf-8-sig")
    if hasattr(opt, 'detailed_history') and opt.detailed_history:
        pd.DataFrame(opt.detailed_history).to_csv(
            os.path.join(run_dir, "particle_history.csv"),
            index=False, encoding="utf-8-sig")


# =========================================================
# 11. U1: MULTI-START MOPSO
# =========================================================

def run_multi_start_mopso(MOPSOClass, bounds, n_runs, seed_base,
                          evaluator, output_dir,
                          n_jobs_outer=1, n_jobs_inner=1):
    ensure(output_dir)
    all_F = []
    qrows = []
    run_param_rows = []
    all_par_rows = []

    merged_obj = os.path.join(output_dir, "pareto_objectives_all.csv")
    for p in [merged_obj]:
        if os.path.exists(p):
            os.remove(p)

    tasks = [(run_id, seed_base + run_id * 101 + 13,
              MOPSOClass, bounds, output_dir, n_jobs_inner, evaluator)
             for run_id in range(n_runs)]

    print("Multi-start MOPSO: %d runs, n_jobs_outer=%d, n_jobs_inner=%d"
          % (n_runs, n_jobs_outer, n_jobs_inner))

    results = _par(n_jobs_outer, tasks, _run_one_mopso)

    for r in results:
        if r is None:
            continue
        run_id = r['run_id']
        F = r['F']
        all_F.append(F)

        qrows.append({'Run_ID': run_id + 1, 'Seed': r['seed'],
                      'Front_Size': len(F)})

        obj_df = pd.DataFrame(
            F, columns=['Obj_RMSE_norm', 'Obj_PBIAS_norm'])
        obj_df.insert(0, 'Run_ID', run_id + 1)
        ho = not os.path.exists(merged_obj)
        obj_df.to_csv(merged_obj, mode='a', header=ho, index=False,
                      encoding="utf-8-sig")

        if r['par_rows']:
            all_par_rows.extend(r['par_rows'])
        if r['best_row'] is not None:
            run_param_rows.append(r['best_row'])

        print("Run %02d: front size = %d" % (run_id + 1, len(F)))

    if not all_F:
        raise RuntimeError("All MOPSO runs failed.")

    if all_par_rows:
        pd.DataFrame(all_par_rows).to_csv(
            os.path.join(output_dir, "pareto_params_all.csv"),
            index=False, encoding="utf-8-sig")

    ref_front = merge_reference_front(all_F)
    all_FF = np.vstack(all_F)
    ref_point = all_FF.max(axis=0) * 1.1

    for row, F in zip(qrows, all_F):
        row['HV'] = hypervolume_2d(F, ref_point)
        row['Spacing'] = spacing_metric(F)
        row['IGD'] = igd_metric(F, ref_front)
        row['CenterDist'] = front_center_distance(F, ref_front)

    qdf = pd.DataFrame(qrows)
    qdf.to_csv(os.path.join(output_dir, "pareto_quality_metrics.csv"),
               index=False, encoding="utf-8-sig")

    summary = []
    for col in ['HV', 'Spacing', 'IGD', 'CenterDist', 'Front_Size']:
        v = qdf[col].values.astype(float)
        m_ = float(np.mean(v))
        s_ = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        cv_ = s_ / m_ * 100.0 if abs(m_) > 1e-12 else np.nan
        if len(v) > 1:
            tc = stats.t.ppf(0.975, df=len(v) - 1)
            half = tc * s_ / np.sqrt(len(v))
        else:
            half = 0.0
        summary.append({'Metric': col, 'Mean': m_, 'Std': s_,
                        'CV_pct': cv_,
                        'CI_Lower_95': m_ - half,
                        'CI_Upper_95': m_ + half})
    sdf = pd.DataFrame(summary)
    sdf.to_csv(os.path.join(output_dir, "pareto_quality_summary.csv"),
               index=False, encoding="utf-8-sig")
    print("\nPareto quality summary:")
    print(sdf.to_string(index=False))

    save_individual_fronts(all_F, output_dir)
    save_pooled_front(ref_front, output_dir)

    if run_param_rows:
        rp = pd.DataFrame(run_param_rows)
        rp.to_csv(os.path.join(output_dir,
                                "multi_start_topsis_params.csv"),
                  index=False, encoding="utf-8-sig")
        param_cols = [c for c in rp.columns
                      if c not in ('Run_ID', 'Seed',
                                   'TOPSIS_Obj_RMSE_norm',
                                   'TOPSIS_Obj_PBIAS_norm',
                                   'TOPSIS_CV', 'TOPSIS_Score')]
        st = rp[param_cols].agg(
            ['mean', 'std', 'min', 'max']).T
        st.columns = ['Mean', 'Std', 'Min', 'Max']
        st['Range'] = st['Max'] - st['Min']
        st['CV_pct'] = st['Std'] / st['Mean'].abs() * 100.0
        st.to_csv(os.path.join(
            output_dir, "multi_start_topsis_param_summary.csv"),
            encoding="utf-8-sig")
        print("\nMulti-start TOPSIS parameter summary:")
        print(st.to_string())

    return qdf, sdf, ref_front, []


# =========================================================
# 12. U2: PARAMETER RESPONSE SURFACE
# =========================================================

def parameter_response_surface(curve_models, configs, calib_df, actual_all,
                               bounds, output_dir, n_grid=30, n_jobs=1):
    ensure(output_dir)
    keys = list(bounds.keys())
    if len(keys) != 2:
        raise ValueError(
            "parameter_response_surface requires exactly 2 params.")

    k1, k2 = keys
    x1 = np.linspace(bounds[k1][0], bounds[k1][1], n_grid)
    x2 = np.linspace(bounds[k2][0], bounds[k2][1], n_grid)

    c0 = pd.Timestamp('2010-01-01')
    c1 = pd.Timestamp('2014-12-01')
    actual_c = actual_all.loc[(actual_all.index >= c0)
                              & (actual_all.index <= c1)]

    tasks = [(v1, v2, k1, k2, curve_models, configs, calib_df, actual_c)
             for v1 in x1 for v2 in x2]

    print("Parameter response surface: %d grid points, n_jobs=%d"
          % (len(tasks), n_jobs))

    results = _par(n_jobs, tasks, _eval_grid_point)

    rows = [r for r in results if r is not None]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir,
                            "parameter_response_surface.csv"),
              index=False, encoding="utf-8-sig")
    print("Parameter response surface saved: %d valid points." % len(df))
    return df


# =========================================================
# 13. U3: DECISION UNCERTAINTY
# =========================================================

def topsis_preference_scan(pareto_objs, pareto_pars, param_names=None,
                           step=0.05, output_dir="decision",
                           save_every=20):
    ensure(output_dir)
    F = np.asarray(pareto_objs, dtype=float)
    P = np.asarray(pareto_pars, dtype=float)
    if param_names is None:
        param_names = ['Param_%d' % j for j in range(P.shape[1])]

    fmin, fmax = F.min(axis=0), F.max(axis=0)
    den = np.where(fmax - fmin > 1e-12, fmax - fmin, 1.0)
    Z = (F - fmin) / den

    rows = []
    csv_path = os.path.join(output_dir, "topsis_preference_scan.csv")
    grid = np.arange(0, 1.0001, step)

    for w1 in grid:
        w2 = 1.0 - w1
        if w2 < -1e-8 or w2 > 1 + 1e-8:
            continue
        w = np.array([w1, max(w2, 0.0)])
        if w.sum() <= 0:
            continue
        w = w / w.sum()

        W = Z * w
        ideal = W.min(axis=0)
        anti = W.max(axis=0)
        dp = np.linalg.norm(W - ideal, axis=1)
        dn = np.linalg.norm(W - anti, axis=1)
        sc = dn / (dp + dn + 1e-12)
        idx = int(np.argmax(sc))

        row = {'w_RMSE': w1, 'w_PBIAS': w2,
               'Best_Index': idx, 'Best_Score': sc[idx],
               'Obj_RMSE': F[idx, 0], 'Obj_PBIAS': F[idx, 1]}
        for j, nm in enumerate(param_names):
            row[nm] = P[idx, j]
        rows.append(row)

        if len(rows) % save_every == 0:
            pd.DataFrame(rows).to_csv(
                csv_path, index=False, encoding="utf-8-sig")

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    drift = df[param_names].agg(['mean', 'std', 'min', 'max']).T
    drift.columns = ['Mean', 'Std', 'Min', 'Max']
    drift['Range'] = drift['Max'] - drift['Min']
    drift.to_csv(os.path.join(output_dir, "topsis_param_drift.csv"),
                 encoding="utf-8-sig")
    print("\nTOPSIS preference drift on parameters:")
    print(drift.to_string())
    return df, drift


def decision_method_cross_check(pareto_objs, pareto_pars,
                                 param_names=None, output_dir="decision"):
    ensure(output_dir)
    F = np.asarray(pareto_objs, dtype=float)
    P = np.asarray(pareto_pars, dtype=float)
    if param_names is None:
        param_names = ['Param_%d' % j for j in range(P.shape[1])]

    fmin, fmax = F.min(axis=0), F.max(axis=0)
    den = np.where(fmax - fmin > 1e-12, fmax - fmin, 1.0)
    Z = (F - fmin) / den

    res = {}

    w = np.array([1, 1]) / 2
    W = Z * w
    ideal = W.min(axis=0); anti = W.max(axis=0)
    dp = np.linalg.norm(W - ideal, axis=1)
    dn = np.linalg.norm(W - anti, axis=1)
    sc = dn / (dp + dn + 1e-12)
    res['TOPSIS_Equal'] = int(np.argmax(sc))

    Pm = Z + 1e-12
    Pm = Pm / Pm.sum(axis=0, keepdims=True)
    k = 1.0 / np.log(len(Z) + 1e-12)
    E = -k * (Pm * np.log(Pm + 1e-12)).sum(axis=0)
    d = 1 - E
    we = d / d.sum()
    W = Z * we
    ideal = W.min(axis=0); anti = W.max(axis=0)
    dp = np.linalg.norm(W - ideal, axis=1)
    dn = np.linalg.norm(W - anti, axis=1)
    sc = dn / (dp + dn + 1e-12)
    res['TOPSIS_Entropy'] = int(np.argmax(sc))

    w = np.array([1, 1]) / 2
    fs = Z.min(axis=0); fn = Z.max(axis=0)
    span = fn - fs + 1e-12
    S = ((Z - fs) / span * w).sum(axis=1)
    R = ((Z - fs) / span * w).max(axis=1)
    v = 0.5
    Q = (v * (S - S.min()) / (S.max() - S.min() + 1e-12)
         + (1 - v) * (R - R.min()) / (R.max() - R.min() + 1e-12))
    res['VIKOR'] = int(np.argmin(Q))

    ref = Z.min(axis=0)
    diff = np.abs(Z - ref)
    rho = 0.5
    grc = ((diff.min(axis=0) + rho * diff.max(axis=0))
           / (diff + rho * diff.max(axis=0) + 1e-12))
    grd = grc.mean(axis=1)
    res['GRA'] = int(np.argmax(grd))

    try:
        n, m = Z.shape
        w = np.array([1, 1]) / 2
        q_th = 0.05 * (Z.max(axis=0) - Z.min(axis=0) + 1e-12)
        conc = np.zeros((n, n))
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                c_w = 0.0
                for j in range(m):
                    diffj = Z[b, j] - Z[a, j]
                    if diffj >= -q_th[j]:
                        c_w += w[j]
                conc[a, b] = c_w
        net = conc.sum(axis=1) / n - conc.sum(axis=0) / n
        res['Simplified_ELECTRE'] = int(np.argmax(net))
    except Exception as e:
        print("Simplified ELECTRE skipped: %s" % e)

    rows = []
    for meth, idx in res.items():
        r = {'Method': meth, 'Selected_Index': idx,
             'Obj_RMSE': F[idx, 0], 'Obj_PBIAS': F[idx, 1]}
        for j, nm in enumerate(param_names):
            r[nm] = P[idx, j]
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir,
                            "decision_method_comparison.csv"),
              index=False, encoding="utf-8-sig")
    print("\nDecision method comparison:")
    print(df.to_string(index=False))
    return df


# =========================================================
# 14. U4: BLOCK BOOTSTRAP
# =========================================================

def block_bootstrap_replicates(series, block_len=12,
                                n_replicates=30, seed=42):
    """Legacy moving-block bootstrap (kept as fallback)."""
    rng = np.random.default_rng(seed)
    vals = series.values
    n = len(vals)
    nb = int(np.ceil(n / block_len))
    reps = []
    for _ in range(n_replicates):
        starts = rng.integers(0, max(n - block_len, 1), size=nb)
        idx = np.concatenate([np.arange(s, s + block_len)
                              for s in starts])
        idx = idx[:n]
        reps.append(pd.Series(vals[idx], index=series.index))
    return reps


def hydrological_year_bootstrap_replicates(series, n_replicates=30,
                                            seed=42):
    """
    Complete hydrological-year block bootstrap:
      - Retain only years with 12 complete months
      - Sample n_years years with replacement
      - Concatenate, then truncate to original series length
    Preserves intra-annual seasonality of the observed series.
    """
    rng = np.random.default_rng(seed)
    series = series.sort_index()

    years = sorted(series.index.year.unique())
    year_blocks = {
        y: series[series.index.year == y].values.copy()
        for y in years}
    year_blocks = {y: v for y, v in year_blocks.items() if len(v) == 12}

    valid_years = list(year_blocks.keys())
    if not valid_years:
        raise ValueError("No complete 12-month years for bootstrap.")

    n_years = len(valid_years)
    reps = []
    for _ in range(n_replicates):
        sampled_years = rng.choice(valid_years, size=n_years,
                                   replace=True)
        vals = np.concatenate([year_blocks[y] for y in sampled_years])
        vals = vals[:len(series)]
        reps.append(pd.Series(vals, index=series.index,
                              name=series.name))
    return reps


def run_bootstrap_analysis(MOPSOClass, bounds, calib_evaluator,
                           calib_obs_series, output_dir,
                           n_replicates=30, block_len=12,
                           seed_base=42,
                           n_jobs_outer=1, n_jobs_inner=1):
    ensure(output_dir)

    reps = hydrological_year_bootstrap_replicates(
        calib_obs_series,
        n_replicates=n_replicates, seed=seed_base)

    tasks = [(rid, reps[rid - 1], MOPSOClass, bounds, output_dir,
              n_jobs_inner, calib_evaluator, seed_base)
             for rid in range(1, n_replicates + 1)]

    print("Bootstrap: %d replicates, n_jobs_outer=%d, n_jobs_inner=%d"
          % (n_replicates, n_jobs_outer, n_jobs_inner))

    results = _par(n_jobs_outer, tasks, _run_one_bootstrap)

    all_records = [r for r in results if r is not None]
    if not all_records:
        raise RuntimeError("All bootstrap replicates failed.")

    all_F = [r['F'] for r in all_records]
    all_FF = np.vstack(all_F)
    ref_point = all_FF.max(axis=0) * 1.10

    rows = []
    for r in all_records:
        F = r['F']
        row = {k: v for k, v in r.items() if k != 'F'}
        row['HV'] = hypervolume_2d(F, ref_point)
        row['Spacing'] = spacing_metric(F)
        row['Front_Size'] = len(F)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "bootstrap_hv.csv"),
              index=False, encoding="utf-8-sig")

    param_cols = [c for c in df.columns if c.startswith('Best_')]
    if param_cols:
        st_rows = []
        for c in param_cols:
            v = df[c].values.astype(float)
            v = v[np.isfinite(v)]
            if len(v) < 2:
                continue
            m_ = float(np.mean(v))
            s_ = float(np.std(v, ddof=1))
            q025 = float(np.quantile(v, 0.025))
            q50 = float(np.quantile(v, 0.50))
            q975 = float(np.quantile(v, 0.975))
            st_rows.append({
                'Parameter': c.replace('Best_', ''),
                'Mean': m_,
                'Median': q50,
                'Std': s_,
                'CI_Lower_95': q025,
                'CI_Upper_95': q975,
                'Min': float(np.min(v)),
                'Max': float(np.max(v))})
        pd.DataFrame(st_rows).to_csv(
            os.path.join(output_dir, "bootstrap_param_ci.csv"),
            index=False, encoding="utf-8-sig")
        print("\nBootstrap parameter CI (percentile 2.5% - 97.5%):")
        print(pd.DataFrame(st_rows).to_string(index=False))

    summary = []
    for col in ['HV', 'Spacing', 'TOPSIS_Obj_RMSE',
                'TOPSIS_Obj_PBIAS', 'TOPSIS_CV']:
        if col not in df.columns:
            continue
        v = df[col].values.astype(float)
        v = v[np.isfinite(v)]
        if len(v) < 2:
            continue
        m_ = float(np.mean(v))
        s_ = float(np.std(v, ddof=1))
        q025 = float(np.quantile(v, 0.025))
        q50 = float(np.quantile(v, 0.50))
        q975 = float(np.quantile(v, 0.975))
        summary.append({'Metric': col, 'Mean': m_, 'Median': q50,
                        'Std': s_,
                        'CI_Lower_95': q025,
                        'CI_Upper_95': q975})
    pd.DataFrame(summary).to_csv(
        os.path.join(output_dir, "bootstrap_quality_summary.csv"),
        index=False, encoding="utf-8-sig")
    print("\nBootstrap summary (percentile 2.5% - 97.5%):")
    print(pd.DataFrame(summary).to_string(index=False))

    return df


# =========================================================
# 15. U4b: PREDICTION UNCERTAINTY BAND  [FIX-7]
# =========================================================

def compute_picp_mpiw(obs, lower, upper):
    obs = np.asarray(obs, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    m = np.isfinite(obs) & np.isfinite(lower) & np.isfinite(upper)
    if not np.any(m):
        return np.nan, np.nan
    inside = (obs[m] >= lower[m]) & (obs[m] <= upper[m])
    picp = float(np.mean(inside))
    mpiw = float(np.mean(upper[m] - lower[m]))
    return picp, mpiw


def prediction_uncertainty_band(curve_models, configs, inflow_df,
                                actual_all, param_sets, output_dir,
                                val_start='2015-01',
                                val_end='2019-12',
                                cal_start='2010-01',
                                cal_end='2014-12',
                                n_jobs=1):
    """
    Prediction uncertainty band with residual-based expansion.

    [FIX-7] Two-component uncertainty:
      (1) Parameter uncertainty: bootstrap ensemble spread
      (2) Residual uncertainty: calibration-period residual std
    The final band is:
      P05_adj = P05 - z90 * sigma_resid
      P95_adj = P95 + z90 * sigma_resid
    where z90 = 1.645 (for 90% normal interval).
    """
    ensure(output_dir)
    tasks = [(i, params, curve_models, configs, inflow_df, output_dir)
             for i, params in enumerate(param_sets, start=1)]

    print("Prediction band: %d ensemble members, n_jobs=%d"
          % (len(tasks), n_jobs))

    results = _par(n_jobs, tasks, _run_one_ensemble_member)
    sim_list = [r for r in results if r is not None]
    if not sim_list:
        raise RuntimeError(
            "No valid ensemble members for prediction band.")

    sim_df = pd.concat(sim_list, axis=1)
    sim_df.columns = ['M%03d' % (i + 1) for i in range(sim_df.shape[1])]
    sim_df.to_csv(os.path.join(
        output_dir, "ensemble_monthly_generation.csv"),
        encoding="utf-8-sig")

    # --- Component 1: parameter uncertainty (raw quantiles) ---
    q05_raw = sim_df.quantile(0.05, axis=1)
    q25_raw = sim_df.quantile(0.25, axis=1)
    q50 = sim_df.quantile(0.50, axis=1)
    q75_raw = sim_df.quantile(0.75, axis=1)
    q95_raw = sim_df.quantile(0.95, axis=1)

    # --- Component 2: residual uncertainty from calibration period ---
    # Use the ensemble median over the calibration period,
    # compute residual std vs observations, then expand band.
    cal_idx = q50.loc[cal_start:cal_end].index
    obs_cal = actual_all.loc[cal_start:cal_end]
    common_cal = cal_idx.intersection(obs_cal.index)

    if len(common_cal) >= 6:
        resid = (obs_cal.loc[common_cal].values
                 - q50.loc[common_cal].values)
        sigma_resid = float(np.std(resid, ddof=1))
    else:
        sigma_resid = 0.0

    # z for 90% two-sided interval
    z90 = 1.6448536269514722
    half_resid = z90 * sigma_resid

    # Save residual info
    pd.DataFrame([{
        'Sigma_resid': sigma_resid,
        'Z90': z90,
        'Half_width_resid': half_resid,
        'N_cal_months': len(common_cal)}]).to_csv(
        os.path.join(output_dir, "prediction_band_residual.csv"),
        index=False, encoding="utf-8-sig")

    print("  Residual sigma = %.4f, Half width (90%%) = %.4f"
          % (sigma_resid, half_resid))

    # --- Final adjusted band ---
    q05 = q05_raw - half_resid
    q25 = q25_raw - 0.5 * half_resid
    q75 = q75_raw + 0.5 * half_resid
    q95 = q95_raw + half_resid

    # Also save raw band for comparison
    band = pd.DataFrame({
        'P05': q05, 'P25': q25, 'P50': q50,
        'P75': q75, 'P95': q95,
        'P05_raw': q05_raw, 'P95_raw': q95_raw})
    band.index.name = 'Date'
    band.to_csv(os.path.join(
        output_dir, "prediction_band_quantiles.csv"),
        encoding="utf-8-sig")

    # --- Validation metrics ---
    obs_val = actual_all.loc[val_start:val_end]
    common_val = band.index.intersection(obs_val.index)
    if len(common_val) > 0:
        picp90, mpiw90 = compute_picp_mpiw(
            obs_val.loc[common_val].values,
            band.loc[common_val, 'P05'].values,
            band.loc[common_val, 'P95'].values)
        picp50, mpiw50 = compute_picp_mpiw(
            obs_val.loc[common_val].values,
            band.loc[common_val, 'P25'].values,
            band.loc[common_val, 'P75'].values)

        # Raw band metrics for reference
        picp90_raw, mpiw90_raw = compute_picp_mpiw(
            obs_val.loc[common_val].values,
            band.loc[common_val, 'P05_raw'].values,
            band.loc[common_val, 'P95_raw'].values)

        sim_med = band.loc[common_val, 'P50'].values
        obs_v = obs_val.loc[common_val].values
        mval = compute_all_metrics(sim_med, obs_v)

        pd.DataFrame([{
            'N_Ensemble': len(sim_list),
            'N_Months_Val': len(common_val),
            'PICP_90': picp90, 'MPIW_90': mpiw90,
            'PICP_50': picp50, 'MPIW_50': mpiw50,
            'PICP_90_raw': picp90_raw,
            'MPIW_90_raw': mpiw90_raw,
            'Sigma_resid': sigma_resid,
            'NSE_P50': mval['NSE'] if mval else np.nan,
            'KGE_P50': mval['KGE'] if mval else np.nan,
            'PBIAS_P50': mval['PBIAS'] if mval else np.nan,
        }]).to_csv(os.path.join(
            output_dir, "prediction_band_metrics.csv"),
            index=False, encoding="utf-8-sig")
        print("Prediction band metrics saved:")
        print("  Raw:       PICP_90=%.3f, MPIW_90=%.3f"
              % (picp90_raw, mpiw90_raw))
        print("  Adjusted:  PICP_90=%.3f, MPIW_90=%.3f"
              % (picp90, mpiw90))

    obs_cal_full = actual_all.loc[cal_start:cal_end]
    common_cal_full = band.index.intersection(obs_cal_full.index)
    if len(common_cal_full) > 0:
        sim_med_c = band.loc[common_cal_full, 'P50'].values
        obs_c = obs_cal_full.loc[common_cal_full].values
        mcal = compute_all_metrics(sim_med_c, obs_c)
        pd.DataFrame([{
            'N_Months_Cal': len(common_cal_full),
            'NSE_P50': mcal['NSE'] if mcal else np.nan,
            'KGE_P50': mcal['KGE'] if mcal else np.nan,
            'PBIAS_P50': mcal['PBIAS'] if mcal else np.nan,
        }]).to_csv(os.path.join(
            output_dir, "prediction_band_metrics_cal.csv"),
            index=False, encoding="utf-8-sig")

    return band


# =========================================================
# 16. U5: DEAD LEVEL OAT
# =========================================================

def dead_level_sensitivity_oat(curve_models, configs_base, inflow_df,
                               actual_all, best_params, output_dir,
                               delta=5.0, n_jobs=1):
    ensure(output_dir)
    names = ["Chahanwusu", "Liushugou", "Dashankou"]
    base_levels = {
        "Chahanwusu": configs_base[0].get("dead_water_level", 80.0),
        "Liushugou": configs_base[1].get("dead_water_level", 76.0),
        "Dashankou": configs_base[2].get("dead_water_level", 53.0)}

    scenarios = [("Base", dict(base_levels))]
    for nm in names:
        for sign, tag in [(-1, "-5"), (+1, "+5")]:
            scen = dict(base_levels)
            scen[nm] = base_levels[nm] + sign * delta
            scenarios.append(("%s%s" % (nm, tag), scen))

    tasks = [(tag, dl, names, configs_base, curve_models,
              inflow_df, actual_all, best_params, output_dir)
             for tag, dl in scenarios]

    print("Dead level OAT: %d scenarios, n_jobs=%d"
          % (len(tasks), n_jobs))

    results = _par(n_jobs, tasks, _run_one_dead_level)
    rows = [r for r in results if r is not None]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir,
                            "dead_level_sensitivity_oat.csv"),
              index=False, encoding="utf-8-sig")

    if 'Base' in df['Scenario'].values:
        base_row = df[df['Scenario'] == 'Base'].iloc[0]
        tor = []
        for _, r in df.iterrows():
            if r['Scenario'] == 'Base':
                continue
            tor.append({'Scenario': r['Scenario'],
                        'Delta_Val_NSE':
                            r['Val_NSE'] - base_row['Val_NSE'],
                        'Delta_Val_KGE':
                            r['Val_KGE'] - base_row['Val_KGE'],
                        'Delta_Val_PBIAS':
                            r['Val_PBIAS'] - base_row['Val_PBIAS']})
        pd.DataFrame(tor).to_csv(
            os.path.join(output_dir, "dead_level_tornado.csv"),
            index=False, encoding="utf-8-sig")

    print("\nDead level sensitivity (OAT):")
    print(df.to_string(index=False))
    return df


# =========================================================
# 17. U6: WET / NORMAL / DRY  [FIX-8]
# =========================================================

def wet_dry_stability(monthly_sim, actual_all, inflow_df, output_dir):
    """
    [FIX-8] Wet / Normal / Dry classification is now restricted
    to years that have observed generation data, so that the
    classification and the evaluation are on the same period.
    """
    ensure(output_dir)

    # Physical annual volume (m^3)
    tmp = inflow_df.copy()
    tmp["Annual_Volume_m3"] = tmp["Data"] * 86400.0
    annual_inflow_full = tmp.groupby(
        tmp["Date"].dt.year)["Annual_Volume_m3"].sum()

    # [FIX-8] Restrict to observed years
    obs_years = set(actual_all.index.year.unique())
    annual_inflow = annual_inflow_full[
        annual_inflow_full.index.isin(obs_years)]

    if len(annual_inflow) < 3:
        print("[WARN] Too few observed years for wet/normal/dry split.")
        df = pd.DataFrame()
        df.to_csv(os.path.join(output_dir, "wet_dry_stability.csv"),
                  index=False, encoding="utf-8-sig")
        return df

    q33 = annual_inflow.quantile(0.33)
    q67 = annual_inflow.quantile(0.67)

    wet_years = annual_inflow[annual_inflow >= q67].index.tolist()
    normal_years = annual_inflow[
        (annual_inflow > q33) & (annual_inflow < q67)].index.tolist()
    dry_years = annual_inflow[annual_inflow <= q33].index.tolist()

    print("  Observed years used for classification: %d"
          % len(annual_inflow))
    print("  q33 = %.3e m3, q67 = %.3e m3" % (q33, q67))

    rows = []
    for tag, years in [("Wet", wet_years),
                       ("Normal", normal_years),
                       ("Dry", dry_years)]:
        if not years:
            continue
        sim_v = monthly_sim[monthly_sim.index.year.isin(years)]
        obs_v = actual_all[actual_all.index.year.isin(years)]
        common = sim_v.index.intersection(obs_v.index)
        if len(common) < 6:
            continue
        m = compute_all_metrics(sim_v.loc[common].values,
                                obs_v.loc[common].values)
        if m is None:
            continue
        rows.append({'Scenario': tag,
                     'Years': ','.join(map(str, years)),
                     'N_Months': len(common),
                     'NSE': m['NSE'], 'KGE': m['KGE'],
                     'PBIAS': m['PBIAS']})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "wet_dry_stability.csv"),
              index=False, encoding="utf-8-sig")
    print("\nWet/Normal/Dry stability (observed years only):")
    print(df.to_string(index=False))
    return df


# =========================================================
# 18. MAIN
# =========================================================

if __name__ == "__main__":

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR = "results_kaidu_%s" % ts
    ensure(RESULTS_DIR)

    print("=" * 70)
    print("KAIDU - PURE HYDROPOWER + 2-OBJ MOPSO + UNCERTAINTY "
          "(PARALLEL + FIXED)")
    print("Output: %s/" % RESULTS_DIR)
    print("=" * 70)

    CURVE_PATHS = [
        "chahanwusu/inverse_ch.xlsx",
        "liushugou/inverse_liushugou.xlsx",
        "dashankou/inverse_dashankou.xlsx"]
    RESERVOIR_NAMES = ["Chahanwusu", "Liushugou", "Dashankou"]
    INFLOW_PATH = "hebin.xlsx"
    ACTUAL_PATH = "dianliang.xlsx"

    bounds = {
        "normal_zone_power_fraction": (0.20, 0.90),
        "storage_ratio": (0.05, 0.80)}

    eco_up = {1: 8.0, 2: 8.0, 3: 8.0, 4: 18.0,
              5: 22.57, 6: 22.57, 7: 22.57, 8: 22.57,
              9: 18.0, 10: 12.0, 11: 8.0, 12: 8.0}
    eco_down = {1: 4.0, 2: 4.0, 3: 6.0, 4: 9.0,
                5: 11.20, 6: 11.20, 7: 11.20, 8: 11.20,
                9: 9.0, 10: 6.0, 11: 4.0, 12: 4.0}

    configs = [
        {"name": "Chahanwusu",
         "impoundment_start_date": "2007-01-01",
         "generation_start_date": "2007-01-01",
         "min_eco_release_m3s": 22.57,
         "monthly_min_eco_release_m3s": eco_up,
         "flood_trigger_inflow_m3s": 220.0,
         "installed_capacity_kw": 330000,
         "tailwater": 5, "efficiency": 0.90, "initial_volume": 0,
         "dead_water_level": 80.0, "spill_level": 97.5,
         "monthly_normal_high": {1: 96.5, 2: 96.5, 3: 96.5,
                                 4: 96.0, 5: 95.5, 6: 96.0,
                                 7: 96.0, 8: 96.0, 9: 95.5,
                                 10: 96.0, 11: 96.5, 12: 96.5},
         "monthly_normal_low": {1: 92.0, 2: 92.0, 3: 92.0,
                                4: 91.5, 5: 91.0, 6: 91.0,
                                7: 91.5, 8: 91.5, 9: 91.5,
                                10: 92.0, 11: 92.0, 12: 92.0},
         "monthly_flood_control_level": {
             1: 96.0, 2: 96.0, 3: 96.0, 4: 95.5,
             5: 95.0, 6: 95.0, 7: 95.0, 8: 95.0,
             9: 95.0, 10: 95.5, 11: 96.0, 12: 96.0},
         "active_drawdown_enabled": True,
         "active_drawdown_rate": 150.0,
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
         "dead_water_level": 76.0, "spill_level": 93.5,
         "monthly_normal_high": {1: 92.5, 2: 92.5, 3: 92.5,
                                 4: 92.0, 5: 91.5, 6: 92.0,
                                 7: 92.0, 8: 92.0, 9: 91.5,
                                 10: 92.0, 11: 92.5, 12: 92.5},
         "monthly_normal_low": {1: 89.5, 2: 89.5, 3: 89.5,
                                4: 89.0, 5: 88.5, 6: 88.5,
                                7: 89.0, 8: 89.0, 9: 89.0,
                                10: 89.5, 11: 89.5, 12: 89.5},
         "monthly_flood_control_level": {
             1: 92.0, 2: 92.0, 3: 92.0, 4: 91.5,
             5: 91.0, 6: 91.0, 7: 91.0, 8: 91.0,
             9: 91.0, 10: 91.5, 11: 92.0, 12: 92.0},
         "active_drawdown_enabled": True,
         "active_drawdown_rate": 120.0,
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
         "dead_water_level": 53.0, "spill_level": 68.0,
         "monthly_normal_high": {1: 67.5, 2: 67.5, 3: 67.5,
                                 4: 67.0, 5: 66.5, 6: 67.0,
                                 7: 67.0, 8: 67.0, 9: 66.5,
                                 10: 67.0, 11: 67.5, 12: 67.5},
         "monthly_normal_low": {1: 63.0, 2: 63.0, 3: 63.0,
                                4: 62.5, 5: 62.0, 6: 62.0,
                                7: 62.5, 8: 62.5, 9: 62.5,
                                10: 63.0, 11: 63.0, 12: 63.0},
         "monthly_flood_control_level": {
             1: 67.0, 2: 67.0, 3: 67.0, 4: 67.0,
             5: 66.5, 6: 66.0, 7: 66.0, 8: 66.0,
             9: 66.5, 10: 67.0, 11: 67.0, 12: 67.0},
         "active_drawdown_enabled": True,
         "active_drawdown_rate": 100.0,
         "use_pid": True, "pid_Kp": 0.15, "pid_Ki": 0.01,
         "pid_Kd": 0.05, "pid_target_offset": 0.6,
         "max_power_fraction": 0.90, "min_power_fraction": 0.20,
         "high_buffer": 1.0, "storage_ratio": 0.30,
         "interpolate_targets": True,
         "routing_coeff": 0.98, "travel_time": 0,
         "interbasin_flow_ratio": 0.0},
    ]

    print("\nLoading reservoir curves...")
    curve_models = [ReservoirCurveModel(CURVE_PATHS[i],
                                        RESERVOIR_NAMES[i])
                    for i in range(3)]

    print("\nLoading inflow and actual power data...")
    inflow_df = load_inflow_data(INFLOW_PATH)
    actual_all = load_actual_power(ACTUAL_PATH)
    print("Data loaded.")

    calib_inflow_df = inflow_df[
        (inflow_df["Date"] >= "2008-01-01") &
        (inflow_df["Date"] <= "2014-12-31")].copy()

    cascade_op = CascadeReservoirOperation(curve_models, configs,
                                            verbose=True)

    def fitness_mopso(params):
        return evaluate_mopso_detailed(cascade_op, calib_inflow_df,
                                       actual_all, params)

    # ---------------- MAIN MOPSO ----------------
    print("\n" + "=" * 70)
    print("MAIN MOPSO (pop=40, iter=50, 2 objectives, parallel)")
    print("=" * 70)
    mopso = MultiObjectivePSO(
        bounds=bounds, pop_size=40, max_iter=50,
        archive_size=80, output_dir=RESULTS_DIR, seed=42,
        n_jobs=N_JOBS_GLOBAL)
    pareto_archive = mopso.optimize(fitness_mopso, verbose=True)
    if not pareto_archive:
        raise RuntimeError("Empty Pareto archive.")

    best_solution = select_topsis_solution(pareto_archive)
    best_params = best_solution['params']

    print("\nBest parameters:")
    for k, v in best_params.items():
        print("  %s: %.6f" % (k, v))
    print("  RMSE_norm=%.6f  PBIAS_norm=%.6f  CV=%.6f  TOPSIS=%.6f"
          % (best_solution['objectives'][0],
             best_solution['objectives'][1],
             best_solution['constraint_violation'],
             best_solution['TOPSIS_Score']))

    # ---------------- FULL SIMULATION ----------------
    print("\n" + "=" * 70)
    print("FULL SIMULATION")
    print("=" * 70)
    final_op = CascadeReservoirOperation(curve_models, configs,
                                          verbose=True)
    final_op.update_calibration_params(best_params, reservoir_idx=2)
    final_op.reset_simulation_state()
    result_full, reservoir_daily_dfs = final_op.simulate_cascade_full(
        inflow_df, output_dir=RESULTS_DIR, write_files=True)

    downstream = result_full['R3_Dashankou_Power_MW']
    monthly_sim = downstream.resample('ME').sum() * 24.0 / 10.0
    monthly_sim.index = monthly_sim.index.to_period('M').to_timestamp()

    cal_sim = monthly_sim.loc['2010-01':'2014-12']
    cal_obs = actual_all.loc['2010-01':'2014-12']
    cal_c = cal_sim.index.intersection(cal_obs.index)
    cal_m = compute_all_metrics(cal_sim.loc[cal_c].values,
                                cal_obs.loc[cal_c].values)

    val_sim = monthly_sim.loc['2015-01':'2019-12']
    val_obs = actual_all.loc['2015-01':'2019-12']
    val_c = val_sim.index.intersection(val_obs.index)
    val_m = compute_all_metrics(val_sim.loc[val_c].values,
                                val_obs.loc[val_c].values)

    print("\nCalibration (2010-2014):")
    if cal_m:
        for k, v in cal_m.items():
            print("  %s: %.4f" % (k, v))
    print("\nValidation (2015-2019):")
    if val_m:
        for k, v in val_m.items():
            print("  %s: %.4f" % (k, v))

    if cal_m and val_m:
        save_df(pd.DataFrame({
            'Metric': list(cal_m.keys()),
            'Calibration': list(cal_m.values()),
            'Validation': [val_m.get(k, np.nan)
                           for k in cal_m.keys()]}),
            os.path.join(RESULTS_DIR, "validation_metrics.csv"))

    save_df(pd.DataFrame([{
        **best_params,
        'Objective_RMSE_norm': best_solution['objectives'][0],
        'Objective_PBIAS_norm': best_solution['objectives'][1],
        'Constraint_Violation': best_solution['constraint_violation'],
        'TOPSIS_Score': best_solution['TOPSIS_Score'],
        'FEASIBLE_TOL_Final': MultiObjectivePSO.FEASIBLE_TOL,
        **best_solution['components']}]),
        os.path.join(RESULTS_DIR, "mopso_topsis_best_solution.csv"))

    # =====================================================
    # UNCERTAINTY ANALYSIS
    # =====================================================
    UD = ensure(os.path.join(RESULTS_DIR, "uncertainty"))

    pareto_objs = np.vstack([x['objectives'] for x in pareto_archive])
    pareto_pars = np.array([[x['params'][k] for k in bounds.keys()]
                            for x in pareto_archive])

    cal_obs_for_boot = actual_all.loc['2010-01':'2014-12']

    def evaluator_calib(params, series):
        return evaluate_mopso_detailed(cascade_op, calib_inflow_df,
                                       series, params)

    # ---------------- U1 ----------------
    print("\n" + "=" * 70)
    print("U1: Multi-start MOPSO (10 runs, outer+inner parallel)")
    print("=" * 70)
    n_runs_u1 = 10
    n_outer_u1 = min(n_runs_u1, 10)
    n_inner_u1 = max(1, N_JOBS_GLOBAL // n_outer_u1)
    qdf, sdf, ref_front_ms, all_arch_ms = run_multi_start_mopso(
        MultiObjectivePSO, bounds, n_runs=n_runs_u1, seed_base=42,
        evaluator=fitness_mopso, output_dir=UD,
        n_jobs_outer=n_outer_u1, n_jobs_inner=n_inner_u1)

    # ---------------- U2 ----------------
    print("\n" + "=" * 70)
    print("U2: Parameter response surface (parallel grid)")
    print("=" * 70)
    parameter_response_surface(
        curve_models, configs, calib_inflow_df, actual_all,
        bounds, output_dir=os.path.join(UD, "param_surface"),
        n_grid=30, n_jobs=N_JOBS_GLOBAL)

    # ---------------- U3 ----------------
    print("\n" + "=" * 70)
    print("U3: Decision uncertainty")
    print("=" * 70)
    topsis_preference_scan(pareto_objs, pareto_pars,
                           param_names=list(bounds.keys()),
                           output_dir=os.path.join(UD, "decision"))
    decision_method_cross_check(pareto_objs, pareto_pars,
                                 param_names=list(bounds.keys()),
                                 output_dir=os.path.join(UD, "decision"))

    # ---------------- U4 ----------------
    print("\n" + "=" * 70)
    print("U4: Complete-year bootstrap (parallel)")
    print("=" * 70)
    n_rep_u4 = 30
    n_outer_u4 = min(n_rep_u4, 10)
    n_inner_u4 = max(1, N_JOBS_GLOBAL // n_outer_u4)
    boot_df = run_bootstrap_analysis(
        MultiObjectivePSO, bounds,
        calib_evaluator=evaluator_calib,
        calib_obs_series=cal_obs_for_boot,
        output_dir=os.path.join(UD, "bootstrap"),
        n_replicates=n_rep_u4, block_len=12, seed_base=42,
        n_jobs_outer=n_outer_u4, n_jobs_inner=n_inner_u4)

    # ---------------- U4b ----------------
    print("\n" + "=" * 70)
    print("U4b: Prediction uncertainty band "
          "(Bootstrap params + residual expansion)")
    print("=" * 70)

    param_sets_for_band = []
    if ('Best_normal_zone_power_fraction' in boot_df.columns and
            'Best_storage_ratio' in boot_df.columns):
        for _, row in boot_df.iterrows():
            params = {
                "normal_zone_power_fraction":
                    float(row["Best_normal_zone_power_fraction"]),
                "storage_ratio":
                    float(row["Best_storage_ratio"])}
            param_sets_for_band.append(params)
        print("  Using %d Bootstrap-calibrated parameter sets."
              % len(param_sets_for_band))
    else:
        param_sets_for_band = [x['params'] for x in pareto_archive]
        print("  [WARN] Bootstrap params not found, "
              "fallback to Pareto archive (%d sets)."
              % len(param_sets_for_band))

    prediction_uncertainty_band(
        curve_models, configs, inflow_df, actual_all,
        param_sets_for_band,
        output_dir=os.path.join(UD, "prediction_band"),
        n_jobs=N_JOBS_GLOBAL)

    # ---------------- U5 ----------------
    print("\n" + "=" * 70)
    print("U5: Dead level OAT sensitivity (parallel)")
    print("=" * 70)
    dead_level_sensitivity_oat(
        curve_models, configs, inflow_df, actual_all, best_params,
        output_dir=os.path.join(UD, "dead_level"),
        delta=5.0, n_jobs=7)

    # ---------------- U6 ----------------
    print("\n" + "=" * 70)
    print("U6: Wet / Normal / Dry year stability "
          "(observed years only)")
    print("=" * 70)
    wet_dry_stability(monthly_sim, actual_all, inflow_df,
                      output_dir=os.path.join(UD, "wet_dry"))

    print("\n" + "=" * 70)
    print("ALL DONE. Results in: %s" % RESULTS_DIR)
    print("=" * 70)
    for root, dirs, files in os.walk(RESULTS_DIR):
        for f in sorted(files):
            print("  %s" % os.path.relpath(
                os.path.join(root, f), RESULTS_DIR))