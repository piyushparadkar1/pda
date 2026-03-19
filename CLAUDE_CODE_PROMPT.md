# SDA UNIT SIMULATOR — CLAUDE CODE HANDOFF
# Propane Deasphalting Digital Twin — Plant Calibration Framework
# ================================================================

## HOW TO USE THIS FILE

This file is the master engineering brief for Claude Code. It describes:
- What the project does (Section 1-2)
- Full call chain and file architecture (Section 3-4)
- The K-value physics engine (Section 5)
- Plant data targets from the Operating Manual (Section 6)
- What's already done (Section 7)
- What's broken and needs fixing — with root causes and fix directions (Section 8)
- Testing commands (Section 9)
- Coding standards (Section 10)

**Give Claude Code: this file + ALL .py/.html/.csv/.txt/.json files in this folder.**

**Start Claude Code with:**
```
Read CLAUDE_CODE_PROMPT.md first. Then fix Issues 1-5 in Section 8, 
in order. After each fix, run the test commands from Section 9. 
Start with Issue 1 (CCR formula) and Issue 2 (density model).
```

---

## 1. PROJECT OVERVIEW

Python-based Propane Deasphalting (PDA/SDA) unit simulator for a refinery. 
Simulates countercurrent liquid-liquid extraction of vacuum residue (VR) 
with propane solvent to produce De-Asphalted Oil (DAO) and asphalt.

- **Platform:** Python 3.9+ on Windows (VS Code terminal)
- **Web UI:** Flask + Plotly (dark-themed, single HTML template)
- **Entry point:** `python run_simulation.py` (launches web UI on port 5000)
- **Optimizer:** scipy.optimize.least_squares (TRF method)

The base simulation engine works. The calibration framework is ~80% done. 
The remaining ~20% is fixing two physics correlations (CCR, density) and 
wiring calibration params through sensitivity analysis.

---

## 2. FILE ARCHITECTURE

All files live in ONE flat directory (no subdirectories except calibration_profiles/).

```
pda_simulator_v3/
│
│── Core simulation modules (bottom-up dependency order):
│   ├── residue_distribution.py  — Feed characterisation: 20 SARA pseudo-components
│   ├── phct_eos.py              — PHCT EOS: density + fugacity (used by lle_solver)
│   ├── lle_solver.py            — K-values + Rachford-Rice flash  ★ MODIFIED
│   ├── asphaltene_kinetics.py   — First-order precipitation kinetics
│   ├── stage_efficiency.py      — Murphree efficiency correction
│   ├── entrainment_model.py     — Asphalt-into-DAO entrainment
│   └── hunter_nash_extractor.py — Countercurrent extractor main loop  ★ MODIFIED
│
│── Application layer:
│   ├── sensitivity_analysis.py  — Parameter sweeps + Plotly plots  ★ NEEDS FIX
│   ├── plant_calibration.py     — Calibration framework  ★ MODIFIED, NEEDS CCR FIX
│   ├── run_simulation.py        — CLI + Flask web server  ★ MODIFIED
│   └── ui_template.html         — Web UI (Simulation + Calibration tabs)  ★ NEW
│
│── Data:
│   ├── sample_plant_data.csv    — 14-point sample plant data  ★ NEEDS REGENERATION
│   ├── requirements.txt         — flask, plotly, scipy, numpy, pandas
│   ├── __init__.py              — Package init
│   └── calibration_profiles/    — JSON calibration profiles
│       ├── sda_default.json
│       ├── sda_lube_dao.json
│       └── sda_fcc_dao.json
│
│── Documentation:
│   ├── CLAUDE_CODE_PROMPT.md    — THIS FILE
│   └── CALIBRATION_GUIDE.md     — User-facing calibration guide
```

★ = Files I (Claude chat) modified or created. All others are original.

---

## 3. SIMULATION CALL CHAIN (read this to understand data flow)

```
User clicks "Run Simulation" in Web UI
  │
  ▼
run_simulation.py :: /api/simulate  (Flask endpoint)
  │  Reads: feed, solvent, S/O, T, stages, K_multiplier, delta_crit, 
  │         E_murphree, C_entrain, k_precip, tau
  │
  ├── residue_distribution.build_residue_distribution(feed_name, n_comp=20, solvent)
  │     → Creates List[PseudoComponent] with fields:
  │       index, MW, z (mole frac), density, delta (solubility param),
  │       sara_class, is_heavy, precipitable, eps_kb
  │     → 5 components per SARA class: saturates, aromatics, resins, asphaltenes
  │     → MW range: ~100 (light sat) to ~2800 (heavy asph)
  │
  └── hunter_nash_extractor.run_extractor(
        components, solvent_name, solvent_ratio, N_stages, T_profile,
        kinetics, efficiency, entrainment,
        K_multiplier, delta_crit)          ← calibration params
        │
        │  Outer loop: up to 60 iterations, damping=0.5
        │  For each stage s (countercurrent):
        │
        ├── lle_solver.solve_lle(components, T, P, solvent, S/O, feed_mass,
        │                        K_multiplier, delta_crit)
        │     │
        │     └── K_value(component, T, solvent, S/O, K_multiplier, delta_crit)
        │           Returns K_i = w_i^DAO / w_i^asphalt
        │           Uses: ln(K) = a + b*ln(MW) + C_T*(T-T_ref) + D_solv
        │           Then: K *= K_multiplier (yield) and SARA-specific delta_crit adjustment (quality)
        │     
        │     → Rachford-Rice flash → psi (DAO fraction), mass_I, mass_II
        │
        ├── asphaltene_kinetics.apply_precipitation_kinetics(A_prev, A_eq, params)
        │     → Exact ODE: A_stage = A_eq + (A_prev - A_eq)*exp(-k*tau)
        │
        ├── stage_efficiency.apply_stage_efficiency(mass_I_eq, mass_II_eq, mass_I_in, E)
        │     → Murphree: mass_I_actual = mass_I_in + E*(mass_I_eq - mass_I_in)
        │
        └── entrainment_model.apply_entrainment(mass_I, mass_II, S/O, params)
              → frac = C_entrain / S_O^n_exp, transfers asphalt into DAO
        
        Returns: DAO_yield_net, asphalt_yield, asphal_contam_pct, density_DAO,
                 MW_DAO_avg, SARA_DAO, stage_results, converged, etc.
```

**Calibration call chain:**
```
plant_calibration.run_calibration(dataset, weights)
  │
  └── scipy.optimize.least_squares(
        fun = _build_residuals(param_vec, dataset, weights, cache, history)
        │
        └── For each PlantDataPoint in dataset:
              _simulate_point(pt, params, comp_cache)
                │
                └── run_extractor(..., K_multiplier=params['K_multiplier'],
                                      delta_crit=params['delta_crit'],
                                      kinetics=KineticParams(params['k_precip']),
                                      efficiency=StageEfficiency(params['E_murphree']),
                                      entrainment=EntrainmentParams(params['C_entrain']))
                │
                └── Returns: {DAO_yield, DAO_density, DAO_CCR, asph_contam}
                     DAO_CCR is computed from SARA_DAO using a correlation ← THIS IS BROKEN
      )
```

---

## 4. THE 5 CALIBRATION PARAMETERS

| Parameter | Default | Bounds | Hooks into | What it controls |
|-----------|---------|--------|------------|------------------|
| K_multiplier | 1.00 | 0.30–3.50 | lle_solver.K_value() | Uniform K scaler → PRIMARY yield lever |
| delta_crit | 2.50 | 0.50–8.00 | lle_solver.K_value() | SARA-selective K penalty → DAO quality |
| E_murphree | 0.70 | 0.30–1.00 | StageEfficiency → stage_efficiency.py | Stage separation sharpness |
| C_entrain | 0.015 | 0.001–0.10 | EntrainmentParams → entrainment_model.py | Asphaltene contamination in DAO |
| k_precip | 0.50 | 0.05–5.00 | KineticParams → asphaltene_kinetics.py | Precipitation kinetics rate |

**Wiring (ALREADY DONE):**
- K_multiplier + delta_crit: `_simulate_point()` → `run_extractor(K_multiplier=, delta_crit=)` → `solve_lle(K_multiplier=, delta_crit=)` → `K_value(K_multiplier=, delta_crit=)`
- E_murphree: `_simulate_point()` → `StageEfficiency(E_murphree)` → `run_extractor(efficiency=)`
- C_entrain: `_simulate_point()` → `EntrainmentParams(C_entrain, 1.20)` → `run_extractor(entrainment=)`
- k_precip: `_simulate_point()` → `KineticParams(k_precip, 10.0)` → `run_extractor(kinetics=)`

---

## 5. K-VALUE MODEL (the core physics — do NOT change without reason)

```python
# In lle_solver.py :: K_value()
# Base K from SARA-class regression:
ln(K) = a + b*ln(MW) + C_T*(T - T_ref) + D_solvent

_K_PARAMS = {
    'saturates':   {'a':  9.868, 'b': -1.30},
    'aromatics':   {'a':  9.456, 'b': -1.30},
    'resins':      {'a':  8.817, 'b': -1.50},
    'asphaltenes': {'a':  6.372, 'b': -1.80},
}
_T_REF  = {'propane': 348.15, 'butane': 413.15}   # K (75°C / 140°C)
_C_T    = -0.007    # lnK per K  (yield drops ~0.7% per °C increase)
_D_SOLV = {'propane': 0.0, 'butane': +0.55}
so_factor = (max(solvent_ratio, 0.5) / 8.0) ** 0.50

# Then calibration adjustments (ALREADY IMPLEMENTED):
K_base = exp(lnK) * so_factor
K = K_base * K_multiplier                                # yield scaler
if sara == 'resins':      K *= exp(-0.10 * (delta_crit - 2.5))
if sara == 'aromatics':   K *= exp(-0.03 * (delta_crit - 2.5))
if sara == 'asphaltenes': K *= exp(-0.05 * (delta_crit - 2.5))
```

**Verified behavior:**
- K_multiplier=0.65, T=60°C → DAO=18% (matches lube target)
- K_multiplier=1.05, T=55°C → DAO=33% (matches FCC target)
- K_multiplier monotonically controls yield ✓
- delta_crit monotonically shifts resin/aromatic partitioning ✓

---

## 6. PLANT DATA TARGETS (Operating Manual Chapter 4)

### Feed: 70/30 Basra-Kuwait Mix VR (Design Case)
| Property | Value | Internal key |
|----------|-------|-------------|
| SG @15.5°C | 1.028 | basra_kuwait_mix |
| API | 6.1 | |
| CCR | 22.8 wt% | |
| Sulphur | 5.0 wt% | |
| Nickel | 28 wppm | |
| Vanadium | 104 wppm | |
| Visc @100°C | 1621 cSt | |
| Visc @135°C | 230 cSt | |

### Feed: Basra Light VR (Check Case)
| Property | Value | Internal key |
|----------|-------|-------------|
| SG @15.5°C | 1.026 | basra_light |
| API | 6.5 | |
| CCR | 22.6 wt% | |
| Visc @100°C | 1137 cSt | |

### DAO Product Targets (propane solvent, 3-stage ROSEMAX extractor):

| Property | Lube Bright Stock | FCC Feed |
|----------|-------------------|----------|
| DAO Yield | 18 wt% | 32 wt% |
| DAO API | 21.3 | 19.7 |
| DAO Density (from API) | 0.926 g/cm³ | 0.936 g/cm³ |
| DAO CCR | 1.5 wt% | 2.5 wt% |
| DAO Sulphur | 2.4 wt% | 2.9 wt% |
| DAO Ni | <<1 wppm | <1 wppm |
| DAO V | <<1 wppm | <1 wppm |
| DAO Visc @100°C | 33 cSt | 45 cSt |
| Asph. in DAO | <100 wppm | <200 wppm |
| Operating T | ~60°C | ~55°C |
| Asphalt Yield | 82 wt% | 68 wt% |
| Asphalt Softening Pt | 62°C | 80°C |

### Operating Limits (Chapter 13):
- S/O ratio: optimum 8:1 vol/vol
- Extractor pressure: min 38 kg/cm²g
- Solvent: Propane only (actual operation; blend with i-butane was planned but not implemented)
- Pre-dilution: 1.8 vol solvent/vol feed (lube), 1.2 (FCC)

---

## 7. WHAT'S ALREADY DONE AND WORKING

### Calibration parameter wiring ✅
K_multiplier and delta_crit flow correctly through the ENTIRE chain:
```
UI slider → /api/simulate → run_base_case() → run_extractor() → solve_lle() → K_value()
                                                                                  ↑
plant_calibration._simulate_point() → run_extractor(K_multiplier=, delta_crit=) ─┘
```

### DAO yield response ✅
```
K_mult=0.65, T=60°C, S/O=8 → DAO=17.9%  (target: 18%)  ✓
K_mult=1.05, T=55°C, S/O=8 → DAO=33.4%  (target: 32%)  ✓
Yield monotonically increases with K_multiplier            ✓
Yield monotonically decreases with temperature (propane)   ✓
```

### Calibration pipeline ✅
- `load_plant_data(csv)` → parses CSV with flexible column handling
- `_simulate_point(pt, params)` → runs extractor with cal params
- `_build_residuals()` → weighted residuals for least_squares
- `run_calibration()` → scipy least_squares with TRF, bounded
- `save_profile()` / `load_profile()` / `list_profiles()` → JSON profiles
- `compute_metrics()` → MAE, RMSE, bias per variable
- `plot_calibration_results()` → 4-panel Plotly parity plots

### CLI ✅
```bash
python run_simulation.py                                    # web UI
python run_simulation.py --no-ui                            # CLI only
python run_simulation.py --calibrate data.csv               # calibrate
python run_simulation.py --calibrate data.csv --profile x   # save to profile
python run_simulation.py --calibrate data.csv --weights dao=1.0,ccr=0.5,rho=50,asp=5
python run_simulation.py --profile sda_lube_dao             # use saved profile
```

### Web UI ✅
- Two tabs: Simulation + Calibration
- Simulation tab: feed/solvent/S-O/T/stages sliders, K_multiplier + delta_crit sliders, 
  profile quick-load, KPIs, SARA bars, stage table, sensitivity plots
- Calibration tab: CSV upload/paste, weight sliders, profile management, 
  run calibration button, results display (KPIs, param table, metrics, parity plots, point-by-point)

### Data privacy ✅
- UI header: "SDA Unit Simulator — Digital Twin" (no plant-specific names)
- Feed labels: "Heavy VR Blend A / B" (not crude source names)
- `GENERIC_LABELS` dict + `generic_feed_label()` in plant_calibration.py
- Internal keys (basra_kuwait_mix) kept for backward compat

### Profiles ✅
- sda_default.json (K=1.00, default params)
- sda_lube_dao.json (K=0.65, delta=3.0 — targets 18% yield)
- sda_fcc_dao.json (K=1.05, delta=2.3 — targets 32% yield)

---

## 8. WHAT'S BROKEN — FIX THESE IN ORDER

### ISSUE 1 (CRITICAL): DAO CCR correlation produces 5.3% instead of 1.5%

**File:** `plant_calibration.py`, function `_simulate_point()`, around line 295-301

**Current code:**
```python
sara    = r.get('SARA_DAO', {})
f_res   = sara.get('resins',      0.0) / 100.0
f_asp   = sara.get('asphaltenes', 0.0) / 100.0
f_aro   = sara.get('aromatics',   0.0) / 100.0
dao_ccr = (0.96 * f_asp + 0.22 * f_res + 0.04 * f_aro) * 100.0
```

**Symptom:** At lube conditions (K=0.65, T=60°C), SARA_DAO is typically:
```
sat≈28%, aro≈61%, res≈10.5%, asp≈0.6%
```
With current coefficients: CCR = 0.96*0.006 + 0.22*0.105 + 0.04*0.61 = 0.054 = 5.4%
But plant target is 1.5% (lube) / 2.5% (FCC).

**Root cause:** The coefficients (0.96, 0.22, 0.04) are for whole VR material. 
DAO-range fractions are lighter cuts within each SARA class — they contribute 
much less CCR per unit mass than whole-feed resins/aromatics.

**Fix direction:** The CCR coefficients need to be ~4-5x lower for DAO-range material:
```python
# Suggested starting point (tune to match both targets):
dao_ccr = (0.90 * f_asp + 0.05 * f_res + 0.008 * f_aro) * 100.0
```

**Verification:** After fixing, check:
- Lube mode (K=0.65, T=60°C): SARA_DAO≈{sat:28,aro:61,res:10.5,asp:0.6} → CCR should ≈ 1.5%
- FCC mode (K=1.05, T=55°C): SARA_DAO will have more resins → CCR should ≈ 2.5%

If you can't get both modes right with fixed coefficients, consider making the 
CCR correlation MW-dependent (DAO MW varies between modes).

### ISSUE 2 (CRITICAL): DAO density produces 0.88 instead of 0.926

**File:** `residue_distribution.py`, function `_density_from_class()`, around line 167-170

**Current code:**
```python
def _density_from_class(MW: float, sara_class: str) -> float:
    base = {'saturates': 0.840, 'aromatics': 0.950,
            'resins': 1.010, 'asphaltenes': 1.080}.get(sara_class, 0.950)
    correction = -0.06 * np.exp(-(MW - 400) / 400)
    return float(np.clip(base + correction, 0.75, 1.15))
```

**Symptom:** Weight-averaged DAO density ≈ 0.88. Plant target: 0.926 (lube, API=21.3).
API 21.3 → ρ = 141.5/(21.3+131.5) = 0.926 g/cm³.

**Root cause:** The base densities for saturates (0.840) and aromatics (0.950) are 
for light/mid-range cuts. VR-derived DAO fractions are heavier. The DAO is ~60% 
aromatics + ~28% saturates, so the weighted average lands at ~0.88.

**Fix direction:** Increase base densities to reflect VR-derived material:
```python
base = {'saturates': 0.900, 'aromatics': 1.000,
        'resins': 1.030, 'asphaltenes': 1.100}.get(sara_class, 0.970)
```

**IMPORTANT:** This change affects ALL simulations (base case, sensitivity, calibration).
After changing, re-verify:
- Lube DAO density target: 0.926 ± 0.01
- FCC DAO density target: 0.936 ± 0.01
- Base case (K=1.0, T=75°C): should give density in 0.90–0.95 range

Also check `phct_eos.py :: calculate_density()` — it has its own density model 
for residue components. Make sure the two are consistent.

### ISSUE 3 (MEDIUM): sensitivity_analysis.py doesn't forward calibration params

**File:** `sensitivity_analysis.py`

**Symptom:** All sensitivity sweeps use default K_multiplier=1.0 and delta_crit=2.5, 
even when the user has loaded a calibration profile.

**Fix:** 
1. Add `K_multiplier=1.0` and `delta_crit=2.5` parameters to `_run()` helper and 
   all `sweep_*` functions.
2. Forward them to `run_extractor()`.
3. In `run_simulation.py :: /api/sensitivity`, read K_multiplier and delta_crit from 
   the request JSON and pass them through.

### ISSUE 4 (LOW): sample_plant_data.csv has unreachable targets

**File:** `sample_plant_data.csv`

**Symptom:** The CSV contains plant-realistic values (density=0.926, CCR=1.5) that 
the simulator can't reproduce until Issues 1-2 are fixed.

**Fix:** After fixing Issues 1-2, regenerate the sample CSV:
```python
# In plant_calibration.py :: make_sample_csv()
# Strategy: run the simulator at design conditions, add noise
from plant_calibration import PlantDataPoint, _simulate_point
import random

def make_consistent_csv():
    """Generate sample data by running the fixed simulator + adding noise."""
    # Run sim at various conditions, add ±3-5% gaussian noise to outputs
    # This ensures calibration can actually converge on the sample
```

Or simpler: just run the fixed simulator at the sample operating conditions and 
use those outputs as the "plant" values with small random perturbations.

### ISSUE 5 (LOW): Generic labels not applied everywhere

**File:** `sensitivity_analysis.py` plot titles, `run_simulation.py` print_summary header

**Symptom:** Some plot titles still show "Basra-Kuwait" from `HPCL_FEEDS[feed_name]['label']`

**Fix:** In sensitivity_analysis plot functions, replace:
```python
HPCL_FEEDS.get(feed_name,{}).get("label", feed_name)
```
with:
```python
# Import from plant_calibration or define locally
GENERIC_LABELS = {'basra_kuwait_mix': 'Heavy VR Blend A', 'basra_light': 'Heavy VR Blend B'}
GENERIC_LABELS.get(feed_name, feed_name)
```

---

## 9. TESTING COMMANDS

Run these after each fix to verify:

```bash
# Install deps (first time)
pip install -r requirements.txt

# Test 1: Basic simulation still works
python -c "
from run_simulation import run_base_case
r = run_base_case()
print(f'DAO={r[\"DAO_yield_net\"]:.1f}%  density={r[\"density_DAO\"]:.4f}')
assert r['DAO_yield_net'] > 10, 'Yield too low'
print('PASS')
"

# Test 2: Lube mode targets (CRITICAL — must pass after Issues 1+2 fixed)
python -c "
from plant_calibration import PlantDataPoint, _simulate_point
pt = PlantDataPoint(SO_ratio=8.0, temperature=60.0, N_stages=3, solvent='propane')
sim = _simulate_point(pt, {'K_multiplier':0.65, 'delta_crit':3.0, 'C_entrain':0.01, 'k_precip':0.5, 'E_murphree':0.75}, {})
print(f'DAO={sim[\"DAO_yield\"]:.1f}%  CCR={sim[\"DAO_CCR\"]:.2f}%  density={sim[\"DAO_density\"]:.4f}')
print(f'Targets: DAO~18%  CCR~1.5%  density~0.926')
assert 16 < sim['DAO_yield'] < 20, f'Yield {sim[\"DAO_yield\"]:.1f} out of 16-20 range'
assert 0.5 < sim['DAO_CCR'] < 3.0, f'CCR {sim[\"DAO_CCR\"]:.2f} out of 0.5-3.0 range'
assert 0.90 < sim['DAO_density'] < 0.95, f'Density {sim[\"DAO_density\"]:.4f} out of 0.90-0.95 range'
print('ALL ASSERTIONS PASSED')
"

# Test 3: FCC mode targets
python -c "
from plant_calibration import PlantDataPoint, _simulate_point
pt = PlantDataPoint(SO_ratio=8.0, temperature=55.0, N_stages=3, solvent='propane')
sim = _simulate_point(pt, {'K_multiplier':1.05, 'delta_crit':2.3, 'C_entrain':0.015, 'k_precip':0.5, 'E_murphree':0.72}, {})
print(f'DAO={sim[\"DAO_yield\"]:.1f}%  CCR={sim[\"DAO_CCR\"]:.2f}%  density={sim[\"DAO_density\"]:.4f}')
print(f'Targets: DAO~32%  CCR~2.5%  density~0.936')
assert 29 < sim['DAO_yield'] < 36, f'Yield {sim[\"DAO_yield\"]:.1f} out of range'
assert 1.5 < sim['DAO_CCR'] < 4.0, f'CCR {sim[\"DAO_CCR\"]:.2f} out of range'
assert 0.91 < sim['DAO_density'] < 0.96, f'Density {sim[\"DAO_density\"]:.4f} out of range'
print('ALL ASSERTIONS PASSED')
"

# Test 4: Full calibration pipeline
python run_simulation.py --calibrate sample_plant_data.csv --profile test_run --no-ui

# Test 5: CLI with profile
python run_simulation.py --profile sda_lube_dao --no-ui --T 60

# Test 6: Web UI (manual)
python run_simulation.py
# → Open http://localhost:5000
# → Test Simulation tab: adjust sliders, click Run
# → Test Calibration tab: click "Download Sample CSV", click "Run Calibration"
# → Verify parity plots appear and params update
```

---

## 10. CODING STANDARDS

- Python 3.9+ compatible
- `python run_simulation.py` (no args) MUST still launch web UI
- All numpy types → Python native before JSON: use _to_json_safe() or float()/int()
- JSON profiles: indent=2 for human readability
- Error messages: clear and actionable (plant engineers will read them)
- Do NOT remove existing function parameters — only add new optional ones
- Use `scipy.optimize.least_squares` (not minimize or differential_evolution)
- Generic labels in all UI-facing text (no proprietary crude/plant names)

---

## 11. DEPENDENCY BETWEEN FIXES

```
Issue 1 (CCR formula)  ──┐
                         ├──→ Issue 4 (regenerate sample CSV) ──→ Test calibration
Issue 2 (density model) ─┘

Issue 3 (sensitivity params) → independent, can be done in parallel

Issue 5 (generic labels) → independent, cosmetic
```

Fix 1+2 first, then 4, then verify calibration, then 3+5.
