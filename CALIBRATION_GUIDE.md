# SDA Unit Simulator — Plant Calibration Guide

## 1. Calibration Parameter Reference

The simulator exposes **5 calibration parameters** that control different aspects of the physics. Here is what each one does and how to tweak it:

### K_multiplier (Range: 0.30 – 3.50, Default: 1.00)
**What it controls:** DAO yield — this is the PRIMARY lever.

**Physics:** Uniformly scales all liquid-liquid equilibrium K-values. K_i determines how much of each pseudo-component partitions to DAO vs asphalt.
- **Increase K_multiplier → More DAO yield** (more material extracted into DAO phase)
- **Decrease K_multiplier → Less DAO yield** (more selective extraction)

**When to adjust:** Always adjust this FIRST. If your plant DAO yield is consistently higher or lower than the simulation, K_multiplier is the fix.

**Typical calibrated values:**
- Lube Bright Stock mode (~18% DAO): K_multiplier ≈ 0.60–0.70
- FCC Feed mode (~32% DAO): K_multiplier ≈ 1.00–1.10

---

### delta_crit (Range: 0.50 – 8.00, Default: 2.50)
**What it controls:** DAO quality (CCR, colour) — this is the QUALITY lever.

**Physics:** Modifies the solubility parameter spread between SARA classes. Higher delta_crit penalises the K-values of resins and aromatics more than saturates, making the DAO "cleaner" but reducing yield slightly.
- **Increase delta_crit → Cleaner DAO** (lower CCR, less resin contamination, but lower yield)
- **Decrease delta_crit → Dirtier DAO** (higher CCR, more resins in DAO, but higher yield)

**When to adjust:** If DAO CCR or colour spec is off while yield matches plant data. Adjust AFTER K_multiplier.

---

### E_murphree (Range: 0.30 – 1.00, Default: 0.70)
**What it controls:** Stage separation sharpness.

**Physics:** Murphree stage efficiency — the fraction of equilibrium separation actually achieved per stage. Accounts for imperfect mixing, axial dispersion, and mass-transfer limitations.
- **Increase E_murphree → Sharper separation** (closer to equilibrium per stage)
- **Decrease E_murphree → Poorer separation** (more contamination, lower effective stages)

**When to adjust:** If the simulator matches single-stage data but diverges at multiple stages, the efficiency is likely wrong.

**Typical values:** 0.60–0.80 for packed SDA columns with ROSEMAX internals.

---

### C_entrain (Range: 0.001 – 0.10, Default: 0.015)
**What it controls:** Asphaltene contamination in DAO product.

**Physics:** Fraction of asphalt phase mechanically entrained upward into the DAO phase. Higher values mean more asphaltene contamination in DAO.
- **Increase C_entrain → More contamination** (colour/viscosity issues in DAO)
- **Decrease C_entrain → Cleaner DAO** (better de-entrainment)

**When to adjust:** If DAO asphaltene content (wppm) is significantly off. This is typically the last parameter to fine-tune.

---

### k_precip (Range: 0.05 – 5.00, Default: 0.50)
**What it controls:** How fast asphaltenes precipitate.

**Physics:** First-order precipitation rate constant. At default k=0.5 s⁻¹ and τ=10s, the system reaches ~99% of equilibrium. Only relevant if residence time is very short.
- **Increase k_precip → Faster precipitation** (closer to equilibrium)
- **Decrease k_precip → Slower precipitation** (kinetically limited)

**When to adjust:** Usually leave at default unless you have evidence of kinetic limitations (e.g., high-throughput operation with short residence times).

---

## 2. Recommended Calibration Strategy

### Step-by-Step Approach:

1. **Start with K_multiplier only** — Set all other params to defaults. Adjust K_multiplier until DAO yield matches plant data within ±2 wt%.

2. **Then adjust delta_crit** — If DAO CCR is still off, increase delta_crit to reduce CCR (or decrease to increase CCR).

3. **Fine-tune E_murphree** — If multi-stage separation quality doesn't match, adjust efficiency.

4. **Finally, adjust C_entrain** — If asphaltene contamination is significantly off from plant analytics.

5. **Leave k_precip at default** unless you have specific kinetic data.

### Recommended Weights for Optimiser:

| Weight | Value | Rationale |
|--------|-------|-----------|
| w_dao | 1.0 | DAO yield is the primary metric — keep at 1.0 |
| w_ccr | 0.5 | CCR matters for quality but is harder to measure |
| w_rho | 50.0 | Density differences are ~0.01 scale; multiply by 50 to balance |
| w_asp | 5.0 | Contamination values are small; amplify to give them weight |

### Separate Profiles for Different Modes:

The simulator works best when you calibrate **separate profiles** for each operating mode:
- **Lube DAO profile**: Use only lube-mode plant data (T~60°C, yield~18%)
- **FCC DAO profile**: Use only FCC-mode plant data (T~55°C, yield~32%)

A single calibration across both modes will compromise on both.

---

## 3. CSV Data Format

```
timestamp,feed_name,feed_API,feed_CCR,SO_ratio,temperature,N_stages,solvent,DAO_yield,DAO_density,DAO_CCR,asph_contam
2024-01-10,basra_kuwait_mix,6.1,22.8,8.0,60,3,propane,17.5,0.929,1.6,0.008
```

**Required columns:** `SO_ratio`, `temperature`, and at least one of: `DAO_yield`, `DAO_density`, `DAO_CCR`, `asph_contam`

**Optional columns:** `timestamp`, `feed_name`, `feed_API`, `feed_CCR`, `N_stages`, `solvent`

**Missing measurements:** Leave blank or write `N/A` — the optimizer will skip those residuals.

---

## 4. Running Calibration from Command Line (Windows)

```bash
# Install dependencies (first time only)
pip install -r requirements.txt

# Run with sample data
python run_simulation.py --calibrate sample_plant_data.csv

# Run with custom weights
python run_simulation.py --calibrate plant_data.csv --weights dao=1.0,ccr=0.5,rho=50,asp=5

# Save to specific profile
python run_simulation.py --calibrate plant_data.csv --profile sda_lube_calibrated

# Run simulation with calibrated profile
python run_simulation.py --profile sda_lube_calibrated --T 60

# Start web UI (default)
python run_simulation.py
```

---

## 5. Data Privacy Notes

The web UI uses **generic labels** throughout:
- Feed types are shown as "Heavy VR Blend A/B" (not crude source names)
- The header shows "SDA Unit Simulator" (not plant-specific identifiers)
- CSV data stays local — nothing is transmitted externally

Internal code still uses short feed keys (`basra_kuwait_mix`, `basra_light`) for backward compatibility, but these never appear in the user-facing UI.

---

## 6. Validation Targets (Design Specifications)

| Mode | DAO Yield | DAO Density | DAO CCR | Asph. in DAO |
|------|-----------|-------------|---------|--------------|
| Lube Bright Stock | 18 wt% | 0.926 g/cm³ | 1.5 wt% | <100 wppm |
| FCC Feed | 32 wt% | 0.936 g/cm³ | 2.5 wt% | <200 wppm |

After calibration, check your MAE values against:
- DAO Yield MAE < 2.0 wt%
- DAO Density MAE < 0.008 g/cm³
- DAO CCR MAE < 0.5 wt%
- Asph. Contamination MAE < 0.05 wt%
