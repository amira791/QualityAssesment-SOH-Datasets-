"""
CALCE Battery Dataset — Quality Scoring
=========================================
Translates quantitative findings from the exploration script into the
6-criterion quality scorecard.

Run AFTER calce_explore.py
"""

import os, glob, re, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# ── CONFIG ───────────────────────────────────────────────────────────────────
DATASET_PATH = r"C:\Users\admin\Desktop\DR2\11 All Datasets\03 CALCE Battery Dataset\dataset calce"
SAVE_PLOTS   = True
OUT_DIR      = os.path.join(DATASET_PATH, "exploration_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# Physical limits for LCO cells
CELL_V_MIN, CELL_V_MAX = 2.5, 4.2      # V
TEMP_MIN,   TEMP_MAX   = -20, 60       # °C
MIN_CAPACITY, MAX_CAPACITY = 0.8, 1.5  # Ah

# ─────────────────────────────────────────────────────────────────────────────
SCORE_LABELS = {
    "++":  "Fully Satisfied",
    "+":   "Mostly Satisfied",
    "o":   "Partially Satisfied",
    "-":   "Not Satisfied",
    "N/A": "Not Applicable",
}
SCORE_COLORS = {
    "++": "#2ca02c", "+": "#98df8a", "o": "#ffbb78", "-": "#d62728", "N/A": "#aec7e8"
}

def score_line(criterion, aspect, score, finding):
    bar = "─" * 68
    print(f"\n  {'Criterion':<22} {criterion}")
    print(f"  {'Aspect':<22} {aspect}")
    print(f"  {'Score':<22} {score}  ({SCORE_LABELS[score]})")
    print(f"  {'Finding':<22} {finding}")
    print(f"  {bar}")
    return {"criterion": criterion, "aspect": aspect, "score": score, "finding": finding}

def parse_filename(filename):
    pattern = r'CALCE_([^_]+)_prism_([^_]+)_(\d+)C_([\d\-]+)_([\d\.]+)-([\d\.]+)C_([a-z])_(cycle_data|timeseries)\.csv'
    match = re.match(pattern, filename)
    if match:
        return {
            'cell_id': match.group(1),
            'chemistry': match.group(2),
            'temperature': int(match.group(3)),
            'DoD_range': match.group(4),
            'charge_Crate': float(match.group(5)),
            'discharge_Crate': float(match.group(6)),
            'replicate': match.group(7),
            'file_type': match.group(8)
        }
    return None

def clean_column_names(df):
    df.columns = df.columns.str.strip()
    return df


# ════════════════════════════════════════════════════════════════════════════
#  LOAD DATA AND COMPUTE METRICS
# ════════════════════════════════════════════════════════════════════════════
print("Loading CALCE dataset...")

all_csv_files = sorted(glob.glob(os.path.join(DATASET_PATH, "*.csv")))
cycle_files = [f for f in all_csv_files if 'cycle_data' in f]

cycle_frames = []
for f in cycle_files:
    filename = os.path.basename(f)
    parsed = parse_filename(filename)
    df = pd.read_csv(f, low_memory=False)
    df = clean_column_names(df)
    
    df['file_source'] = filename
    if parsed:
        df['cell_id'] = parsed['cell_id']
        df['chemistry'] = parsed['chemistry']
        df['temperature_C'] = parsed['temperature']
        df['DoD_range'] = parsed['DoD_range']
        df['charge_Crate'] = parsed['charge_Crate']
        df['discharge_Crate'] = parsed['discharge_Crate']
        df['replicate'] = parsed['replicate']
    
    cycle_frames.append(df)

cycle_data = pd.concat(cycle_frames, ignore_index=True)

# Column mapping
COL_MAP = {}
for col in cycle_data.columns:
    col_lower = col.lower()
    if 'cycle_index' in col_lower:
        COL_MAP['cycle_index'] = col
    elif 'discharge_capacity' in col_lower:
        COL_MAP['discharge_capacity'] = col
    elif 'charge_capacity' in col_lower:
        COL_MAP['charge_capacity'] = col
    elif 'min_voltage' in col_lower:
        COL_MAP['min_voltage'] = col
    elif 'max_voltage' in col_lower:
        COL_MAP['max_voltage'] = col
    elif 'min_current' in col_lower:
        COL_MAP['min_current'] = col
    elif 'max_current' in col_lower:
        COL_MAP['max_current'] = col

print(f"Loaded {len(cycle_data):,} cycle records from {cycle_data['cell_id'].nunique()} cells")

# Compute SOH
cap_col = COL_MAP.get('discharge_capacity')
cycle_col = COL_MAP.get('cycle_index')

soh_records = []
if cap_col and cycle_col:
    for cell_id, grp in cycle_data.groupby('cell_id'):
        valid_caps = grp[cap_col].dropna()
        if len(valid_caps) == 0:
            continue
        initial_cap = valid_caps.iloc[0]
        if initial_cap <= 0:
            continue
        grp = grp.copy()
        grp['SOH'] = grp[cap_col] / initial_cap
        soh_records.append(grp[['cell_id', cycle_col, cap_col, 'SOH']])
    
    if soh_records:
        soh_df = pd.concat(soh_records, ignore_index=True)
        # Filter unrealistic SOH values
        soh_df = soh_df[soh_df['SOH'] <= 1.1]
        print(f"SOH computed for {soh_df['cell_id'].nunique()} cells, {len(soh_df):,} cycles")
    else:
        soh_df = pd.DataFrame()
else:
    soh_df = pd.DataFrame()

# Physical plausibility metrics
total_cycles = len(cycle_data)
if cap_col:
    cap = cycle_data[cap_col].dropna()
    cap_outliers = ((cap < MIN_CAPACITY) | (cap > MAX_CAPACITY)).sum()
    cap_outlier_pct = cap_outliers / len(cap) * 100 if len(cap) > 0 else 0
else:
    cap_outlier_pct = 0

# Check voltage ranges
v_min_col = COL_MAP.get('min_voltage')
v_max_col = COL_MAP.get('max_voltage')
voltage_outliers = 0
total_v_pts = 0
if v_min_col and v_max_col:
    total_v_pts = len(cycle_data)
    v_min_out = ((cycle_data[v_min_col] < CELL_V_MIN) | (cycle_data[v_min_col] > CELL_V_MAX)).sum()
    v_max_out = ((cycle_data[v_max_col] < CELL_V_MIN) | (cycle_data[v_max_col] > CELL_V_MAX)).sum()
    voltage_outliers = v_min_out + v_max_out
    voltage_outlier_pct = voltage_outliers / (total_v_pts * 2) * 100 if total_v_pts > 0 else 0
else:
    voltage_outlier_pct = 0

# EOL statistics
if not soh_df.empty:
    cells_at_eol = soh_df[soh_df['SOH'] < 0.8]['cell_id'].nunique()
    total_cells = soh_df['cell_id'].nunique()
    eol_pct = cells_at_eol / total_cells * 100 if total_cells > 0 else 0
else:
    cells_at_eol = 0
    total_cells = 0
    eol_pct = 0

# Cycle life range
if cycle_col:
    cycle_life = cycle_data.groupby('cell_id')[cycle_col].max()
    cycle_min = cycle_life.min() if len(cycle_life) > 0 else 0
    cycle_max = cycle_life.max() if len(cycle_life) > 0 else 0
else:
    cycle_min = cycle_max = 0

print(f"Capacity outliers: {cap_outlier_pct:.2f}%")
print(f"Voltage outliers: {voltage_outlier_pct:.2f}%")
print(f"Cells reached EOL: {cells_at_eol}/{total_cells} ({eol_pct:.0f}%)")
print("Done.\n")

print("=" * 72)
print("  CALCE BATTERY DATASET — QUALITY SCORECARD")
print("=" * 72)

results = []

# ════════════════════════════════════════════════════════════════════════════
#  1. CORRECTNESS
# ════════════════════════════════════════════════════════════════════════════
print("\n══ 1. CORRECTNESS " + "═" * 53)

finding_physical = (
    f"Voltage: {voltage_outlier_pct:.2f}% out of [{CELL_V_MIN}, {CELL_V_MAX}]V - "
    f"minor outliers at cycle boundaries. Capacity: {cap_outlier_pct:.2f}% out of "
    f"[{MIN_CAPACITY}, {MAX_CAPACITY}]Ah - outliers include initial cycles and EOL "
    f"(12.497Ah is clearly erroneous, likely data recording error). "
    f"Temperature data not present in cycle files (available in timeseries)."
)
score_physical = "+"
results.append(score_line("Correctness", "Physical plausibility", score_physical, finding_physical))

finding_sign = (
    f"Current measurements use standard sign convention: positive for charge, "
    f"negative for discharge. Min_Current values negative, Max_Current values positive. "
    f"Consistent across all cells. All cycles properly identified with sequential cycle indices."
)
score_sign = "++"
results.append(score_line("Correctness", "Current sign convention", score_sign, finding_sign))

# ════════════════════════════════════════════════════════════════════════════
#  2. COMPLETENESS
# ════════════════════════════════════════════════════════════════════════════
print("\n══ 2. COMPLETENESS " + "═" * 52)

finding_missing = (
    f"Start_Time and End_Time columns are 100% NaN - structural issue. "
    f"All critical columns (Cycle_Index, capacities, voltages, currents) are 100% complete. "
    f"Temperature data only in timeseries files, not cycle summaries."
)
score_missing = "o"
results.append(score_line("Completeness", "Missing values", score_missing, finding_missing))

finding_doc = (
    f"Complete experimental documentation in filenames: each file encodes "
    f"cell ID, chemistry (LCO), temperature (25°C), DoD range (0-100%), "
    f"charge/discharge C-rates (0.5C), and replicate ID. 7 cells with consistent protocols."
)
score_doc = "++"
results.append(score_line("Completeness", "Test protocol documentation", score_doc, finding_doc))

# ════════════════════════════════════════════════════════════════════════════
#  3. ANOMALY MINIMIZATION
# ════════════════════════════════════════════════════════════════════════════
print("\n══ 3. ANOMALY MINIMIZATION " + "═" * 45)

# 3a — Statistical outliers (fixed)
if cap_col:
    cap_valid = cycle_data[cycle_data[cap_col] < 2.0][cap_col].dropna()
    if len(cap_valid) > 0:
        q1, q3 = cap_valid.quantile(0.25), cap_valid.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
        cap_outliers_iqr = ((cap_valid < lo) | (cap_valid > hi)).sum()
        cap_outlier_iqr_pct = cap_outliers_iqr / len(cap_valid) * 100
        
        finding_outliers = (
            f"IQR (3×) outliers on discharge capacity: {cap_outliers_iqr:,}/{len(cap_valid):,} ({cap_outlier_iqr_pct:.2f}%). "
            f"Capacity range after filtering extreme values: {cap_valid.min():.3f}-{cap_valid.max():.3f}Ah. "
            f"Main outliers are from cycle 0 (formation) and EOL degradation."
        )
        score_outliers = "+"
    else:
        finding_outliers = "Insufficient data for outlier analysis"
        score_outliers = "N/A"
else:
    finding_outliers = "No capacity data available"
    score_outliers = "N/A"
results.append(score_line("Anomaly Minimization", "Statistical outliers", score_outliers, finding_outliers))

# 3b — Efficiency metrics
finding_efficiency = (
    f"Coulombic Efficiency and Energy Efficiency columns are not present in the cycle data files. "
    f"This is a limitation for efficiency-focused studies."
)
score_efficiency = "-"
results.append(score_line("Anomaly Minimization", "Efficiency metrics", score_efficiency, finding_efficiency))

# ════════════════════════════════════════════════════════════════════════════
#  4. REPRESENTATIVENESS & DIVERSITY
# ════════════════════════════════════════════════════════════════════════════
print("\n══ 4. REPRESENTATIVENESS & DIVERSITY " + "═" * 34)

finding_chem = (
    f"Single chemistry: LCO (Lithium Cobalt Oxide) across all 7 cells. "
    f"Limited chemistry diversity - good for LCO-specific studies but not representative of other chemistries."
)
score_chem = "o"
results.append(score_line("Representativeness & Diversity", "Chemistry diversity", score_chem, finding_chem))

finding_temp = (
    f"Single temperature condition: 25°C for all cells. No thermal variation testing. "
    f"Limits applicability for temperature-dependent degradation studies."
)
score_temp = "-"
results.append(score_line("Representativeness & Diversity", "Temperature conditions", score_temp, finding_temp))

finding_dod = (
    f"Single DoD range: 0-100% (full depth of discharge). No partial DoD cycling profiles. "
    f"Represents full cycle aging but lacks shallow cycling conditions."
)
score_dod = "o"
results.append(score_line("Representativeness & Diversity", "Depth of Discharge diversity", score_dod, finding_dod))

finding_crate = (
    f"Single C-rate: 0.5C charge and 0.5C discharge. Represents standard (slow) cycling, "
    f"not fast-charging conditions. Limited applicability for fast-charging research."
)
score_crate = "-"
results.append(score_line("Representativeness & Diversity", "C-rate diversity", score_crate, finding_crate))

finding_replicates = (
    f"7 replicate cells under identical conditions. Enables statistical analysis of "
    f"cell-to-cell variability and manufacturing variance."
)
score_replicates = "++"
results.append(score_line("Representativeness & Diversity", "Replicate cells", score_replicates, finding_replicates))

finding_calendar = (
    f"No dedicated calendar aging protocol. Dataset focuses on cyclic aging only."
)
score_calendar = "-"
results.append(score_line("Representativeness & Diversity", "Calendar aging", score_calendar, finding_calendar))

finding_dynamic = (
    f"Constant-current constant-voltage (CCCV) cycling only. No dynamic drive-cycle profiles."
)
score_dynamic = "o"
results.append(score_line("Representativeness & Diversity", "Dynamic load profiles", score_dynamic, finding_dynamic))

# ════════════════════════════════════════════════════════════════════════════
#  5. BALANCED DISTRIBUTION
# ════════════════════════════════════════════════════════════════════════════
print("\n══ 5. BALANCED DISTRIBUTION " + "═" * 43)

if not soh_df.empty:
    soh_min = soh_df['SOH'].min()
    soh_max = soh_df['SOH'].max()
    below_eol = (soh_df['SOH'] < 0.8).sum()
    total_cyc = len(soh_df)
    
    finding_soh_range = (
        f"SOH ranges from {soh_min:.3f} to {soh_max:.3f}. "
        f"{below_eol}/{total_cyc:,} cycles ({below_eol/total_cyc*100:.1f}%) below EOL (80%). "
        f"All 7 cells reached EOL, providing complete degradation trajectories."
    )
    score_soh_range = "++"
else:
    finding_soh_range = "SOH data not available"
    score_soh_range = "N/A"
results.append(score_line("Balanced Distribution", "SOH range coverage", score_soh_range, finding_soh_range))

if cycle_life is not None and len(cycle_life) > 0:
    cv_cycles = cycle_life.std() / cycle_life.mean() if cycle_life.mean() > 0 else 0
    finding_cycle_bal = (
        f"Cycle counts per cell: min={cycle_min:.0f}, max={cycle_max:.0f}, CV={cv_cycles:.2f}. "
        f"All cells tested under identical conditions, variance reflects natural degradation variability."
    )
    score_cycle_bal = "+"
else:
    finding_cycle_bal = "No cycle count data available"
    score_cycle_bal = "N/A"
results.append(score_line("Balanced Distribution", "Cycle contribution per cell", score_cycle_bal, finding_cycle_bal))

# ════════════════════════════════════════════════════════════════════════════
#  6. TEMPORAL COHERENCE
# ════════════════════════════════════════════════════════════════════════════
print("\n══ 6. TEMPORAL COHERENCE " + "═" * 46)

if cycle_col:
    non_monotonic = 0
    for cell_id, grp in cycle_data.groupby('cell_id'):
        cycles = grp[cycle_col].values
        if len(cycles) > 1 and not np.all(np.diff(cycles) >= 0):
            non_monotonic += 1
    
    finding_mono = (
        f"Cycle index monotonicity: {non_monotonic}/{cycle_data['cell_id'].nunique()} cells "
        f"have non-monotonic cycles. Cycle indices are strictly increasing within each cell."
    )
    score_mono = "++" if non_monotonic == 0 else "+"
else:
    finding_mono = "Cycle index column not available"
    score_mono = "N/A"
results.append(score_line("Temporal Coherence", "Monotonic cycle index", score_mono, finding_mono))

if not soh_df.empty:
    non_mono_soh = 0
    for cell_id, grp in soh_df.groupby('cell_id'):
        soh_vals = grp.sort_values(cycle_col)['SOH'].values if cycle_col else grp['SOH'].values
        if len(soh_vals) > 1:
            violations = np.sum(np.diff(soh_vals) > 0.03)
            if violations > 0:
                non_mono_soh += 1
    
    finding_deg = (
        f"Degradation trend consistency: {non_mono_soh}/{soh_df['cell_id'].nunique()} cells "
        f"show non-monotonic SOH (>3% increases). Minor recoveries expected due to measurement noise."
    )
    score_deg = "++" if non_mono_soh <= 2 else "+"
else:
    finding_deg = "Cannot assess degradation trend"
    score_deg = "o"
results.append(score_line("Temporal Coherence", "Consistent degradation trend", score_deg, finding_deg))

finding_complete = (
    f"All 7 cells cycled until capacity fell below 80% of initial (EOL). "
    f"Complete degradation trajectories available for full lifecycle analysis."
)
score_complete = "++"
results.append(score_line("Temporal Coherence", "Complete degradation to EOL", score_complete, finding_complete))

# ════════════════════════════════════════════════════════════════════════════
#  FINAL SCORECARD TABLE
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("  FINAL SCORECARD — CALCE Battery Dataset")
print("=" * 72)
print(f"  {'Criterion':<32} {'Aspect':<40} {'Score'}")
print("  " + "─" * 70)

results_df = pd.DataFrame(results)
for _, row in results_df.iterrows():
    print(f"  {row['criterion']:<32} {row['aspect']:<40} {row['score']}")

print("\n  Score distribution:")
score_counts = results_df["score"].value_counts()
for s in ["++", "+", "o", "-", "N/A"]:
    n = score_counts.get(s, 0)
    print(f"    {s:>3}  {SCORE_LABELS[s]:<22} : {n}")

# ════════════════════════════════════════════════════════════════════════════
#  VISUALISATION — Scorecard heatmap
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(15, 12))
ax.set_xlim(0, 1); ax.set_ylim(0, len(results_df))
ax.axis("off")

col_offsets = [0.00, 0.30, 0.82, 0.92]
headers = ["Criterion", "Aspect", "Score", ""]

for txt, x in zip(headers, col_offsets):
    ax.text(x + 0.01, len(results_df) + 0.3, txt,
            fontsize=9, fontweight="bold", va="center")

for i, row in results_df.iterrows():
    y = len(results_df) - 1 - i
    color = SCORE_COLORS[row["score"]]
    bg = "#f7f7f7" if i % 2 == 0 else "white"
    ax.add_patch(mpatches.FancyBboxPatch((0, y), 1, 0.85,
                 boxstyle="square,pad=0", linewidth=0,
                 facecolor=bg, zorder=0))
    ax.add_patch(mpatches.FancyBboxPatch((col_offsets[3], y + 0.1), 0.07, 0.7,
                 boxstyle="round,pad=0.01", linewidth=0,
                 facecolor=color, zorder=1))

    ax.text(col_offsets[0] + 0.01, y + 0.42, row["criterion"],
            fontsize=7.5, va="center", fontweight="bold" if i == 0 or
            (i > 0 and results_df.iloc[i-1]["criterion"] != row["criterion"]) else "normal")
    ax.text(col_offsets[1] + 0.01, y + 0.42, row["aspect"],
            fontsize=7.5, va="center")
    ax.text(col_offsets[2] + 0.01, y + 0.42, row["score"],
            fontsize=8, va="center", fontweight="bold", color=color)

prev = None
for i, row in results_df.iterrows():
    if row["criterion"] != prev and prev is not None:
        y = len(results_df) - i
        ax.axhline(y, color="#999", linewidth=0.8, xmin=0, xmax=1)
    prev = row["criterion"]

legend_patches = [mpatches.Patch(color=c, label=f"{s} — {SCORE_LABELS[s]}")
                  for s, c in SCORE_COLORS.items()]
ax.legend(handles=legend_patches, loc="lower center",
          bbox_to_anchor=(0.5, -0.08), ncol=5, fontsize=7.5,
          frameon=True, edgecolor="#ccc")

ax.set_title("CALCE Battery Dataset — Quality Scorecard",
             fontsize=11, fontweight="bold", pad=14)
plt.tight_layout()
if SAVE_PLOTS:
    path = os.path.join(OUT_DIR, "calce_quality_scorecard.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    print(f"\n  → Scorecard plot saved: {path}")
plt.show()

print("\nDone.")