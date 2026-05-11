"""
SNL Battery Dataset — Exploration Script
=========================================
Run from the folder containing this script, or set DATASET_PATH below.
Covers:
  1. Dataset inventory (files, sizes, naming convention parsing)
  2. Schema & dtypes inspection (cycle vs timeseries files)
  3. Basic statistics per file
  4. Cycle-level summary (capacity fade per cell)
  5. Missing-value audit
  6. Temperature analysis (min/max/avg profiles)
  7. Efficiency analysis (Coulombic & Energy efficiency)
  8. Test condition extraction (temp, DoD, C-rate)
  9. SOH estimation & degradation tracking
  10. Per-chemistry comparison (LFP, NCA, NMC)
"""

import os
import glob
import re
import warnings
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── ① USER CONFIG ────────────────────────────────────────────────────────────
DATASET_PATH = r"C:\Users\admin\Desktop\DR2\11 All Datasets\13 SNL Battery Dataset\SNL"
SAVE_PLOTS   = True
OUT_DIR      = os.path.join(DATASET_PATH, "exploration_outputs")
# ─────────────────────────────────────────────────────────────────────────────

# ── Physical plausibility limits (18650 cells) ───────────────────────────────
TEMP_MIN,  TEMP_MAX    = -20, 60    # °C operating bounds
# ─────────────────────────────────────────────────────────────────────────────

matplotlib.rcParams.update({"figure.dpi": 120, "font.size": 9})

if SAVE_PLOTS:
    os.makedirs(OUT_DIR, exist_ok=True)

def savefig(name: str):
    if SAVE_PLOTS:
        path = os.path.join(OUT_DIR, name)
        plt.savefig(path, bbox_inches="tight")
        print(f"  → saved: {path}")
    plt.show()

def parse_filename(filename):
    """
    Parse SNL filename convention:
    SNL_18650_{CHEMISTRY}_{TEMP}C_{DOD}_{CHARGE_C}-{DISCHARGE_C}_{REPLICATE}_{TYPE}.csv
    
    Example: SNL_18650_LFP_25C_0-100_0.5-1C_a_cycle_data.csv
    """
    pattern = r'SNL_18650_(\w+)_(\d+)C_([\d\-]+)_([\d\.]+)-([\d\.]+)C_([a-z])_(cycle_data|timeseries)\.csv'
    match = re.match(pattern, filename)
    
    if match:
        return {
            'chemistry': match.group(1),
            'temperature': int(match.group(2)),
            'DoD_range': match.group(3),
            'charge_Crate': float(match.group(4)),
            'discharge_Crate': float(match.group(5)),
            'replicate': match.group(6),
            'file_type': match.group(7)
        }
    else:
        return None

def clean_column_names(df):
    """Clean column names by stripping whitespace and standardizing"""
    df.columns = df.columns.str.strip()
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — INVENTORY
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SECTION 1 — FILE INVENTORY")
print("=" * 70)

# Find all CSV files
all_csv_files = sorted(glob.glob(os.path.join(DATASET_PATH, "**", "*.csv"), recursive=True))

cycle_files = [f for f in all_csv_files if 'cycle_data' in f.lower()]
timeseries_files = [f for f in all_csv_files if 'timeseries' in f.lower()]

print(f"\nTotal CSV files found: {len(all_csv_files)}")
print(f"  Cycle data files:   {len(cycle_files)}")
print(f"  Timeseries files:   {len(timeseries_files)}")

# Parse filenames and build inventory
inventory = []
for f in cycle_files + timeseries_files:
    size_kb = os.path.getsize(f) / 1024
    filename = os.path.basename(f)
    parsed = parse_filename(filename)
    
    # Get row count (fast method)
    with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
        nrows = sum(1 for _ in fh) - 1
    
    row = {
        'file': filename,
        'size_KB': round(size_kb, 1),
        'rows': nrows,
        'type': 'cycle' if 'cycle_data' in filename.lower() else 'timeseries'
    }
    
    if parsed:
        row.update(parsed)
    else:
        row['chemistry'] = 'unknown'
        row['temperature'] = -999
        row['DoD_range'] = 'unknown'
        row['charge_Crate'] = -1
        row['discharge_Crate'] = -1
        row['replicate'] = 'unknown'
    
    inventory.append(row)

inv_df = pd.DataFrame(inventory)
print(f"\nFile inventory summary:")
if len(inv_df) > 0:
    summary = inv_df.groupby(['chemistry', 'type']).agg({
        'file': 'count',
        'size_KB': 'sum',
        'rows': 'sum'
    }).round(1)
    print(summary)

print(f"\nTotal data volume:")
print(f"  Total rows  : {inv_df['rows'].sum():,}")
print(f"  Total size  : {inv_df['size_KB'].sum()/1024:.1f} MB")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — SCHEMA & DTYPES INSPECTION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 2 — SCHEMA & DTYPES")
print("=" * 70)

# Inspect one cycle data file and one timeseries file
sample_cycle = None
sample_timeseries = None

for f in cycle_files[:1]:
    sample_cycle = f
    break
for f in timeseries_files[:1]:
    sample_timeseries = f
    break

# Dictionary to store column mappings
COL_MAP = {}

if sample_cycle:
    print(f"\n--- CYCLE DATA FILE SCHEMA ---")
    print(f"File: {os.path.basename(sample_cycle)}")
    df_cycle_sample = pd.read_csv(sample_cycle, nrows=5)
    df_cycle_sample = clean_column_names(df_cycle_sample)
    print(f"Columns ({len(df_cycle_sample.columns)}): {list(df_cycle_sample.columns)}")
    print("\nFirst 3 rows:")
    print(df_cycle_sample.head(3).to_string())
    print("\nData types:")
    print(df_cycle_sample.dtypes)
    
    # Build column mapping for cycle data
    for col in df_cycle_sample.columns:
        col_lower = col.lower()
        if 'cycle' in col_lower and 'index' in col_lower:
            COL_MAP['cycle_index'] = col
        elif 'discharge' in col_lower and 'capacity' in col_lower:
            COL_MAP['discharge_capacity'] = col
        elif 'charge' in col_lower and 'capacity' in col_lower:
            COL_MAP['charge_capacity'] = col
        elif 'discharge' in col_lower and 'energy' in col_lower:
            COL_MAP['discharge_energy'] = col
        elif 'charge' in col_lower and 'energy' in col_lower:
            COL_MAP['charge_energy'] = col
        elif 'coulombic' in col_lower or ('ce' in col_lower and 'efficiency' in col_lower):
            COL_MAP['coulombic_efficiency'] = col
        elif 'energy' in col_lower and 'efficiency' in col_lower:
            COL_MAP['energy_efficiency'] = col
        elif 'max' in col_lower and 'temp' in col_lower:
            COL_MAP['max_temp'] = col
        elif 'min' in col_lower and 'temp' in col_lower:
            COL_MAP['min_temp'] = col
        elif 'avg' in col_lower and 'temp' in col_lower:
            COL_MAP['avg_temp'] = col
        elif 'average' in col_lower and 'temp' in col_lower:
            COL_MAP['avg_temp'] = col
    
    print(f"\nDetected column mapping: {COL_MAP}")

if sample_timeseries:
    print(f"\n--- TIMESERIES FILE SCHEMA ---")
    print(f"File: {os.path.basename(sample_timeseries)}")
    df_ts_sample = pd.read_csv(sample_timeseries, nrows=5)
    df_ts_sample = clean_column_names(df_ts_sample)
    print(f"Columns ({len(df_ts_sample.columns)}): {list(df_ts_sample.columns)}")
    print("\nFirst 3 rows:")
    print(df_ts_sample.head(3).to_string())
    print("\nData types:")
    print(df_ts_sample.dtypes)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — LOAD ALL CYCLE DATA FILES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 3 — LOADING CYCLE DATA")
print("=" * 70)

if len(cycle_files) == 0:
    print("ERROR: No cycle data files found!")
    exit(1)

print("Loading all cycle_data CSV files...")
cycle_frames = []

for f in cycle_files:
    filename = os.path.basename(f)
    parsed = parse_filename(filename)
    
    try:
        df = pd.read_csv(f, low_memory=False)
        df = clean_column_names(df)
        
        # Add metadata columns
        df['file_source'] = filename
        if parsed:
            df['chemistry'] = parsed['chemistry']
            df['temperature_C'] = parsed['temperature']
            df['DoD_range'] = parsed['DoD_range']
            df['charge_Crate'] = parsed['charge_Crate']
            df['discharge_Crate'] = parsed['discharge_Crate']
            df['replicate'] = parsed['replicate']
        else:
            df['chemistry'] = 'unknown'
            df['temperature_C'] = -999
            df['DoD_range'] = 'unknown'
            df['charge_Crate'] = -1
            df['discharge_Crate'] = -1
            df['replicate'] = 'unknown'
        
        cycle_frames.append(df)
    except Exception as e:
        print(f"  Warning: Could not load {filename}: {e}")

if cycle_frames:
    cycle_data = pd.concat(cycle_frames, ignore_index=True)
    print(f"Total cycle data shape: {cycle_data.shape}")
    print(f"Unique batteries/cells: {cycle_data['file_source'].nunique()}")
    print(f"Unique chemistries: {cycle_data['chemistry'].unique()}")
else:
    print("ERROR: No cycle data could be loaded!")
    exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — MISSING VALUE AUDIT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 4 — MISSING VALUES")
print("=" * 70)

null_counts = cycle_data.isnull().sum()
null_pct = (null_counts / len(cycle_data) * 100).round(2)
null_df = pd.DataFrame({"null_count": null_counts, "null_%": null_pct})
null_df = null_df[null_df["null_count"] > 0].sort_values("null_%", ascending=False)

if null_df.empty:
    print("No missing values found in cycle data.")
else:
    print("Missing values in cycle data:")
    print(null_df.to_string())

# Per-file missing check - only use columns that exist
critical_cols = []
for col in ['discharge_capacity', 'charge_capacity', 'coulombic_efficiency', 'max_temp']:
    if col in COL_MAP and COL_MAP[col] in cycle_data.columns:
        critical_cols.append(COL_MAP[col])

if critical_cols and 'file_source' in cycle_data.columns:
    missing_per_file = cycle_data.groupby('file_source')[critical_cols].apply(
        lambda g: g.isnull().sum()
    )
    missing_with_missing = missing_per_file[missing_per_file.sum(axis=1) > 0]
    if not missing_with_missing.empty:
        print("\nMissing values per file (critical columns):")
        print(missing_with_missing.to_string())
    else:
        print("\nNo missing values in critical columns across all files")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — BASIC STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 5 — BASIC STATISTICS")
print("=" * 70)

# Get numeric columns
numeric_cols = cycle_data.select_dtypes(include=[np.number]).columns.tolist()
# Remove metadata columns
exclude_cols = ['temperature_C', 'charge_Crate', 'discharge_Crate', 'Cycle_Index' if 'Cycle_Index' in cycle_data.columns else 'dummy']
stats_cols = [c for c in numeric_cols if c not in exclude_cols]

if stats_cols:
    stats = cycle_data[stats_cols].describe().T
    print("Overall statistics for all cycle data:")
    print(stats.to_string())
else:
    print("No numeric columns found for statistics")

# Per-chemistry statistics
if 'chemistry' in cycle_data.columns:
    print("\n" + "-" * 50)
    print("Statistics by chemistry:")
    for chem in cycle_data['chemistry'].unique():
        if chem != 'unknown':
            chem_data = cycle_data[cycle_data['chemistry'] == chem]
            print(f"\n{chem} (n={len(chem_data):,} cycles):")
            
            if COL_MAP.get('discharge_capacity') and COL_MAP['discharge_capacity'] in chem_data.columns:
                cap_col = COL_MAP['discharge_capacity']
                print(f"  Discharge Capacity: {chem_data[cap_col].mean():.2f} ± {chem_data[cap_col].std():.2f} Ah")
            
            if COL_MAP.get('coulombic_efficiency') and COL_MAP['coulombic_efficiency'] in chem_data.columns:
                ce_col = COL_MAP['coulombic_efficiency']
                print(f"  Coulombic Efficiency: {chem_data[ce_col].mean():.2f} ± {chem_data[ce_col].std():.2f} %")
            
            if COL_MAP.get('max_temp') and COL_MAP['max_temp'] in chem_data.columns:
                temp_col = COL_MAP['max_temp']
                print(f"  Max Temperature: {chem_data[temp_col].mean():.1f} ± {chem_data[temp_col].std():.1f} °C")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — PHYSICAL PLAUSIBILITY CHECKS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 6 — PHYSICAL PLAUSIBILITY CHECKS")
print("=" * 70)

# Check discharge capacity plausibility
if COL_MAP.get('discharge_capacity') and COL_MAP['discharge_capacity'] in cycle_data.columns:
    cap_col = COL_MAP['discharge_capacity']
    cap = cycle_data[cap_col].dropna()
    # Typical 18650 cells: 1.5-3.5 Ah range
    cap_outliers = ((cap < 1.0) | (cap > 4.0)).sum()
    cap_pct = cap_outliers / len(cap) * 100 if len(cap) > 0 else 0
    print(f"  Discharge Capacity:")
    print(f"    Range checked : [1.0, 4.0] Ah")
    print(f"    Valid rows    : {len(cap):,}")
    print(f"    Out of range  : {cap_outliers:,} ({cap_pct:.3f}%)")
    if cap_outliers > 0:
        print(f"    Outlier values: {cap[cap < 1.0].head(5).tolist()} (low), {cap[cap > 4.0].head(5).tolist()} (high)")

# Check efficiencies
if COL_MAP.get('coulombic_efficiency') and COL_MAP['coulombic_efficiency'] in cycle_data.columns:
    ce_col = COL_MAP['coulombic_efficiency']
    ce = cycle_data[ce_col].dropna()
    ce_outliers = ((ce < 80) | (ce > 102)).sum()
    ce_pct = ce_outliers / len(ce) * 100 if len(ce) > 0 else 0
    print(f"\n  Coulombic Efficiency:")
    print(f"    Range checked : [80, 102] %")
    print(f"    Valid rows    : {len(ce):,}")
    print(f"    Out of range  : {ce_outliers:,} ({ce_pct:.3f}%)")
    if len(ce) > 0 and ce.mean() < 99:
        print(f"    WARNING: Low mean CE = {ce.mean():.2f}% (expected >99% for healthy cells)")

# Check temperatures
if COL_MAP.get('max_temp') and COL_MAP['max_temp'] in cycle_data.columns:
    temp_col = COL_MAP['max_temp']
    temp_max = cycle_data[temp_col].dropna()
    temp_outliers = ((temp_max < TEMP_MIN) | (temp_max > TEMP_MAX)).sum()
    temp_pct = temp_outliers / len(temp_max) * 100 if len(temp_max) > 0 else 0
    print(f"\n  Temperature (Max):")
    print(f"    Range checked : [{TEMP_MIN}, {TEMP_MAX}] °C")
    print(f"    Valid rows    : {len(temp_max):,}")
    print(f"    Out of range  : {temp_outliers:,} ({temp_pct:.3f}%)")
    if len(temp_max) > 0:
        print(f"    Temperature range: {temp_max.min():.1f}°C – {temp_max.max():.1f}°C")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — CAPACITY FADE & SOH ANALYSIS (FIXED)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 7 — CAPACITY FADE & SOH")
print("=" * 70)

if COL_MAP.get('discharge_capacity') and COL_MAP['discharge_capacity'] in cycle_data.columns:
    cap_col = COL_MAP['discharge_capacity']
    cycle_col = COL_MAP.get('cycle_index', 'Cycle_Index')
    
    # Try to find cycle column
    if cycle_col not in cycle_data.columns:
        # Look for any column with 'cycle' in name
        for col in cycle_data.columns:
            if 'cycle' in col.lower():
                cycle_col = col
                break
    
    # For each cell, find initial capacity and compute SOH
    soh_records = []
    
    for cell_id, grp in cycle_data.groupby('file_source'):
        # Get first valid discharge capacity (skip zero values which are reference cycles)
        valid_caps = grp[grp[cap_col] > 0.1][cap_col].dropna()  # Ignore zeros (reference cycles)
        if len(valid_caps) == 0:
            continue
        
        initial_cap = valid_caps.iloc[0]
        if initial_cap <= 0:
            continue
        
        # Compute SOH for each cycle (cap/initial_cap), but clamp to reasonable range
        grp = grp.copy()
        grp['SOH'] = (grp[cap_col] / initial_cap).clip(0, 1.2)  # Cap at 1.2 to handle measurement noise
        
        # Get cycle number if available
        if cycle_col in grp.columns:
            grp['cycle_number'] = grp[cycle_col]
        else:
            grp['cycle_number'] = grp.index
        
        # Only keep rows with valid SOH
        valid_grp = grp[grp['SOH'] > 0].copy()
        
        if len(valid_grp) > 0:
            soh_records.append(valid_grp[['file_source', 'chemistry', 'temperature_C', 
                                          'DoD_range', 'charge_Crate', 'discharge_Crate',
                                          'cycle_number', cap_col, 'SOH']])
    
    if soh_records:
        soh_df = pd.concat(soh_records, ignore_index=True)
        print(f"SOH computed for {soh_df['file_source'].nunique()} cells")
        print(f"Total cycles with valid SOH: {len(soh_df):,}")
        
        # SOH statistics per cell (keep chemistry info)
        soh_stats = soh_df.groupby('file_source').agg({
            'SOH': ['min', 'max', 'mean'],
            'cycle_number': 'max',
            'chemistry': 'first',  # Keep chemistry
            'temperature_C': 'first',  # Keep temperature
            'DoD_range': 'first'  # Keep DoD range
        }).round(3)
        soh_stats.columns = ['SOH_min', 'SOH_max', 'SOH_mean', 'total_cycles', 
                            'chemistry', 'temperature_C', 'DoD_range']
        
        print("\nSOH statistics per cell (first 10):")
        print(soh_stats.head(10).to_string())
        
        # Count cells reaching EOL (SOH < 0.8)
        cells_at_eol = soh_stats[soh_stats['SOH_min'] < 0.8].shape[0]
        print(f"\nCells that reached EOL (SOH < 80%): {cells_at_eol} / {len(soh_stats)}")
        
        # Capacity fade visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Capacity fade by chemistry (filter out zeros)
        ax1 = axes[0, 0]
        for chem in soh_df['chemistry'].unique():
            if chem != 'unknown':
                chem_data = soh_df[soh_df['chemistry'] == chem]
                chem_data = chem_data[chem_data[cap_col] > 0.1]  # Filter out zeros
                # Sample every 10th cycle for cleaner plot if many cycles
                if len(chem_data) > 0:
                    sample = chem_data.iloc[::max(1, len(chem_data)//500)]
                    ax1.plot(sample['cycle_number'], sample[cap_col], 
                            '.', markersize=1, alpha=0.5, label=chem)
        ax1.set_xlabel('Cycle Number')
        ax1.set_ylabel('Discharge Capacity (Ah)')
        ax1.set_title('Capacity Fade by Chemistry')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: SOH degradation by temperature
        ax2 = axes[0, 1]
        for temp in sorted(soh_df['temperature_C'].unique()):
            if temp > -100:  # valid temperature
                temp_data = soh_df[soh_df['temperature_C'] == temp]
                temp_data = temp_data[temp_data['SOH'] > 0]  # Filter valid SOH
                if not temp_data.empty:
                    sample = temp_data.iloc[::max(1, len(temp_data)//300)]
                    ax2.plot(sample['cycle_number'], sample['SOH'], 
                            '.', markersize=1, alpha=0.5, label=f'{temp}°C')
        ax2.axhline(0.8, color='red', linestyle='--', linewidth=1, label='EOL (80%)')
        ax2.set_xlabel('Cycle Number')
        ax2.set_ylabel('State of Health (SOH)')
        ax2.set_title('SOH Degradation by Temperature')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.1)
        
        # Plot 3: Capacity fade by DoD range
        ax3 = axes[1, 0]
        for dod in soh_df['DoD_range'].unique():
            if dod != 'unknown':
                dod_data = soh_df[soh_df['DoD_range'] == dod]
                dod_data = dod_data[dod_data[cap_col] > 0.1]
                if not dod_data.empty:
                    sample = dod_data.iloc[::max(1, len(dod_data)//300)]
                    ax3.plot(sample['cycle_number'], sample[cap_col], 
                            '.', markersize=1, alpha=0.5, label=dod)
        ax3.set_xlabel('Cycle Number')
        ax3.set_ylabel('Discharge Capacity (Ah)')
        ax3.set_title('Capacity Fade by Depth of Discharge')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Cycle life distribution by chemistry
        ax4 = axes[1, 1]
        # Filter to cells that actually degraded to EOL for meaningful cycle life
        cycle_life = soh_stats[soh_stats['SOH_min'] < 0.8].groupby('chemistry')['total_cycles'].agg(['mean', 'std', 'count'])
        if len(cycle_life) > 0:
            x_pos = np.arange(len(cycle_life))
            bars = ax4.bar(x_pos, cycle_life['mean'], yerr=cycle_life['std'], 
                          capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(cycle_life.index)
            ax4.set_ylabel('Cycle Life (cycles to EOL)')
            ax4.set_title('Cycle Life by Chemistry (cells reaching EOL)')
            ax4.grid(True, alpha=0.3, axis='y')
        else:
            ax4.text(0.5, 0.5, 'No cells reached EOL yet', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Cycle Life by Chemistry')
        
        plt.suptitle('SNL Battery Dataset — Capacity Fade & SOH Analysis', fontsize=12, y=1.02)
        plt.tight_layout()
        savefig("01_capacity_fade_analysis.png")
        
        # Individual cell fade plots (for selected cells)
        n_cells = min(9, len(soh_df['file_source'].unique()))
        selected_cells = soh_df['file_source'].unique()[:n_cells]
        
        fig, axes = plt.subplots(3, 3, figsize=(12, 10))
        axes_flat = axes.flatten()
        
        for i, cell in enumerate(selected_cells):
            cell_data = soh_df[soh_df['file_source'] == cell]
            cell_data = cell_data[cell_data[cap_col] > 0.1]  # Filter zeros
            if len(cell_data) > 0:
                chem = cell_data['chemistry'].iloc[0] if 'chemistry' in cell_data.columns else 'unknown'
                temp = cell_data['temperature_C'].iloc[0] if 'temperature_C' in cell_data.columns else 0
                
                axes_flat[i].plot(cell_data['cycle_number'], cell_data[cap_col], 
                                 'b-', linewidth=1, alpha=0.7)
                initial_cap = cell_data[cap_col].iloc[0] if len(cell_data) > 0 else 1
                axes_flat[i].axhline(0.8 * initial_cap, 
                                    color='r', linestyle='--', linewidth=0.8, label='EOL (80%)')
                axes_flat[i].set_title(f'{os.path.basename(cell)[:30]}\n{chem}, {temp}°C')
                axes_flat[i].set_xlabel('Cycle')
                axes_flat[i].set_ylabel('Capacity (Ah)')
                axes_flat[i].grid(True, alpha=0.3)
                axes_flat[i].legend(fontsize=6)
        
        for i in range(len(selected_cells), len(axes_flat)):
            axes_flat[i].set_visible(False)
        
        plt.suptitle('Individual Cell Capacity Fade Curves', fontsize=12)
        plt.tight_layout()
        savefig("02_individual_cell_fade.png")
    else:
        print("Could not compute SOH - no valid capacity data found")
else:
    print("Discharge capacity column not found - skipping SOH analysis")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — EFFICIENCY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 8 — EFFICIENCY ANALYSIS")
print("=" * 70)

if COL_MAP.get('coulombic_efficiency') and COL_MAP['coulombic_efficiency'] in cycle_data.columns:
    ce_col = COL_MAP['coulombic_efficiency']
    ce_by_chem = cycle_data.groupby('chemistry')[ce_col].agg(['mean', 'std', 'min', 'max'])
    print("Coulombic Efficiency by Chemistry:")
    print(ce_by_chem.round(2).to_string())
    
    # Efficiency vs Cycle plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Coulombic efficiency over cycles
    ax1 = axes[0]
    cycle_col = COL_MAP.get('cycle_index', 'Cycle_Index')
    if cycle_col not in cycle_data.columns:
        for col in cycle_data.columns:
            if 'cycle' in col.lower():
                cycle_col = col
                break
    
    for chem in cycle_data['chemistry'].unique():
        if chem != 'unknown':
            chem_data = cycle_data[cycle_data['chemistry'] == chem].dropna(subset=[ce_col])
            if not chem_data.empty:
                sample = chem_data.iloc[::max(1, len(chem_data)//300)]
                ax1.plot(sample[cycle_col] if cycle_col in sample.columns else range(len(sample)), 
                        sample[ce_col], '.', markersize=1, alpha=0.5, label=chem)
    ax1.set_xlabel('Cycle Number')
    ax1.set_ylabel('Coulombic Efficiency (%)')
    ax1.set_title('Coulombic Efficiency vs Cycle')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(95, 101)
    
    # Efficiency distribution
    ax2 = axes[1]
    for chem in cycle_data['chemistry'].unique():
        if chem != 'unknown':
            chem_ce = cycle_data[cycle_data['chemistry'] == chem][ce_col].dropna()
            if not chem_ce.empty:
                ax2.hist(chem_ce, bins=50, alpha=0.5, label=chem)
    ax2.set_xlabel('Coulombic Efficiency (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Coulombic Efficiency Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('SNL Battery Dataset — Efficiency Analysis', fontsize=12)
    plt.tight_layout()
    savefig("03_efficiency_analysis.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — TEMPERATURE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 9 — TEMPERATURE ANALYSIS")
print("=" * 70)

temp_cols_found = []
for temp_type in ['max_temp', 'min_temp', 'avg_temp']:
    if temp_type in COL_MAP and COL_MAP[temp_type] in cycle_data.columns:
        temp_cols_found.append(COL_MAP[temp_type])

if temp_cols_found:
    temp_stats = cycle_data[temp_cols_found].describe()
    print("Temperature statistics (all data):")
    print(temp_stats.round(1).to_string())
    
    # Temperature rise (max - min)
    if COL_MAP.get('max_temp') and COL_MAP.get('min_temp'):
        if COL_MAP['max_temp'] in cycle_data.columns and COL_MAP['min_temp'] in cycle_data.columns:
            cycle_data['Temp_Rise(°C)'] = cycle_data[COL_MAP['max_temp']] - cycle_data[COL_MAP['min_temp']]
            print(f"\nAverage temperature rise per cycle: {cycle_data['Temp_Rise(°C)'].mean():.1f}°C")
            print(f"Max temperature rise: {cycle_data['Temp_Rise(°C)'].max():.1f}°C")
    
    # Temperature by chemistry and C-rate
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Max temperature by chemistry
    ax1 = axes[0]
    if COL_MAP.get('max_temp'):
        max_temp_col = COL_MAP['max_temp']
        data_to_plot = [cycle_data[cycle_data['chemistry'] == chem][max_temp_col].dropna() 
                       for chem in cycle_data['chemistry'].unique() if chem != 'unknown']
        labels = [chem for chem in cycle_data['chemistry'].unique() if chem != 'unknown']
        if data_to_plot:
            ax1.boxplot(data_to_plot, labels=labels)
            ax1.set_title('Max Temperature Distribution by Chemistry')
            ax1.set_ylabel('Temperature (°C)')
            ax1.grid(True, alpha=0.3)
    
    # Temperature vs Discharge C-rate
    ax2 = axes[1]
    if COL_MAP.get('max_temp'):
        for chem in cycle_data['chemistry'].unique():
            if chem != 'unknown':
                chem_data = cycle_data[cycle_data['chemistry'] == chem].dropna(subset=['discharge_Crate', COL_MAP['max_temp']])
                if not chem_data.empty:
                    ax2.scatter(chem_data['discharge_Crate'], chem_data[COL_MAP['max_temp']], 
                               alpha=0.3, s=10, label=chem)
        ax2.set_xlabel('Discharge C-rate')
        ax2.set_ylabel('Max Temperature (°C)')
        ax2.set_title('Temperature vs Discharge C-rate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle('SNL Battery Dataset — Thermal Analysis', fontsize=12)
    plt.tight_layout()
    savefig("04_temperature_analysis.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — SUMMARY REPORT (FIXED VERSION)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 10 — SUMMARY REPORT")
print("=" * 70)

# Safely build summary statistics to avoid f-string brace issues
n_batteries = cycle_data['file_source'].nunique() if 'file_source' in cycle_data.columns else 'N/A'

chemistries = 'N/A'
if 'chemistry' in cycle_data.columns:
    chem_list = [c for c in cycle_data['chemistry'].unique() if c != 'unknown']
    chemistries = ', '.join(chem_list) if chem_list else 'N/A'

temperatures = 'N/A'
if 'temperature_C' in cycle_data.columns:
    temp_list = sorted([t for t in cycle_data['temperature_C'].unique() if t > -100])
    temperatures = str(temp_list) if temp_list else 'N/A'

dod_ranges = 'N/A'
if 'DoD_range' in cycle_data.columns:
    dod_ranges = str(cycle_data['DoD_range'].unique().tolist())

charge_crates = 'N/A'
if 'charge_Crate' in cycle_data.columns:
    charge_crates = str(sorted(cycle_data['charge_Crate'].unique()))

discharge_crates = 'N/A'
if 'discharge_Crate' in cycle_data.columns:
    discharge_crates = str(sorted(cycle_data['discharge_Crate'].unique()))

coulombic_str = "N/A"
if 'coulombic_efficiency' in COL_MAP and COL_MAP['coulombic_efficiency'] in cycle_data.columns:
    ce_col = COL_MAP['coulombic_efficiency']
    if len(cycle_data[ce_col].dropna()) > 0:
        coulombic_str = f"{cycle_data[ce_col].mean():.2f} ± {cycle_data[ce_col].std():.2f}%"

energy_str = "Available" if ('energy_efficiency' in COL_MAP and COL_MAP['energy_efficiency'] in cycle_data.columns) else "Not available"

temp_range_str = "N/A"
if ('min_temp' in COL_MAP and 'max_temp' in COL_MAP and 
    COL_MAP['min_temp'] in cycle_data.columns and 
    COL_MAP['max_temp'] in cycle_data.columns):
    min_col = COL_MAP['min_temp']
    max_col = COL_MAP['max_temp']
    if len(cycle_data[min_col].dropna()) > 0 and len(cycle_data[max_col].dropna()) > 0:
        temp_range_str = f"{cycle_data[min_col].min():.1f} – {cycle_data[max_col].max():.1f} °C"

print(f"""
Dataset      : SNL Battery Dataset (Sandia National Laboratories)
Files        : {len(cycle_files)} cycle data files, {len(timeseries_files)} timeseries files
Total cycles : {len(cycle_data):,}
Batteries    : {n_batteries}
Chemistries  : {chemistries}

Test Conditions:
  Temperatures      : {temperatures} °C
  DoD Ranges        : {dod_ranges}
  Charge C-rates    : {charge_crates}C
  Discharge C-rates : {discharge_crates}C

Efficiency:
  Coulombic efficiency  : {coulombic_str}
  Energy efficiency     : {energy_str}

Temperature:
  Operating range       : {temp_range_str}
""")

if SAVE_PLOTS:
    print(f"All plots saved to: {OUT_DIR}")
print("Done.")