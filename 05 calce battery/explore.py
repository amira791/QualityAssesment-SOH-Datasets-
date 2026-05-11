"""
CALCE Battery Dataset — Exploration Script
============================================
Run from the folder containing this script, or set DATASET_PATH below.
"""

import os
import glob
import re
import warnings
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

# ── ① USER CONFIG ────────────────────────────────────────────────────────────
DATASET_PATH = r"C:\Users\admin\Desktop\DR2\11 All Datasets\03 CALCE Battery Dataset\dataset calce"
SAVE_PLOTS   = True
OUT_DIR      = os.path.join(DATASET_PATH, "exploration_outputs")
# ─────────────────────────────────────────────────────────────────────────────

# ── Physical plausibility limits (LCO and LFP cells) ─────────────────────────
CELL_V_MIN, CELL_V_MAX = 2.5, 4.2      # V (covering both chemistries)
TEMP_MIN,  TEMP_MAX    = -20, 60       # °C operating bounds
MIN_CAPACITY, MAX_CAPACITY = 0.8, 1.5  # Ah (based on 1100-1350mAh cells)
MIN_CE, MAX_CE = 80, 102              # Coulombic efficiency range (%)

# Typical capacities by battery type
NOMINAL_CAP = {
    'CX2-16': 1.1,
    'CX2-25': 1.1,
    'CX2-33': 1.35,
    'CX2-34': 1.35,
    'CX2-36': 1.1,
    'CX2-37': 1.1,
    'CX2-38': 1.1
}
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
    Parse CALCE filename convention:
    CALCE_{CELL_ID}_prism_{CHEMISTRY}_{TEMP}C_{DOD}_{CHARGE_C}-{DISCHARGE_C}_{REPLICATE}_{TYPE}.csv
    """
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
    else:
        return None

def clean_column_names(df):
    """Clean column names by stripping whitespace"""
    df.columns = df.columns.str.strip()
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — INVENTORY
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SECTION 1 — FILE INVENTORY")
print("=" * 70)

all_csv_files = sorted(glob.glob(os.path.join(DATASET_PATH, "*.csv")))

cycle_files = [f for f in all_csv_files if 'cycle_data' in f]
timeseries_files = [f for f in all_csv_files if 'timeseries' in f]

print(f"\nTotal CSV files found: {len(all_csv_files)}")
print(f"  Cycle data files:   {len(cycle_files)}")
print(f"  Timeseries files:   {len(timeseries_files)}")

inventory = []
for f in cycle_files + timeseries_files:
    size_kb = os.path.getsize(f) / 1024
    filename = os.path.basename(f)
    parsed = parse_filename(filename)
    
    with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
        nrows = sum(1 for _ in fh) - 1
    
    row = {
        'file': filename,
        'size_KB': round(size_kb, 1),
        'rows': nrows,
        'type': 'cycle' if 'cycle_data' in filename else 'timeseries'
    }
    
    if parsed:
        row.update(parsed)
    else:
        row['cell_id'] = 'unknown'
        row['chemistry'] = 'unknown'
        row['temperature'] = -999
        row['DoD_range'] = 'unknown'
        row['charge_Crate'] = -1
        row['discharge_Crate'] = -1
        row['replicate'] = 'unknown'
    
    inventory.append(row)

inv_df = pd.DataFrame(inventory)
print(f"\nFile inventory summary:")
summary = inv_df.groupby(['cell_id', 'type']).agg({
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

sample_cycle = None
sample_timeseries = None

for f in cycle_files[:1]:
    sample_cycle = f
    break
for f in timeseries_files[:1]:
    sample_timeseries = f
    break

COL_MAP = {}

if sample_cycle:
    print(f"\n--- CYCLE DATA FILE SCHEMA ---")
    print(f"File: {os.path.basename(sample_cycle)}")
    df_cycle_sample = pd.read_csv(sample_cycle, nrows=5)
    df_cycle_sample = clean_column_names(df_cycle_sample)
    print(f"Columns ({len(df_cycle_sample.columns)}): {list(df_cycle_sample.columns)}")
    print("\nFirst 5 rows:")
    print(df_cycle_sample.head(5).to_string())
    print("\nData types:")
    print(df_cycle_sample.dtypes)
    
    for col in df_cycle_sample.columns:
        col_lower = col.lower()
        if 'cycle_index' in col_lower or col_lower == 'cycle':
            COL_MAP['cycle_index'] = col
        elif 'discharge_capacity' in col_lower:
            COL_MAP['discharge_capacity'] = col
        elif 'charge_capacity' in col_lower:
            COL_MAP['charge_capacity'] = col
        elif 'discharge_energy' in col_lower:
            COL_MAP['discharge_energy'] = col
        elif 'charge_energy' in col_lower:
            COL_MAP['charge_energy'] = col
        elif 'coulombic_efficiency' in col_lower:
            COL_MAP['coulombic_efficiency'] = col
        elif 'energy_efficiency' in col_lower:
            COL_MAP['energy_efficiency'] = col
        elif 'max_temperature' in col_lower:
            COL_MAP['max_temp'] = col
        elif 'min_temperature' in col_lower:
            COL_MAP['min_temp'] = col
        elif 'avg_temperature' in col_lower:
            COL_MAP['avg_temp'] = col
    
    print(f"\nDetected column mapping: {COL_MAP}")

if sample_timeseries:
    print(f"\n--- TIMESERIES FILE SCHEMA ---")
    print(f"File: {os.path.basename(sample_timeseries)}")
    df_ts_sample = pd.read_csv(sample_timeseries, nrows=5)
    df_ts_sample = clean_column_names(df_ts_sample)
    print(f"Columns ({len(df_ts_sample.columns)}): {list(df_ts_sample.columns)}")
    print("\nFirst 5 rows:")
    print(df_ts_sample.head(5).to_string())
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
        
        df['file_source'] = filename
        if parsed:
            df['cell_id'] = parsed['cell_id']
            df['chemistry'] = parsed['chemistry']
            df['temperature_C'] = parsed['temperature']
            df['DoD_range'] = parsed['DoD_range']
            df['charge_Crate'] = parsed['charge_Crate']
            df['discharge_Crate'] = parsed['discharge_Crate']
            df['replicate'] = parsed['replicate']
        else:
            df['cell_id'] = 'unknown'
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
    print(f"Unique batteries/cells: {cycle_data['cell_id'].nunique()}")
    print(f"Unique chemistries: {cycle_data['chemistry'].unique()}")
    print(f"Unique cells: {sorted(cycle_data['cell_id'].unique())}")
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

critical_cols = [COL_MAP.get('discharge_capacity', 'Discharge_Capacity(Ah)'),
                 COL_MAP.get('coulombic_efficiency', 'Coulombic_Efficiency(%)'),
                 COL_MAP.get('max_temp', 'Max_Temperature(°C)')]

critical_cols_exist = [c for c in critical_cols if c in cycle_data.columns]

if critical_cols_exist and 'cell_id' in cycle_data.columns:
    missing_per_file = cycle_data.groupby('cell_id')[critical_cols_exist].apply(
        lambda g: g.isnull().sum()
    )
    missing_with_missing = missing_per_file[missing_per_file.sum(axis=1) > 0]
    if not missing_with_missing.empty:
        print("\nMissing values per cell (critical columns):")
        print(missing_with_missing.to_string())
    else:
        print("\nNo missing values in critical columns across all cells")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — BASIC STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 5 — BASIC STATISTICS")
print("=" * 70)

numeric_cols = cycle_data.select_dtypes(include=[np.number]).columns.tolist()
exclude_cols = ['temperature_C', 'charge_Crate', 'discharge_Crate']
stats_cols = [c for c in numeric_cols if c not in exclude_cols]

if stats_cols:
    stats = cycle_data[stats_cols].describe().T
    print("Overall statistics for all cycle data:")
    print(stats.to_string())
else:
    print("No numeric columns found for statistics")

print("\n" + "-" * 50)
print("Statistics by cell:")
for cell in sorted(cycle_data['cell_id'].unique()):
    cell_data_subset = cycle_data[cycle_data['cell_id'] == cell]
    print(f"\n{cell}:")
    
    if COL_MAP.get('discharge_capacity') and COL_MAP['discharge_capacity'] in cell_data_subset.columns:
        cap_col = COL_MAP['discharge_capacity']
        print(f"  Cycles: {len(cell_data_subset):,}")
        print(f"  Discharge Capacity: {cell_data_subset[cap_col].mean():.3f} ± {cell_data_subset[cap_col].std():.3f} Ah")
        print(f"  Initial Capacity: {cell_data_subset[cap_col].iloc[0]:.3f} Ah")
        print(f"  Final Capacity: {cell_data_subset[cap_col].iloc[-1]:.3f} Ah")
    
    if COL_MAP.get('coulombic_efficiency') and COL_MAP['coulombic_efficiency'] in cell_data_subset.columns:
        ce_col = COL_MAP['coulombic_efficiency']
        print(f"  Coulombic Efficiency: {cell_data_subset[ce_col].mean():.2f} ± {cell_data_subset[ce_col].std():.2f} %")
    
    if COL_MAP.get('max_temp') and COL_MAP['max_temp'] in cell_data_subset.columns:
        temp_col = COL_MAP['max_temp']
        print(f"  Max Temperature: {cell_data_subset[temp_col].mean():.1f} ± {cell_data_subset[temp_col].std():.1f} °C")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — PHYSICAL PLAUSIBILITY CHECKS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 6 — PHYSICAL PLAUSIBILITY CHECKS")
print("=" * 70)

if COL_MAP.get('discharge_capacity') and COL_MAP['discharge_capacity'] in cycle_data.columns:
    cap_col = COL_MAP['discharge_capacity']
    cap = cycle_data[cap_col].dropna()
    cap_outliers = ((cap < MIN_CAPACITY) | (cap > MAX_CAPACITY)).sum()
    cap_pct = cap_outliers / len(cap) * 100 if len(cap) > 0 else 0
    print(f"  Discharge Capacity [{MIN_CAPACITY}, {MAX_CAPACITY}] Ah:")
    print(f"    Out of range: {cap_outliers:,}/{len(cap):,} ({cap_pct:.3f}%)")
    print(f"    Range: {cap.min():.3f} – {cap.max():.3f} Ah")

if COL_MAP.get('coulombic_efficiency') and COL_MAP['coulombic_efficiency'] in cycle_data.columns:
    ce_col = COL_MAP['coulombic_efficiency']
    ce = cycle_data[ce_col].dropna()
    ce_outliers = ((ce < MIN_CE) | (ce > MAX_CE)).sum()
    ce_pct = ce_outliers / len(ce) * 100 if len(ce) > 0 else 0
    print(f"\n  Coulombic Efficiency [{MIN_CE}, {MAX_CE}] %:")
    print(f"    Out of range: {ce_outliers:,}/{len(ce):,} ({ce_pct:.3f}%)")
    print(f"    Mean CE: {ce.mean():.2f}%")

if COL_MAP.get('max_temp') and COL_MAP['max_temp'] in cycle_data.columns:
    temp_col = COL_MAP['max_temp']
    temp_max = cycle_data[temp_col].dropna()
    temp_outliers = ((temp_max < TEMP_MIN) | (temp_max > TEMP_MAX)).sum()
    temp_pct = temp_outliers / len(temp_max) * 100 if len(temp_max) > 0 else 0
    print(f"\n  Temperature (Max) [{TEMP_MIN}, {TEMP_MAX}] °C:")
    print(f"    Out of range: {temp_outliers:,}/{len(temp_max):,} ({temp_pct:.3f}%)")
    print(f"    Range: {temp_max.min():.1f} – {temp_max.max():.1f} °C")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — CAPACITY FADE & SOH ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 7 — CAPACITY FADE & SOH")
print("=" * 70)

if COL_MAP.get('discharge_capacity') and COL_MAP.get('cycle_index'):
    cap_col = COL_MAP['discharge_capacity']
    cycle_col = COL_MAP['cycle_index']
    
    soh_records = []
    
    for cell_id, grp in cycle_data.groupby('cell_id'):
        valid_caps = grp[cap_col].dropna()
        if len(valid_caps) == 0:
            continue
        
        initial_cap = valid_caps.iloc[0]
        if initial_cap <= 0:
            continue
        
        grp = grp.copy()
        grp['SOH'] = grp[cap_col] / initial_cap
        nominal = NOMINAL_CAP.get(cell_id, 1.1)
        grp['SOH_vs_nominal'] = grp[cap_col] / nominal
        
        soh_records.append(grp[['cell_id', 'chemistry', 'temperature_C', 
                                'DoD_range', 'charge_Crate', 'discharge_Crate',
                                cycle_col, cap_col, 'SOH', 'SOH_vs_nominal']])
    
    if soh_records:
        soh_df = pd.concat(soh_records, ignore_index=True)
        print(f"SOH computed for {soh_df['cell_id'].nunique()} cells")
        print(f"Total cycles with SOH: {len(soh_df):,}")
        
        soh_stats = soh_df.groupby('cell_id').agg({
            'SOH': ['min', 'max', 'mean'],
            cycle_col: 'max',
            'SOH_vs_nominal': ['min', 'max', 'mean']
        }).round(3)
        soh_stats.columns = ['SOH_min', 'SOH_max', 'SOH_mean', 'total_cycles',
                            'SOH_nom_min', 'SOH_nom_max', 'SOH_nom_mean']
        
        print("\nSOH statistics per cell:")
        print(soh_stats.to_string())
        
        cells_at_eol = soh_stats[soh_stats['SOH_min'] < 0.8].shape[0]
        print(f"\nCells that reached EOL (SOH < 80% of initial): {cells_at_eol} / {len(soh_stats)}")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        ax1 = axes[0, 0]
        for cell in soh_df['cell_id'].unique():
            cell_data = soh_df[soh_df['cell_id'] == cell]
            ax1.plot(cell_data[cycle_col], cell_data[cap_col], 
                    '-', linewidth=1, label=cell, alpha=0.7)
        ax1.set_xlabel('Cycle Number')
        ax1.set_ylabel('Discharge Capacity (Ah)')
        ax1.set_title('Capacity Fade by Cell')
        ax1.legend(fontsize=7, loc='best')
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        for cell in soh_df['cell_id'].unique():
            cell_data = soh_df[soh_df['cell_id'] == cell]
            ax2.plot(cell_data[cycle_col], cell_data['SOH'], 
                    '-', linewidth=1, label=cell, alpha=0.7)
        ax2.axhline(0.8, color='red', linestyle='--', linewidth=1, label='EOL (80%)')
        ax2.set_xlabel('Cycle Number')
        ax2.set_ylabel('State of Health (SOH)')
        ax2.set_title('SOH Degradation')
        ax2.legend(fontsize=7, loc='best')
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[1, 0]
        cells = soh_stats.index.tolist()
        eol_cycles = soh_stats['total_cycles'].values
        bars = ax3.bar(cells, eol_cycles, alpha=0.7, color='steelblue')
        ax3.set_xlabel('Cell ID')
        ax3.set_ylabel('Total Cycles')
        ax3.set_title('Cycle Life Comparison')
        ax3.set_xticklabels(cells, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, eol_cycles):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f'{int(val)}', ha='center', va='bottom', fontsize=8)
        
        ax4 = axes[1, 1]
        x = np.arange(len(cells))
        width = 0.35
        initial_caps = [soh_df[soh_df['cell_id'] == cell][cap_col].iloc[0] for cell in cells]
        final_caps = [soh_df[soh_df['cell_id'] == cell][cap_col].iloc[-1] for cell in cells]
        
        ax4.bar(x - width/2, initial_caps, width, label='Initial', alpha=0.7, color='green')
        ax4.bar(x + width/2, final_caps, width, label='Final', alpha=0.7, color='orange')
        ax4.set_xlabel('Cell ID')
        ax4.set_ylabel('Capacity (Ah)')
        ax4.set_title('Capacity Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(cells, rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('CALCE Battery Dataset - Capacity Fade & SOH Analysis', fontsize=12, y=1.02)
        plt.tight_layout()
        savefig("01_capacity_fade_analysis.png")
        
        # Individual cell fade plots
        n_cells = len(soh_df['cell_id'].unique())
        n_cols = min(3, n_cells)
        n_rows = (n_cells + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4))
        if n_cells == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, cell in enumerate(soh_df['cell_id'].unique()):
            cell_data = soh_df[soh_df['cell_id'] == cell]
            ax = axes[i]
            ax.plot(cell_data[cycle_col], cell_data[cap_col], 'b-', linewidth=1.5)
            initial_cap = cell_data[cap_col].iloc[0]
            ax.axhline(0.8 * initial_cap, color='r', linestyle='--', linewidth=1, label='EOL (80%)')
            ax.set_xlabel('Cycle')
            ax.set_ylabel('Capacity (Ah)')
            ax.set_title(f"{cell}\nInitial: {initial_cap:.3f}Ah, Final: {cell_data[cap_col].iloc[-1]:.3f}Ah")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        
        for i in range(len(soh_df['cell_id'].unique()), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Individual Cell Capacity Fade', fontsize=12)
        plt.tight_layout()
        savefig("02_individual_cell_fade.png")
    else:
        print("Could not compute SOH - no valid capacity data found")
else:
    print("Discharge capacity or cycle index column not found - skipping SOH analysis")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — EFFICIENCY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 8 — EFFICIENCY ANALYSIS")
print("=" * 70)

if COL_MAP.get('coulombic_efficiency') and COL_MAP['coulombic_efficiency'] in cycle_data.columns:
    ce_col = COL_MAP['coulombic_efficiency']
    ce_by_cell = cycle_data.groupby('cell_id')[ce_col].agg(['mean', 'std', 'min', 'max'])
    print("Coulombic Efficiency by Cell:")
    print(ce_by_cell.round(2).to_string())
    
    if COL_MAP.get('energy_efficiency') and COL_MAP['energy_efficiency'] in cycle_data.columns:
        ee_col = COL_MAP['energy_efficiency']
        ee_by_cell = cycle_data.groupby('cell_id')[ee_col].agg(['mean', 'std', 'min', 'max'])
        print("\nEnergy Efficiency by Cell:")
        print(ee_by_cell.round(2).to_string())
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1 = axes[0]
    cycle_col = COL_MAP.get('cycle_index', 'Cycle_Index')
    for cell in cycle_data['cell_id'].unique():
        cell_data = cycle_data[cycle_data['cell_id'] == cell].dropna(subset=[ce_col])
        if not cell_data.empty:
            ax1.plot(cell_data[cycle_col], cell_data[ce_col], 
                    '.', markersize=1, alpha=0.5, label=cell)
    ax1.set_xlabel('Cycle Number')
    ax1.set_ylabel('Coulombic Efficiency (%)')
    ax1.set_title('Coulombic Efficiency vs Cycle')
    ax1.legend(fontsize=7, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(80, 102)
    
    ax2 = axes[1]
    for cell in cycle_data['cell_id'].unique():
        cell_ce = cycle_data[cycle_data['cell_id'] == cell][ce_col].dropna()
        if not cell_ce.empty:
            ax2.hist(cell_ce, bins=50, alpha=0.5, label=cell, density=True)
    ax2.set_xlabel('Coulombic Efficiency (%)')
    ax2.set_ylabel('Density')
    ax2.set_title('Coulombic Efficiency Distribution')
    ax2.legend(fontsize=7, loc='best')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('CALCE Battery Dataset - Efficiency Analysis', fontsize=12)
    plt.tight_layout()
    savefig("03_efficiency_analysis.png")
else:
    print("Coulombic efficiency column not found - skipping efficiency analysis")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — TEMPERATURE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 9 — TEMPERATURE ANALYSIS")
print("=" * 70)

temp_cols = []
for temp_type in ['max_temp', 'min_temp', 'avg_temp']:
    if temp_type in COL_MAP and COL_MAP[temp_type] in cycle_data.columns:
        temp_cols.append(COL_MAP[temp_type])

if temp_cols:
    temp_stats = cycle_data[temp_cols].describe()
    print("Temperature statistics (all data):")
    print(temp_stats.round(1).to_string())
    
    if COL_MAP.get('max_temp') and COL_MAP.get('min_temp'):
        if COL_MAP['max_temp'] in cycle_data.columns and COL_MAP['min_temp'] in cycle_data.columns:
            cycle_data['Temp_Rise(°C)'] = cycle_data[COL_MAP['max_temp']] - cycle_data[COL_MAP['min_temp']]
            print(f"\nAverage temperature rise per cycle: {cycle_data['Temp_Rise(°C)'].mean():.1f}°C")
            print(f"Max temperature rise: {cycle_data['Temp_Rise(°C)'].max():.1f}°C")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1 = axes[0]
    if COL_MAP.get('max_temp'):
        max_temp_col = COL_MAP['max_temp']
        cells = sorted(cycle_data['cell_id'].unique())
        temp_data = [cycle_data[cycle_data['cell_id'] == cell][max_temp_col].dropna() for cell in cells]
        bp = ax1.boxplot(temp_data, labels=cells, rot=45)
        ax1.set_title('Max Temperature Distribution by Cell')
        ax1.set_ylabel('Temperature (°C)')
        ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    if 'Temp_Rise(°C)' in cycle_data.columns:
        rise_by_cell = cycle_data.groupby('cell_id')['Temp_Rise(°C)'].agg(['mean', 'std'])
        cells = rise_by_cell.index.tolist()
        x = np.arange(len(cells))
        ax2.bar(x, rise_by_cell['mean'], yerr=rise_by_cell['std'], 
                capsize=5, alpha=0.7, color='coral')
        ax2.set_xlabel('Cell ID')
        ax2.set_ylabel('Temperature Rise (°C)')
        ax2.set_title('Average Temperature Rise per Cycle')
        ax2.set_xticks(x)
        ax2.set_xticklabels(cells, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('CALCE Battery Dataset - Temperature Analysis', fontsize=12)
    plt.tight_layout()
    savefig("04_temperature_analysis.png")
else:
    print("Temperature columns not found - skipping temperature analysis")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — TEST CONDITION DIVERSITY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 10 — TEST CONDITION DIVERSITY")
print("=" * 70)

condition_summary = cycle_data.groupby(['cell_id', 'chemistry', 'temperature_C', 
                                        'DoD_range', 'charge_Crate', 'discharge_Crate']).agg({
    'file_source': 'nunique'
}).rename(columns={'file_source': 'n_files'})

print("Test condition matrix:")
print(f"  Total unique conditions: {len(condition_summary)}")
print(f"\nCondition summary by cell:")
print(condition_summary.to_string())


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 11 — SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 11 — SUMMARY REPORT")
print("=" * 70)

# Build summary strings safely
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
Dataset      : CALCE Battery Dataset (University of Maryland)
Files        : {len(cycle_files)} cycle data files, {len(timeseries_files)} timeseries files
Total cycles : {len(cycle_data):,}
Batteries    : {cycle_data['cell_id'].nunique()}
Chemistries  : {', '.join([c for c in cycle_data['chemistry'].unique() if c != 'unknown'])}

Test Conditions:
  Temperatures  : {sorted([t for t in cycle_data['temperature_C'].unique() if t > -100])} °C
  DoD Ranges    : {cycle_data['DoD_range'].unique().tolist()}
  Charge C-rates: {sorted(cycle_data['charge_Crate'].unique())}C
  Discharge C-rates: {sorted(cycle_data['discharge_Crate'].unique())}C

Capacity & SOH:
  Nominal capacities: 1.1 Ah (1100 mAh) for CX2-16/25/36/37/38, 1.35 Ah (1350 mAh) for CX2-33/34
""")

if 'soh_df' in locals() and not soh_df.empty:
    print(f"""
  SOH range             : {soh_df['SOH'].min():.3f} – {soh_df['SOH'].max():.3f}
  Cells reaching EOL    : {cells_at_eol if 'cells_at_eol' in locals() else 'N/A'} / {len(soh_df['cell_id'].unique())}

Efficiency:
  Coulombic efficiency  : {coulombic_str}
  Energy efficiency     : {energy_str}

Temperature:
  Operating range       : {temp_range_str}
""")

if SAVE_PLOTS:
    print(f"All plots saved to: {OUT_DIR}")
print("Done.")