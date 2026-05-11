"""
Real-World Electric Vehicle Operation Dataset — Exploration Script
====================================================================
Run from the folder containing this script, or set DATASET_PATH below.
Covers:
  1. Dataset inventory (files, sizes)
  2. Schema & dtypes inspection
  3. Basic statistics per vehicle
  4. Missing-value audit
  5. Physical plausibility checks
  6. Driving vs Charging analysis
  7. SOC distribution analysis
  8. Temperature analysis
  9. Voltage and current analysis
  10. Mileage progression
  11. Summary report
"""

"""
Real-World Electric Vehicle Operation Dataset — Exploration Script
====================================================================
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
DATASET_PATH = r"C:\Users\admin\Desktop\DR2\11 All Datasets\12 Real-World 10EVs dataset\Electric-vehicle-operation-data"
SAVE_PLOTS   = True
OUT_DIR      = os.path.join(DATASET_PATH, "exploration_outputs")
# ─────────────────────────────────────────────────────────────────────────────

# ── Physical plausibility limits ─────────────────────────────────────────────
VOLTAGE_MIN, VOLTAGE_MAX = 200, 800    # V
CURRENT_MIN, CURRENT_MAX = -500, 500   # A
SOC_MIN, SOC_MAX = 0, 100              # %
TEMPERATURE_MIN, TEMPERATURE_MAX = -20, 60  # °C
SPEED_MAX = 150                        # km/h
# ─────────────────────────────────────────────────────────────────────────────

matplotlib.rcParams.update({"figure.dpi": 120, "font.size": 9})

if SAVE_PLOTS:
    os.makedirs(OUT_DIR, exist_ok=True)

def savefig(name: str):
    if SAVE_PLOTS:
        path = os.path.join(OUT_DIR, name)
        plt.savefig(path, bbox_inches="tight")
        print(f"  → saved: {path}")
    plt.close()

def parse_vehicle_number(filename):
    """Extract vehicle number from filename"""
    match = re.search(r'vehicle#(\d+)', filename.lower())
    if match:
        return f"Vehicle#{match.group(1)}"
    return None

# Vehicle information dictionary
VEHICLE_INFO = {
    'Vehicle#1': {'type': 'Passenger', 'chemistry': 'NCM', 'capacity_Ah': 150, 'cells_series': 91, 'sampling_hz': 0.1},
    'Vehicle#2': {'type': 'Passenger', 'chemistry': 'NCM', 'capacity_Ah': 150, 'cells_series': 91, 'sampling_hz': 0.1},
    'Vehicle#3': {'type': 'Passenger', 'chemistry': 'NCM', 'capacity_Ah': 160, 'cells_series': 91, 'sampling_hz': 0.1},
    'Vehicle#4': {'type': 'Passenger', 'chemistry': 'NCM', 'capacity_Ah': 160, 'cells_series': 91, 'sampling_hz': 0.1},
    'Vehicle#5': {'type': 'Passenger', 'chemistry': 'NCM', 'capacity_Ah': 160, 'cells_series': 91, 'sampling_hz': 0.1},
    'Vehicle#6': {'type': 'Passenger', 'chemistry': 'NCM', 'capacity_Ah': 160, 'cells_series': 91, 'sampling_hz': 0.1},
    'Vehicle#7': {'type': 'Passenger', 'chemistry': 'LFP', 'capacity_Ah': 120, 'cells_series': None, 'sampling_hz': 0.5},
    'Vehicle#8': {'type': 'Bus', 'chemistry': 'LFP', 'capacity_Ah': 645, 'cells_series': None, 'sampling_hz': 0.1},
    'Vehicle#9': {'type': 'Bus', 'chemistry': 'LFP', 'capacity_Ah': 505, 'cells_series': 360, 'sampling_hz': 0.1},
    'Vehicle#10': {'type': 'Bus', 'chemistry': 'LFP', 'capacity_Ah': 505, 'cells_series': 324, 'sampling_hz': 0.1},
}

# Column name mapping
COLUMN_MAP = {
    'time': ['time', 'Time', 'timestamp', 'Timestamp'],
    'speed': ['vhc_speed', 'speed', 'Speed', 'vehicle_speed'],
    'charging_signal': ['charging_signal', 'charge_status', 'Charging_Status', 'charge_signal'],
    'mileage': ['vhc_totalMile', 'mileage', 'Mileage', 'total_mileage'],
    'voltage': ['hv_voltage', 'voltage', 'Voltage', 'pack_voltage'],
    'current': ['hv_current', 'current', 'Current', 'pack_current'],
    'soc': ['bcell_soc', 'soc', 'SOC', 'state_of_charge'],
    'max_voltage': ['bcell_maxVoltage', 'max_voltage', 'Max_Voltage'],
    'min_voltage': ['bcell_minVoltage', 'min_voltage', 'Min_Voltage'],
    'max_temp': ['bcell_maxTemp', 'max_temp', 'Max_Temperature', 'bcell_Temp'],
    'min_temp': ['bcell_minTemp', 'min_temp', 'Min_Temperature'],
}

def find_column(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — INVENTORY
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SECTION 1 — FILE INVENTORY")
print("=" * 70)

excel_files = sorted(glob.glob(os.path.join(DATASET_PATH, "*.xlsx")))
print(f"\nFound {len(excel_files)} Excel files")

inventory = []
for f in excel_files:
    size_mb = os.path.getsize(f) / (1024 * 1024)
    filename = os.path.basename(f)
    vehicle = parse_vehicle_number(filename)
    
    inventory.append({
        'file': filename,
        'vehicle': vehicle,
        'size_MB': round(size_mb, 1),
    })

inv_df = pd.DataFrame(inventory)
print(inv_df.to_string(index=False))
print(f"\nTotal size: {inv_df['size_MB'].sum():.1f} MB")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — LOAD AND INSPECT EACH VEHICLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 2 — LOADING VEHICLE DATA")
print("=" * 70)

vehicle_data = {}
vehicle_columns = {}

for f in excel_files:
    filename = os.path.basename(f)
    vehicle = parse_vehicle_number(filename)
    
    if vehicle is None:
        vehicle = filename.replace('.xlsx', '')
    
    print(f"\nLoading {vehicle}...")
    
    try:
        df = pd.read_excel(f)
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        col_map = {}
        for key, possible_names in COLUMN_MAP.items():
            found = find_column(df, possible_names)
            if found:
                col_map[key] = found
        
        print(f"  Mapped columns: {col_map}")
        
        vehicle_data[vehicle] = df
        vehicle_columns[vehicle] = col_map
        
    except Exception as e:
        print(f"  Error loading {filename}: {e}")

print(f"\nSuccessfully loaded {len(vehicle_data)} vehicles")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — SCHEMA INSPECTION (first vehicle)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 3 — SCHEMA INSPECTION")
print("=" * 70)

if vehicle_data:
    sample_vehicle = list(vehicle_data.keys())[0]
    sample_df = vehicle_data[sample_vehicle]
    
    print(f"\nSample vehicle: {sample_vehicle}")
    print(f"Shape: {sample_df.shape}")
    print("\nFirst 5 rows:")
    print(sample_df.head(5).to_string())
    print("\nData types:")
    print(sample_df.dtypes)
    print("\nBasic statistics:")
    print(sample_df.describe().to_string())


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — DATASET OVERVIEW PER VEHICLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 4 — DATASET OVERVIEW PER VEHICLE")
print("=" * 70)

vehicle_summary = []

for vehicle, df in vehicle_data.items():
    col_map = vehicle_columns[vehicle]
    info = VEHICLE_INFO.get(vehicle, {})
    
    summary = {
        'vehicle': vehicle,
        'rows': len(df),
        'type': info.get('type', 'N/A'),
        'chemistry': info.get('chemistry', 'N/A'),
        'capacity_Ah': info.get('capacity_Ah', 'N/A'),
    }
    
    if 'mileage' in col_map:
        mile_col = col_map['mileage']
        summary['mileage_km'] = round(df[mile_col].max(), 1)
    
    if 'soc' in col_map:
        soc_col = col_map['soc']
        summary['soc_range'] = f"{df[soc_col].min():.1f}-{df[soc_col].max():.1f}%"
    
    if 'charging_signal' in col_map:
        charge_col = col_map['charging_signal']
        if 3 in df[charge_col].values:
            driving_pct = (df[charge_col] == 3).sum() / len(df) * 100
            summary['driving_pct'] = f"{driving_pct:.1f}%"
        if 1 in df[charge_col].values:
            charging_pct = (df[charge_col] == 1).sum() / len(df) * 100
            summary['charging_pct'] = f"{charging_pct:.1f}%"
    
    vehicle_summary.append(summary)

summary_df = pd.DataFrame(vehicle_summary)
print(summary_df.to_string(index=False))

total_rows = sum(len(df) for df in vehicle_data.values())
print(f"\nTotal data across all vehicles: {total_rows:,} rows")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — MISSING VALUE AUDIT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 5 — MISSING VALUES")
print("=" * 70)

for vehicle, df in vehicle_data.items():
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df) * 100).round(2)
    null_df = pd.DataFrame({"col": null_counts.index, "null_count": null_counts.values, "null_%": null_pct.values})
    null_df = null_df[null_df["null_count"] > 0]
    
    if not null_df.empty:
        print(f"\n{vehicle}:")
        print(null_df.to_string(index=False))
    else:
        print(f"\n{vehicle}: No missing values found")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — PHYSICAL PLAUSIBILITY CHECKS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 6 — PHYSICAL PLAUSIBILITY CHECKS")
print("=" * 70)

plausibility_results = []

for vehicle, df in vehicle_data.items():
    col_map = vehicle_columns[vehicle]
    info = VEHICLE_INFO.get(vehicle, {})
    chemistry = info.get('chemistry', 'Unknown')
    
    results = {'vehicle': vehicle, 'chemistry': chemistry}
    
    if 'voltage' in col_map:
        v = df[col_map['voltage']].dropna()
        v_out = ((v < VOLTAGE_MIN) | (v > VOLTAGE_MAX)).sum()
        v_pct = v_out / len(v) * 100 if len(v) > 0 else 0
        results['voltage_outliers'] = f"{v_out:,} ({v_pct:.3f}%)"
        results['voltage_range'] = f"{v.min():.1f}-{v.max():.1f}V"
    
    if 'current' in col_map:
        i = df[col_map['current']].dropna()
        i_out = ((i < CURRENT_MIN) | (i > CURRENT_MAX)).sum()
        i_pct = i_out / len(i) * 100 if len(i) > 0 else 0
        results['current_outliers'] = f"{i_out:,} ({i_pct:.3f}%)"
        results['current_range'] = f"{i.min():.1f}-{i.max():.1f}A"
    
    if 'soc' in col_map:
        soc = df[col_map['soc']].dropna()
        soc_out = ((soc < SOC_MIN) | (soc > SOC_MAX)).sum()
        soc_pct = soc_out / len(soc) * 100 if len(soc) > 0 else 0
        results['soc_outliers'] = f"{soc_out:,} ({soc_pct:.3f}%)"
        results['soc_range'] = f"{soc.min():.1f}-{soc.max():.1f}%"
    
    if 'max_temp' in col_map:
        t = df[col_map['max_temp']].dropna()
        t_out = ((t < TEMPERATURE_MIN) | (t > TEMPERATURE_MAX)).sum()
        t_pct = t_out / len(t) * 100 if len(t) > 0 else 0
        results['temp_outliers'] = f"{t_out:,} ({t_pct:.3f}%)"
        results['temp_range'] = f"{t.min():.1f}-{t.max():.1f}°C"
    
    if 'speed' in col_map:
        sp = df[col_map['speed']].dropna()
        sp_out = (sp > SPEED_MAX).sum()
        sp_pct = sp_out / len(sp) * 100 if len(sp) > 0 else 0
        results['speed_outliers'] = f"{sp_out:,} ({sp_pct:.3f}%)"
        results['max_speed'] = f"{sp.max():.1f}km/h"
    
    plausibility_results.append(results)

plaus_df = pd.DataFrame(plausibility_results)
print("\nPlausibility check results:")
print(plaus_df.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — VISUALIZATIONS (FIXED)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 7 — VISUALIZATIONS")
print("=" * 70)

# Figure 1: SOC Distribution by Vehicle (boxplot)
if len(vehicle_data) > 0:
    fig, ax = plt.subplots(figsize=(14, 6))
    
    soc_data = []
    labels = []
    for vehicle, df in vehicle_data.items():
        col_map = vehicle_columns[vehicle]
        if 'soc' in col_map:
            soc_vals = df[col_map['soc']].dropna()
            if len(soc_vals) > 0:
                soc_data.append(soc_vals)
                labels.append(vehicle)
    
    if soc_data:
        bp = ax.boxplot(soc_data, labels=labels)
        ax.set_ylabel('State of Charge (%)')
        ax.set_title('SOC Distribution by Vehicle')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        savefig("01_soc_distribution.png")

# Figure 2: Max Temperature by Vehicle
fig, ax = plt.subplots(figsize=(12, 5))
temp_data = []
temp_labels = []
for vehicle, df in vehicle_data.items():
    col_map = vehicle_columns[vehicle]
    if 'max_temp' in col_map:
        temp_vals = df[col_map['max_temp']].dropna()
        if len(temp_vals) > 0:
            temp_data.append(temp_vals)
            temp_labels.append(vehicle)

if temp_data:
    bp = ax.boxplot(temp_data, labels=temp_labels)
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Max Temperature Distribution by Vehicle')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    savefig("02_temperature_distribution.png")

# Figure 3: Current Distribution (sample vehicle)
sample_vehicle = list(vehicle_data.keys())[0]
sample_df = vehicle_data[sample_vehicle]
sample_cols = vehicle_columns[sample_vehicle]

if 'current' in sample_cols:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Current time series
    ax1 = axes[0]
    current_col = sample_cols['current']
    sample_size = min(5000, len(sample_df))
    ax1.plot(sample_df[current_col].iloc[:sample_size].values, linewidth=0.5, alpha=0.7)
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Current (A)')
    ax1.set_title(f'{sample_vehicle} - Current Profile (first {sample_size} samples)')
    ax1.grid(True, alpha=0.3)
    
    # Current histogram
    ax2 = axes[1]
    current_vals = sample_df[current_col].dropna()
    ax2.hist(current_vals, bins=100, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', linewidth=1, label='Zero (Rest)')
    ax2.set_xlabel('Current (A)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'{sample_vehicle} - Current Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    savefig("03_current_analysis.png")

# Figure 4: SOC vs Current (sample vehicle)
if 'current' in sample_cols and 'soc' in sample_cols:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    current_col = sample_cols['current']
    soc_col = sample_cols['soc']
    
    # Sample for performance
    sample_size = min(10000, len(sample_df))
    sample_idx = np.random.choice(len(sample_df), sample_size, replace=False)
    
    scatter = ax.scatter(sample_df[current_col].iloc[sample_idx], 
                        sample_df[soc_col].iloc[sample_idx],
                        c=sample_df[current_col].iloc[sample_idx], 
                        cmap='RdYlGn', s=1, alpha=0.5)
    ax.set_xlabel('Current (A)')
    ax.set_ylabel('State of Charge (%)')
    ax.set_title(f'{sample_vehicle} - SOC vs Current')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Current (A)')
    plt.tight_layout()
    savefig("04_soc_vs_current.png")

# Figure 5: Cumulative Mileage Progression
fig, ax = plt.subplots(figsize=(12, 5))

has_mileage = False
for vehicle, df in vehicle_data.items():
    col_map = vehicle_columns[vehicle]
    if 'mileage' in col_map:
        mile_col = col_map['mileage']
        mileage = df[mile_col].dropna()
        if len(mileage) > 0:
            ax.plot(mileage.values, label=vehicle, linewidth=1, alpha=0.7)
            has_mileage = True

if has_mileage:
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Cumulative Mileage (km)')
    ax.set_title('Cumulative Mileage Progression by Vehicle')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig("05_mileage_progression.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 8 — SUMMARY REPORT")
print("=" * 70)

ncm_vehicles = [v for v in vehicle_data.keys() if VEHICLE_INFO.get(v, {}).get('chemistry') == 'NCM']
lfp_vehicles = [v for v in vehicle_data.keys() if VEHICLE_INFO.get(v, {}).get('chemistry') == 'LFP']

print(f"""
Dataset: Real-World Electric Vehicle Operation Data
Source: Public EV operation dataset (1 month per vehicle)
Total vehicles: {len(vehicle_data)}
  - NCM vehicles: {len(ncm_vehicles)} (Passenger cars)
  - LFP vehicles: {len(lfp_vehicles)} (Passenger + Buses)
Total data points: {total_rows:,}

Vehicle Summary:
""")

for vehicle, df in vehicle_data.items():
    info = VEHICLE_INFO.get(vehicle, {})
    col_map = vehicle_columns[vehicle]
    
    mileage = '-'
    if 'mileage' in col_map:
        mileage = f"{df[col_map['mileage']].max():,.0f} km"
    
    print(f"  {vehicle}: {info.get('type', 'N/A')} | {info.get('chemistry', 'N/A')} | "
          f"{len(df):,} rows | {info.get('capacity_Ah', 'N/A')}Ah | Mileage: {mileage}")

# Data Quality Summary
if len(plaus_df) > 0:
    print(f"\nData Quality Summary:")
    # Extract percentages safely
    voltage_pcts = []
    for val in plaus_df['voltage_outliers']:
        try:
            pct = float(val.split('(')[1].split('%')[0])
            voltage_pcts.append(pct)
        except:
            voltage_pcts.append(0)
    
    soc_pcts = []
    for val in plaus_df['soc_outliers']:
        try:
            pct = float(val.split('(')[1].split('%')[0])
            soc_pcts.append(pct)
        except:
            soc_pcts.append(0)
    
    temp_pcts = []
    for val in plaus_df['temp_outliers']:
        if val != 'nan':
            try:
                pct = float(val.split('(')[1].split('%')[0])
                temp_pcts.append(pct)
            except:
                temp_pcts.append(0)
    
    if voltage_pcts:
        print(f"  Average voltage outliers: {np.mean(voltage_pcts):.2f}%")
    if soc_pcts:
        print(f"  Average SOC outliers: {np.mean(soc_pcts):.2f}%")
    if temp_pcts:
        print(f"  Average temperature outliers: {np.mean(temp_pcts):.2f}%")

if SAVE_PLOTS:
    print(f"\nAll plots saved to: {OUT_DIR}")
print("Done.")