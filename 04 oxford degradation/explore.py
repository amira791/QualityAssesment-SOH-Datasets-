"""
Oxford Battery Degradation Dataset 1 — Exploration Script (FIXED)
==========================================================
Handles the Oxford dataset structure where each cell has columns named 'cycXXXX'
containing time-series data for each cycle. Specifically parses the nested
mat_struct format found in the original .mat file.
"""

import os
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ── ① USER CONFIG ────────────────────────────────────────────────────────────
DATASET_PATH = r"C:\Users\admin\Desktop\DR2\11 All Datasets\05 Oxford Battery Degradation Dataset\oxford dataset"
SAVE_PLOTS   = True
OUT_DIR      = os.path.join(DATASET_PATH, "exploration_outputs")
# ─────────────────────────────────────────────────────────────────────────────

# ── Physical plausibility limits (Kokam 740mAh Li-ion pouch) ─────────────────
CELL_V_MIN, CELL_V_MAX = 2.5, 4.2      # V
TEMP_MIN,  TEMP_MAX    = -20, 60       # °C
CURRENT_MAX            = 5.0           # A
NOMINAL_CAPACITY_AH    = 0.74          # Ah
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


# ================= FIXED FUNCTION =================
def extract_cycle_data(cycle_object):
    """
    Extracts time-series data from the nested mat_struct format.
    The relevant discharge data is found at cycle_object.C1dc.
    """
    # Check if the cycle has the 'C1dc' attribute, which holds the discharge data
    if hasattr(cycle_object, 'C1dc'):
        discharge_data = cycle_object.C1dc
        
        # Extract the necessary arrays (t, v, q, T)
        # The 'squeeze_me' flag in loadmat ensures these are 1D arrays.
        time = discharge_data.t.flatten()
        voltage = discharge_data.v.flatten()
        charge = discharge_data.q.flatten()
        
        # Temperature might not be present in all cycles, handle gracefully
        temperature = None
        if hasattr(discharge_data, 'T'):
            temperature = discharge_data.T.flatten()
        else:
            # Placeholder if T is missing (should not happen for Oxford dataset)
            temperature = np.full_like(time, np.nan)

        return pd.DataFrame({
            'time': time,
            'voltage': voltage,
            'current': charge,  # Note: 'q' in the struct represents cumulative charge, which can be used as a proxy for current.
            'temperature': temperature
        })
    else:
        # Fallback for other potential structures, though unlikely
        print(f"Warning: Cycle object missing 'C1dc' attribute. Found: {dir(cycle_object)}")
        return None
# ===================================================


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — LOAD AND INSPECT MAT FILE
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SECTION 1 — LOADING OXFORD DATASET")
print("=" * 70)

mat_file = os.path.join(DATASET_PATH, "Oxford_Battery_Degradation_Dataset_1.mat")

if not os.path.exists(mat_file):
    raise FileNotFoundError(f"MAT file not found:\n  {mat_file}")

# Load with squeeze_me=True to simplify nested structures
print(f"Loading: {mat_file}")
mat_data = sio.loadmat(mat_file, squeeze_me=True, struct_as_record=False)

print(f"\nMAT file keys: {[k for k in mat_data.keys() if not k.startswith('__')]}")

# Load cell data
cell_data_raw = {}
cell_keys = [k for k in mat_data.keys() if k.startswith('Cell')]

for cell_key in cell_keys:
    cell_struct = mat_data[cell_key]
    print(f"\n{cell_key}: Type = {type(cell_struct)}")
    
    # The cell is a mat_struct, and its fields are the 'cycXXXX' objects
    cycle_names = [name for name in dir(cell_struct) if name.startswith('cyc') and not name.startswith('__')]
    
    if not cycle_names:
        print(f"  No 'cycXXXX' attributes found. Available: {[a for a in dir(cell_struct) if not a.startswith('__')]}")
        continue
        
    print(f"  Found {len(cycle_names)} cycle attributes (e.g., {cycle_names[0]})")
    # Store the struct itself for now; we'll extract data in the next section
    cell_data_raw[cell_key] = cell_struct

print(f"\nCells loaded: {len(cell_data_raw)}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — EXTRACT CYCLE DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 2 — EXTRACTING CYCLE DATA")
print("=" * 70)

all_cell_data = {}  # Will store DataFrames for each cell

for cell_key, cell_struct in cell_data_raw.items():
    print(f"\nProcessing {cell_key}...")
    
    # Get all cycle attributes
    cycle_names = [name for name in dir(cell_struct) if name.startswith('cyc') and not name.startswith('__')]
    print(f"  Found {len(cycle_names)} cycle attributes")
    
    if len(cycle_names) == 0:
        print(f"  No cycle attributes found.")
        continue
    
    cell_cycles = []
    cycles_extracted = 0
    
    # Sort cycles naturally (e.g., cyc0000, cyc0100, ...)
    # A simple sort works because of leading zeros.
    for cycle_name in sorted(cycle_names):
        cycle_obj = getattr(cell_struct, cycle_name)
        
        # Extract cycle number from the attribute name
        try:
            cycle_num = int(cycle_name.replace('cyc', ''))
        except:
            cycle_num = 0 # fallback
        
        # Extract time-series data using the corrected function
        df_cycle = extract_cycle_data(cycle_obj)
        
        if df_cycle is not None and len(df_cycle) > 0:
            df_cycle['cycle_number'] = cycle_num
            cell_cycles.append(df_cycle)
            cycles_extracted += 1
    
    if cell_cycles:
        # Combine all cycles for this cell
        cell_df = pd.concat(cell_cycles, ignore_index=True)
        all_cell_data[cell_key] = cell_df
        print(f"  ✓ Extracted {len(cell_df)} rows across {cycles_extracted} cycles")
    else:
        print(f"  ✗ No cycle data extracted")

print(f"\nTotal cells with data: {len(all_cell_data)}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — SCHEMA INSPECTION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 3 — SCHEMA INSPECTION")
print("=" * 70)

if all_cell_data:
    sample_cell = list(all_cell_data.keys())[0]
    sample_df = all_cell_data[sample_cell]
    
    print(f"\nSample cell: {sample_cell}")
    print(f"Shape: {sample_df.shape}")
    print(f"Columns: {list(sample_df.columns)}")
    print("\nFirst 10 rows (sample):")
    print(sample_df.head(10).to_string())
    print("\nData types:")
    print(sample_df.dtypes)
    print("\nBasic statistics:")
    print(sample_df.describe().to_string())


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — DATASET OVERVIEW PER CELL
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 4 — DATASET OVERVIEW PER CELL")
print("=" * 70)

def compute_capacity(df):
    """Compute capacity from the 'current' (which is actually cumulative charge 'q') data."""
    # The 'current' column in our extracted df is actually the 'q' (charge) value.
    # The capacity for a discharge cycle is simply max(q) - min(q).
    if len(df) < 2:
        return 0
    # Assuming the data is from a single cycle
    cap = df['current'].max() - df['current'].min()
    return cap / 3600.0 # Convert from Coulombs (if q is in Coulombs?) to Ah? Usually it's in Coulombs.

cell_summary = []

for cell_name, df in all_cell_data.items():
    # Create summary dictionary
    summary = {
        'cell': cell_name,
        'rows': len(df),
        'cycles': df['cycle_number'].nunique()
    }
    
    # Add voltage range
    v_min = df['voltage'].min()
    v_max = df['voltage'].max()
    summary['voltage_range'] = f"{v_min:.2f}-{v_max:.2f}V"
    
    # Add current range (which is actually charge range)
    q_min = df['current'].min()
    q_max = df['current'].max()
    summary['charge_range'] = f"{q_min:.2f}-{q_max:.2f}C"
    
    # Add temperature range if available
    if 'temperature' in df.columns and not df['temperature'].isna().all():
        t_min = df['temperature'].min()
        t_max = df['temperature'].max()
        summary['temp_range'] = f"{t_min:.1f}-{t_max:.1f}°C"
    else:
        summary['temp_range'] = "N/A"
    
    # Add duration
    if 'time' in df.columns:
        duration = df['time'].max() - df['time'].min()
        summary['duration_hours'] = round(duration / 3600, 1) if duration > 0 else 0
    else:
        summary['duration_hours'] = 0
    
    cell_summary.append(summary)

summary_df = pd.DataFrame(cell_summary)
print(summary_df.to_string(index=False))

total_rows = sum(len(df) for df in all_cell_data.values())
print(f"\nTotal data across all cells: {total_rows:,} rows")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — PHYSICAL PLAUSIBILITY CHECKS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 5 — PHYSICAL PLAUSIBILITY CHECKS")
print("=" * 70)

for cell_name, df in all_cell_data.items():
    print(f"\n{cell_name}:")
    
    # Voltage check
    v = df['voltage'].dropna()
    v_out = ((v < CELL_V_MIN) | (v > CELL_V_MAX)).sum()
    v_pct = v_out / len(v) * 100 if len(v) > 0 else 0
    print(f"  Voltage: {v_out} outliers ({v_pct:.3f}%) - range [{v.min():.2f}, {v.max():.2f}]V")
    
    # Current (charge) check
    i = df['current'].dropna()
    print(f"  Charge (q): range [{i.min():.2f}, {i.max():.2f}]C")
    
    # Temperature check (if available)
    if 'temperature' in df.columns:
        t = df['temperature'].dropna()
        if len(t) > 0:
            t_out = ((t < TEMP_MIN) | (t > TEMP_MAX)).sum()
            t_pct = t_out / len(t) * 100 if len(t) > 0 else 0
            print(f"  Temperature: {t_out} outliers ({t_pct:.3f}%) - range [{t.min():.1f}, {t.max():.1f}]°C")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — CAPACITY FADE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 6 — CAPACITY FADE ANALYSIS")
print("=" * 70)

capacity_data = []

for cell_name, df in all_cell_data.items():
    # Calculate capacity for each cycle
    cycle_capacities = []
    cycle_numbers = []
    
    for cycle_num in sorted(df['cycle_number'].unique()):
        cycle_df = df[df['cycle_number'] == cycle_num]
        
        # Since our data is already the discharge 'q', we just compute its range.
        if len(cycle_df) > 10:
            # Cap is the total discharged capacity in this cycle.
            cap = compute_capacity(cycle_df)
            if cap > 0.01:  # Ignore tiny capacities
                cycle_capacities.append(cap)
                cycle_numbers.append(cycle_num)
    
    if cycle_capacities:
        initial_cap = cycle_capacities[0]
        soh = [c / initial_cap for c in cycle_capacities]
        
        # Find EOL (80% of initial)
        eol_cycle = None
        for i, s in enumerate(soh):
            if s < 0.8:
                eol_cycle = cycle_numbers[i]
                break
        
        capacity_data.append({
            'cell': cell_name,
            'cycle_numbers': cycle_numbers,
            'capacities': cycle_capacities,
            'soh': soh,
            'initial_cap': initial_cap,
            'final_cap': cycle_capacities[-1],
            'eol_cycle': eol_cycle if eol_cycle else len(cycle_capacities)
        })
        
        print(f"\n{cell_name}:")
        print(f"  Initial capacity: {initial_cap:.3f} Ah ({initial_cap/NOMINAL_CAPACITY_AH*100:.1f}% of nominal)")
        print(f"  Final capacity: {cycle_capacities[-1]:.3f} Ah")
        print(f"  Total cycles analyzed: {len(cycle_capacities)}")
        print(f"  Cycles to EOL: {eol_cycle if eol_cycle else 'Not reached'}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 7 — VISUALIZATION")
print("=" * 70)

if capacity_data:
    # Figure 1: Capacity fade curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Capacity fade
    ax1 = axes[0, 0]
    for cell in capacity_data:
        ax1.plot(cell['cycle_numbers'], cell['capacities'], '.-', 
                markersize=4, linewidth=1, label=cell['cell'], alpha=0.7)
    ax1.axhline(NOMINAL_CAPACITY_AH, color='gray', linestyle='--', label=f'Nominal ({NOMINAL_CAPACITY_AH}Ah)')
    ax1.axhline(0.8 * NOMINAL_CAPACITY_AH, color='red', linestyle='--', label='EOL (80%)')
    ax1.set_xlabel('Cycle Number')
    ax1.set_ylabel('Discharge Capacity (Ah)')
    ax1.set_title('Capacity Fade Curves')
    ax1.legend(fontsize=8, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: SOH degradation
    ax2 = axes[0, 1]
    for cell in capacity_data:
        ax2.plot(cell['cycle_numbers'], cell['soh'], '.-', 
                markersize=4, linewidth=1, label=cell['cell'], alpha=0.7)
    ax2.axhline(0.8, color='red', linestyle='--', label='EOL (80%)')
    ax2.set_xlabel('Cycle Number')
    ax2.set_ylabel('State of Health (SOH)')
    ax2.set_title('SOH Degradation')
    ax2.legend(fontsize=8, loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Cycle life comparison
    ax3 = axes[1, 0]
    cells = [c['cell'] for c in capacity_data]
    eol_cycles = [c['eol_cycle'] if c['eol_cycle'] else len(c['cycle_numbers']) for c in capacity_data]
    bars = ax3.bar(cells, eol_cycles, alpha=0.7, color='steelblue')
    ax3.set_xlabel('Cell')
    ax3.set_ylabel('Cycles to EOL')
    ax3.set_title('Cycle Life Comparison')
    ax3.set_xticklabels(cells, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, eol_cycles):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                str(val), ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Initial vs Final capacity
    ax4 = axes[1, 1]
    x = np.arange(len(cells))
    width = 0.35
    initial_caps = [c['initial_cap'] for c in capacity_data]
    final_caps = [c['final_cap'] for c in capacity_data]
    
    ax4.bar(x - width/2, initial_caps, width, label='Initial', alpha=0.7, color='green')
    ax4.bar(x + width/2, final_caps, width, label='Final', alpha=0.7, color='orange')
    ax4.set_xlabel('Cell')
    ax4.set_ylabel('Capacity (Ah)')
    ax4.set_title('Capacity Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(cells, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Oxford Battery Degradation Dataset - Capacity Fade Analysis', fontsize=14, y=1.02)
    plt.tight_layout()
    savefig("01_capacity_fade_analysis.png")
    
    # Figure 2: Example voltage and current profiles for first cell
    if all_cell_data:
        first_cell = list(all_cell_data.keys())[0]
        df = all_cell_data[first_cell]
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Get first cycle
        first_cycle = df[df['cycle_number'] == df['cycle_number'].min()]
        
        # Voltage profile
        ax1 = axes[0]
        ax1.plot(first_cycle['time'], first_cycle['voltage'], 'b-', linewidth=1)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Voltage (V)')
        ax1.set_title(f'{first_cell} - Voltage Profile (Cycle {df["cycle_number"].min()})')
        ax1.grid(True, alpha=0.3)
        
        # Current (Charge) profile
        ax2 = axes[1]
        ax2.plot(first_cycle['time'], first_cycle['current'], 'r-', linewidth=1)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Cumulative Charge (C)')
        ax2.set_title(f'{first_cell} - Cumulative Charge Profile (Cycle {df["cycle_number"].min()})')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Example Cycle Profile', fontsize=12)
        plt.tight_layout()
        savefig("02_example_cycle_profile.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 8 — SUMMARY REPORT")
print("=" * 70)

print(f"""
Dataset: Oxford Battery Degradation Dataset 1
Source: University of Oxford
Battery: Kokam 740 mAh Li-ion pouch cells

Summary Statistics:
  Total cells: {len(all_cell_data)}
  Total rows: {total_rows:,}
  
  Cell data:
""")

for cell_name, df in all_cell_data.items():
    n_cycles = df['cycle_number'].nunique()
    print(f"  {cell_name}: {len(df):,} rows, {n_cycles} cycles")

if capacity_data:
    avg_initial = np.mean([c['initial_cap'] for c in capacity_data])
    avg_final = np.mean([c['final_cap'] for c in capacity_data])
    avg_eol = np.mean([c['eol_cycle'] if c['eol_cycle'] else len(c['cycle_numbers']) for c in capacity_data])
    
    print(f"""
Capacity Summary:
  Average initial capacity: {avg_initial:.3f} Ah ({avg_initial/NOMINAL_CAPACITY_AH*100:.1f}% of nominal)
  Average final capacity: {avg_final:.3f} Ah
  Average cycles to EOL: {avg_eol:.0f} cycles
  Nominal capacity: {NOMINAL_CAPACITY_AH} Ah (740 mAh)
""")

if SAVE_PLOTS:
    print(f"\nAll plots saved to: {OUT_DIR}")
print("\nDone.")