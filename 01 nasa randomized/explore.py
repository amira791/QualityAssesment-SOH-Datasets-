"""
NASA Randomized & Recommissioned Battery Dataset — Exploration Script
======================================================================
Run from the folder containing this script, or set DATASET_PATH below.
Covers:
  1. Dataset inventory (files, sizes)
  2. Schema & dtypes inspection
  3. Basic statistics per file
  4. Cycle-level summary (capacity per cycle, per battery)
  5. Missing-value audit
  6. Mode distribution (-1 / 0 / 1)
  7. Voltage / current / temperature range checks
  8. Cycle count & EOL coverage
  9. Per-pack timeline overview
"""

import os
import glob
import warnings
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

warnings.filterwarnings("ignore")

# ── ① USER CONFIG ────────────────────────────────────────────────────────────
DATASET_PATH = r"C:\Users\admin\Desktop\DR2\11 All Datasets\02 NASA Randomized Battery Dataset\battery_alt_dataset\regular_alt_batteries"
SAVE_PLOTS   = True          # set False to only show plots interactively
OUT_DIR      = os.path.join(DATASET_PATH, "exploration_outputs")
# ─────────────────────────────────────────────────────────────────────────────

# ── Physical plausibility limits (LFP A123 APR18650M1A) ──────────────────────
LFP_V_MIN, LFP_V_MAX   = 2.0,  3.65   # V  (charge cut-off ~ 3.6 V, discharge 2.0 V)
TEMP_MIN,  TEMP_MAX     = -20,  60     # °C operating bounds
I_DISCHARGE_MAX         = 25.0         # A  (4C × 1.1 Ah × pack-level, conservative)
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


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — INVENTORY
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SECTION 1 — FILE INVENTORY")
print("=" * 70)

csv_files = sorted(glob.glob(os.path.join(DATASET_PATH, "battery*.csv")))
if not csv_files:
    raise FileNotFoundError(f"No battery*.csv files found in:\n  {DATASET_PATH}")

inventory = []
for f in csv_files:
    size_kb = os.path.getsize(f) / 1024
    # quick row count without loading full file
    with open(f, "r") as fh:
        nrows = sum(1 for _ in fh) - 1        # subtract header
    name = os.path.basename(f)
    inventory.append({"file": name, "size_KB": round(size_kb, 1), "rows": nrows})

inv_df = pd.DataFrame(inventory)
print(inv_df.to_string(index=False))
print(f"\nTotal files : {len(csv_files)}")
print(f"Total rows  : {inv_df['rows'].sum():,}")
print(f"Total size  : {inv_df['size_KB'].sum()/1024:.1f} MB")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — SCHEMA & DTYPES  (inspect one file)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 2 — SCHEMA (from first file)")
print("=" * 70)

sample = pd.read_csv(csv_files[0], nrows=5)
print(f"\nFile: {os.path.basename(csv_files[0])}")
print(f"Columns ({len(sample.columns)}): {list(sample.columns)}")
print("\nHead (5 rows):")
print(sample.to_string())
print("\ndtypes:")
print(sample.dtypes)


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD ALL FILES  (with battery ID tag)
# ══════════════════════════════════════════════════════════════════════════════
print("\nLoading all CSV files …")
frames = []
for f in csv_files:
    batt_id = os.path.basename(f).replace(".csv", "")   # e.g. "battery00"
    df = pd.read_csv(f, low_memory=False)
    df["battery_id"] = batt_id
    frames.append(df)

data = pd.concat(frames, ignore_index=True)
print(f"Full dataset shape: {data.shape}")

# Normalise column names (strip whitespace)
data.columns = data.columns.str.strip()

# Identify key columns (flexible matching)
COL_MAP = {}
for col in data.columns:
    cl = col.lower()
    if "relative" in cl and "time" in cl:   COL_MAP["rel_time"]    = col
    elif "mode" in cl:                       COL_MAP["mode"]        = col
    elif "voltage" in cl and "charger" in cl:COL_MAP["v_charger"]   = col
    elif "voltage" in cl and "load" in cl:   COL_MAP["v_load"]      = col
    elif "current" in cl:                    COL_MAP["current"]     = col
    elif "temp" in cl and "battery" in cl:   COL_MAP["temp_batt"]   = col
    elif "temp" in cl and "mosfet" in cl:    COL_MAP["temp_mosfet"] = col
    elif "temp" in cl and "resistor" in cl:  COL_MAP["temp_res"]    = col
    elif "mission" in cl:                    COL_MAP["mission_type"]= col

print(f"\nColumn mapping detected: {COL_MAP}")

# ── Coerce all measurement columns to numeric (handles stray strings/headers) ─
numeric_keys = ("rel_time","v_charger","v_load","current","temp_batt","temp_mosfet","temp_res","mode","mission_type")
for key in numeric_keys:
    if key in COL_MAP and COL_MAP[key] in data.columns:
        col = COL_MAP[key]
        before = data[col].dtype
        data[col] = pd.to_numeric(data[col], errors="coerce")
        after = data[col].dtype
        n_coerced = data[col].isna().sum()
        if str(before) == "object":
            print(f"  [coerce] '{col}': {before} → {after}  (NaNs introduced: {n_coerced:,})")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — MISSING VALUE AUDIT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 3 — MISSING VALUES")
print("=" * 70)

null_counts = data.isnull().sum()
null_pct    = (null_counts / len(data) * 100).round(2)
null_df = pd.DataFrame({"null_count": null_counts, "null_%": null_pct})
null_df = null_df[null_df["null_count"] > 0].sort_values("null_%", ascending=False)
if null_df.empty:
    print("No missing values found across all columns.")
else:
    print(null_df.to_string())

# Per-battery missing check on critical columns
critical = [v for v in COL_MAP.values() if v in data.columns]
print(f"\nMissing per battery (critical cols: {critical}):")
miss_per_batt = (
    data.groupby("battery_id")[critical]
    .apply(lambda g: g.isnull().sum())
)
print(miss_per_batt[miss_per_batt.sum(axis=1) > 0].to_string()
      if not miss_per_batt[miss_per_batt.sum(axis=1) > 0].empty
      else "  — none —")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — MODE DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 4 — OPERATIONAL MODE DISTRIBUTION (-1=Discharge, 0=Rest, 1=Charge)")
print("=" * 70)

if "mode" in COL_MAP:
    mode_col = COL_MAP["mode"]
    mode_vc  = data[mode_col].value_counts().sort_index()
    mode_pct = (mode_vc / len(data) * 100).round(2)
    mode_summary = pd.DataFrame({"count": mode_vc, "%": mode_pct})
    mode_summary.index.name = "mode"
    print(mode_summary.to_string())

    # Per-battery mode breakdown
    mode_per_batt = (
        data.groupby(["battery_id", mode_col])
        .size()
        .unstack(fill_value=0)
    )
    print("\nMode rows per battery:")
    print(mode_per_batt.to_string())

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    mode_summary["%"].plot(kind="bar", ax=axes[0], color=["#d62728","#7f7f7f","#2ca02c"])
    axes[0].set_title("Overall mode distribution (%)")
    axes[0].set_xlabel("Mode"); axes[0].set_ylabel("%")
    axes[0].set_xticklabels(["-1 (Discharge)","0 (Rest)","1 (Charge)"], rotation=0)

    mode_per_batt.plot(kind="bar", ax=axes[1], stacked=True,
                       color=["#d62728","#7f7f7f","#2ca02c"],
                       legend=True)
    axes[1].set_title("Mode rows per battery")
    axes[1].set_xlabel("Battery"); axes[1].set_ylabel("Rows")
    axes[1].legend(labels=["Discharge","Rest","Charge"], loc="upper right")
    plt.tight_layout()
    savefig("01_mode_distribution.png")
else:
    print("  'mode' column not found — skipping.")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — BASIC STATISTICS (voltage, current, temperature)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 5 — BASIC STATISTICS")
print("=" * 70)

stat_cols = [v for k, v in COL_MAP.items()
             if k in ("v_charger","v_load","current","temp_batt","temp_mosfet","temp_res")
             and v in data.columns]
stats = data[stat_cols].describe().T
print(stats.to_string())


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — PHYSICAL PLAUSIBILITY CHECKS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 6 — PHYSICAL PLAUSIBILITY FLAGS")
print("=" * 70)

flags = {}

# ── Voltage load (valid during discharge & rest: mode == -1 or 0) ─────────────
if "v_load" in COL_MAP and COL_MAP["v_load"] in data.columns:
    col  = COL_MAP["v_load"]
    # Only check rows where the sensor is expected to be active (discharge & rest)
    if "mode" in COL_MAP and COL_MAP["mode"] in data.columns:
        active = data[data[COL_MAP["mode"]].isin([-1, 0])].copy()
    else:
        active = data.copy()
    valid   = active[col].dropna()
    out_mask = (valid < LFP_V_MIN) | (valid > LFP_V_MAX)
    n_out   = out_mask.sum()
    pct     = n_out / len(valid) * 100 if len(valid) else 0
    flags["Voltage (load)"] = (n_out, round(pct, 4))
    print(f"  Voltage (load) [discharge+rest rows only]:")
    print(f"    Range checked : [{LFP_V_MIN}, {LFP_V_MAX}] V")
    print(f"    Rows checked  : {len(valid):,}")
    print(f"    Out-of-range  : {n_out:,}  ({pct:.3f}%)")
    if n_out > 0:
        print(f"    Sample bad values : {valid[out_mask].head(10).tolist()}")

# ── Voltage charger (only meaningful during charge: mode == 1) ────────────────
if "v_charger" in COL_MAP and COL_MAP["v_charger"] in data.columns:
    col = COL_MAP["v_charger"]
    if "mode" in COL_MAP and COL_MAP["mode"] in data.columns:
        charge_rows = data[data[COL_MAP["mode"]] == 1].copy()
    else:
        charge_rows = data.copy()
    valid   = charge_rows[col].dropna()
    out_mask = (valid < LFP_V_MIN) | (valid > LFP_V_MAX)
    n_out   = out_mask.sum()
    pct     = n_out / len(valid) * 100 if len(valid) else 0
    flags["Voltage (charger)"] = (n_out, round(pct, 4))
    print(f"\n  Voltage (charger) [charge rows only]:")
    print(f"    Range checked : [{LFP_V_MIN}, {LFP_V_MAX}] V")
    print(f"    Rows checked  : {len(valid):,}")
    print(f"    Out-of-range  : {n_out:,}  ({pct:.3f}%)")
    if n_out > 0:
        print(f"    Sample bad values : {valid[out_mask].head(10).tolist()}")

# ── Temperature checks ────────────────────────────────────────────────────────
print()
for tkey, tlabel in [("temp_batt","Battery"), ("temp_mosfet","MOSFET"), ("temp_res","Resistor")]:
    if tkey in COL_MAP and COL_MAP[tkey] in data.columns:
        col  = COL_MAP[tkey]
        valid = data[col].dropna()
        out_mask = (valid < TEMP_MIN) | (valid > TEMP_MAX)
        n_out = out_mask.sum()
        pct   = n_out / len(valid) * 100 if len(valid) else 0
        print(f"  Temp ({tlabel:10s}): {n_out:,} rows out of [{TEMP_MIN},{TEMP_MAX}] °C  ({pct:.3f}%)"
              + (f"  ← sample: {valid[out_mask].head(5).tolist()}" if n_out > 0 else "  ✓ OK"))

# ── Current sign vs. mode consistency ─────────────────────────────────────────
print()
if "mode" in COL_MAP and "current" in COL_MAP:
    mc = COL_MAP["mode"]
    cc = COL_MAP["current"]
    # Only check rows with valid numeric values in both columns
    check = data[[mc, cc]].dropna()
    check[mc] = pd.to_numeric(check[mc], errors="coerce")
    check[cc] = pd.to_numeric(check[cc], errors="coerce")
    check = check.dropna()

    dis_rows  = check[check[mc] == -1]
    chg_rows  = check[check[mc] ==  1]
    dis_wrong = dis_rows[dis_rows[cc] > 0]
    chg_wrong = chg_rows[chg_rows[cc] < 0]

    print(f"  Current sign vs. mode consistency:")
    print(f"    Discharge rows (mode=-1)  : {len(dis_rows):,}")
    print(f"    → with I > 0 (wrong sign) : {len(dis_wrong):,}  ({len(dis_wrong)/max(len(dis_rows),1)*100:.3f}%)")
    print(f"    Charge rows (mode=+1)     : {len(chg_rows):,}")
    print(f"    → with I < 0 (wrong sign) : {len(chg_wrong):,}  ({len(chg_wrong)/max(len(chg_rows),1)*100:.3f}%)")
    if len(dis_wrong) > 0:
        print(f"    Sample discharge sign-wrong current values: {dis_wrong[cc].head(5).tolist()}")
    if len(chg_wrong) > 0:
        print(f"    Sample charge sign-wrong current values:    {chg_wrong[cc].head(5).tolist()}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — CYCLE COUNTING & CAPACITY ESTIMATION (per battery)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 7 — CYCLE COUNT & DISCHARGE CAPACITY (per battery)")
print("=" * 70)

"""
Strategy:
  - A cycle boundary is detected when mode transitions from Discharge → Rest/Charge
    (i.e. end of a discharge segment).
  - Discharge capacity [Ah] ≈ ∫ I·dt  (trapezoid rule over discharge rows)
    We use the relative_time column and current_load column.
"""

cycle_records = []

for batt_id, grp in data.groupby("battery_id"):
    grp = grp.copy().reset_index(drop=True)

    if "mode" not in COL_MAP or COL_MAP["mode"] not in grp.columns:
        continue
    mc = COL_MAP["mode"]
    tc = COL_MAP.get("rel_time")
    cc = COL_MAP.get("current")

    # Detect mode transitions to label cycles
    grp["mode_shift"] = (grp[mc] != grp[mc].shift()).cumsum()

    # Each contiguous block of mode == -1 is one discharge segment
    discharge_segs = grp[grp[mc] == -1].groupby("mode_shift")

    cycle_idx = 0
    for seg_id, seg in discharge_segs:
        cycle_idx += 1
        cap_ah = np.nan
        if tc and cc and tc in seg.columns and cc in seg.columns:
            t = seg[tc].values.astype(float)
            i = np.abs(seg[cc].values.astype(float))   # absolute discharge current
            if len(t) > 1:
                cap_ah = np.trapz(i, t) / 3600.0       # Coulombs → Ah

        t_start = seg[tc].iloc[0] if (tc and tc in seg.columns) else np.nan
        t_end   = seg[tc].iloc[-1] if (tc and tc in seg.columns) else np.nan
        cycle_records.append({
            "battery_id":  batt_id,
            "cycle":       cycle_idx,
            "n_rows":      len(seg),
            "t_start_s":   t_start,
            "t_end_s":     t_end,
            "duration_s":  t_end - t_start if not np.isnan(t_start) else np.nan,
            "capacity_Ah": cap_ah,
        })

cyc_df = pd.DataFrame(cycle_records)

print("\nCycles per battery:")
cyc_summary = cyc_df.groupby("battery_id").agg(
    n_cycles   = ("cycle",       "max"),
    cap_mean   = ("capacity_Ah", "mean"),
    cap_std    = ("capacity_Ah", "std"),
    cap_min    = ("capacity_Ah", "min"),
    cap_max    = ("capacity_Ah", "max"),
).round(4)
print(cyc_summary.to_string())

print(f"\nTotal discharge segments detected: {len(cyc_df):,}")
print(f"Cycle life range: {cyc_summary['n_cycles'].min()} – {cyc_summary['n_cycles'].max()}")


# ── Plot capacity fade per battery ───────────────────────────────────────────
batteries = cyc_df["battery_id"].unique()
ncols = 4
nrows_plot = int(np.ceil(len(batteries) / ncols))
fig, axes = plt.subplots(nrows_plot, ncols,
                          figsize=(ncols*4, nrows_plot*3),
                          sharex=False, sharey=False)
axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

for ax, batt_id in zip(axes_flat, batteries):
    sub = cyc_df[cyc_df["battery_id"] == batt_id].dropna(subset=["capacity_Ah"])
    if sub.empty:
        ax.set_visible(False)
        continue
    ax.plot(sub["cycle"], sub["capacity_Ah"], ".-", markersize=2, linewidth=0.8, color="#1f77b4")
    # EOL line at 80% of nominal (1.1 Ah) → 0.88 Ah
    ax.axhline(0.88, color="red", linestyle="--", linewidth=0.8, label="EOL 80%")
    ax.set_title(batt_id, fontsize=8)
    ax.set_xlabel("Cycle", fontsize=7)
    ax.set_ylabel("Ah", fontsize=7)
    ax.tick_params(labelsize=7)

for ax in axes_flat[len(batteries):]:
    ax.set_visible(False)

plt.suptitle("Discharge Capacity Fade — NASA Randomized Dataset", fontsize=11, y=1.01)
plt.tight_layout()
savefig("02_capacity_fade_per_battery.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — SOH ESTIMATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 8 — SOH ESTIMATION")
print("=" * 70)

NOMINAL_CAP = 1.1   # Ah

# For each battery, reference capacity = first valid measured capacity
soh_records = []
for batt_id, grp in cyc_df.groupby("battery_id"):
    grp = grp.dropna(subset=["capacity_Ah"]).copy()
    if grp.empty:
        continue
    ref_cap = grp["capacity_Ah"].iloc[0]
    if ref_cap <= 0:
        continue
    grp["SOH"] = grp["capacity_Ah"] / ref_cap
    soh_records.append(grp)

soh_df = pd.concat(soh_records, ignore_index=True) if soh_records else pd.DataFrame()

if not soh_df.empty:
    soh_stats = soh_df.groupby("battery_id")["SOH"].agg(["min","max","mean"]).round(3)
    print(soh_stats.to_string())

    # SOH distribution histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(soh_df["SOH"].clip(0, 1.1), bins=50, edgecolor="white", color="#2ca02c", alpha=0.8)
    ax.axvline(0.8, color="red", linestyle="--", label="EOL (SOH=0.80)")
    ax.set_title("SOH Distribution across all cycles & batteries")
    ax.set_xlabel("SOH"); ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    savefig("03_SOH_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — TEMPERATURE OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 9 — TEMPERATURE OVERVIEW")
print("=" * 70)

temp_cols = {k: v for k, v in COL_MAP.items() if "temp" in k and v in data.columns}
if temp_cols:
    temp_data = data[list(temp_cols.values())].describe().T
    temp_data.index = list(temp_cols.keys())
    print(temp_data.round(2).to_string())

    # Per-battery temperature spread (battery temp only)
    if "temp_batt" in COL_MAP and COL_MAP["temp_batt"] in data.columns:
        tc = COL_MAP["temp_batt"]
        temp_per_batt = data.groupby("battery_id")[tc].agg(["min","mean","max"]).round(2)
        print("\nBattery temperature range per battery:")
        print(temp_per_batt.to_string())

        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(temp_per_batt))
        ax.bar(x, temp_per_batt["max"] - temp_per_batt["min"],
               bottom=temp_per_batt["min"], color="#ff7f0e", alpha=0.7, label="Min–Max range")
        ax.plot(x, temp_per_batt["mean"], "o-", color="black", markersize=4, label="Mean")
        ax.set_xticks(x)
        ax.set_xticklabels(temp_per_batt.index, rotation=45, ha="right")
        ax.set_title("Battery Temperature Range per Battery")
        ax.set_ylabel("°C")
        ax.legend()
        plt.tight_layout()
        savefig("04_temperature_per_battery.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — MISSION TYPE DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 10 — MISSION TYPE (0=Reference, 1=Regular Mission)")
print("=" * 70)

if "mission_type" in COL_MAP and COL_MAP["mission_type"] in data.columns:
    mc = COL_MAP["mission_type"]
    # Only meaningful during discharge
    if "mode" in COL_MAP:
        dis = data[data[COL_MAP["mode"]] == -1]
        mt_vc = dis[mc].value_counts().sort_index()
        print(mt_vc.rename({0: "Reference (0)", 1: "Regular mission (1)"}))
        print(f"\nTotal discharge rows: {len(dis):,}")
else:
    print("  'mission_type' column not found — skipping.")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 11 — SIGNAL NOISE & ABRUPT CHANGES (sample: one battery)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 11 — SIGNAL SMOOTHNESS (sample battery)")
print("=" * 70)

sample_batt = frames[0].copy()
sample_batt.columns = sample_batt.columns.str.strip()

if COL_MAP.get("v_load") and COL_MAP["v_load"] in sample_batt.columns:
    vc  = COL_MAP["v_load"]
    tc  = COL_MAP.get("rel_time")
    mc  = COL_MAP.get("mode")

    # Focus on one discharge block
    if mc and mc in sample_batt.columns:
        dis_block = sample_batt[sample_batt[mc] == -1].copy()
    else:
        dis_block = sample_batt.copy()

    dis_block = dis_block.dropna(subset=[vc])
    diff_v = dis_block[vc].diff().abs()
    print(f"  Voltage diff stats (discharge, {os.path.basename(csv_files[0])}):")
    print(f"  mean |dV|={diff_v.mean():.5f}  std={diff_v.std():.5f}  max|dV|={diff_v.max():.4f}")

    # Quick plot: raw voltage trace of first discharge segment
    first_seg_id = (dis_block[mc] != dis_block[mc].shift()).cumsum().iloc[0] if mc and mc in dis_block.columns else 0
    seg_rows = dis_block.iloc[:500]   # first 500 discharge rows

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    x_axis = seg_rows[tc].values if (tc and tc in seg_rows.columns) else np.arange(len(seg_rows))
    axes[0].plot(x_axis, seg_rows[vc], linewidth=0.8, color="#1f77b4")
    axes[0].set_ylabel("Voltage (load) [V]")
    axes[0].set_title(f"Voltage & Current trace — {os.path.basename(csv_files[0])} (first 500 discharge rows)")

    if COL_MAP.get("current") and COL_MAP["current"] in seg_rows.columns:
        axes[1].plot(x_axis, seg_rows[COL_MAP["current"]], linewidth=0.8, color="#d62728")
        axes[1].set_ylabel("Current [A]")
    axes[1].set_xlabel("Relative time [s]")
    plt.tight_layout()
    savefig("05_signal_trace_sample.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 12 — SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 12 — SUMMARY REPORT")
print("=" * 70)

print(f"""
Dataset      : NASA Randomized & Recommissioned Battery Dataset
Files        : {len(csv_files)} CSV files
Total rows   : {len(data):,}
Batteries    : {data['battery_id'].nunique()}
Columns      : {list(data.columns)}

Detected mapping:
{chr(10).join(f'  {k:15s} → {v}' for k,v in COL_MAP.items())}

Cycle life   : {cyc_summary['n_cycles'].min()} – {cyc_summary['n_cycles'].max()} cycles
Avg cycles   : {cyc_summary['n_cycles'].mean():.0f}

Capacity (Ah):
  Min  = {cyc_df['capacity_Ah'].min():.4f}
  Mean = {cyc_df['capacity_Ah'].mean():.4f}
  Max  = {cyc_df['capacity_Ah'].max():.4f}

Nominal capacity : {NOMINAL_CAP} Ah
EOL threshold    : {0.8*NOMINAL_CAP:.3f} Ah  (80% of nominal)
""")

if SAVE_PLOTS:
    print(f"All plots saved to: {OUT_DIR}")
print("Done.")