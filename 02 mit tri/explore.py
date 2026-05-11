"""
MIT–Stanford–TRI Fast-Charging Dataset — Exploration Script (v3)
=================================================================
Robust version: auto-discovers the HDF5 structure instead of
assuming a fixed path, so it works regardless of exact nesting.

Run this first; it will print the full tree + all extracted data.
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import h5py

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({"figure.dpi": 110, "font.size": 9})

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATASET_PATH = r"C:\Users\admin\Desktop\DR2\11 All Datasets\04 MIT–Stanford–TRI Fast-Charging Dataset\mit_dataset"
SAVE_PLOTS   = True
OUT_DIR      = os.path.join(DATASET_PATH, "exploration_outputs")

NOMINAL_CAP  = 2.54
V_MIN, V_MAX = 2.0,  3.65
I_MAX        = 10.0
T_MIN, T_MAX = 10.0, 50.0

if SAVE_PLOTS:
    os.makedirs(OUT_DIR, exist_ok=True)

def savefig(name):
    if SAVE_PLOTS:
        p = os.path.join(OUT_DIR, name)
        plt.savefig(p, bbox_inches="tight")
        print(f"  -> saved: {p}")
    plt.show()
    plt.close()

mat_files = sorted([
    os.path.join(DATASET_PATH, f)
    for f in os.listdir(DATASET_PATH) if f.endswith(".mat")
])

# ======================================================================
#  SECTION 1 - FILE INVENTORY
# ======================================================================
print("=" * 70)
print("SECTION 1 - FILE INVENTORY")
print("=" * 70)
for f in mat_files:
    print(f"  {os.path.basename(f):<65} {os.path.getsize(f)/1024**2:.1f} MB")
print(f"\nTotal : {len(mat_files)} files, "
      f"{sum(os.path.getsize(f) for f in mat_files)/1024**2:.1f} MB")


# ======================================================================
#  SECTION 2 - FULL HDF5 TREE
# ======================================================================
print("\n" + "=" * 70)
print("SECTION 2 - HDF5 TREE (first file, depth <= 4)")
print("=" * 70)

def _tree(node, h5root, prefix="", depth=0, max_depth=4):
    if depth > max_depth:
        return
    keys = list(node.keys()) if hasattr(node, "keys") else []
    for key in keys[:25]:
        item = node[key]
        if isinstance(item, h5py.Group):
            print(f"{prefix}[G] {key}/  ({len(item)} children)")
            _tree(item, h5root, prefix + "    ", depth + 1, max_depth)
        elif isinstance(item, h5py.Dataset):
            print(f"{prefix}[D] {key}  shape={item.shape}  dtype={item.dtype}")
            if item.dtype.kind == "O" and item.size > 0:
                ref = item[()].flatten()[0]
                try:
                    target = h5root[ref]
                    if isinstance(target, h5py.Group):
                        print(f"{prefix}     L-[ref->G] keys={list(target.keys())[:12]}")
                        _tree(target, h5root, prefix + "         ", depth + 1, max_depth)
                    elif isinstance(target, h5py.Dataset):
                        print(f"{prefix}     L-[ref->D] shape={target.shape}  dtype={target.dtype}")
                except Exception as e:
                    print(f"{prefix}     L- ref ERR: {e}")

with h5py.File(mat_files[0], "r") as f:
    print(f"Top-level keys: {list(f.keys())}\n")
    _tree(f, f)


# ======================================================================
#  HELPERS - flexible HDF5 navigation
# ======================================================================

def resolve_ref(h5, ref):
    try:
        return h5[ref]
    except Exception:
        return None

def to_float_array(ds):
    try:
        return ds[()].flatten().astype(float)
    except Exception:
        return np.array([np.nan])

def safe_stat(arr, func):
    v = arr[~np.isnan(arr)] if len(arr) else np.array([])
    return func(v) if len(v) else np.nan

def get_n_cells_from_batch(h5):
    try:
        batch = h5["batch"]
        for key in batch.keys():
            ds = batch[key]
            if isinstance(ds, h5py.Dataset) and ds.dtype.kind == "O":
                return int(ds[()].flatten().shape[0])
        return 0
    except Exception:
        return 0

def get_summary_field(h5, cell_idx, field):
    """
    Two layouts:
      A) batch/summary is a (N_cells,) object-ref array;
         each ref -> Group with fields like QDischarge, cycle ...
      B) batch/summary is a Group;
         batch/summary/<field> is a (N_cells,) object-ref array
    """
    try:
        batch = h5["batch"]
        if "summary" not in batch:
            return np.array([np.nan])
        summ = batch["summary"]

        # Layout B: summary is a Group
        if isinstance(summ, h5py.Group):
            if field not in summ:
                return np.array([np.nan])
            ds   = summ[field]
            data = ds[()].flatten()
            if data.dtype.kind == "O":
                if cell_idx >= len(data):
                    return np.array([np.nan])
                target = resolve_ref(h5, data[cell_idx])
                return to_float_array(target) if target else np.array([np.nan])
            return data.astype(float)

        # Layout A: summary is a Dataset of object refs
        if isinstance(summ, h5py.Dataset) and summ.dtype.kind == "O":
            flat = summ[()].flatten()
            if cell_idx >= len(flat):
                return np.array([np.nan])
            cell_ref = flat[cell_idx]
            cell_grp = resolve_ref(h5, cell_ref)
            if cell_grp is None or not isinstance(cell_grp, h5py.Group):
                return np.array([np.nan])
            if field not in cell_grp:
                return np.array([np.nan])
            return to_float_array(cell_grp[field])

        return np.array([np.nan])
    except Exception:
        return np.array([np.nan])

def _get_cell_cycles_node(h5, cell_idx):
    """
    Return the node that holds per-cycle data for one cell.
    Returns (cell_node, layout) where layout is 'data_ds' or 'direct'.
    """
    try:
        batch = h5["batch"]
        if "cycles" not in batch:
            return None, None

        cyc_ds = batch["cycles"]
        if not (isinstance(cyc_ds, h5py.Dataset) and cyc_ds.dtype.kind == "O"):
            return None, None

        flat = cyc_ds[()].flatten()
        if cell_idx >= len(flat):
            return None, None

        cell_ref = flat[cell_idx]
        cell_grp = resolve_ref(h5, cell_ref)
        if cell_grp is None:
            return None, None

        # Does it have a 'data' key that is a ref array of individual cycles?
        if "data" in cell_grp:
            data_ds = cell_grp["data"]
            if isinstance(data_ds, h5py.Dataset) and data_ds.dtype.kind == "O":
                return cell_grp, "data_ds"

        # Otherwise fields (V, I, t ...) are directly inside and each is
        # a (N_cycles,) ref array
        return cell_grp, "direct"

    except Exception:
        return None, None

def get_n_cycles_for_cell(h5, cell_idx):
    node, layout = _get_cell_cycles_node(h5, cell_idx)
    if node is None:
        return 0
    try:
        if layout == "data_ds":
            return int(node["data"][()].flatten().shape[0])
        # direct: any field is a (N_cycles,) ref array
        for key in node.keys():
            ds = node[key]
            if isinstance(ds, h5py.Dataset) and ds.dtype.kind == "O":
                return int(ds[()].flatten().shape[0])
    except Exception:
        pass
    return 0

def get_raw_cycle_field(h5, cell_idx, cycle_idx, field):
    node, layout = _get_cell_cycles_node(h5, cell_idx)
    if node is None:
        return np.array([np.nan])
    try:
        if layout == "data_ds":
            flat    = node["data"][()].flatten()
            if cycle_idx >= len(flat):
                return np.array([np.nan])
            cyc_ref = flat[cycle_idx]
            cyc_grp = resolve_ref(h5, cyc_ref)
            if cyc_grp is None or not isinstance(cyc_grp, h5py.Group):
                return np.array([np.nan])
            if field not in cyc_grp:
                return np.array([np.nan])
            return to_float_array(cyc_grp[field])

        # direct layout: node[field] is a (N_cyc,) ref array
        if field not in node:
            return np.array([np.nan])
        field_ds = node[field]
        if isinstance(field_ds, h5py.Dataset):
            if field_ds.dtype.kind == "O":
                flat = field_ds[()].flatten()
                if cycle_idx >= len(flat):
                    return np.array([np.nan])
                target = resolve_ref(h5, flat[cycle_idx])
                return to_float_array(target) if target else np.array([np.nan])
            else:
                arr = field_ds[()].flatten().astype(float)
                return arr[cycle_idx:cycle_idx+1]
    except Exception:
        pass
    return np.array([np.nan])


# ======================================================================
#  SECTION 3 - CELL COUNT & FIELD PROBE
# ======================================================================
print("\n" + "=" * 70)
print("SECTION 3 - CELL COUNT & AVAILABLE FIELDS")
print("=" * 70)

print("\nCell counts per batch file:")
batch_meta = []
for f in mat_files:
    with h5py.File(f, "r") as h5:
        n = get_n_cells_from_batch(h5)
        batch_meta.append({"file": os.path.basename(f)[:20], "n_cells": n})
        print(f"  {os.path.basename(f):<65} {n} cells")
total_cells = sum(b["n_cells"] for b in batch_meta)
print(f"\nTotal cells: {total_cells}")

print("\nSummary fields (cell 0, first file):")
with h5py.File(mat_files[0], "r") as h5:
    for sf in ["QDischarge","QCharge","cycle","IR","Tmax","Tavg","Tmin","chargetime"]:
        arr   = get_summary_field(h5, 0, sf)
        valid = arr[~np.isnan(arr)]
        print(f"  {sf:<15}: len={len(arr)}, sample={valid[:4].tolist()}" if len(valid)
              else f"  {sf:<15}: all NaN / not found")

print("\nRaw cycle fields (cell 0, cycle 0, first file):")
with h5py.File(mat_files[0], "r") as h5:
    nc = get_n_cycles_for_cell(h5, 0)
    print(f"  Cycles in cell 0: {nc}")
    for rf in ["t","V","I","T","Qc","Qd"]:
        arr   = get_raw_cycle_field(h5, 0, 0, rf)
        valid = arr[~np.isnan(arr)]
        print(f"  {rf:<6}: len={len(arr)}, sample={valid[:4].tolist()}" if len(valid)
              else f"  {rf:<6}: empty / all NaN")


# ======================================================================
#  SECTION 4 - CYCLE COUNTS PER CELL
# ======================================================================
print("\n" + "=" * 70)
print("SECTION 4 - CYCLE COUNTS PER CELL")
print("=" * 70)

all_cells = []
for f in mat_files:
    bname = os.path.basename(f)[:10]
    with h5py.File(f, "r") as h5:
        n_cells = get_n_cells_from_batch(h5)
        for ci in range(n_cells):
            n_cyc = get_n_cycles_for_cell(h5, ci)
            policy = "N/A"
            try:
                p_ds  = h5["batch"]["policy_readable"]
                p_ref = p_ds[()].flatten()[ci]
                p_arr = h5[p_ref][()]
                policy = "".join(chr(int(c)) for c in p_arr.flatten()
                                 if 32 <= int(c) < 127)
            except Exception:
                pass
            all_cells.append({
                "batch": bname, "cell_id": f"{bname}_c{ci:03d}",
                "n_cycles": n_cyc, "policy": policy,
            })

cells_df = pd.DataFrame(all_cells)
print(cells_df[["batch","cell_id","n_cycles","policy"]].to_string(index=False))
print(f"\nTotal cells : {len(cells_df)}")
if len(cells_df):
    print(f"Cycle range : {cells_df['n_cycles'].min()} - {cells_df['n_cycles'].max()}")
    print(f"Mean/std    : {cells_df['n_cycles'].mean():.1f} / {cells_df['n_cycles'].std():.1f}")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(cells_df["n_cycles"], bins=30, edgecolor="white", color="#1f77b4", alpha=0.85)
    ax.axvline(cells_df["n_cycles"].mean(), color="red", ls="--",
               label=f"Mean={cells_df['n_cycles'].mean():.0f}")
    ax.set_xlabel("Cycle life"); ax.set_ylabel("# cells")
    ax.set_title("Cycle Life Distribution - MIT Fast-Charging Dataset")
    ax.legend(); plt.tight_layout()
    savefig("01_cycle_life_distribution.png")


# ======================================================================
#  SECTION 5 - CAPACITY FADE & SOH
# ======================================================================
print("\n" + "=" * 70)
print("SECTION 5 - CAPACITY FADE & SOH")
print("=" * 70)

cap_records = []
for f in mat_files:
    bname = os.path.basename(f)[:10]
    with h5py.File(f, "r") as h5:
        n_cells = get_n_cells_from_batch(h5)
        for ci in range(n_cells):
            Q   = get_summary_field(h5, ci, "QDischarge")
            cyc = get_summary_field(h5, ci, "cycle")
            valid = ~np.isnan(Q) & (Q > 0)
            if not valid.any():
                continue
            ref_cap = Q[valid][0]
            SOH     = Q / ref_cap
            for idx, (c, q, s) in enumerate(zip(cyc, Q, SOH)):
                cap_records.append({
                    "batch": bname, "cell_id": f"{bname}_c{ci:03d}",
                    "cycle": int(c) if not np.isnan(c) else idx,
                    "Q_Ah": q, "SOH": s,
                })

cap_df = pd.DataFrame(cap_records)
if not cap_df.empty:
    print(f"Total cycle records : {len(cap_df):,}")
    print(f"Q_discharge range   : {cap_df['Q_Ah'].min():.3f} - {cap_df['Q_Ah'].max():.3f} Ah")
    print(f"SOH range           : {cap_df['SOH'].min():.4f} - {cap_df['SOH'].max():.4f}")
    print(f"Below EOL (SOH<0.80): {(cap_df['SOH']<0.80).sum():,}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].hist(cap_df["SOH"].clip(0,1.1), bins=40, edgecolor="white",
                 color="#2ca02c", alpha=0.85)
    axes[0].axvline(0.80, color="red", ls="--", label="EOL SOH=0.80")
    axes[0].set_title("SOH Distribution"); axes[0].set_xlabel("SOH"); axes[0].legend()

    sample = cap_df["cell_id"].unique()[:25]
    cmap   = plt.cm.viridis(np.linspace(0,1,len(sample)))
    for cid, col in zip(sample, cmap):
        sub = cap_df[cap_df["cell_id"]==cid].sort_values("cycle")
        axes[1].plot(sub["cycle"], sub["Q_Ah"], lw=0.7, alpha=0.7, color=col)
    axes[1].axhline(0.8*NOMINAL_CAP, color="red", ls="--",
                    label=f"EOL={0.8*NOMINAL_CAP:.2f}Ah")
    axes[1].set_title("Capacity Fade (sample)"); axes[1].set_xlabel("Cycle")
    axes[1].set_ylabel("Q [Ah]"); axes[1].legend(fontsize=7)
    plt.tight_layout()
    savefig("02_capacity_fade.png")
else:
    print("  No capacity data - check Section 3 summary field output.")


# ======================================================================
#  SECTION 6 - MISSING / NaN AUDIT
# ======================================================================
print("\n" + "=" * 70)
print("SECTION 6 - MISSING / NaN AUDIT")
print("=" * 70)

summary_fields = ["QDischarge","QCharge","cycle","IR","Tmax","Tavg","Tmin","chargetime"]
null_agg = {f: {"nan":0,"total":0} for f in summary_fields}

for f in mat_files:
    with h5py.File(f, "r") as h5:
        n_cells = get_n_cells_from_batch(h5)
        for ci in range(n_cells):
            for sf in summary_fields:
                arr = get_summary_field(h5, ci, sf)
                null_agg[sf]["nan"]   += int(np.isnan(arr).sum())
                null_agg[sf]["total"] += len(arr)

print("\nSummary field NaN counts (all cells, all batches):")
for sf, v in null_agg.items():
    pct  = v["nan"]/v["total"]*100 if v["total"] else 0
    flag = "  OK" if v["nan"]==0 else f"  <- {v['nan']:,} NaN"
    print(f"  {sf:<15}: {v['nan']:6,} / {v['total']:6,}  ({pct:.2f}%){flag}")

print("\nRaw field NaN check (first file, 3 cells x 10 cycles):")
raw_fields = ["t","V","I","T","Qc","Qd"]
with h5py.File(mat_files[0], "r") as h5:
    for ci in range(min(get_n_cells_from_batch(h5), 3)):
        n_cyc = get_n_cycles_for_cell(h5, ci)
        agg = {rf: {"nan":0,"total":0} for rf in raw_fields}
        for cyc_i in range(min(n_cyc, 10)):
            for rf in raw_fields:
                arr = get_raw_cycle_field(h5, ci, cyc_i, rf)
                agg[rf]["nan"]   += int(np.isnan(arr).sum())
                agg[rf]["total"] += len(arr)
        print(f"\n  Cell {ci} ({min(n_cyc,10)} cycles):")
        for rf in raw_fields:
            pct  = agg[rf]["nan"]/agg[rf]["total"]*100 if agg[rf]["total"] else 0
            flag = "  OK" if pct==0 else f"  <- {agg[rf]['nan']} NaN"
            print(f"    {rf}: {agg[rf]['nan']:,}/{agg[rf]['total']:,} ({pct:.1f}%){flag}")


# ======================================================================
#  SECTION 7 - PHYSICAL PLAUSIBILITY
# ======================================================================
print("\n" + "=" * 70)
print("SECTION 7 - PHYSICAL PLAUSIBILITY CHECKS")
print("=" * 70)

plaus = []
MAX_CELLS = 40

with h5py.File(mat_files[0], "r") as h5:
    n_cells = get_n_cells_from_batch(h5)
    for ci in range(min(n_cells, MAX_CELLS)):
        n_cyc = get_n_cycles_for_cell(h5, ci)
        if n_cyc == 0:
            continue
        V_all, I_all, T_all = [], [], []
        for cyc_i in range(n_cyc):
            for dest, field in [(V_all,"V"),(I_all,"I"),(T_all,"T")]:
                arr = get_raw_cycle_field(h5, ci, cyc_i, field)
                # Only add if real data (not just a [NaN] placeholder)
                if len(arr) > 1 or (len(arr)==1 and not np.isnan(arr[0])):
                    dest.extend(arr.tolist())

        V = np.array(V_all, dtype=float)
        I = np.array(I_all, dtype=float)
        T = np.array(T_all, dtype=float)

        if len(V) == 0:
            print(f"  Cell {ci:03d}: no V/I/T extracted (check field names in Section 3)")
            continue

        plaus.append({
            "cell":  ci, "n_pts": len(V),
            "V_min": safe_stat(V, np.min), "V_max": safe_stat(V, np.max),
            "V_bad": int(np.nansum((V<V_MIN)|(V>V_MAX))),
            "I_min": safe_stat(I, np.min), "I_max": safe_stat(I, np.max),
            "I_bad": int(np.nansum(np.abs(I)>I_MAX)) if len(I) else 0,
            "T_min": safe_stat(T, np.min), "T_max": safe_stat(T, np.max),
            "T_bad": int(np.nansum((T<T_MIN)|(T>T_MAX))) if len(T) else 0,
            "T_pts": len(T),
        })

def _pct(bad, total):
    return f"{bad:,}/{total:,} ({bad/total*100:.3f}%)" if total else "no data"

if plaus:
    plaus_df = pd.DataFrame(plaus)
    tot      = plaus_df["n_pts"].sum()
    print(f"\nChecked {len(plaus_df)} cells  ({tot:,} total data points)")

    print(f"\nVoltage [{V_MIN}, {V_MAX}] V:")
    print(f"  min={plaus_df['V_min'].dropna().min():.4f}  max={plaus_df['V_max'].dropna().max():.4f}")
    print(f"  Out-of-range: {_pct(plaus_df['V_bad'].sum(), tot)}")

    print(f"\nCurrent [+/-{I_MAX}] A:")
    print(f"  min={plaus_df['I_min'].dropna().min():.4f}  max={plaus_df['I_max'].dropna().max():.4f}")
    print(f"  |I|>{I_MAX}: {_pct(plaus_df['I_bad'].sum(), tot)}")
    neg = (plaus_df["I_min"].dropna() < 0).sum()
    print(f"  Cells with I<0 (discharge): {neg}/{len(plaus_df)}")

    print(f"\nTemperature [{T_MIN}, {T_MAX}] C:")
    t_pts = plaus_df["T_pts"].sum()
    print(f"  min={plaus_df['T_min'].dropna().min():.2f}  max={plaus_df['T_max'].dropna().max():.2f}")
    print(f"  Out-of-range: {_pct(plaus_df['T_bad'].sum(), t_pts)}")
else:
    print("\n  No plausibility data extracted.")
    print("  -> Check Section 2 HDF5 tree for actual key names (V/I/T may differ).")


# ======================================================================
#  SECTION 8 - SIGNAL SMOOTHNESS & NOISE
# ======================================================================
print("\n" + "=" * 70)
print("SECTION 8 - SIGNAL SMOOTHNESS & NOISE")
print("=" * 70)

noise_rows = []
with h5py.File(mat_files[0], "r") as h5:
    for ci in range(min(get_n_cells_from_batch(h5), 8)):
        for cyc_i in range(min(get_n_cycles_for_cell(h5, ci), 15)):
            t = get_raw_cycle_field(h5, ci, cyc_i, "t")
            V = get_raw_cycle_field(h5, ci, cyc_i, "V")
            I = get_raw_cycle_field(h5, ci, cyc_i, "I")
            if len(V) < 5: continue
            dV = np.abs(np.diff(V[~np.isnan(V)]))
            dI = np.abs(np.diff(I[~np.isnan(I)])) if len(I)>1 else np.array([np.nan])
            dt = np.diff(t[~np.isnan(t)]) if len(t)>1 else np.array([np.nan])
            noise_rows.append({
                "cell": ci, "cycle": cyc_i, "n_pts": len(V),
                "mean_dV": float(np.mean(dV)), "std_dV": float(np.std(dV)),
                "max_dV":  float(np.max(dV)),
                "mean_dI": float(np.nanmean(dI)), "max_dI": float(np.nanmax(dI)),
                "mean_dt": float(np.nanmean(dt)),
            })

if noise_rows:
    nd = pd.DataFrame(noise_rows)
    print(f"\n|dV| per step: mean={nd['mean_dV'].mean():.5f}  std={nd['std_dV'].mean():.5f}  max={nd['max_dV'].max():.4f}")
    print(f"|dI| per step: mean={nd['mean_dI'].mean():.4f}  max={nd['max_dI'].max():.4f}")
    print(f"Sampling dt  : mean={nd['mean_dt'].mean():.3f} s")
    print(f"Points/cycle : mean={nd['n_pts'].mean():.0f}")

    with h5py.File(mat_files[0], "r") as h5:
        n_cyc = get_n_cycles_for_cell(h5, 0)
        sample_cycs = [c for c in [0,10,50,100,150,200] if c < n_cyc]
        fig, axes = plt.subplots(3, 1, figsize=(11,8), sharex=False)
        cmap = plt.cm.Blues(np.linspace(0.4, 0.95, len(sample_cycs)))
        for cyc_i, col in zip(sample_cycs, cmap):
            t = get_raw_cycle_field(h5, 0, cyc_i, "t")
            V = get_raw_cycle_field(h5, 0, cyc_i, "V")
            I = get_raw_cycle_field(h5, 0, cyc_i, "I")
            T = get_raw_cycle_field(h5, 0, cyc_i, "T")
            axes[0].plot(t, V, lw=0.7, color=col, label=f"Cyc {cyc_i}")
            axes[1].plot(t, I, lw=0.7, color=col)
            axes[2].plot(t, T, lw=0.7, color=col)
        axes[0].set_ylabel("Voltage [V]"); axes[0].legend(fontsize=7, ncol=3)
        axes[1].set_ylabel("Current [A]")
        axes[2].set_ylabel("Temp [C]"); axes[2].set_xlabel("Time [s]")
        axes[0].set_title("Raw cycle traces - cell 0")
        plt.tight_layout()
        savefig("03_raw_traces.png")
else:
    print("  No noise data extracted.")


# ======================================================================
#  SECTION 9 - TEMPERATURE OVERVIEW
# ======================================================================
print("\n" + "=" * 70)
print("SECTION 9 - TEMPERATURE OVERVIEW")
print("=" * 70)

temp_rows = []
for f in mat_files:
    bname = os.path.basename(f)[:10]
    with h5py.File(f, "r") as h5:
        for ci in range(get_n_cells_from_batch(h5)):
            Tmax = get_summary_field(h5, ci, "Tmax")
            Tavg = get_summary_field(h5, ci, "Tavg")
            Tmin = get_summary_field(h5, ci, "Tmin")
            temp_rows.append({
                "cell":  f"{bname}_c{ci:03d}",
                "Tmax":  float(np.nanmax(Tmax)) if len(Tmax) else np.nan,
                "Tavg":  float(np.nanmean(Tavg)) if len(Tavg) else np.nan,
                "Tmin":  float(np.nanmin(Tmin)) if len(Tmin) else np.nan,
            })

temp_df = pd.DataFrame(temp_rows).dropna()
if not temp_df.empty:
    print(f"Global T range  : {temp_df['Tmin'].min():.2f} - {temp_df['Tmax'].max():.2f} C")
    print(f"Mean Tavg       : {temp_df['Tavg'].mean():.2f} C")
    print(f"Cells Tmax>35 C : {(temp_df['Tmax']>35).sum()} / {len(temp_df)}")
    fig, ax = plt.subplots(figsize=(12,4))
    x = np.arange(len(temp_df))
    ax.bar(x, temp_df["Tmax"]-temp_df["Tmin"], bottom=temp_df["Tmin"],
           alpha=0.5, color="#ff7f0e", label="Tmin-Tmax")
    ax.plot(x, temp_df["Tavg"], "o-", ms=2, lw=0.6, color="black", label="Tavg")
    ax.axhline(25, color="blue", ls="--", lw=0.8, label="25C setpoint")
    ax.set_xlabel("Cell index"); ax.set_ylabel("C")
    ax.set_title("Temperature per Cell"); ax.legend(fontsize=8)
    plt.tight_layout()
    savefig("04_temperature_per_cell.png")
else:
    print("  No temperature data extracted.")


# ======================================================================
#  SECTION 10 - TEMPORAL COHERENCE
# ======================================================================
print("\n" + "=" * 70)
print("SECTION 10 - TEMPORAL COHERENCE")
print("=" * 70)

non_mono_t = 0; non_mono_Q = 0; total_segs = 0
with h5py.File(mat_files[0], "r") as h5:
    n_cells = get_n_cells_from_batch(h5)
    for ci in range(min(n_cells, 20)):
        n_cyc = get_n_cycles_for_cell(h5, ci)
        total_segs += n_cyc
        for cyc_i in range(n_cyc):
            t = get_raw_cycle_field(h5, ci, cyc_i, "t")
            tv = t[~np.isnan(t)]
            if len(tv) > 1 and not np.all(np.diff(tv) >= 0):
                non_mono_t += 1
        Q = get_summary_field(h5, ci, "QDischarge")
        Qv = Q[~np.isnan(Q)]
        if len(Qv) > 2 and np.sum(np.diff(Qv) > 0.02*NOMINAL_CAP) > 0:
            non_mono_Q += 1

cyc_idx_issues = 0
with h5py.File(mat_files[0], "r") as h5:
    n_cells = get_n_cells_from_batch(h5)
    for ci in range(n_cells):
        idx = get_summary_field(h5, ci, "cycle")
        v = idx[~np.isnan(idx)]
        if len(v) > 1 and not np.all(np.diff(v) >= 0):
            cyc_idx_issues += 1

print(f"  Non-monotonic time vectors    : {non_mono_t} / {total_segs} segments")
print(f"  Cells with Q recovery >2%     : {non_mono_Q} / {min(n_cells,20)}")
print(f"  Non-sequential cycle index    : {cyc_idx_issues} / {n_cells}")


# ======================================================================
#  SECTION 11 - SUMMARY
# ======================================================================
print("\n" + "=" * 70)
print("SECTION 11 - SUMMARY REPORT")
print("=" * 70)
cr = f"{cells_df['n_cycles'].min()} - {cells_df['n_cycles'].max()}" if len(cells_df) else "N/A"
print(f"""
Dataset     : MIT-Stanford-TRI Fast-Charging (Severson et al. 2019)
Files       : {len(mat_files)} .mat  (MATLAB HDF5 v7.3)
Total cells : {total_cells}
Cycle range : {cr}
Mean cycles : {cells_df['n_cycles'].mean():.1f if len(cells_df) else 'N/A'}
Nominal cap : {NOMINAL_CAP} Ah
EOL thresh  : {0.8*NOMINAL_CAP:.3f} Ah (80%)
""")
if SAVE_PLOTS:
    print(f"Plots -> {OUT_DIR}")
print("Done.")