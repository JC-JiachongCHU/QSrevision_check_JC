import streamlit as st
import pandas as pd
import string
import numpy as np
import matplotlib.pyplot as plt
import io, os
from typing import List
import matplotlib.lines as mlines
import glob
import matplotlib as mpl
from matplotlib.colors import PowerNorm
import re
from pathlib import Path


@st.cache_data(show_spinner=False)
def load_quantstudio(uploaded_file) -> pd.DataFrame:
    """
    Read QuantStudio exports from CSV or Excel (xlsx/xls) and standardize columns.
    Removes comment lines for CSV; for Excel, reads the first sheet and
    auto-recovers header if needed.
    """
    from pathlib import Path

    suffix = Path(uploaded_file.name).suffix.lower()

    def _standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
        rename = {}
        for c in df.columns:
            cl = str(c).lower().strip()
            if cl == "well position": rename[c] = "Well"
            elif cl == "cycle number": rename[c] = "Cycle"
            elif cl == "well":         rename[c] = "Index"  # numeric index, not A1 label
            elif cl == "stage number": rename[c] = "Stage"
            elif cl == "step number":  rename[c] = "Step"
        if rename:
            df = df.rename(columns=rename)
        if "Index" in df.columns and "Well" in df.columns:
            df = df.drop(columns=["Index"])
        return df

    if suffix == ".csv":
        txt = uploaded_file.getvalue().decode("utf-8", errors="replace")
        # strip blank lines and comment-like prefaces
        lines = [ln for ln in txt.splitlines()
                 if (s := ln.strip()) and not s.startswith("#") and s not in {"...", "…"}]
        df = pd.read_csv(io.StringIO("\n".join(lines)), engine="python")
        return _standardize_cols(df)

    elif suffix in (".xlsx", ".xls"):
        # try normal read first
        uploaded_file.seek(0)
        try:
            df = pd.read_excel(uploaded_file, sheet_name=0)  # engine auto
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file, sheet_name=0, engine="openpyxl")

        # if header wasn't detected, try to find it
        cols_str = pd.Index([str(c).lower() for c in df.columns])
        if not any(k in " ".join(cols_str) for k in ["well position", "cycle number", "stage number", "step number", "well"]):
            uploaded_file.seek(0)
            tmp = pd.read_excel(uploaded_file, sheet_name=0, header=None)
            # find a row that contains likely header tokens
            header_row = None
            tokens = {"well position", "cycle number", "stage number", "step number", "well"}
            for i in range(min(len(tmp), 30)):  # scan first 30 rows
                row_vals = tmp.iloc[i].astype(str).str.lower().str.strip()
                if any(v in tokens for v in row_vals):
                    header_row = i
                    break
            uploaded_file.seek(0)
            if header_row is not None:
                df = pd.read_excel(uploaded_file, sheet_name=0, header=header_row)
        return _standardize_cols(df)

    else:
        raise ValueError(f"Unsupported file type: {suffix}")

# ---- Global Matplotlib sizing (tweak as you like) ----
# FIGSIZE = (7,5)   # inches
DPI = 110              # pixels per inch

mpl.rcParams.update({
    
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
})

# === Here we GO! ===




# timestamp = datetime.datetime.now().strftime("%Y/%m/%d %H:%M")
version = "v1.0.0"

st.set_page_config(layout="wide")
st.title("QS5 incorrect revision check")
st.markdown(f"**Version:** {version}")
# st.markdown(f"**Last updated:** {timestamp}")
st.write("Contact JIACHONG CHU for questions.")



st.title("qPCR Data Importer")
st.markdown("Please import ALL exported file from QuantStudio")
uploaded_files = []
uploaded_files = st.file_uploader("QuantStudio exported files",type=["csv", "xlsx", "xls"],accept_multiple_files=True)


def make_plate_df(plate_format: str) -> pd.DataFrame:
    """Create a boolean DataFrame for the plate shape."""
    if plate_format.startswith("384"):
        rows = list(string.ascii_uppercase[:16])  # A–P
        cols = list(range(1, 24+1))              # 1–24
    else:  # 96
        rows = list(string.ascii_uppercase[:8])  # A–H
        cols = list(range(1, 12+1))              # 1–12
    df = pd.DataFrame(False, index=rows, columns=cols)
    return df

def wells_from_df(df: pd.DataFrame) -> list[str]:
    """Extract selected wells from a boolean DataFrame in A1 style."""
    wells = []
    for r in df.index:
        for c in df.columns:
            if bool(df.loc[r, c]):
                wells.append(f"{r}{c}")
    return wells

# --- UI: Plate format ---
plate_format = st.radio(
    "Plate format",
    ["384-well (16×24)", "96-well (8×12)"],
    horizontal=True,
)

# Initialize plate grid
grid_key = f"well_grid_{'384' if plate_format.startswith('384') else '96'}"
if grid_key not in st.session_state:
    st.session_state[grid_key] = make_plate_df(plate_format)

plate_df = st.session_state[grid_key]

def full_plate_select(df: pd.DataFrame,
                      row_rule: str = "All rows",
                      col_rule: str = "All cols",
                      select: bool = True) -> pd.DataFrame:
    """
    Apply a bulk selection to the entire plate with row/col filters.
    - df: boolean DataFrame; index are row letters (A..), columns are ints (1..)
    - row_rule: "All rows" | "Odd rows only" | "Even rows only"
    - col_rule: "All cols" | "Odd cols only" | "Even cols only"
    - select: True -> set checked; False -> uncheck
    Returns a modified copy of df.
    """
    out = df.copy()

    def row_ok(r_label: str) -> bool:
        # 1-based row position: A=1, B=2, ...
        pos = df.index.get_loc(r_label) + 1
        if row_rule == "Odd rows only":
            return (pos % 2) == 1
        if row_rule == "Even rows only":
            return (pos % 2) == 0
        return True  # All rows

    def col_ok(c_label) -> bool:
        c = int(c_label)
        if col_rule == "Odd cols only":
            return (c % 2) == 1
        if col_rule == "Even cols only":
            return (c % 2) == 0
        return True  # All cols

    for r in df.index:
        if not row_ok(r):
            continue
        # vectorized assignment for allowed columns
        allowed_cols = [c for c in df.columns if col_ok(c)]
        if allowed_cols:
            out.loc[r, allowed_cols] = select
    return out


# ---- UI controls for full-plate selection ----
cc1, cc2, cc3, cc4 = st.columns([1, 1, 1, 1])

with cc1:
    row_rule = st.selectbox(
        "Row rule",
        ["All rows", "Odd rows only", "Even rows only"],
        index=0,
        help="Choose which rows to affect when bulk selecting."
    )
with cc2:
    col_rule = st.selectbox(
        "Column rule",
        ["All cols", "Odd cols only", "Even cols only"],
        index=0,
        help="Choose which columns to affect when bulk selecting."
    )
with cc3:
    mode = st.radio("Mode", ["Select", "Deselect"], horizontal=True)
with cc4:
    applied = st.button("Apply full-plate selection", use_container_width=True)

if applied:
    st.session_state[grid_key] = full_plate_select(
        st.session_state[grid_key],
        row_rule=row_rule,
        col_rule=col_rule,
        select=(mode == "Select")
    )
    st.success(f"Applied: {mode.lower()} | {row_rule} × {col_rule}")
    plate_df = st.session_state[grid_key]
# --- Row/Column selectors ---
row_choice = st.multiselect("Select entire rows", plate_df.index)
col_choice = st.multiselect("Select entire columns", plate_df.columns)

if st.button("Apply row/col selection"):
    for r in row_choice:
        plate_df.loc[r, :] = True
    for c in col_choice:
        plate_df.loc[:, c] = True
    st.session_state[grid_key] = plate_df

# --- Grid editor ---
column_config = {c: st.column_config.CheckboxColumn() for c in plate_df.columns}
edited_grid = st.data_editor(
    plate_df,
    use_container_width=True,
    num_rows="fixed",
    hide_index=False,
    column_config=column_config,
    key=f"editor_{grid_key}",
)

st.session_state[grid_key] = edited_grid
plate_df = st.session_state[grid_key]
# --- Extract selected wells ---
selected_wells = wells_from_df(edited_grid)

with st.expander(f"Selected wells ({len(selected_wells)})", expanded=False):
    st.write(", ".join(selected_wells) if selected_wells else "None")

st.info(f"{len(selected_wells)} wells selected.")


# ===== qPOS (reference) well selector =====
fam_raw_ch = 'X1_M1'   # pre: FAM raw
rox_raw_ch = 'X4_M4'   # pre: ROX raw
FAM_post_ch = 'FAM'    # post: FAM multicomponent
ROX_post_ch = 'ROX'

# derive row/col size from plate_format
if plate_format.startswith("384"):
    n_rows, n_cols = 16, 24
    default_ref = "P24"
    default_style_idx = 1  # Up–Down default for 384
else:
    n_rows, n_cols = 8, 12
    default_ref = "H12"
    default_style_idx = 0  # Left–Right default for 96

rows = list(string.ascii_uppercase[:n_rows])
cols = list(range(1, n_cols + 1))
all_wells = [f"{r}{c}" for r in rows for c in cols]

# make sure default exists
if default_ref not in all_wells:
    default_ref = all_wells[-1]

refwell = st.selectbox(
    "qPOS (reference) well",
    options=all_wells,
    index=all_wells.index(default_ref),
    help="Used to compute relative FRC from its Cq."
)


results_file = None
raw_file = None
multicomponent_file = None

for f in uploaded_files:  # uploaded_files comes from st.file_uploader
    name = f.name.lower()
    if "results" in name and "replicate" not in name and "sample" not in name and "rq" not in name:
        results_file = f
    elif "raw data" in name:
        raw_file = f
    elif "multicomponent" in name:
        multicomponent_file = f

if not results_file:
    st.error("No Results file found (expected something like *_Results_*.csv)")
if not raw_file:
    st.error("No Raw Data file found (expected something like *_Raw Data_*.csv)")
if not multicomponent_file:
    st.error("No Multicomponent file found (expected something like *_Multicomponent_*.csv)")

runname = Path(raw_file.name).stem.split('_Raw Data', 1)[0]

if results_file and raw_file and multicomponent_file:
    results = load_quantstudio(results_file)
    df = load_quantstudio(raw_file)
    df_m = load_quantstudio(multicomponent_file)

    st.success("All key files loaded successfully!")

# pull the FAM Cq of the reference well
sub_ref = results[results["Well"].astype(str) == str(refwell)]
sub_ref_fam = sub_ref[sub_ref["Reporter"].astype(str) == "FAM"]
ref_vals = pd.to_numeric(sub_ref_fam["Cq"], errors="coerce").dropna().to_numpy()
ref_cq = float(ref_vals[0]) if ref_vals.size else np.nan


mask = edited_grid.astype(bool)
selected_rows = [r for r in mask.index if mask.loc[r].any()]
selected_cols = [c for c in mask.columns if mask[c].any()]

FRC = np.full((len(selected_rows), len(selected_cols)), np.nan, dtype=float)
avg = np.full_like(FRC, np.nan, dtype=float)
std = np.full_like(FRC, np.nan, dtype=float)
# ---------- Fill FRC and avg only for checked wells ----------
for i, r in enumerate(selected_rows):
    for j, c in enumerate(selected_cols):
        if not bool(mask.loc[r, c]):
            continue  # not selected, leave NaN
        well = f"{r}{c}"

        # Cq from Results -> FAM only
        sub_res = results[results["Well"].astype(str) == str(well)]
        sub_fam = sub_res[sub_res["Reporter"].astype(str) == "FAM"]
        vals = pd.to_numeric(sub_fam["Cq"], errors="coerce").dropna().to_numpy()
        Cq = float(vals[0]) if vals.size else np.nan
        FRC[i,j] = 30000 / (2**(Cq - ref_cq))
        # avg metric from Raw vs Multicomponent ROX (cycles 15–40)
        sub_raw = df[df["Well"].astype(str) == str(well)]
        sub_mc  = df_m[df_m["Well"].astype(str) == str(well)]

        rox_y = pd.to_numeric(sub_raw[rox_raw_ch], errors="coerce").to_numpy()
        rox_post = pd.to_numeric(sub_mc[ROX_post_ch], errors="coerce").to_numpy()
        flag = np.abs(1.0 - rox_post / rox_y)
        std[i, j] = np.nanstd(rox_y[0:15])
        avg[i, j] = np.nanmean(flag[14:40])  # cycles 15–40


# ----- calculate the replicates
replicate_style = st.radio(
    "Replicate style",
    ["Left–Right (halves)", "Up–Down (neighbors)"],
    index=default_style_idx,
    help="Left–Right: split columns into 2 halves; Up–Down: pair adjacent rows (A↔B, C↔D, ...).",
    horizontal=True,
)

r, c = FRC.shape
if replicate_style.startswith("Left"):
    mid = c // 2
    left, right = FRC[:, :mid], FRC[:, mid:]
    stack = np.stack([left, right], axis=0)          # (2, rows, mid)
    pair_avg = np.nanmean(stack, axis=0)             # (rows, mid)
    pair_std = np.nanstd(stack, axis=0, ddof=0)
    pair_cv = (pair_std / pair_avg) * 100.0

    aux_left, aux_right = avg[:, :mid], avg[:, mid:]
    pair_color = np.maximum(aux_left, aux_right)     # (rows, mid)

else:  # Up–Down neighbors: A↔B, C↔D, ...
    top, bottom = FRC[0::2, :], FRC[1::2, :]         # (rows/2, cols)
    stack = np.stack([top, bottom], axis=0)          # (2, rows/2, cols)
    pair_avg = np.nanmean(stack, axis=0)             # (rows/2, cols)
    pair_std = np.nanstd(stack, axis=0, ddof=0)
    pair_cv = (pair_std / pair_avg) * 100.0

    aux_top, aux_bottom = avg[0::2, :], avg[1::2, :]
    pair_color = np.maximum(aux_top, aux_bottom)  

X = pair_avg.ravel()
Y = pair_cv.ravel()
C = pair_color.ravel()

sort_idx = np.argsort(C)
X_sorted = X[sort_idx]
Y_sorted = Y[sort_idx]
C_sorted = C[sort_idx]

# Colormap + norm
norm = mpl.colors.Normalize(vmin=0, vmax=0.5)
cmap = plt.get_cmap("plasma_r")

# Plot in Streamlit
fig, ax = plt.subplots(figsize=(7,5))  # uses global FIGSIZE/DPI above
sc = ax.scatter(
    X_sorted, Y_sorted,
    s=28, alpha=0.9,
    c=C_sorted, cmap=cmap, norm=norm
)
ax.set_xscale("log")
ax.set_xlabel("Mean FRC")
ax.set_ylabel("FRC CV%")
ax.set_title(f"{runname} · Rep style: {replicate_style}")
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Pair min AVG of |1 − (ROX/X4_M4)| cycles 15–40")
st.pyplot(fig, use_container_width=False)


fig, ax = plt.subplots(figsize=(14,6))  # uses global FIGSIZE/DPI above
im = ax.imshow(np.ma.masked_invalid(avg), cmap="Reds", aspect="auto", vmin=0, vmax=0.7) # masks NaNs
ax.set_xticks(np.arange(len(selected_cols)))
ax.set_xticklabels(selected_cols)
ax.set_yticks(np.arange(len(selected_rows)))
ax.set_yticklabels(selected_rows)
ax.set_xlabel("Column")
ax.set_ylabel("Row")
cbar = plt.colorbar(im, ax=ax)
plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
for i in range(avg.shape[0]):
    for j in range(avg.shape[1]):
        ax.text(j, i, f"{avg[i, j]:.2f}", 
                ha="center", va="center", fontsize=7, color="black")
cbar.set_label(f"|1-ROX/X4_M4| for cycle 15-40")
ax.set_title(f"{runname} - |1-ROX/X4_M4| for cycle 15-40")
st.pyplot(fig, use_container_width=False)

fig, ax = plt.subplots(figsize=(14,6))
im = ax.imshow(np.ma.masked_invalid(std), cmap="Reds", aspect="auto", vmin=0, vmax=30000) # masks NaNs
ax.set_xticks(np.arange(len(selected_cols)))
ax.set_xticklabels(selected_cols)
ax.set_yticks(np.arange(len(selected_rows)))
ax.set_yticklabels(selected_rows)
ax.set_xlabel("Column")
ax.set_ylabel("Row")
cbar = plt.colorbar(im, ax=ax)
plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
for i in range(std.shape[0]):
    for j in range(std.shape[1]):
        ax.text(j, i, f"{std[i, j]:.2f}", 
                ha="center", va="center", fontsize=7, color="black")
cbar.set_label(f"Standard deviation (X4_M4) for first 10 cycles")
ax.set_title(f"{runname} - std (X4_M4) for first 10 cycles")
st.pyplot(fig, use_container_width=False)

