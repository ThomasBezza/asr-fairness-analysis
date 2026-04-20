from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "raw" / "ASR_Fairness_Master_Spreadsheet_v5_descriptives.xlsx"
OUT_DIR = ROOT / "outputs" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
df = pd.read_excel(DATA_FILE, sheet_name="master_data").copy()

# -----------------------------
# Basic cleaning
# -----------------------------
for col in [
    "asr_model_name_standardized",
    "benchmark_name_standardized",
    "transcript_type_binary",
    "resource_level_tertile",
]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

if "resource_level_tertile" in df.columns:
    df["resource_level_tertile"] = df["resource_level_tertile"].str.title()

# -----------------------------
# Keep rows with WER
# -----------------------------
work_df = df[df["wer_reported"].notna()].copy()

# -----------------------------
# Define grouping key for computed fairness metrics
# -----------------------------
group_cols = [
    "asr_model_name_standardized",
    "benchmark_name_standardized",
    "transcript_type_binary",
]

# -----------------------------
# Optional weight column search for micro-average
# -----------------------------
candidate_weight_cols = [
    "sample_size",
    "subgroup_sample_size",
    "n_samples",
    "utterance_count",
    "word_count",
    "speaker_count",
]

weight_col = None
for c in candidate_weight_cols:
    if c in work_df.columns:
        if pd.to_numeric(work_df[c], errors="coerce").notna().sum() > 0:
            weight_col = c
            work_df[c] = pd.to_numeric(work_df[c], errors="coerce")
            break

# -----------------------------
# Compute group-level fairness metrics
# -----------------------------
metric_rows = []

for keys, g in work_df.groupby(group_cols, dropna=False):
    g = g.copy()
    g["wer_reported"] = pd.to_numeric(g["wer_reported"], errors="coerce")
    g = g[g["wer_reported"].notna()].copy()

    if len(g) == 0:
        continue

    model_name, benchmark_name, transcript_binary = keys

    delta_wer = g["wer_reported"].max() - g["wer_reported"].min()
    worst_group_wer = g["wer_reported"].max()
    macro_avg_wer = g["wer_reported"].mean()

    # Micro-average only if a valid weight column exists
    micro_avg_wer = np.nan
    micro_notes = "not_computed_no_weight_column"

    if weight_col is not None:
        valid_weights = g[weight_col].notna() & (g[weight_col] > 0)
        if valid_weights.sum() > 0:
            tmp = g.loc[valid_weights].copy()
            micro_avg_wer = np.average(tmp["wer_reported"], weights=tmp[weight_col])
            micro_notes = f"computed_with_{weight_col}"
        else:
            micro_notes = f"weight_column_present_but_missing_or_nonpositive_{weight_col}"

    metric_rows.append({
        "asr_model_name_standardized": model_name,
        "benchmark_name_standardized": benchmark_name,
        "transcript_type_binary": transcript_binary,
        "group_row_count": len(g),
        "delta_wer_reported": delta_wer,
        "worst_group_wer_reported": worst_group_wer,
        "macro_avg_wer_reported": macro_avg_wer,
        "micro_avg_wer_reported": micro_avg_wer,
        "micro_avg_notes": micro_notes,
    })

group_metrics = pd.DataFrame(metric_rows)

# -----------------------------
# Merge computed metrics back onto row-level dataset
# -----------------------------
merged_df = df.merge(
    group_metrics,
    on=[
        "asr_model_name_standardized",
        "benchmark_name_standardized",
        "transcript_type_binary",
    ],
    how="left",
    suffixes=("", "_computed")
)

# If original columns exist and are empty, fill them with computed values
for metric in [
    "delta_wer_reported",
    "worst_group_wer_reported",
    "macro_avg_wer_reported",
    "micro_avg_wer_reported",
]:
    computed_col = f"{metric}_computed"
    if computed_col in merged_df.columns:
        if metric in merged_df.columns:
            merged_df[metric] = merged_df[metric].combine_first(merged_df[computed_col])
            merged_df.drop(columns=[computed_col], inplace=True)
        else:
            merged_df.rename(columns={computed_col: metric}, inplace=True)

# -----------------------------
# Save outputs
# -----------------------------
group_metrics.to_csv(OUT_DIR / "computed_fairness_metrics_by_group.csv", index=False)
merged_df.to_csv(OUT_DIR / "master_data_with_computed_fairness_metrics.csv", index=False)

excel_path = OUT_DIR / "computed_fairness_metrics.xlsx"
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    group_metrics.to_excel(writer, sheet_name="group_metrics", index=False)
    merged_df.to_excel(writer, sheet_name="master_with_metrics", index=False)

# -----------------------------
# Quick audit table
# -----------------------------
audit_rows = []
for metric in [
    "delta_wer_reported",
    "worst_group_wer_reported",
    "macro_avg_wer_reported",
    "micro_avg_wer_reported",
]:
    audit_rows.append({
        "metric": metric,
        "non_missing_rows": int(merged_df[metric].notna().sum()) if metric in merged_df.columns else 0
    })

audit_df = pd.DataFrame(audit_rows)
audit_df.to_csv(OUT_DIR / "computed_fairness_metric_availability.csv", index=False)

print("\n=== COMPUTED FAIRNESS METRICS COMPLETE ===\n")
print("Grouping key:", group_cols)
print("Weight column used for micro average:", weight_col if weight_col is not None else "NONE")
print("\nMetric availability after computation:")
print(audit_df.to_string(index=False))

print("\nSaved files:")
print("-", OUT_DIR / "computed_fairness_metrics_by_group.csv")
print("-", OUT_DIR / "master_data_with_computed_fairness_metrics.csv")
print("-", OUT_DIR / "computed_fairness_metric_availability.csv")
print("-", excel_path)