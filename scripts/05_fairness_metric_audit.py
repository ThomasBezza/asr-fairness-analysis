from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "raw" / "ASR_Fairness_Master_Spreadsheet_v5_descriptives.xlsx"
OUT_DIR = ROOT / "outputs" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(DATA_FILE, sheet_name="master_data")

# Standardize grouping variables
df["resource_level_tertile"] = df["resource_level_tertile"].astype(str).str.strip().str.title()
df["transcript_type_binary"] = df["transcript_type_binary"].astype(str).str.strip()

# Eligible sample
eligible_df = df.copy()
eligible_df = eligible_df[eligible_df["eligible_for_main_analysis"] == True].copy()
eligible_df = eligible_df[eligible_df["resource_level_tertile"].isin(["Low", "Medium", "High"])].copy()
eligible_df = eligible_df[eligible_df["transcript_type_binary"].isin(["standard_reference", "curated_reference"])].copy()

metrics = [
    "wer_reported",
    "cer_reported",
    "delta_wer_reported",
    "worst_group_wer_reported",
    "macro_avg_wer_reported",
    "micro_avg_wer_reported",
]

rows = []
for metric in metrics:
    rows.append({
        "metric": metric,
        "non_missing_all_rows": int(df[metric].notna().sum()) if metric in df.columns else 0,
        "non_missing_eligible_rows": int(eligible_df[metric].notna().sum()) if metric in eligible_df.columns else 0,
    })

availability = pd.DataFrame(rows)

# By resource level
resource_tables = []
for metric in metrics:
    if metric not in eligible_df.columns:
        continue
    tmp = eligible_df[eligible_df[metric].notna()].copy()
    if len(tmp) == 0:
        continue
    summary = (
        tmp.groupby("resource_level_tertile")[metric]
        .agg(sample_size="count", mean="mean", median="median")
        .reset_index()
    )
    summary.insert(0, "metric", metric)
    resource_tables.append(summary)

resource_summary = pd.concat(resource_tables, ignore_index=True) if resource_tables else pd.DataFrame()

# By transcript/reference condition
reference_tables = []
for metric in metrics:
    if metric not in eligible_df.columns:
        continue
    tmp = eligible_df[eligible_df[metric].notna()].copy()
    if len(tmp) == 0:
        continue
    summary = (
        tmp.groupby("transcript_type_binary")[metric]
        .agg(sample_size="count", mean="mean", median="median")
        .reset_index()
    )
    summary.insert(0, "metric", metric)
    reference_tables.append(summary)

reference_summary = pd.concat(reference_tables, ignore_index=True) if reference_tables else pd.DataFrame()

# Save
availability.to_csv(OUT_DIR / "fairness_metric_availability.csv", index=False)
resource_summary.to_csv(OUT_DIR / "fairness_metrics_by_resource_level.csv", index=False)
reference_summary.to_csv(OUT_DIR / "fairness_metrics_by_reference_condition.csv", index=False)

with pd.ExcelWriter(OUT_DIR / "fairness_metric_audit.xlsx", engine="openpyxl") as writer:
    availability.to_excel(writer, sheet_name="availability", index=False)
    resource_summary.to_excel(writer, sheet_name="by_resource_level", index=False)
    reference_summary.to_excel(writer, sheet_name="by_reference_condition", index=False)

print("\n=== FAIRNESS METRIC AUDIT ===\n")
print(availability.to_string(index=False))
print("\nSaved files to:", OUT_DIR)