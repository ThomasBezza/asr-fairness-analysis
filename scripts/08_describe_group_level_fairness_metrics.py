from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DELTA_FILE = OUT_DIR / "group_level_delta_wer_sample.csv"
WORST_FILE = OUT_DIR / "group_level_worst_group_wer_sample.csv"
MACRO_FILE = OUT_DIR / "group_level_macro_avg_wer_sample.csv"

# -----------------------------
# Load files
# -----------------------------
delta_df = pd.read_csv(DELTA_FILE)
worst_df = pd.read_csv(WORST_FILE)
macro_df = pd.read_csv(MACRO_FILE)

# -----------------------------
# Basic cleaning
# -----------------------------
for df in [delta_df, worst_df, macro_df]:
    if "resource_level_tertile" in df.columns:
        df["resource_level_tertile"] = df["resource_level_tertile"].astype(str).str.strip().str.title()
    if "transcript_type_binary" in df.columns:
        df["transcript_type_binary"] = df["transcript_type_binary"].astype(str).str.strip()

# -----------------------------
# Helper functions
# -----------------------------
def overall_summary(df: pd.DataFrame, metric_col: str, metric_name: str) -> pd.DataFrame:
    x = pd.to_numeric(df[metric_col], errors="coerce").dropna()
    return pd.DataFrame([{
        "metric": metric_name,
        "sample_size": len(x),
        "mean": x.mean() if len(x) > 0 else None,
        "median": x.median() if len(x) > 0 else None,
        "min": x.min() if len(x) > 0 else None,
        "max": x.max() if len(x) > 0 else None,
    }])

def grouped_summary(df: pd.DataFrame, metric_col: str, group_col: str, metric_name: str) -> pd.DataFrame:
    tmp = df.copy()
    tmp[metric_col] = pd.to_numeric(tmp[metric_col], errors="coerce")
    tmp = tmp[tmp[metric_col].notna()].copy()

    if len(tmp) == 0 or group_col not in tmp.columns:
        return pd.DataFrame(columns=["metric", group_col, "sample_size", "mean", "median", "min", "max"])

    out = (
        tmp.groupby(group_col)[metric_col]
        .agg(sample_size="count", mean="mean", median="median", min="min", max="max")
        .reset_index()
    )
    out.insert(0, "metric", metric_name)
    return out

# -----------------------------
# Overall summaries
# -----------------------------
overall_tables = [
    overall_summary(delta_df, "delta_wer_reported", "delta_wer_reported"),
    overall_summary(worst_df, "worst_group_wer_reported", "worst_group_wer_reported"),
    overall_summary(macro_df, "macro_avg_wer_reported", "macro_avg_wer_reported"),
]
overall_summary_df = pd.concat(overall_tables, ignore_index=True)

# -----------------------------
# By transcript/reference condition
# -----------------------------
reference_tables = [
    grouped_summary(delta_df, "delta_wer_reported", "transcript_type_binary", "delta_wer_reported"),
    grouped_summary(worst_df, "worst_group_wer_reported", "transcript_type_binary", "worst_group_wer_reported"),
    grouped_summary(macro_df, "macro_avg_wer_reported", "transcript_type_binary", "macro_avg_wer_reported"),
]
reference_summary_df = pd.concat(reference_tables, ignore_index=True)

# -----------------------------
# By resource level
# Only keep summaries when there is actual variation
# -----------------------------
resource_tables = []

for data, metric_col, metric_name in [
    (delta_df, "delta_wer_reported", "delta_wer_reported"),
    (worst_df, "worst_group_wer_reported", "worst_group_wer_reported"),
    (macro_df, "macro_avg_wer_reported", "macro_avg_wer_reported"),
]:
    if "resource_level_tertile" in data.columns:
        n_unique = data["resource_level_tertile"].dropna().nunique()
        if n_unique >= 2:
            resource_tables.append(
                grouped_summary(data, metric_col, "resource_level_tertile", metric_name)
            )

if resource_tables:
    resource_summary_df = pd.concat(resource_tables, ignore_index=True)
else:
    resource_summary_df = pd.DataFrame(
        columns=["metric", "resource_level_tertile", "sample_size", "mean", "median", "min", "max"]
    )

# -----------------------------
# Small audit table
# -----------------------------
audit_rows = []
for metric_name, df_obj in [
    ("delta_wer_reported", delta_df),
    ("worst_group_wer_reported", worst_df),
    ("macro_avg_wer_reported", macro_df),
]:
    audit_rows.append({
        "metric": metric_name,
        "n_groups": len(df_obj),
        "n_unique_resource_levels": df_obj["resource_level_tertile"].dropna().nunique() if "resource_level_tertile" in df_obj.columns else 0,
        "n_unique_reference_conditions": df_obj["transcript_type_binary"].dropna().nunique() if "transcript_type_binary" in df_obj.columns else 0,
    })

audit_df = pd.DataFrame(audit_rows)

# -----------------------------
# Save outputs
# -----------------------------
overall_summary_df.to_csv(OUT_DIR / "group_level_fairness_overall_summary.csv", index=False)
reference_summary_df.to_csv(OUT_DIR / "group_level_fairness_by_reference_condition.csv", index=False)
resource_summary_df.to_csv(OUT_DIR / "group_level_fairness_by_resource_level.csv", index=False)
audit_df.to_csv(OUT_DIR / "group_level_fairness_descriptive_audit.csv", index=False)

excel_path = OUT_DIR / "group_level_fairness_descriptives.xlsx"
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    overall_summary_df.to_excel(writer, sheet_name="overall_summary", index=False)
    reference_summary_df.to_excel(writer, sheet_name="by_reference_condition", index=False)
    resource_summary_df.to_excel(writer, sheet_name="by_resource_level", index=False)
    audit_df.to_excel(writer, sheet_name="audit", index=False)

# -----------------------------
# Print terminal summary
# -----------------------------
print("\n=== GROUP-LEVEL FAIRNESS DESCRIPTIVES COMPLETE ===\n")
print("Overall summary:")
print(overall_summary_df.to_string(index=False))

print("\nAudit:")
print(audit_df.to_string(index=False))

print("\nSaved files:")
print("-", OUT_DIR / "group_level_fairness_overall_summary.csv")
print("-", OUT_DIR / "group_level_fairness_by_reference_condition.csv")
print("-", OUT_DIR / "group_level_fairness_by_resource_level.csv")
print("-", OUT_DIR / "group_level_fairness_descriptive_audit.csv")
print("-", excel_path)