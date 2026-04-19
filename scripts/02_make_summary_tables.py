from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "raw" / "ASR_Fairness_Master_Spreadsheet_v5_descriptives.xlsx"
OUT_DIR = ROOT / "outputs" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_excel(DATA_FILE, sheet_name="master_data")

# Keep only rows eligible for main analysis and with WER present
df = df[df["eligible_for_main_analysis"] == True].copy()
df = df[df["wer_reported"].notna()].copy()

# Standardize labels for grouping
df["resource_level_tertile"] = df["resource_level_tertile"].astype(str).str.strip().str.title()
df["transcript_type_binary"] = df["transcript_type_binary"].astype(str).str.strip()

# -----------------------------
# Table 1: WER by resource level
# -----------------------------
resource_order = ["Low", "Medium", "High"]
resource_df = df[df["resource_level_tertile"].isin(resource_order)].copy()

resource_summary = (
    resource_df.groupby("resource_level_tertile", dropna=False)["wer_reported"]
    .agg(sample_size="count", mean_wer="mean", median_wer="median")
    .reset_index()
)

resource_summary["resource_level_tertile"] = pd.Categorical(
    resource_summary["resource_level_tertile"],
    categories=resource_order,
    ordered=True
)
resource_summary = resource_summary.sort_values("resource_level_tertile")

# -----------------------------
# Table 2: WER by transcript/reference condition
# -----------------------------
reference_order = ["standard_reference", "curated_reference"]
reference_df = df[df["transcript_type_binary"].isin(reference_order)].copy()

reference_summary = (
    reference_df.groupby("transcript_type_binary", dropna=False)["wer_reported"]
    .agg(sample_size="count", mean_wer="mean", median_wer="median")
    .reset_index()
)

reference_summary["transcript_type_binary"] = pd.Categorical(
    reference_summary["transcript_type_binary"],
    categories=reference_order,
    ordered=True
)
reference_summary = reference_summary.sort_values("transcript_type_binary")

# -----------------------------
# Table 3: Training hours vs WER scatterplot sample summary
# -----------------------------
scatter_df = df[df["training_hours_numeric"].notna()].copy()

scatter_summary = pd.DataFrame({
    "rows_used_for_scatterplot": [len(scatter_df)],
    "mean_training_hours": [scatter_df["training_hours_numeric"].mean()],
    "median_training_hours": [scatter_df["training_hours_numeric"].median()],
    "mean_wer": [scatter_df["wer_reported"].mean()],
    "median_wer": [scatter_df["wer_reported"].median()],
})

# -----------------------------
# Save CSV files
# -----------------------------
resource_summary.to_csv(OUT_DIR / "wer_by_resource_level_summary.csv", index=False)
reference_summary.to_csv(OUT_DIR / "wer_by_reference_condition_summary.csv", index=False)
scatter_summary.to_csv(OUT_DIR / "training_hours_vs_wer_scatter_summary.csv", index=False)

# -----------------------------
# Save combined Excel workbook
# -----------------------------
excel_path = OUT_DIR / "basic_visuals_summary_tables.xlsx"
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    resource_summary.to_excel(writer, sheet_name="wer_by_resource_level", index=False)
    reference_summary.to_excel(writer, sheet_name="wer_by_reference_condition", index=False)
    scatter_summary.to_excel(writer, sheet_name="training_hours_vs_wer", index=False)

print("Saved summary tables to:", OUT_DIR)
print("Created files:")
print("-", OUT_DIR / "wer_by_resource_level_summary.csv")
print("-", OUT_DIR / "wer_by_reference_condition_summary.csv")
print("-", OUT_DIR / "training_hours_vs_wer_scatter_summary.csv")
print("-", excel_path)