from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "raw" / "ASR_Fairness_Master_Spreadsheet_v5_descriptives.xlsx"
OUT_DIR = ROOT / "outputs" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
df = pd.read_excel(DATA_FILE, sheet_name="master_data")

# -----------------------------
# Basic cleaning
# -----------------------------
df["resource_level_tertile"] = df["resource_level_tertile"].astype(str).str.strip().str.title()
df["transcript_type_binary"] = df["transcript_type_binary"].astype(str).str.strip()

# -----------------------------
# Keep only rows eligible for regression
# -----------------------------
reg_df = df.copy()
reg_df = reg_df[reg_df["eligible_for_main_analysis"] == True].copy()
reg_df = reg_df[reg_df["wer_reported"].notna()].copy()
reg_df = reg_df[reg_df["resource_level_tertile"].isin(["Low", "Medium", "High"])].copy()
reg_df = reg_df[reg_df["transcript_type_binary"].isin(["standard_reference", "curated_reference"])].copy()

# -----------------------------
# Table 1: overall regression sample size
# -----------------------------
sample_size_table = pd.DataFrame({
    "metric": [
        "rows_in_master_data",
        "eligible_for_main_analysis",
        "eligible_with_wer",
        "final_regression_sample"
    ],
    "value": [
        len(df),
        int((df["eligible_for_main_analysis"] == True).sum()),
        int(((df["eligible_for_main_analysis"] == True) & (df["wer_reported"].notna())).sum()),
        len(reg_df)
    ]
})

# -----------------------------
# Table 2: counts by resource level
# -----------------------------
resource_counts = (
    reg_df["resource_level_tertile"]
    .value_counts()
    .rename_axis("resource_level_tertile")
    .reset_index(name="count")
)

resource_counts["resource_level_tertile"] = pd.Categorical(
    resource_counts["resource_level_tertile"],
    categories=["Low", "Medium", "High"],
    ordered=True
)
resource_counts = resource_counts.sort_values("resource_level_tertile")

# -----------------------------
# Table 3: counts by transcript/reference condition
# -----------------------------
reference_counts = (
    reg_df["transcript_type_binary"]
    .value_counts()
    .rename_axis("transcript_type_binary")
    .reset_index(name="count")
)

reference_counts["transcript_type_binary"] = pd.Categorical(
    reference_counts["transcript_type_binary"],
    categories=["standard_reference", "curated_reference"],
    ordered=True
)
reference_counts = reference_counts.sort_values("transcript_type_binary")

# -----------------------------
# Table 4: interaction cell counts
# -----------------------------
interaction_counts = pd.crosstab(
    reg_df["resource_level_tertile"],
    reg_df["transcript_type_binary"],
    dropna=False
)

interaction_counts = interaction_counts.reindex(
    index=["Low", "Medium", "High"],
    columns=["standard_reference", "curated_reference"],
    fill_value=0
)

interaction_counts_reset = interaction_counts.reset_index()

# -----------------------------
# Flag tiny cells
# -----------------------------
tiny_cells = interaction_counts.stack().reset_index()
tiny_cells.columns = ["resource_level_tertile", "transcript_type_binary", "count"]
tiny_cells["cell_flag"] = tiny_cells["count"].apply(
    lambda x: "EMPTY" if x == 0 else ("TINY(<5)" if x < 5 else "OK")
)

# -----------------------------
# Print results to terminal
# -----------------------------
print("\n=== REGRESSION HEALTH CHECK ===\n")

print("1. Sample size table")
print(sample_size_table.to_string(index=False))

print("\n2. Counts by resource level")
print(resource_counts.to_string(index=False))

print("\n3. Counts by transcript/reference condition")
print(reference_counts.to_string(index=False))

print("\n4. Interaction cell counts")
print(interaction_counts)

print("\n5. Interaction cell flags")
print(tiny_cells.to_string(index=False))

# -----------------------------
# Save outputs
# -----------------------------
sample_size_table.to_csv(OUT_DIR / "regression_health_sample_size.csv", index=False)
resource_counts.to_csv(OUT_DIR / "regression_health_resource_counts.csv", index=False)
reference_counts.to_csv(OUT_DIR / "regression_health_reference_counts.csv", index=False)
interaction_counts_reset.to_csv(OUT_DIR / "regression_health_interaction_counts.csv", index=False)
tiny_cells.to_csv(OUT_DIR / "regression_health_interaction_flags.csv", index=False)

with pd.ExcelWriter(OUT_DIR / "regression_health_check.xlsx", engine="openpyxl") as writer:
    sample_size_table.to_excel(writer, sheet_name="sample_size", index=False)
    resource_counts.to_excel(writer, sheet_name="resource_counts", index=False)
    reference_counts.to_excel(writer, sheet_name="reference_counts", index=False)
    interaction_counts_reset.to_excel(writer, sheet_name="interaction_counts", index=False)
    tiny_cells.to_excel(writer, sheet_name="interaction_flags", index=False)

print("\nSaved health check tables to:", OUT_DIR)