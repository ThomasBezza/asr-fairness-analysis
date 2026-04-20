from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "ASR_Fairness_Master_Spreadsheet_v5_descriptives.xlsx"
GROUP_FILE = ROOT / "outputs" / "tables" / "computed_fairness_metrics_by_group.csv"
OUT_DIR = ROOT / "outputs" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load files
# -----------------------------
raw_df = pd.read_excel(RAW_FILE, sheet_name="master_data").copy()
group_df = pd.read_csv(GROUP_FILE).copy()

# -----------------------------
# Clean key columns
# -----------------------------
for df in [raw_df, group_df]:
    df["asr_model_name_standardized"] = df["asr_model_name_standardized"].astype(str).str.strip()
    df["benchmark_name_standardized"] = df["benchmark_name_standardized"].astype(str).str.strip()
    df["transcript_type_binary"] = df["transcript_type_binary"].astype(str).str.strip()

if "resource_level_tertile" in raw_df.columns:
    raw_df["resource_level_tertile"] = raw_df["resource_level_tertile"].astype(str).str.strip().str.title()

# -----------------------------
# Create one resource-level lookup per group
# -----------------------------
lookup_cols = [
    "asr_model_name_standardized",
    "benchmark_name_standardized",
    "transcript_type_binary",
    "resource_level_tertile",
]

resource_lookup = raw_df[lookup_cols].drop_duplicates().copy()

# If duplicates still exist because rows in same group have mixed resource levels,
# keep only groups where the resource level is uniquely defined.
resource_counts = (
    resource_lookup.groupby(
        ["asr_model_name_standardized", "benchmark_name_standardized", "transcript_type_binary"]
    )["resource_level_tertile"]
    .nunique()
    .reset_index(name="n_resource_levels")
)

resource_lookup = resource_lookup.merge(
    resource_counts,
    on=["asr_model_name_standardized", "benchmark_name_standardized", "transcript_type_binary"],
    how="left"
)

resource_lookup = resource_lookup[resource_lookup["n_resource_levels"] == 1].copy()
resource_lookup = resource_lookup.drop(columns=["n_resource_levels"]).drop_duplicates()

# -----------------------------
# Merge group fairness metrics with resource level
# -----------------------------
fairness_df = group_df.merge(
    resource_lookup,
    on=["asr_model_name_standardized", "benchmark_name_standardized", "transcript_type_binary"],
    how="left"
)

# Keep clean categories
fairness_df = fairness_df[fairness_df["resource_level_tertile"].isin(["Low", "Medium", "High"])].copy()
fairness_df = fairness_df[fairness_df["transcript_type_binary"].isin(["standard_reference", "curated_reference"])].copy()

# -----------------------------
# Build metric-specific analysis subsets
# -----------------------------
delta_df = fairness_df[
    fairness_df["delta_wer_reported"].notna() &
    (fairness_df["group_row_count"] >= 2)
].copy()

worst_df = fairness_df[
    fairness_df["worst_group_wer_reported"].notna()
].copy()

macro_df = fairness_df[
    fairness_df["macro_avg_wer_reported"].notna()
].copy()

# -----------------------------
# Sample-size audit
# -----------------------------
audit = pd.DataFrame([
    {"metric": "delta_wer_reported", "n_groups": len(delta_df)},
    {"metric": "worst_group_wer_reported", "n_groups": len(worst_df)},
    {"metric": "macro_avg_wer_reported", "n_groups": len(macro_df)},
])

# -----------------------------
# Save outputs
# -----------------------------
fairness_df.to_csv(OUT_DIR / "group_level_fairness_master.csv", index=False)
delta_df.to_csv(OUT_DIR / "group_level_delta_wer_sample.csv", index=False)
worst_df.to_csv(OUT_DIR / "group_level_worst_group_wer_sample.csv", index=False)
macro_df.to_csv(OUT_DIR / "group_level_macro_avg_wer_sample.csv", index=False)
audit.to_csv(OUT_DIR / "group_level_fairness_sample_sizes.csv", index=False)

with pd.ExcelWriter(OUT_DIR / "group_level_fairness_data.xlsx", engine="openpyxl") as writer:
    fairness_df.to_excel(writer, sheet_name="master", index=False)
    delta_df.to_excel(writer, sheet_name="delta_wer", index=False)
    worst_df.to_excel(writer, sheet_name="worst_group_wer", index=False)
    macro_df.to_excel(writer, sheet_name="macro_avg_wer", index=False)
    audit.to_excel(writer, sheet_name="sample_sizes", index=False)

print("\n=== GROUP-LEVEL FAIRNESS DATA READY ===\n")
print(audit.to_string(index=False))
print("\nSaved files:")
print("-", OUT_DIR / "group_level_fairness_master.csv")
print("-", OUT_DIR / "group_level_delta_wer_sample.csv")
print("-", OUT_DIR / "group_level_worst_group_wer_sample.csv")
print("-", OUT_DIR / "group_level_macro_avg_wer_sample.csv")
print("-", OUT_DIR / "group_level_fairness_sample_sizes.csv")
print("-", OUT_DIR / "group_level_fairness_data.xlsx")