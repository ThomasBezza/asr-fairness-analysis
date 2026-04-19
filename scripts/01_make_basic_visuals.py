from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "raw" / "ASR_Fairness_Master_Spreadsheet_v5_descriptives.xlsx"
FIG_DIR = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load the master sheet
df = pd.read_excel(DATA_FILE, sheet_name="master_data")

# Keep only rows eligible for main analysis and with WER present
df = df[df["eligible_for_main_analysis"] == True].copy()
df = df[df["wer_reported"].notna()].copy()

# Standardize labels for plotting
df["resource_level_tertile"] = df["resource_level_tertile"].astype(str).str.strip().str.title()
df["transcript_type_binary"] = df["transcript_type_binary"].astype(str).str.strip()

resource_order = ["Low", "Medium", "High"]
reference_order = ["standard_reference", "curated_reference"]

# Plot 1: WER by resource level
plot_df = df[df["resource_level_tertile"].isin(resource_order)].copy()

plt.figure(figsize=(8, 6))
groups = [plot_df.loc[plot_df["resource_level_tertile"] == lvl, "wer_reported"] for lvl in resource_order]
plt.boxplot(groups, labels=resource_order)
plt.title("WER by Resource Level")
plt.xlabel("Resource Level")
plt.ylabel("WER")
plt.tight_layout()
plt.savefig(FIG_DIR / "wer_by_resource_level_boxplot.png", dpi=300)
plt.close()

# Plot 2: WER by transcript/reference condition
plot_df = df[df["transcript_type_binary"].isin(reference_order)].copy()

plt.figure(figsize=(8, 6))
groups = [plot_df.loc[plot_df["transcript_type_binary"] == cond, "wer_reported"] for cond in reference_order]
plt.boxplot(groups, labels=reference_order)
plt.title("WER by Transcript/Reference Condition")
plt.xlabel("Transcript/Reference Condition")
plt.ylabel("WER")
plt.tight_layout()
plt.savefig(FIG_DIR / "wer_by_reference_condition_boxplot.png", dpi=300)
plt.close()

# Plot 3: Training hours vs WER
plot_df = df[df["training_hours_numeric"].notna()].copy()

plt.figure(figsize=(8, 6))
plt.scatter(plot_df["training_hours_numeric"], plot_df["wer_reported"])
plt.title("Training Hours vs WER")
plt.xlabel("Training Hours")
plt.ylabel("WER")
plt.tight_layout()
plt.savefig(FIG_DIR / "training_hours_vs_wer_scatter.png", dpi=300)
plt.close()

print("Saved figures to:", FIG_DIR)
print("Rows used for WER analysis:", len(df))
print("Rows used for scatterplot:", len(plot_df))