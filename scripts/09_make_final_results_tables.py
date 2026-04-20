from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "tables"

# -----------------------------
# Input files
# -----------------------------
REG_FIT_FILE = OUT_DIR / "basic_regression_model_fit.csv"
REG_COEF_FILE = OUT_DIR / "basic_regression_coefficients.csv"
GROUP_FAIRNESS_OVERALL_FILE = OUT_DIR / "group_level_fairness_overall_summary.csv"
GROUP_FAIRNESS_AUDIT_FILE = OUT_DIR / "group_level_fairness_descriptive_audit.csv"
HEALTH_SAMPLE_FILE = OUT_DIR / "regression_health_sample_size.csv"

# -----------------------------
# Load files
# -----------------------------
reg_fit = pd.read_csv(REG_FIT_FILE)
reg_coef = pd.read_csv(REG_COEF_FILE)
fairness_overall = pd.read_csv(GROUP_FAIRNESS_OVERALL_FILE)
fairness_audit = pd.read_csv(GROUP_FAIRNESS_AUDIT_FILE)
health_sample = pd.read_csv(HEALTH_SAMPLE_FILE)

# -----------------------------
# Clean regression model fit table
# -----------------------------
reg_fit_clean = reg_fit.copy()
reg_fit_clean = reg_fit_clean.rename(columns={
    "model": "Model",
    "n": "N",
    "r_squared": "R_squared",
    "adj_r_squared": "Adj_R_squared",
    "f_statistic": "F_statistic",
    "f_p_value": "Model_p_value",
    "aic": "AIC",
    "bic": "BIC",
})

# -----------------------------
# Clean regression coefficient table
# -----------------------------
reg_coef_clean = reg_coef.copy()
reg_coef_clean = reg_coef_clean.rename(columns={
    "model": "Model",
    "term": "Term",
    "coef": "Coefficient",
    "std_err": "Std_Error",
    "t_value": "t_value",
    "p_value": "p_value",
    "ci_lower": "CI_lower",
    "ci_upper": "CI_upper",
})

# Optional: drop intercept into separate emphasis if wanted
reg_coef_non_intercept = reg_coef_clean[reg_coef_clean["Term"] != "Intercept"].copy()

# -----------------------------
# Clean fairness summary tables
# -----------------------------
fairness_overall_clean = fairness_overall.copy()
fairness_overall_clean = fairness_overall_clean.rename(columns={
    "metric": "Metric",
    "sample_size": "N_groups",
    "mean": "Mean",
    "median": "Median",
    "min": "Min",
    "max": "Max",
})

fairness_audit_clean = fairness_audit.copy()
fairness_audit_clean = fairness_audit_clean.rename(columns={
    "metric": "Metric",
    "n_groups": "N_groups",
    "n_unique_resource_levels": "Unique_resource_levels",
    "n_unique_reference_conditions": "Unique_reference_conditions",
})

# -----------------------------
# Build one compact sample summary
# -----------------------------
health_sample_clean = health_sample.copy()
health_sample_clean = health_sample_clean.rename(columns={
    "metric": "Sample_metric",
    "value": "Value",
})

# Pivot the regression health sample file into one row if possible
sample_summary = health_sample_clean.copy()

# -----------------------------
# Build one narrative-ready summary table
# -----------------------------
summary_rows = []

# Main regression result anchor = Model 2
model2 = reg_fit_clean[reg_fit_clean["Model"] == "Model 2: WER ~ resource level + transcript/reference condition"]
if not model2.empty:
    row = model2.iloc[0]
    summary_rows.append({
        "Section": "Main inferential result",
        "Result": "WER regression (Model 2)",
        "N_or_groups": row["N"],
        "Key_statistic": f"R² = {row['R_squared']:.3f}",
        "p_value_or_note": f"Model p = {row['Model_p_value']:.3f}",
        "Interpretation_note": "No statistically significant overall model detected."
    })

# Exploratory interaction = Model 3
model3 = reg_fit_clean[reg_fit_clean["Model"] == "Model 3: WER ~ resource level * transcript/reference condition"]
if not model3.empty:
    row = model3.iloc[0]
    summary_rows.append({
        "Section": "Exploratory inferential result",
        "Result": "WER interaction model (Model 3)",
        "N_or_groups": row["N"],
        "Key_statistic": f"R² = {row['R_squared']:.3f}",
        "p_value_or_note": f"Model p = {row['Model_p_value']:.3f}",
        "Interpretation_note": "Exploratory only; interaction sample is sparse."
    })

# Fairness descriptives
for _, row in fairness_overall_clean.iterrows():
    summary_rows.append({
        "Section": "Supplemental fairness descriptives",
        "Result": row["Metric"],
        "N_or_groups": row["N_groups"],
        "Key_statistic": f"Mean = {row['Mean']:.3f}; Median = {row['Median']:.3f}",
        "p_value_or_note": "Descriptive only",
        "Interpretation_note": f"Range: {row['Min']:.3f} to {row['Max']:.3f}"
    })

final_summary_table = pd.DataFrame(summary_rows)

# -----------------------------
# Save outputs
# -----------------------------
reg_fit_clean.to_csv(OUT_DIR / "final_results_model_fit_table.csv", index=False)
reg_coef_clean.to_csv(OUT_DIR / "final_results_coefficients_table.csv", index=False)
reg_coef_non_intercept.to_csv(OUT_DIR / "final_results_coefficients_no_intercept.csv", index=False)
fairness_overall_clean.to_csv(OUT_DIR / "final_results_fairness_overall_table.csv", index=False)
fairness_audit_clean.to_csv(OUT_DIR / "final_results_fairness_audit_table.csv", index=False)
final_summary_table.to_csv(OUT_DIR / "final_results_summary_table.csv", index=False)

excel_path = OUT_DIR / "final_results_tables.xlsx"
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    sample_summary.to_excel(writer, sheet_name="sample_summary", index=False)
    reg_fit_clean.to_excel(writer, sheet_name="regression_model_fit", index=False)
    reg_coef_clean.to_excel(writer, sheet_name="regression_coefficients", index=False)
    reg_coef_non_intercept.to_excel(writer, sheet_name="coefficients_no_intercept", index=False)
    fairness_overall_clean.to_excel(writer, sheet_name="fairness_overall", index=False)
    fairness_audit_clean.to_excel(writer, sheet_name="fairness_audit", index=False)
    final_summary_table.to_excel(writer, sheet_name="final_summary", index=False)

print("\n=== FINAL RESULTS TABLES COMPLETE ===\n")
print("Created files:")
print("-", OUT_DIR / "final_results_model_fit_table.csv")
print("-", OUT_DIR / "final_results_coefficients_table.csv")
print("-", OUT_DIR / "final_results_coefficients_no_intercept.csv")
print("-", OUT_DIR / "final_results_fairness_overall_table.csv")
print("-", OUT_DIR / "final_results_fairness_audit_table.csv")
print("-", OUT_DIR / "final_results_summary_table.csv")
print("-", excel_path)

print("\nFinal summary preview:")
print(final_summary_table.to_string(index=False))