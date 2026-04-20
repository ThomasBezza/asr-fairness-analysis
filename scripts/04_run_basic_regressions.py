from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "raw" / "ASR_Fairness_Master_Spreadsheet_v5_descriptives.xlsx"
OUT_DIR = ROOT / "outputs" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
df = pd.read_excel(DATA_FILE, sheet_name="master_data")

# -----------------------------
# Clean and filter to regression sample
# -----------------------------
df["resource_level_tertile"] = df["resource_level_tertile"].astype(str).str.strip().str.title()
df["transcript_type_binary"] = df["transcript_type_binary"].astype(str).str.strip()

reg_df = df.copy()
reg_df = reg_df[reg_df["eligible_for_main_analysis"] == True].copy()
reg_df = reg_df[reg_df["wer_reported"].notna()].copy()
reg_df = reg_df[reg_df["resource_level_tertile"].isin(["Low", "Medium", "High"])].copy()
reg_df = reg_df[reg_df["transcript_type_binary"].isin(["standard_reference", "curated_reference"])].copy()

# -----------------------------
# Set categorical reference levels
# -----------------------------
reg_df["resource_level_tertile"] = pd.Categorical(
    reg_df["resource_level_tertile"],
    categories=["Low", "Medium", "High"],
    ordered=True
)

reg_df["transcript_type_binary"] = pd.Categorical(
    reg_df["transcript_type_binary"],
    categories=["standard_reference", "curated_reference"],
    ordered=True
)

# -----------------------------
# Run models
# -----------------------------
model_1 = smf.ols(
    formula="wer_reported ~ C(resource_level_tertile, Treatment(reference='Low'))",
    data=reg_df
).fit()

model_2 = smf.ols(
    formula="wer_reported ~ C(resource_level_tertile, Treatment(reference='Low')) + C(transcript_type_binary, Treatment(reference='standard_reference'))",
    data=reg_df
).fit()

model_3 = smf.ols(
    formula="wer_reported ~ C(resource_level_tertile, Treatment(reference='Low')) * C(transcript_type_binary, Treatment(reference='standard_reference'))",
    data=reg_df
).fit()

models = {
    "Model 1: WER ~ resource level": model_1,
    "Model 2: WER ~ resource level + transcript/reference condition": model_2,
    "Model 3: WER ~ resource level * transcript/reference condition": model_3,
}

# -----------------------------
# Save full text summaries
# -----------------------------
for model_name, model in models.items():
    safe_name = (
        model_name.lower()
        .replace(" ", "_")
        .replace("~", "")
        .replace("/", "_")
        .replace("*", "interaction")
        .replace("+", "plus")
        .replace(":", "")
    )
    summary_path = OUT_DIR / f"{safe_name}_summary.txt"
    with open(summary_path, "w") as f:
        f.write(model.summary().as_text())

# -----------------------------
# Build coefficients table
# -----------------------------
coef_rows = []

for model_name, model in models.items():
    conf = model.conf_int()
    for term in model.params.index:
        coef_rows.append({
            "model": model_name,
            "term": term,
            "coef": model.params[term],
            "std_err": model.bse[term],
            "t_value": model.tvalues[term],
            "p_value": model.pvalues[term],
            "ci_lower": conf.loc[term, 0],
            "ci_upper": conf.loc[term, 1],
        })

coef_table = pd.DataFrame(coef_rows)

# -----------------------------
# Build model fit table
# -----------------------------
fit_rows = []

for model_name, model in models.items():
    fit_rows.append({
        "model": model_name,
        "n": int(model.nobs),
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "f_statistic": model.fvalue,
        "f_p_value": model.f_pvalue,
        "aic": model.aic,
        "bic": model.bic,
    })

fit_table = pd.DataFrame(fit_rows)

# -----------------------------
# Save CSV outputs
# -----------------------------
coef_table.to_csv(OUT_DIR / "basic_regression_coefficients.csv", index=False)
fit_table.to_csv(OUT_DIR / "basic_regression_model_fit.csv", index=False)

# -----------------------------
# Save Excel workbook
# -----------------------------
excel_path = OUT_DIR / "basic_regression_results.xlsx"
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    coef_table.to_excel(writer, sheet_name="coefficients", index=False)
    fit_table.to_excel(writer, sheet_name="model_fit", index=False)
    reg_df.to_excel(writer, sheet_name="regression_sample", index=False)

# -----------------------------
# Print quick terminal summary
# -----------------------------
print("\n=== BASIC REGRESSIONS COMPLETE ===\n")
print("Regression sample size:", len(reg_df))

print("\nModel fit summary:")
print(fit_table.to_string(index=False))

print("\nSaved files:")
print("-", OUT_DIR / "basic_regression_coefficients.csv")
print("-", OUT_DIR / "basic_regression_model_fit.csv")
print("-", excel_path)

print("\nSaved text summaries:")
for model_name in models.keys():
    safe_name = (
        model_name.lower()
        .replace(" ", "_")
        .replace("~", "")
        .replace("/", "_")
        .replace("*", "interaction")
        .replace("+", "plus")
        .replace(":", "")
    )
    print("-", OUT_DIR / f"{safe_name}_summary.txt")