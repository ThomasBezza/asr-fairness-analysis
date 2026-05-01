from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.nonparametric.smoothers_lowess import lowess
import patsy

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "raw" / "ASR_Fairness_Master_Spreadsheet_v5_descriptives.xlsx"
OUT_TABLES = ROOT / "outputs" / "tables"
OUT_FIGURES = ROOT / "outputs" / "figures"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGURES.mkdir(parents=True, exist_ok=True)

# =========================================================
# Load and clean data (identical to script 04)
# =========================================================
df = pd.read_excel(DATA_FILE, sheet_name="master_data")

df["resource_level_tertile"] = df["resource_level_tertile"].astype(str).str.strip().str.title()
df["transcript_type_binary"] = df["transcript_type_binary"].astype(str).str.strip()

reg_df = df.copy()
reg_df = reg_df[reg_df["eligible_for_main_analysis"] == True].copy()
reg_df = reg_df[reg_df["wer_reported"].notna()].copy()
reg_df = reg_df[reg_df["resource_level_tertile"].isin(["Low", "Medium", "High"])].copy()
reg_df = reg_df[reg_df["transcript_type_binary"].isin(["standard_reference", "curated_reference"])].copy()

n = len(reg_df)
print(f"Regression sample n = {n}")
assert n == 50, f"Expected n=50, got n={n}"

reg_df = reg_df.reset_index(drop=True)

# =========================================================
# Fit models
# =========================================================
FORMULA_1 = "wer_reported ~ C(resource_level_tertile, Treatment(reference='Low'))"
FORMULA_2 = (
    "wer_reported ~ C(resource_level_tertile, Treatment(reference='Low'))"
    " + C(transcript_type_binary, Treatment(reference='standard_reference'))"
)
FORMULA_3 = (
    "wer_reported ~ C(resource_level_tertile, Treatment(reference='Low'))"
    " * C(transcript_type_binary, Treatment(reference='standard_reference'))"
)

model_1 = smf.ols(FORMULA_1, data=reg_df).fit()
model_2 = smf.ols(FORMULA_2, data=reg_df).fit()
model_3 = smf.ols(FORMULA_3, data=reg_df).fit()

models = {"Model 1": model_1, "Model 2": model_2, "Model 3": model_3}

saved_files = []

# =========================================================
# Diagnostic 1: Variance Inflation Factors (Model 2)
# =========================================================
rhs_2 = FORMULA_2.split("~", 1)[1].strip()
dm = patsy.dmatrix(rhs_2, data=reg_df, return_type="dataframe")
dm_no_intercept = dm.drop(columns=["Intercept"])

vif_data = pd.DataFrame({
    "variable": dm_no_intercept.columns,
    "vif": [
        variance_inflation_factor(dm_no_intercept.values, i)
        for i in range(dm_no_intercept.shape[1])
    ],
})

vif_path = OUT_TABLES / "diagnostics_vif_model2.csv"
vif_data.to_csv(vif_path, index=False)
saved_files.append(vif_path)
print(f"\nDiagnostic 1 — VIF (Model 2):\n{vif_data.to_string(index=False)}")

# =========================================================
# Diagnostic 2: Breusch-Pagan test (all three models)
# =========================================================
bp_rows = []
for name, model in models.items():
    lm_stat, lm_pval, f_stat, f_pval = het_breuschpagan(model.resid, model.model.exog)
    bp_rows.append({
        "model": name,
        "lm_stat": lm_stat,
        "lm_pvalue": lm_pval,
        "f_stat": f_stat,
        "f_pvalue": f_pval,
    })

bp_df = pd.DataFrame(bp_rows)
bp_path = OUT_TABLES / "diagnostics_breusch_pagan.csv"
bp_df.to_csv(bp_path, index=False)
saved_files.append(bp_path)
print(f"\nDiagnostic 2 — Breusch-Pagan:\n{bp_df.to_string(index=False)}")

# =========================================================
# Diagnostic 3: Cook's distance + sensitivity refit (Model 2)
# =========================================================
cooks_d = model_2.get_influence().cooks_distance[0]
threshold = 4 / n  # 0.08

cooks_df = reg_df[["row_id", "observation_unit_label", "wer_reported",
                    "resource_level_tertile", "transcript_type_binary"]].copy()
cooks_df["cooks_d"] = cooks_d
cooks_df["exceeds_threshold"] = cooks_df["cooks_d"] > threshold
cooks_df = cooks_df.sort_values("cooks_d", ascending=False).reset_index(drop=True)

cooks_path = OUT_TABLES / "diagnostics_cooks_distance_model2.csv"
cooks_df.to_csv(cooks_path, index=False)
saved_files.append(cooks_path)

n_flagged = cooks_df["exceeds_threshold"].sum()
print(f"\nDiagnostic 3 — Cook's distance (threshold={threshold:.2f}): {n_flagged} observations exceed threshold")

# Sensitivity refit on reduced dataset
drop_mask = cooks_df[cooks_df["exceeds_threshold"]].index
drop_labels = cooks_df.loc[cooks_df["exceeds_threshold"], "observation_unit_label"].tolist()

flagged_row_ids = cooks_df.loc[cooks_df["exceeds_threshold"], "row_id"].tolist()
reg_df_reduced = reg_df[~reg_df["row_id"].isin(flagged_row_ids)].copy()
n_reduced = len(reg_df_reduced)
n_dropped = n - n_reduced

model_1_r = smf.ols(FORMULA_1, data=reg_df_reduced).fit()
model_2_r = smf.ols(FORMULA_2, data=reg_df_reduced).fit()
model_3_r = smf.ols(FORMULA_3, data=reg_df_reduced).fit()

models_full = {"Model 1": model_1, "Model 2": model_2, "Model 3": model_3}
models_reduced = {"Model 1": model_1_r, "Model 2": model_2_r, "Model 3": model_3_r}

sensitivity_rows = []
for name in ["Model 1", "Model 2", "Model 3"]:
    mf = models_full[name]
    mr = models_reduced[name]
    sensitivity_rows.append({
        "model": name,
        "n_full": int(mf.nobs),
        "n_reduced": int(mr.nobs),
        "n_dropped": n_dropped,
        "r_squared_full": mf.rsquared,
        "r_squared_reduced": mr.rsquared,
        "adj_r_squared_full": mf.rsquared_adj,
        "adj_r_squared_reduced": mr.rsquared_adj,
        "model_p_full": mf.f_pvalue,
        "model_p_reduced": mr.f_pvalue,
        "aic_full": mf.aic,
        "aic_reduced": mr.aic,
    })

sensitivity_df = pd.DataFrame(sensitivity_rows)
sensitivity_path = OUT_TABLES / "diagnostics_sensitivity_refit.csv"
sensitivity_df.to_csv(sensitivity_path, index=False)
saved_files.append(sensitivity_path)
print(f"  Sensitivity refit: n_full={n}, n_reduced={n_reduced}, n_dropped={n_dropped}")
print(sensitivity_df[["model", "r_squared_full", "r_squared_reduced",
                       "model_p_full", "model_p_reduced"]].to_string(index=False))

# =========================================================
# Diagnostic 4: Residual and Q-Q plots (all three models)
# =========================================================
for i, (name, model) in enumerate(models.items(), start=1):
    # Residuals vs. fitted
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(model.fittedvalues, model.resid, alpha=0.6, edgecolors="k", linewidths=0.4)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    smoothed = lowess(model.resid, model.fittedvalues, frac=0.6)
    ax.plot(smoothed[:, 0], smoothed[:, 1], color="red", linewidth=1.5, label="LOWESS")
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title(f"{name}: Residuals vs. Fitted")
    ax.legend()
    fig.tight_layout()
    rvf_path = OUT_FIGURES / f"diagnostics_residuals_vs_fitted_model{i}.png"
    fig.savefig(rvf_path, dpi=300)
    plt.close(fig)
    saved_files.append(rvf_path)

    # Q-Q plot
    fig = sm.qqplot(model.resid, line="s")
    fig.suptitle(f"{name}: Q-Q Plot of Residuals")
    fig.tight_layout()
    qq_path = OUT_FIGURES / f"diagnostics_qq_model{i}.png"
    fig.savefig(qq_path, dpi=300)
    plt.close(fig)
    saved_files.append(qq_path)

print("\nDiagnostic 4 — Residual and Q-Q plots saved for Models 1, 2, 3")

# =========================================================
# Diagnostic 5: Nested model comparison (LR tests)
# =========================================================
# statsmodels convention: unrestricted.compare_lr_test(restricted)
# Model 2 is unrestricted relative to Model 1; Model 3 relative to Model 2.
lr_m1_m2 = model_2.compare_lr_test(model_1)
lr_m2_m3 = model_3.compare_lr_test(model_2)

nested_rows = [
    {
        "comparison": "M1 vs M2 (adds reference condition)",
        "lr_statistic": lr_m1_m2[0],
        "p_value": lr_m1_m2[1],
        "df_diff": int(abs(lr_m1_m2[2])),
        "aic_restricted": model_1.aic,
        "aic_unrestricted": model_2.aic,
        "bic_restricted": model_1.bic,
        "bic_unrestricted": model_2.bic,
    },
    {
        "comparison": "M2 vs M3 (adds interaction)",
        "lr_statistic": lr_m2_m3[0],
        "p_value": lr_m2_m3[1],
        "df_diff": int(abs(lr_m2_m3[2])),
        "aic_restricted": model_2.aic,
        "aic_unrestricted": model_3.aic,
        "bic_restricted": model_2.bic,
        "bic_unrestricted": model_3.bic,
    },
]

nested_df = pd.DataFrame(nested_rows)
nested_path = OUT_TABLES / "diagnostics_nested_model_comparison.csv"
nested_df.to_csv(nested_path, index=False)
saved_files.append(nested_path)
print(f"\nDiagnostic 5 — Nested model comparison:\n{nested_df.to_string(index=False)}")

# =========================================================
# Summary
# =========================================================
print("\n=== ALL DIAGNOSTICS COMPLETE ===")
print("\nSaved files:")
for f in saved_files:
    print("-", f)
