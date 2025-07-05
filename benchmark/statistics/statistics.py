import pandas as pd, numpy as np, re, ace_tools_open as tools
from scipy.stats import ttest_ind, t
from os.path import abspath, dirname, join

results_path = join(dirname(abspath(__file__)), "pvalues-input-.xlsx")
df = pd.read_excel(results_path)
df["scenario_num"] = df["SCENARIO"].str.extract(r"(\d+)").astype(int)

# long format
df_long = df.melt(
    id_vars=["MODEL", "scenario_num"], value_vars=df.columns[2:], var_name="sample_id", value_name="f1"
).dropna(subset=["f1"])
df_long["f1"] = df_long["f1"].astype(float)


def welch(x, y, alt="greater"):
    res = ttest_ind(x, y, equal_var=False)
    t_stat, p_two = res.statistic, res.pvalue
    p = p_two / 2 if ((alt == "greater" and t_stat > 0) or (alt == "less" and t_stat < 0)) else 1 - p_two / 2
    n_x, n_y = len(x), len(y)
    mean_x, mean_y = x.mean(), y.mean()
    var_x, var_y = x.var(ddof=1), y.var(ddof=1)
    pooled = np.sqrt(((n_x - 1) * var_x + (n_y - 1) * var_y) / (n_x + n_y - 2))
    d = (mean_x - mean_y) / pooled if pooled else np.nan
    se = np.sqrt(var_x / n_x + var_y / n_y)
    df_d = (var_x / n_x + var_y / n_y) ** 2 / ((var_x**2) / (n_x**2 * (n_x - 1)) + (var_y**2) / (n_y**2 * (n_y - 1)))
    ci = t.ppf(0.975, df_d) * se
    return mean_x, mean_y, mean_x - mean_y, mean_x - mean_y - ci, mean_x - mean_y + ci, p, d


imp_rows, drop_rows = [], []
for model, grp in df_long.groupby("MODEL"):
    s1 = grp[grp["scenario_num"] == 1]["f1"].to_numpy()
    s2 = grp[grp["scenario_num"] == 2]["f1"].to_numpy()
    s3 = grp[grp["scenario_num"] == 3]["f1"].to_numpy()

    if len(s3) and len(s2):
        m3, m2, diff, cil, cih, p, d = welch(s3, s2, "greater")
        imp_rows.append([model, m2, m3, diff, cil, cih, p, d])
    if len(s2) and len(s1):
        m2b, m1, diff, cil, cih, p, d = welch(s2, s1, "less")
        drop_rows.append([model, m1, m2b, diff, cil, cih, p, d])

imp_df = pd.DataFrame(imp_rows, columns=["Model", "Mean S2", "Mean S3", "Delta", "CI Low", "CI High", "p", "Cohen d"])
drop_df = pd.DataFrame(drop_rows, columns=["Model", "Mean S1", "Mean S2", "Delta", "CI Low", "CI High", "p", "Cohen d"])

order_imp = ["layoutlmv2", "layoutlmv3", "layoutxlm", "donut-en", "donut-es", "paligemma-en", "paligemma-es"]
order_drop = ["layoutlmv2", "layoutlmv3", "layoutxlm", "donut-en", "donut-es", "paligemma-en", "paligemma-es"]

imp_df["key"] = imp_df["Model"].str.lower().map({m: i for i, m in enumerate(order_imp)})
imp_df = imp_df.sort_values("key").drop(columns="key")

drop_df["key"] = drop_df["Model"].str.lower().map({m: i for i, m in enumerate(order_drop)})
drop_df = drop_df.sort_values("key").drop(columns="key")

alpha = 0.05
imp_df["Reject H0?"] = np.where(imp_df["p"] < alpha, "Yes", "No")
drop_df["Reject H0?"] = np.where(drop_df["p"] < alpha, "Yes", "No")

imp_df["95% CI Delta"] = imp_df.apply(lambda r: f"[{r['CI Low']:.3f}, {r['CI High']:.3f}]", axis=1)
drop_df["95% CI Delta"] = drop_df.apply(lambda r: f"[{r['CI Low']:.3f}, {r['CI High']:.3f}]", axis=1)

imp_df = imp_df[["Model", "Mean S2", "Mean S3", "Delta", "95% CI Delta", "p", "Cohen d", "Reject H0?"]]
drop_df = drop_df[["Model", "Mean S1", "Mean S2", "Delta", "95% CI Delta", "p", "Cohen d", "Reject H0?"]]

# round
for c in ["Mean S1", "Mean S2", "Mean S3", "Delta", "Cohen d"]:
    if c in imp_df:
        imp_df[c] = imp_df[c].round(3)
    if c in drop_df:
        drop_df[c] = drop_df[c].round(3)

latex_imp = imp_df.to_latex(
    index=False,
    column_format="lrrrrrrc",
    float_format="%.3f",
    escape=False,
    caption="Welch one-sided $t$-test (Scenario 3 $>$ Scenario 2). $\\Delta$ is the mean F1 gain.",
    label="tab:imp_s3_s2",
)
latex_drop = drop_df.to_latex(
    index=False,
    column_format="lrrrrrrc",
    float_format="%.3f",
    escape=False,
    caption="Welch one-sided $t$-test (Scenario 2 $<$ Scenario 1). $\\Delta$ is the mean F1 loss.",
    label="tab:drop_s2_s1",
)


tools.display_dataframe_to_user("Updated degradation table", drop_df)
tools.display_dataframe_to_user("Updated improvement table", imp_df)
