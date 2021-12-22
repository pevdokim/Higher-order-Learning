import pandas as pd
import statsmodels.formula.api as sm
from stargazer.stargazer import Stargazer

# load data

df = pd.read_csv("data_ee.csv")

# measure difference between normalized first- and second-order beliefs

df["guess_truth"] = df.apply(lambda row: row["guess"] if row["urn"] == 1 else 1-row["guess"], axis=1)
df["guess2_truth"] = df.apply(lambda row: row["guess2"] if row["urn"] == 1 else 1-row["guess2"], axis=1)
df["dif"] = df["guess_truth"]-df["guess2_truth"]

# create subject-level means and medians

df_means = df[(df.mturk == 1) & (df.lon == 0)].groupby(['private', 'unique']).dif.agg(['mean', 'median']).reset_index()

# and dummy variables for whether means and medians are positive

df_means["mean_pos"] = df_means.apply(lambda row: 1 if row["mean"] > 0 else 0, axis=1)
df_means["mean_neg"] = df_means.apply(lambda row: 1 if row["mean"] < 0 else 0, axis=1)
df_means["mean_zero"] = df_means.apply(lambda row: 1 if row["mean"] == 0 else 0, axis=1)
df_means["median_pos"] = df_means.apply(lambda row: 1 if row["median"] > 0 else 0, axis=1)
df_means["median_neg"] = df_means.apply(lambda row: 1 if row["median"] < 0 else 0, axis=1)
df_means["median_zero"] = df_means.apply(lambda row: 1 if row["median"] == 0 else 0, axis=1)

print(df_means["mean_neg"].mean())

# fit the regressions

reg1 = sm.ols(formula="mean_neg ~ private", data=df_means).fit(cov_type="cluster", cov_kwds={"groups": df_means["unique"]}, use_t=True)
reg2 = sm.ols(formula="mean_pos ~ private", data=df_means).fit(cov_type="cluster", cov_kwds={"groups": df_means["unique"]}, use_t=True)
reg3 = sm.ols(formula="mean_zero ~ private", data=df_means).fit(cov_type="cluster", cov_kwds={"groups": df_means["unique"]}, use_t=True)
reg4 = sm.ols(formula="median_neg ~ private", data=df_means).fit(cov_type="cluster", cov_kwds={"groups": df_means["unique"]}, use_t=True)
reg5 = sm.ols(formula="median_pos ~ private", data=df_means).fit(cov_type="cluster", cov_kwds={"groups": df_means["unique"]}, use_t=True)
reg6 = sm.ols(formula="median_zero ~ private", data=df_means).fit(cov_type="cluster", cov_kwds={"groups": df_means["unique"]}, use_t=True)

#

print(Stargazer([reg1, reg2, reg3, reg4, reg5, reg6]).render_html())

