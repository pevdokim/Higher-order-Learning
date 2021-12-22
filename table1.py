import statsmodels.formula.api as sm
import pandas as pd
from stargazer.stargazer import Stargazer

# load data

df = pd.read_csv("data_ee.csv")

# define the needed variables

df["guess_truth"]=df.apply(lambda row: row["guess"] if row.urn==1 else 1-row["guess"], axis=1)
df["dum2"]=df.apply(lambda row: 1 if row["type"] == 2 else 0, axis=1)
df["dum3"]=df.apply(lambda row: 1 if row["type"] == 3 else 0, axis=1)

# estimate the regression models

reg1 = sm.ols(formula='guess_truth ~ private', data=df[(df.mturk == 0) & (df.type>1)]).fit(cov_type='cluster', cov_kwds={'groups': df[(df.mturk==0) & (df.type>1)]["unique"]}, use_t=True)
reg2 = sm.ols(formula='guess_truth ~ private * gameperiod', data=df[(df.mturk == 0) & (df.type>1)]).fit(cov_type='cluster', cov_kwds={'groups': df[(df.mturk==0) & (df.type>1)]['unique']}, use_t=True)
reg3 = sm.ols(formula='guess_truth ~ dum2 + dum3', data=df[(df.private==1) & (df.mturk==0)]).fit(cov_type="cluster", cov_kwds={'groups': df[(df.mturk==0) & (df.private==1)]["unique"]}, use_t=True)
reg4 = sm.ols(formula='guess_truth ~ gameperiod * dum2 + gameperiod * dum3', data=df[(df.mturk == 0) & (df.private==1)]).fit(cov_type="cluster", cov_kwds={"groups": df[(df.mturk==0) & (df.private==1)]["unique"]}, use_t=True)
reg5 = sm.ols(formula='guess_truth ~ dum2 + dum3', data=df[(df.private==0) & (df.mturk==0)]).fit(cov_type="cluster", cov_kwds={'groups': df[(df.mturk==0) & (df.private==0)]["unique"]}, use_t=True)
reg6 = sm.ols(formula='guess_truth ~ gameperiod * dum2 + gameperiod * dum3', data=df[(df.mturk == 0) & (df.private==0)]).fit(cov_type="cluster", cov_kwds={"groups": df[(df.mturk==0) & (df.private==0)]["unique"]}, use_t=True)

print(Stargazer([reg1, reg2, reg3, reg4, reg5, reg6]).render_html())













