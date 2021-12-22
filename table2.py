import pandas as pd
import statsmodels.formula.api as sm
from stargazer.stargazer import Stargazer

df = pd.read_csv('data_ee.csv')

# sort the data and create a variable keeping track of the partner's guess (row above or below, depending on condition)

df = df.sort_values(by=["lon", "mturk", "private", "ses", "round", "gameperiod", "group", "type"]).reset_index()
df["guess_prev"] = df.shift(1).guess
df["guess_next"] = df.shift(-1).guess

# define a function for computing belief accuracy

def acc_compute(row):
    if row.mturk == 0:
        return 1-abs(row["guess"]-row["guess_prev"])
    elif (row.mturk == 1) & (row.type == 2):
        return 1 - abs(row["guess2"] - row["guess_prev"])
    elif (row.mturk == 1) & (row.type == 1):
        return 1 - abs(row["guess2"]-row["guess_next"])


# apply this function to every row
df["acc"] = df.apply(acc_compute, axis=1)

# run the regressions
reg1 = sm.ols(formula="acc ~ private", data=df[(df.mturk==0) & (df.type>1)]).fit(cov_type="cluster", cov_kwds={"groups": df[(df.mturk==0) & (df.type>1)]["unique"]}, use_t=True)
reg2 = sm.ols(formula="acc ~ private", data=df[(df.mturk == 0) & (df.type == 2)]).fit(cov_type="cluster", cov_kwds={"groups": df[(df.mturk == 0) & (df.type == 2)]["unique"]}, use_t=True)
reg3 = sm.ols(formula="acc ~ private", data=df[(df.mturk==0) & (df.type==3)]).fit(cov_type="cluster", cov_kwds={"groups": df[(df.mturk==0) & (df.type==3)]["unique"]}, use_t=True)
reg4 = sm.ols(formula="acc ~ private * gameperiod", data=df[(df.mturk==0) & (df.type>1)]).fit(cov_type="cluster", cov_kwds={"groups": df[(df.mturk==0) & (df.type>1)]["unique"]}, use_t=True)
reg5 = sm.ols(formula="acc ~ private", data=df[(df.mturk==1) & (df.lon==0)]).fit(cov_type="cluster", cov_kwds={"groups": df[(df.mturk==1) & (df.lon==0)]["unique"]}, use_t=True)
reg6 = sm.ols(formula="acc ~ private * gameperiod", data=df[(df.mturk==1) & (df.lon==0)]).fit(cov_type="cluster", cov_kwds={"groups": df[(df.mturk==1) & (df.lon==0)]["unique"]}, use_t=True)