import pandas as pd
import numpy
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt

# load data
df = pd.read_csv("data_ee.csv")

# focus on public lab treatment
df = df[(df.mturk == 0) & (df.private == 0)]

# sort chronologically
df = df.sort_values(by=["unique", "round", "gameperiod"])

# create variables
df["prior"] = df["guess"].shift(1)
df["x2"] = df.apply(lambda row: numpy.log(2) if row["orange"] == 1 else numpy.log(0.5), axis=1)


# replace 0's and 1's
def replace(value):
    if value == 0:
        return 0.01
    elif value == 1:
        return 0.99
    else:
        return value


df["guess_tr"] = df.apply(lambda row: replace(row["guess"]), axis=1)
df["prior_tr"] = df.apply(lambda row: replace(row["prior"]), axis=1)
df["y"] = numpy.log(df["guess_tr"]/(1-df["guess_tr"]))
df["x1"] = numpy.log(df["prior_tr"]/(1-df["prior_tr"]))
df["unique"] = pd.to_numeric(df["unique"])

# set the period 1 prior to 1/2
df.loc[(df.gameperiod == 1), 'x1'] = 0

# create a list of subjects in each role
sub_list = df.groupby(["type", "unique"]).count().reset_index()

# initialize dictionaries
betas_prior = dict()
betas_lr = dict()

for type in [1, 2, 3]:
    # the dictionary key is the player type, the value is a list of estimated parameters
    betas_prior[type]=list()
    betas_lr[type]=list()
    # run a loop to populate the list
    for sub in list(sub_list.unique[(sub_list.type==type)]):
        reg = sm.ols(formula="y ~ x1 + x2", data=df[df["unique"]==sub]).fit()
        betas_prior[type].append(reg.params[1])
        betas_lr[type].append(reg.params[2])

# make the plots

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(6, 6))

plt.subplot(2,3,1, title="Player 1, beta_prior")
plt.hist(betas_prior[1], range=(-.5,1.5), bins=15)

plt.subplot(2,3,2, title="Player 2, beta_prior")
plt.hist(betas_prior[2], range=(-.5,1.5), bins=15)

plt.subplot(2,3,3, title="Player 3, beta_prior")
plt.hist(betas_prior[3], range=(-.5,1.5), bins=15)

plt.subplot(2,3,4, title="Player 1, beta_lr")
plt.hist(betas_lr[1], range=(-2,4), bins=15)

plt.subplot(2,3,5, title="Player 2, beta_lr")
plt.hist(betas_lr[2], range=(-2,4), bins=15)

plt.subplot(2,3,6, title="Player 3, beta_lr")
plt.hist(betas_lr[3], range=(-2,4), bins=15)

plt.subplots_adjust(wspace=.5)
plt.subplots_adjust(hspace=.5)

plt.show()





