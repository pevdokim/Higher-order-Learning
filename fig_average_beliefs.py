import pandas as pd
from matplotlib import pyplot as plt

# load the data

data = pd.read_csv("data_ee.csv")

# create the normalized beliefs variables

# note second-order beliefs are saved as guess2 in the mturk treatments, so the averaging there needs to be done separately

data["guess_truth"] = data.apply(lambda row: row["guess"] if row["urn"] == 1 else 1-row["guess"], axis=1)
data["guess2_truth"] = data.apply(lambda row: row["guess2"] if row["urn"] == 1 else 1-row["guess2"], axis=1)

# average the beliefs by treatment, player type, and period

# note that type is meaningless in the mturk treatments, so the averaging there is done separately

data_mean = data.groupby(["mturk", "private", "lon", "gameperiod", "type"]).guess_truth.mean().reset_index()
mturk_mean = data.groupby(["mturk", "private", "lon", "gameperiod"]).guess_truth.mean().reset_index()
mturk2_mean = data.groupby(["mturk", "private", "lon", "gameperiod"]).guess2_truth.mean().reset_index()

# ensure that the data are sorted correctly

data_mean.sort_values(by=["mturk", "lon", "private", "type", "gameperiod"]).reset_index()
mturk_mean.sort_values(by=["mturk", "lon", "private", "gameperiod"]).reset_index()
mturk2_mean.sort_values(by=["mturk", "lon", "private", "gameperiod"]).reset_index()

# make the plots

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))

plt.subplot(3, 2, 1, title="Public (lab)")

plt.plot(range(1,31), data_mean.guess_truth[(data_mean.mturk==0) & (data_mean.lon==0) & (data_mean.private==0) & (data_mean.type==1)])
plt.plot(range(1,31), data_mean.guess_truth[(data_mean.mturk==0) & (data_mean.lon==0) & (data_mean.private==0) & (data_mean.type==2)])
plt.plot(range(1,31), data_mean.guess_truth[(data_mean.mturk==0) & (data_mean.lon==0) & (data_mean.private==0) & (data_mean.type==3)])
plt.xlabel("Period")
plt.ylabel("Normalized belief")
plt.axis([0,31,.45,.81])

plt.subplot(3, 2, 2, title="Private (lab)")

plt.plot(range(1,31), data_mean.guess_truth[(data_mean.mturk==0) & (data_mean.lon==0) & (data_mean.private==1) & (data_mean.type==1)])
plt.plot(range(1,31), data_mean.guess_truth[(data_mean.mturk==0) & (data_mean.lon==0) & (data_mean.private==1) & (data_mean.type==2)])
plt.plot(range(1,31), data_mean.guess_truth[(data_mean.mturk==0) & (data_mean.lon==0) & (data_mean.private==1) & (data_mean.type==3)])
plt.xlabel("Period")
plt.ylabel("Normalized belief")
plt.axis([0,31,.45,.81])

plt.subplot(3, 2, 3, title="Public (MTurk)")

plt.plot(range(1,31), mturk_mean.guess_truth[(mturk_mean.mturk==1) & (mturk_mean.lon==0) & (mturk_mean.private==0)])
plt.plot(range(1,31), mturk2_mean.guess2_truth[(mturk2_mean.mturk==1) & (mturk2_mean.lon==0) & (mturk2_mean.private==0)])
plt.xlabel("Period")
plt.ylabel("Normalized belief")
plt.axis([0,31,.45,.81])

plt.subplot(3, 2, 4, title="Private (MTurk)")

plt.plot(range(1,31), mturk_mean.guess_truth[(mturk_mean.mturk==1) & (mturk_mean.lon==0) & (mturk_mean.private==1)])
plt.plot(range(1,31), mturk2_mean.guess2_truth[(mturk2_mean.mturk==1) & (mturk2_mean.lon==0) & (mturk2_mean.private==1)])
plt.xlabel("Period")
plt.ylabel("Normalized belief")
plt.axis([0,31,.45,.81])

plt.subplot(3, 2, 5, title="Public (long treatment)")

plt.plot(range(1,31), mturk_mean.guess_truth[(mturk_mean.mturk==1) & (mturk_mean.lon==1) & (mturk_mean.private==0)])
plt.plot(range(1,31), mturk2_mean.guess2_truth[(mturk2_mean.mturk==1) & (mturk2_mean.lon==1) & (mturk2_mean.private==0)])
plt.xlabel("Period")
plt.ylabel("Normalized belief")
plt.axis([0,31,.45,.81])

plt.subplots_adjust(wspace=.5)
plt.subplots_adjust(hspace=1.1)

fig.legend(["Player 1", "Player 2", "Player 3"], loc='lower right', bbox_to_anchor=(0.8, .15)
)

plt.show()