import pandas as pd
from matplotlib import pyplot as plt

# load the data
df = pd.read_csv("data_ee.csv")

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

# create a variable indexing player role in both lab and mturk treatments


def labtype(row):
    if row["mturk"]==0:
        return row["type"]
    else:
        return 2


df["labtype"] = df.apply(labtype, axis=1)

# compute mean accuracy by player role
acc_mean = df[df.labtype>1].groupby(["mturk", "lon", "private", "gameperiod", "labtype"]).acc.mean().reset_index()

# make the plots

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))

plt.subplot(3, 2, 1, title="Public (lab)")

plt.plot(range(1,31), acc_mean[(acc_mean.mturk==0) & (acc_mean.lon==0) & (acc_mean.private==0) & (acc_mean.labtype==2)].acc)
plt.plot(range(1,31), acc_mean[(acc_mean.mturk==0) & (acc_mean.lon==0) & (acc_mean.private==0) & (acc_mean.labtype==3)].acc)
plt.axis([0,31,.6,.8])
plt.xlabel("Period")
plt.ylabel("Accuracy")

plt.subplot(3, 2, 2, title="Private (lab)")

plt.plot(range(1,31), acc_mean[(acc_mean.mturk==0) & (acc_mean.lon==0) & (acc_mean.private==1) & (acc_mean.labtype==2)].acc)
plt.plot(range(1,31), acc_mean[(acc_mean.mturk==0) & (acc_mean.lon==0) & (acc_mean.private==1) & (acc_mean.labtype==3)].acc)
plt.axis([0,31,.6,.8])
plt.xlabel("Period")
plt.ylabel("Accuracy")

plt.subplot(3, 2, 3, title="Public (turk)")

plt.plot(range(1,31), acc_mean[(acc_mean.mturk==1) & (acc_mean.lon==0) & (acc_mean.private==0) & (acc_mean.labtype==2)].acc)
plt.axis([0,31,.6,.8])
plt.xlabel("Period")
plt.ylabel("Accuracy")

plt.subplot(3, 2, 4, title="Private (turk)")

plt.plot(range(1,31), acc_mean[(acc_mean.mturk==1) & (acc_mean.lon==0) & (acc_mean.private==1) & (acc_mean.labtype==2)].acc)
plt.axis([0,31,.6,.8])
plt.xlabel("Period")
plt.ylabel("Accuracy")

plt.subplot(3, 2, 5, title="Public (long)")

plt.plot(range(1,31), acc_mean[(acc_mean.mturk==1) & (acc_mean.lon==1) & (acc_mean.private==0) & (acc_mean.labtype==2)].acc)
plt.axis([0,31,.6,.8])
plt.xlabel("Period")
plt.ylabel("Accuracy")

plt.subplots_adjust(wspace=.5)
plt.subplots_adjust(hspace=1.1)

fig.legend(["Player 1", "Player 2", "Player 3"], loc='lower right', bbox_to_anchor=(0.8, .15)
)

plt.show()