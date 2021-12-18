import pandas as pd
from matplotlib import pyplot as plt

# load the data

data = pd.read_csv("data_ee.csv")

# separate it by player

data1 = data[data.type == 1]
data2 = data[data.type == 2]
data3 = data[data.type == 3]

# merge by group

data_merged = pd.merge(data1,
                       data2,
                       on=(["mturk", "lon", "private", "ses", "round", "gameperiod", "group"]),
                       how="outer"
                       )

data_merged = pd.merge(data_merged,
                       data3,
                       on=(["mturk", "lon", "private", "ses", "round", "gameperiod", "group"]),
                       how="outer"
                       )

# compute the belief accuracy variable

data_merged["accuracy2"] = 1-abs(data_merged["guess_x"]-data_merged["guess_y"])
data_merged["accuracy3"] = 1-abs(data_merged["guess_y"]-data_merged["guess"])
# accuracy of Mturk p2 beliefs arbitarily assigned a type of 1
data_merged["accuracy2_turk_1"] = 1-abs(data_merged["guess2_x"]-data_merged["guess_y"])
# accuracy of Mturk p2 beliefs arbitarily assigned a type of 2
data_merged["accuracy2_turk_2"] = 1-abs(data_merged["guess2_y"]-data_merged["guess_x"])
# overall accuracy of Mturk p2 beliefs
data_merged["accuracy2_turk"] = 0.5 * (data_merged["accuracy2_turk_1"] + data_merged["accuracy2_turk_2"])

# average belief accuracy by treatment and period

acc2_mean = data_merged.groupby(["mturk", "lon", "private", "gameperiod"]).accuracy2.mean().reset_index()
acc3_mean = data_merged.groupby(["mturk", "lon", "private", "gameperiod"]).accuracy3.mean().reset_index()
acc2_turk_mean = data_merged.groupby(["mturk", "lon", "private", "gameperiod"]).accuracy2_turk.mean().reset_index()

# make the plots

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))

plt.subplot(3,2,1, title="Public (lab)")

plt.plot(range(1,31),acc2_mean[(acc2_mean.mturk==0) & (acc2_mean.lon==0) & (acc2_mean.private==0)].accuracy2)
plt.plot(range(1,31),acc3_mean[(acc2_mean.mturk==0) & (acc2_mean.lon==0) & (acc2_mean.private==0)].accuracy3)
plt.axis([0,31,.6,.8])
plt.xlabel("Period")
plt.ylabel("Accuracy")

plt.subplot(3,2,2, title="Private (lab)")

plt.plot(range(1,31),acc2_mean[(acc2_mean.mturk==0) & (acc2_mean.lon==0) & (acc2_mean.private==1)].accuracy2)
plt.plot(range(1,31),acc3_mean[(acc2_mean.mturk==0) & (acc2_mean.lon==0) & (acc2_mean.private==1)].accuracy3)
plt.axis([0,31,.6,.8])
plt.xlabel("Period")
plt.ylabel("Accuracy")

plt.subplot(3,2,3, title="Public (turk)")

plt.plot(range(1,31),acc2_turk_mean[(acc2_turk_mean.mturk==1) & (acc2_turk_mean.lon==0) & (acc2_turk_mean.private==0)].accuracy2_turk)
plt.axis([0,31,.6,.8])
plt.xlabel("Period")
plt.ylabel("Accuracy")

plt.subplot(3,2,4, title="Private (turk)")

plt.plot(range(1,31),acc2_turk_mean[(acc2_turk_mean.mturk==1) & (acc2_turk_mean.lon==0) & (acc2_turk_mean.private==1)].accuracy2_turk)
plt.axis([0,31,.6,.8])
plt.xlabel("Period")
plt.ylabel("Accuracy")

plt.subplot(3,2,5, title="Public (long)")

plt.plot(range(1,31),acc2_turk_mean[(acc2_turk_mean.mturk==1) & (acc2_turk_mean.lon==1) & (acc2_turk_mean.private==0)].accuracy2_turk)
plt.axis([0,31,.6,.8])
plt.xlabel("Period")
plt.ylabel("Accuracy")

plt.subplots_adjust(wspace=.5)
plt.subplots_adjust(hspace=1.1)

fig.legend(["Player 1", "Player 2", "Player 3"], loc='lower right', bbox_to_anchor=(0.8, .15)
)

plt.show()