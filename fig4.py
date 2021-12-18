import pandas as pd
from matplotlib import pyplot as plt

# load the data

data = pd.read_csv("data_ee.csv")

data.info()

# break data up according to player type

data1 = data[(data.lon==0) & (data.mturk==0) & (data.type==1)]
data2 = data[(data.lon==0) & (data.mturk==0) & (data.type==2)]
data3 = data[(data.lon==0) & (data.mturk==0) & (data.type==3)]

# merge the three datasets by groups

data_merged=pd.merge(
    data1,
    data2,
    on=["ses", "round", "gameperiod", "group"]
)

data_merged=pd.merge(
    data_merged,
    data3,
    on=["ses", "round", "gameperiod", "group"]
)

# define the accuracy variables

data_merged["acc2"]=1-abs(data_merged["guess_x"]-data_merged["guess_y"])
data_merged["acc3"]=1-abs(data_merged["guess_y"]-data_merged["guess"])

# average accuracy by period and treatment variables

acc2_mean = data_merged.groupby(["private", "gameperiod"]).acc2.mean().reset_index()
acc3_mean = data_merged.groupby(["private", "gameperiod"]).acc3.mean().reset_index()

# make the plots

plt.subplot(1, 2, 1)

plt.plot(range(1,31), acc2_mean[acc2_mean.private==0]["acc2"])
plt.plot(range(1,31), acc3_mean[acc2_mean.private==0]["acc3"])
plt.axis([0,31,.65,.85])
plt.xlabel("Period")
plt.ylabel("Belief accuracy")
plt.legend(['Player 2', 'Player 3'])

plt.subplot(1, 2, 2)

plt.plot(range(1,31), acc2_mean[acc2_mean.private==1]["acc2"])
plt.plot(range(1,31), acc3_mean[acc2_mean.private==1]["acc3"])
plt.axis([0,31,.65,.85])
plt.xlabel("Period")
plt.ylabel("Belief accuracy")
plt.legend(['Player 2', 'Player 3'])

plt.subplots_adjust(wspace=.5)

plt.show()

