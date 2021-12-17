import pandas as pd
from matplotlib import pyplot as plt

# load the data

data = pd.read_csv("data_ee.csv")

# create the normalized guess variable

data["guess_truth"] = data.apply(lambda row: row["guess"] if row["urn"] == 1 else 1-row["guess"], axis=1)

# average guesses by treatment and player type

mean_guess = data.groupby(["lon", "mturk", "private", "type", "gameperiod"]).guess_truth.mean().reset_index()

print(max(mean_guess.guess_truth))
print(min(mean_guess.guess_truth))

# ensure the data are sort correctly

mean_guess = mean_guess.sort_values(by=["lon", "mturk", "private", "type"])

# create a dictionary of y-values to be plotted

dct=dict()
for treatment in range(0,2):
    for type in range(1,4):
        dct[str(treatment)+str(type)]=mean_guess[(mean_guess.lon==0) & (mean_guess.mturk==0) & (mean_guess.private==treatment) & (mean_guess.type==type)].guess_truth

# make the plots

plt.subplot(1, 2, 1, title="Public signals")

plt.plot(range(1,31), dct["01"])
plt.plot(range(1,31), dct["02"])
plt.plot(range(1,31), dct["03"])
plt.legend(['Player 1', 'Player 2', 'Player 3'], loc=4)
plt.xlabel("Period")
plt.ylabel("Normalized belief")

plt.subplot(1, 2, 2, title="Private signals")

plt.plot(range(1,31), dct["11"])
plt.plot(range(1,31), dct["12"])
plt.plot(range(1,31), dct["13"])
plt.legend(['Player 1', 'Player 2', 'Player 3'], loc=4)
plt.xlabel("Period")
plt.ylabel("Normalized belief")

plt.subplots_adjust(wspace=.5)

plt.show()







# make Figure 3


