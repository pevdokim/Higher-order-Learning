import pandas as pd
import numpy
import statsmodels.formula.api as sm
import random
import matplotlib.pyplot as plt

# load data
df = pd.read_csv("data_ee.csv")

# focus on public lab treatment
df = df[(df.mturk == 0) & (df.private == 0)]

# sort chronologically
df = df.sort_values(by=["unique", "round", "gameperiod"])

# create prior and LR variables for the Grether regressions
df["prior"] = df["guess"].shift(1)
df["x2"] = df.apply(lambda row: numpy.log(2) if row["orange"] == 1 else numpy.log(0.5), axis=1)


# replace 0's and 1's to make all logs well-defined
def replace(value):
    if value == 0:
        return 0.01
    elif value == 1:
        return 0.99
    else:
        return value

# transform prior and posterior using the function above
df["guess_tr"] = df.apply(lambda row: replace(row["guess"]), axis=1)
df["prior_tr"] = df.apply(lambda row: replace(row["prior"]), axis=1)
df["y"] = numpy.log(df["guess_tr"]/(1-df["guess_tr"]))
df["x1"] = numpy.log(df["prior_tr"]/(1-df["prior_tr"]))
df["unique"] = pd.to_numeric(df["unique"])

# set the period 1 prior to 1/2
df.loc[(df.gameperiod == 1), 'x1'] = 0

# create a list of subjects in each role
sub_list = df.groupby(["type", "unique"]).count().reset_index()

# initialize dictionaries for saving regression output
inter = dict()
betas_prior = dict()
betas_lr = dict()

for type in [1, 2, 3]:
    # the dictionary key is the player type, the value is a list of estimated parameters
    inter[type]=list()
    betas_prior[type]=list()
    betas_lr[type]=list()
    # run a loop to populate the list
    for sub in list(sub_list.unique[(sub_list.type==type)]):
        reg = sm.ols(formula="y ~ x1 + x2", data=df[df["unique"]==sub]).fit()
        inter[type].append(reg.params[0])
        betas_prior[type].append(reg.params[1])
        betas_lr[type].append(reg.params[2])

df1 = pd.DataFrame(list(zip(inter[1], betas_prior[1], betas_lr[1])), columns=['inter', 'betas_prior', 'betas_lr'])
df2 = pd.DataFrame(list(zip(inter[2], betas_prior[2], betas_lr[2])), columns=['inter', 'betas_prior', 'betas_lr'])

# simulate the paths of belief accuracies

# define numbers of groups and periods in the simulation
numgroups = 300
numperiods = 300

# initialize dictionary for saving the results for every group (key is group label)
group_dict = dict()


# define the Grether-style updating function
def posterior(prior, lr, bp, bl, inter):
    return (prior ** bp) / ((prior ** bp) + ((numpy.exp(-inter)) * ((1-prior) ** bp) * (lr ** (-bl))))


# create a dictionary, where the keys are groups and the values are simulated paths of belief accuracies
for group in range(1,numgroups+1):
    # initialize the path
    group_path = list()
    period = 1

    # initialize beliefs
    belief1 = 0.5
    belief2 = 0.5
    belief3 = 0.5
    df1["guess"] = 0.5
    df2["guess"] = 0.5

    # draw the urn
    urn = random.choice([1, 2])

    # draw the updating parameters for the regular players
    inter1, bp1, bl1 = random.choice(list(zip(inter[1], betas_prior[1], betas_lr[1])))
    inter2, bp2, bl2 = random.choice(list(zip(inter[2], betas_prior[2], betas_lr[2])))
    inter3, bp3, bl3 = random.choice(list(zip(inter[3], betas_prior[3], betas_lr[3])))

    # then simulate an experiment for the group
    while period <= numperiods:

        # draw a signal in every period
        if urn == 1:
            signal = random.choice([1, 1, 0])
        else:
            signal = random.choice([1, 0, 0])
        if signal == 1:
            lr = 2
        else:
            lr = 1/2

        # compute regular players' beliefs
        belief1 = posterior(belief1, lr, bp1, bl1, inter1)
        belief2 = posterior(belief2, lr, bp2, bl2, inter2)
        belief3 = posterior(belief3, lr, bp3, bl3, inter3)

        # compute optimal players' beliefs
        df1["guess"] = posterior(df1["guess"], lr, df1["betas_prior"], df1["betas_lr"], df1["inter"])
        df2["guess"] = posterior(df2["guess"], lr, df2["betas_prior"], df2["betas_lr"], df2["inter"])
        belief_opt2 = numpy.mean(df1["guess"])
        belief_opt3 = numpy.mean(df2["guess"])

        # compute belief accuracies
        acc2 = 1-abs(belief2-belief1)
        acc3 = 1-abs(belief3-belief2)
        acc2_opt = 1-abs(belief_opt2-belief1)
        acc3_opt = 1-abs(belief_opt3-belief2)

        # save the accuracies for the period
        group_path.append([acc2, acc3, acc2_opt, acc3_opt])
        period = period+1

    # save the whole path of accuracies for the group as a dictionary value
    group_dict[group] = group_path


# to create the per-period averages, first initialize the lists
acc2_mean=list()
acc3_mean=list()
acc2opt_mean=list()
acc3opt_mean=list()

# then compute the average for every period across groups, saving the result as a value in the list
for period in range(1, numperiods+1):
    acc2_list=list()
    acc3_list=list()
    acc2opt_list=list()
    acc3opt_list=list()

    for group in range(1, numgroups+1):
        acc2_list.append(group_dict[group][period-1][0])
        acc3_list.append(group_dict[group][period-1][1])
        acc2opt_list.append(group_dict[group][period - 1][2])
        acc3opt_list.append(group_dict[group][period - 1][3])

    acc2_mean.append(numpy.mean(acc2_list))
    acc3_mean.append(numpy.mean(acc3_list))
    acc2opt_mean.append(numpy.mean(acc2opt_list))
    acc3opt_mean.append(numpy.mean(acc3opt_list))

print(numpy.mean(acc2_mean[0:31]))
print(numpy.mean(acc3_mean[0:31]))

# plot the results

plt.plot(range(1,numperiods+1), acc2_mean)
plt.plot(range(1,numperiods+1), acc3_mean)
plt.plot(range(1,numperiods+1), acc2opt_mean)
plt.plot(range(1,numperiods+1), acc3opt_mean)
plt.legend(["Player 2", "Player 3", "Player 2 (optimal)", "Player 3 (optimal)"])
plt.axis([0,300,.67,.9])
plt.title("Simulated belief accuracy")
plt.xlabel("Period")
plt.ylabel("Accuracy")

plt.show()