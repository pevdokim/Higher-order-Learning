import random
from numpy import mean
import matplotlib.pyplot as plt


# define bayes' rule as function of the signal and the prior
def bayes(orange, prior):
    if orange == 1:
        return ( (2/3) * prior ) / ( (2/3) * prior + (1/3) * (1 - prior))
    if orange == 0:
        return ( (1/3) * prior ) / ( (1/3) * prior + (2/3) * (1 - prior))


# define a function that computes the simulated accuracy path for a given distribution of parameters
def accuracy(par1, par2, par3, num_teams, num_periods):
    acc_dct = dict()

    for team in range(1,num_teams+1):
        # initialize periods at 1
        period = 1
        # initialize priors to 1/2
        prior1 = 1/2
        prior2 = 1/2

        # draw an urn, which can be orange or not
        urn = random.choice([0,1])

        # draw a lambda for player 1 and a lambda from player 2 randomly
        lambda1 = random.choice([par1, par2, par3])
        lambda2 = random.choice([par1, par2, par3])

        # initialize paths of accuracies for the two players
        acc_lst = list()

        while period <= num_periods:
            if urn == 1:
                orange = random.choice([1, 1, 0])
            else:
                orange = random.choice([1, 0, 0])

            belief1 = lambda1 * bayes(orange, prior1) + (1-lambda1) * prior1
            belief2 = lambda2 * bayes(orange, prior2) + (1-lambda2) * prior2
            # append belief accuracy for the given period to the list
            acc_lst.append(1 - abs(belief1-belief2))
            # update period and priors
            period = period + 1
            prior1 = belief1
            prior2 = belief2

        acc_dct[team]=acc_lst

    # create a list of average accuracies in each period
    average_acc = list()
    for index in range(num_periods):
        # to compute average accuracy in period, first put all the teams' accuracies for that period in a list
        period_accs = list()
        # loop over the dictionary and add each team's accuracy for that period to the list above
        for k, v in acc_dct.items():
            period_accs.append(v[index])
        # add the mean across teams to the average accuracies list
        average_acc.append(mean(period_accs))

    # return acc_dct, average_acc
    return average_acc


# the code for generating the path of belief accuracies for a player
# who optimally takes into account the distribution of types
def optimal_accuracy(par1, par2, par3, num_teams, num_periods):
    acc_opt_dct = dict()

    for team in range(1,num_teams+1):
        # initialize periods at 1
        period = 1
        # initialize priors to 1/2
        prior = 1/2
        prior1 = 1/2
        prior2 = 1/2
        prior3 = 1/2

        # draw an urn, which can be orange or not
        urn = random.choice([0, 1])

        # draw a lambda for the non-optimal player randomly
        lamb = random.choice([par1, par2, par3])

        # initialize paths of accuracies for the two players
        acc_opt_lst = list()

        while period <= num_periods:
            if urn == 1:
                orange = random.choice([1, 1, 0])
            else:
                orange = random.choice([1, 0, 0])

            belief = lamb * bayes(orange, prior) + (1 - lamb) * prior
            belief1 = par1 * bayes(orange, prior1) + (1 - par1) * prior1
            belief2 = par2 * bayes(orange, prior2) + (1 - par2) * prior2
            belief3 = par3 * bayes(orange, prior3) + (1 - par3) * prior3
            belief_opt = (belief1 + belief2 + belief3)/3
            # append belief accuracy for the given period to the list
            acc_opt_lst.append(1 - abs(belief - belief_opt))
            # update period and priors
            period = period + 1
            prior = belief
            prior1 = belief1
            prior2 = belief2
            prior3 = belief3

        acc_opt_dct[team] = acc_opt_lst
    # create a list of average accuracies in each period
    average_acc = list()
    for index in range(num_periods):
        # to compute average accuracy in period, first put all the teams' accuracies for that period in a list
        period_accs = list()
        # loop over the dictionary and add each team's accuracy for that period to the list above
        for k, v in acc_opt_dct.items():
            period_accs.append(v[index])
        # add the mean across teams to the average accuracies list
        average_acc.append(mean(period_accs))

    # return acc_dct, average_acc
    return average_acc


# DRAW THE PATHS!

periods = list(range(1,301))
accuracies1 = accuracy(1, .9, .8, 5000, 300)
accuracies2 = accuracy(1, .55, .1, 5000, 300)
accuracies3 = optimal_accuracy(1, .55, .1, 5000, 300)

print(accuracies1)
print(accuracies2)
print(accuracies3)

plt.plot(periods, accuracies1)
plt.plot(periods, accuracies2)
plt.plot(periods, accuracies3)
plt.axis([0, 301, 0.84, 1.01])
params = {'mathtext.default': 'regular' }
plt.xlabel('Period')
plt.ylabel('Accuracy')
plt.legend(['$\lambda_1=1$, $\lambda_2=0.9$, $\lambda_1=0.8$', '$\lambda_1=1$, $\lambda_2=0.55$, $\lambda_1=0.1$', 'Optimal beliefs'],
           loc='best')
plt.show()

