# import packages
from scipy.stats import binom
import matplotlib.pyplot as plt


# set the number of periods
periods = list(range(1,31))


# define bayes' rule
def bayes(num_balls, period):
    return (((2/3) ** num_balls) * ((1/3) ** (period-num_balls))) \
    /(((2 / 3) ** num_balls) * ((1 / 3) ** (period - num_balls))+((1 / 3) ** num_balls) * ((2 / 3) ** (period - num_balls)))


# initialize public beliefs list
beliefs_public = list()


# loop through periods and for each period compute expected 1-order belief
for period in periods:
    expected_belief = 0
    for num_balls in range(period+1):
        belief = bayes(num_balls, period)
        expected_belief += belief * binom.pmf(num_balls,period,2/3)
    beliefs_public.append(expected_belief)
print(beliefs_public)

# define function for private second-order beliefs computed using bayes' rule
def bayes2(num_balls, period):
    # expectation over histories for each state
    sum1 = 0
    sum2 = 0
    for num in range(period+1):
        sum1 += bayes(num, period) * binom.pmf(num, period, 2 / 3)
        sum2 += bayes(num, period) * binom.pmf(num, period, 1 / 3)
    # expectation over states
    return bayes(num_balls, period) * sum1 \
           + (1 - bayes(num_balls, period)) * sum2


# initialize the list of private second-order beliefs
beliefs_private2 = list()


# loop through periods and for each period compute expected private second-
# order belief
for period in periods:
    expected_belief2 = 0
    for num_balls in range(period+1):
        expected_belief2 += bayes2(num_balls, period) * binom.pmf(num_balls,period,2/3)
    beliefs_private2.append(expected_belief2)
print(beliefs_private2)

# define function for private third-order beliefs computed using Bayes' rule
def bayes3(num_balls, period):
    # expectation over histories for each state
    sum1 = 0
    sum2 = 0
    for num in range(period+1):
        sum1 += bayes2(num, period) * binom.pmf(num, period, 2 / 3)
        sum2 += bayes2(num, period) * binom.pmf(num, period, 1 / 3)
    # expectation over states
    return bayes(num_balls, period) * sum1 \
           + (1- bayes(num_balls,period)) * sum2


# initialize the list of private second-order beliefs
beliefs_private3 = list()


# loop through periods and for each period compute expected private third-
# order belief
for period in periods:
    expected_belief3 = 0
    for num_balls in range(period+1):
        expected_belief3 += bayes3(num_balls, period) * binom.pmf(num_balls,period,2/3)
    beliefs_private3.append(expected_belief3)


# MAKE THE PLOT!


plt.plot(periods, beliefs_public)
plt.plot(periods, beliefs_private2)
plt.plot(periods, beliefs_private3)
plt.axis([0, 32, 0.5, 1])
plt.xlabel('Period')
plt.ylabel('Belief')
plt.legend(['First-order', 'Second-order', 'Third-order'],
           loc='best')
plt.show()