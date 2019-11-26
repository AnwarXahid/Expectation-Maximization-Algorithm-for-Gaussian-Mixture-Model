import numpy as np
import pandas as pd
import random as rand
import matplotlib.pyplot as plt
from scipy.stats import norm
import math as mt

#same random values for every new execution
rand.seed(42)

#taking 3 clusters
mu1 = [0, 5]
sig1 = [ [2, 0], [0, 3] ]

mu2 = [5, 0]
sig2 = [ [4, 0], [0, 1] ]

mu3 = [5, 5]
sig3 = [ [2, 0], [0, 4] ]

#generating samples
x3, y3 = np.random.multivariate_normal(mu1, sig1, 100).T
x4, y4 = np.random.multivariate_normal(mu2, sig2, 100).T
x5, y5 = np.random.multivariate_normal(mu3, sig3, 100).T

x1 = np.concatenate((x3, x4, x5))
x2 = np.concatenate((y3, y4, y5))
labels = ([1] * 100) + ([2] * 100) + ([3] * 100)

#generating data
data = {'x1': x1, 'x2': x2, 'label': labels}
df = pd.DataFrame(data)

#showing data pictorially
fig = plt.figure()
plt.scatter(data['x1'], data['x2'], c=data['label'])
fig.savefig("showingData.png")



#Implementation of Expectation Maximization algorithm


#the probability of generation of x from Gaussian distribution
def prob_N(x, mu, sig, wei):
    pro = wei
    for i in range(len(x)):
        pro *= norm.pdf(x[i], mu[i], sig[i][i])
    return pro



#E-Step function
def expectationStep(x, mu, sig, wei):
    pro_clu1 = prob_N(x, mu1, sig1, w[0])
    pro_clu2 = prob_N(x, mu2, sig2, w[1])
    pro_clu3 = prob_N(x, mu3, sig3, w[2])
    p_i_j = prob_N(x, mu, sig, wei) / (pro_clu1 + pro_clu2 + pro_clu3)
    return p_i_j


#M-Step function
def MaximizationStep(dataset, mu, sig, wei):
    accProb = 0
    temp = [0, 0]
    temp_sigma = [ [0, 0], [0, 0] ]

    for j in range(dataset.shape[0]):
        x_1 = dataset['x1'][j]
        x_2 = dataset['x2'][j]
        exp = expectationStep([x_1, x_2], mu, sig, wei)
        accProb += exp
        temp1 = exp * np.array([x_1, x_2])
        temp = np.add(temp, temp1)
        temp2 = 1 * np.array([x_1, x_2])
        temp3 = -1 * np.array(mu)
        vect = np.add(temp2,temp3)


        temp_sig = np.multiply(vect, np.transpose(vect))
        temp_sig = exp * temp_sig
        temp_sigma = np.add(temp_sigma, temp_sig)

    temp_mu = (1 /accProb) * temp
    ret_sigma = (1 /accProb) * temp_sigma
    temp_weight = accProb / dataset.shape[0]

    return temp_mu, ret_sigma, temp_weight

#get distance between old and new mean to chack convergence
def params_converge(old_params, new_params):
    distance = [0, 0, 0]
    for param in range(3):
        for i in range(2):
            distance[param] += (old_params[param][i] - new_params[param][i]) ** 2
        distance[param] = mt.sqrt(distance[param])
    return distance[0] + distance[1] + distance[2]



def logLikelihood(dataset):
    likelihoodProb = 0
    for j in range(dataset.shape[0]):
        x_1 = dataset['x1'][j]
        x_2 = dataset['x2'][j]
        pro_clu1 = prob_N([x_1, x_2], mu1, sig1, w[0])
        pro_clu2 = prob_N([x_1, x_2], mu2, sig2, w[1])
        pro_clu3 = prob_N([x_1, x_2], mu3, sig3, w[2])
        lnProb = mt.log((pro_clu1 + pro_clu2 + pro_clu3), 2)
        likelihoodProb += lnProb

    return likelihoodProb




#let's start the main function loop
#initilization
mu1 = [1,1]
sig1 = [ [1, 0], [0, 1] ]
mu2 = [4,4]
sig2 = [ [1, 0], [0, 1] ]
mu3 = [0, 5]
sig3 = [ [1, 0], [0, 1] ]
w = [0.2, 0.3, 0.5]
loglihood = logLikelihood(df.copy())
print("initial likelihood", loglihood)
dis = 10000
count = 0
track = 0
maxlikelihood = 0
while dis > 0.01:
    #E-step and M-step
    new_mu1, new_sigma1, new_weight1 = MaximizationStep(df.copy(), mu1, sig1, w[0])
    new_mu2, new_sigma2, new_weight2 = MaximizationStep(df.copy(), mu2, sig2, w[1])
    new_mu3, new_sigma3, new_weight3 = MaximizationStep(df.copy(), mu3, sig3, w[2])


    #check convergenge
    dis = params_converge([mu1, mu2, mu3], [new_mu1, new_mu2, new_mu3])
    mu1 = new_mu1
    sig1 = new_sigma1
    mu2 = new_mu2
    sig2 = new_sigma2
    mu3 = new_mu3
    sig3 = new_sigma3
    w = [new_weight1, new_weight2, new_weight3]
    #showing result pictorially

    loglihood = logLikelihood(df.copy())
    if mt.fabs(loglihood) < mt.fabs(maxlikelihood) :
        print("Maximum Likelihood")
        track = 1
        break
    maxlikelihood = loglihood
    print(mt.fabs(maxlikelihood))
    count += 1
    print(count)
    print("  ")
    print(loglihood, mu1, sig1, mu2, sig2, mu3, sig3, dis)
    u = mu1[0]
    v = mu1[1]
    a = mt.sqrt( ( mu1[0] - sig1[0][0] ) ** 2 + ( mu1[1] - sig1[0][1] ) ** 2 )
    b = mt.sqrt( ( mu1[0] - sig1[1][0] ) ** 2 + ( mu1[1] - sig1[1][1] ) ** 2 )

    t = np.linspace(0, 2 * mt.pi, 100)



    fig = plt.figure()
    plt.scatter(data['x1'], data['x2'], c=data['label'])
    plt.plot(u + a * np.cos(t), v + b * np.sin(t))
    u = mu2[0]
    v = mu2[1]
    a = mt.sqrt((mu2[0] - sig2[0][0]) ** 2 + (mu2[1] - sig2[0][1]) ** 2)
    b = mt.sqrt((mu2[0] - sig2[1][0]) ** 2 + (mu2[1] - sig2[1][1]) ** 2)

    t = np.linspace(0, 2 * mt.pi, 100)
    plt.plot(u + a * np.cos(t), v + b * np.sin(t))

    u = mu3[0]
    v = mu3[1]
    a = mt.sqrt((mu3[0] - sig3[0][0]) ** 2 + (mu3[1] - sig3[0][1]) ** 2)
    b = mt.sqrt((mu3[0] - sig3[1][0]) ** 2 + (mu3[1] - sig3[1][1]) ** 2)

    t = np.linspace(0, 2 * mt.pi, 100)
    plt.plot(u + a * np.cos(t), v + b * np.sin(t))

    #plt.grid(color='lightgray', linestyle='--')
    fig.savefig("output{}.png".format(count))

if track == 0:
    print("convergence of parameters")





