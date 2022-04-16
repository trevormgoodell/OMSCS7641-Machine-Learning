from math import gamma
import gym
import numpy as np

import matplotlib.pyplot as plt
from random import uniform
import hiive
from hiive.mdptoolbox.mdp import PolicyIteration, ValueIteration, QLearning
from hiive.mdptoolbox.example import forest

np.random.seed(42)

def testPolicyIteration(P, R, gamma=0.99):
    # Train Policy Iteration
    PI = PolicyIteration(transitions=P, reward=R, gamma=gamma, max_iter=100000)
    PI.run()
    policy = PI.policy

    # Test Policy
    values = PI.V
    iters = PI.iter
    time = PI.time

    return values, policy, iters, time

def testValueIteration(P, R, gamma=0.99, epsilon=1e-5):
    # Train Value Iteration
    VI = ValueIteration(transitions=P, reward=R, gamma=gamma, epsilon=epsilon, max_iter=100000)
    VI.run()
    policy = VI.policy

    # Test Policy
    values = VI.V
    iters = VI.iter
    time = VI.time

    return values, policy, iters, time

def testQLearning(P, R, discount=0.99, alpha=0.1, alpha_decay=0.99, alpha_min=0.001, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.99, n_iter=100000):
    # Train Q Learning
    QL = QLearning(transitions=P, reward=R,
                   gamma=discount,
                   alpha=alpha,
                   alpha_decay=alpha_decay,
                   alpha_min=alpha_min,
                   epsilon=epsilon,
                   epsilon_min=epsilon_min, 
                   epsilon_decay=epsilon_decay,
                   n_iter=n_iter)
    QL.run()
    policy = QL.policy

    # Test Policy
    values = QL.V
    time = QL.time

    return values, policy, n_iter, time


def ForestManagementRunner(main_test = False, second_test = False, third_test = True):
    if main_test:
        info = {}
        for numStates in [16, 64, 256, 1024, 4096]:
            info[numStates] = {}
            print("NumStates:", numStates)
            P, R = forest(S=numStates)

            print('########################')
            print("### Policy Iteration ###")
            print('########################')
            values, policy, iters, time = testPolicyIteration(P, R)
            info[numStates]['PolicyIteration'] = (tuple(values), tuple(policy), iters, time)

            print('#######################')
            print("### Value Iteration ###")
            print('#######################')
            values, policy, iters, time = testValueIteration(P, R)
            info[numStates]['ValueIteration'] = (tuple(values), tuple(policy), iters, time)

            print("##################")
            print("### Q-Learning ###")
            print("##################")
            values, policy, iters, time = testQLearning(P, R)
            info[numStates]['QLearning'] = (tuple(values), tuple(policy), iters, time)

        for num_states in info.keys():
            print("Num States:", num_states)
            for algo in info[num_states].keys():
                print("Algorithm:", algo)

                plt.bar(np.arange(num_states), info[num_states][algo][0])
                plt.title("State Values")
                plt.savefig("./Images/" + algo + "/ForestManagement/" + str(num_states) + "_Values.png")
                plt.clf()

                plt.bar(np.arange(num_states), info[num_states][algo][1])
                plt.title("Policies")
                plt.savefig("./Images/" + algo + "/ForestManagement/" + str(num_states) + "_Policy.png")
                plt.clf()

                print("Iterations:", info[num_states][algo][2])
                print("Time:", info[num_states][algo][3])

    if second_test:
        info = {}
        for r1 in [0.8, 0.9, 0.99, 0.9999]:
            info[r1] = {}
            print("r1:", r1)
            P, R = forest(S=64)

            print('########################')
            print("### Policy Iteration ###")
            print('########################')
            values, policy, iters, time = testPolicyIteration(P, R, gamma=r1)
            info[r1]['PolicyIteration'] = (tuple(values), tuple(policy), iters, time)

            print('#######################')
            print("### Value Iteration ###")
            print('#######################')
            values, policy, iters, time = testValueIteration(P, R, gamma=r1)
            info[r1]['ValueIteration'] = (tuple(values), tuple(policy), iters, time)

        for r1 in info.keys():
            print("R1:", r1)
            for algo in info[r1].keys():
                print("Algorithm:", algo)

                plt.bar(np.arange(64), info[r1][algo][0])
                plt.title("State Values")
                plt.savefig("./Images/FM2/"  + algo + "/" + str(r1) + "_Values.png")
                plt.clf()

                plt.bar(np.arange(64), info[r1][algo][1])
                plt.title("Policies")
                plt.savefig("./Images/FM2/"  + algo + "/" + str(r1) + "_Policy.png")
                plt.clf()

                print("Iterations:", info[r1][algo][2])
                print("Time:", info[r1][algo][3])
    
    if third_test:
        info = {}
        P, R = forest(S=64)
        for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            for alpha_decay in [0.9, 0.99, 0.999]:
                for epsilon_min in [0.001, 0.01, 0.1, 0.2, 0.4]:
                    values, policy, iters, time = testQLearning(P, R,
                                                                alpha=alpha,
                                                                alpha_decay=alpha_decay, 
                                                                epsilon_min=epsilon_min)
                plt.bar(np.arange(64), values)
                plt.title("State Values")
                plt.savefig("./Images/FM3/" + str(alpha) + "_" + str(alpha_decay) + "_" + str(epsilon_min) + "_Values.png")
                plt.clf()

                plt.bar(np.arange(64), policy)
                plt.title("Policies")
                plt.savefig("./Images/FM3/" + str(alpha) + "_" + str(alpha_decay) + "_" + str(epsilon_min) + "_Policy.png")
                plt.clf()

                print("Iterations:", iters)
                print("Time:", time)

        

        for r1 in info.keys():
            print("R1:", r1)
            for algo in info[r1].keys():
                print("Algorithm:", algo)

                plt.bar(np.arange(64), info[r1][algo][0])
                plt.title("State Values")
                plt.savefig("./Images/FM2/"  + algo + "/" + str(r1) + "_Values.png")
                plt.clf()

                plt.bar(np.arange(64), info[r1][algo][1])
                plt.title("Policies")
                plt.savefig("./Images/FM2/"  + algo + "/" + str(r1) + "_Policy.png")
                plt.clf()

                print("Iterations:", info[r1][algo][2])
                print("Time:", info[r1][algo][3])
    

    
    x = 1

if __name__ == "__main__":
    ForestManagementRunner()