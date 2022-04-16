import gym
import numpy as np

import matplotlib.pyplot as plt
from random import uniform

from gym.envs.toy_text.frozen_lake import generate_random_map, FrozenLakeEnv

import time

np.random.seed(42)

def _policy_evaluation(policy, env, discount=0.99, theta=1e-9):
    numStates = env.observation_space.n

    S = range(numStates)
    V = np.zeros(numStates)

    iters = 0

    while True:
        iters += 1
        delta = 0

        for s in S:
            v = 0

            for a, p in enumerate(policy[s]):
                for state_prob, s_prime, r, done in env.P[s][a]:
                    v += p * state_prob * (r + discount * V[s_prime])

            delta = max(delta, np.abs(V[s] - v))

            V[s] = v
            
        if delta < theta:
            return V

def _one_step_lookahead(env, s, V, discount):
    numActions = env.action_space.n
    A = np.zeros(numActions)

    for a in range(numActions):
        for p, s_prime, r, done in env.P[s][a]:
            A[a] += p * (r + discount * V[s_prime])

    return A

def testPolicyIteration(env, discount=0.99):
    numStates = env.observation_space.n
    numActions = env.action_space.n

    S = range(numStates)

    policy = np.ones((numStates, numActions)) / numActions

    iters = 0

    while True:
        stablePolicy = True

        V = _policy_evaluation(policy, env)

        for s in S:
            curr_a = np.argmax(policy[s])

            A = _one_step_lookahead(env, s, V, discount)

            a = np.argmax(A)

            if curr_a != a:
                stablePolicy = False
                policy[s] = np.eye(numActions)[a]

        iters += 1

        if stablePolicy:
            return policy, V, iters

def testValueIteration(env, discount=0.99, theta=1e-9):
    numStates = env.observation_space.n
    numActions = env.action_space.n

    V = np.zeros(numStates)

    iters = 0
    
    while True:
        delta = 0

        for s in range(numStates):
            A = _one_step_lookahead(env, s, V, discount)

            best_a = np.max(A)

            delta = max(delta, np.abs(V[s] - best_a))

            V[s] = best_a

        iters += 1

        if delta < theta:
            break

    policy = np.zeros((numStates, numActions))

    for s in range(numStates):
        A = _one_step_lookahead(env, s, V, discount)
        a = np.argmax(A)

        policy[s, a] = 1.0

    return policy, V, iters

def _get_action(env, Q, s, epsilon):
    val = np.random.uniform(0, 1)
    if val > epsilon:
        actions = np.arange(env.action_space.n)
        action = np.random.choice(actions[Q[s, :] == np.max(Q[s, :])])

    else:
        action = env.action_space.sample()
    return action

def testQLearning(env, gamma=0.9, alpha=0.8, n_episodes=250000, max_steps=500, theta=1e-8):
    numStates = env.observation_space.n
    numActions = env.action_space.n

    Q = np.zeros((numStates, numActions))

    epsilon = 1.0
    decay = 0.005
    low = 0.001
    high = 1.0

    max_step = 0

    visitedStates = {}

    total_steps = 0

    for s in range(numStates):
        visitedStates[s] = 0

    for episode in range(n_episodes):
        done = False
        s = env.reset()

        prev_Q = np.copy(Q)
        
        for step in range(max_steps):
            a = _get_action(env, Q, s, epsilon) 
            s_prime, r, done, info = env.step(a)

            Q[s, a] += alpha * (r + gamma * np.max(Q[s_prime]) - Q[s, a])

            visitedStates[s] += 1
            
            s = s_prime

            if done:
                max_step = step if step > max_step else max_step
                break

            total_steps += 1

        epsilon = low + (high - low) * np.exp(-decay * episode)

    return Q, total_steps, visitedStates

def letter_to_num(c):
    if c == "F":
        return 0
    elif c == "S":
        return 1
    elif c == "H":
        return 2
    else:
        return 3

def action_to_arrow(a):
    # Left
    if a == 0:
        return "\u2190"
    # Down
    if a == 1:
        return "\u2193"
    # Right
    if a == 2:
        return "\u2192"
    # Up
    if a == 3:
        return "\u2191"

def visualize_policy(desc, policy, outputDir, imgName):
    plt.clf()
    np_map = np.array([list(map(letter_to_num, line)) for line in desc])
    plt.imshow(np_map)

    i, j = 0, 0

    for p in policy:
        if len(set(p)) == 1:
            arrow = 'x'
        else:
            a = np.argmax(p)
            arrow = action_to_arrow(a)

        plt.text(j, i, arrow, ha="center", va="center", color="white")

        j += 1

        if j == np.sqrt(len(policy)):
            j = 0
            i += 1

    outputFile = outputDir + imgName + "Policy.png"
    figure = plt.gcf()

    if imgName == "4x4":
        figure.set_size_inches(2,2)
    elif imgName == "8x8":
        figure.set_size_inches(3,3)
    elif imgName == "16x16":
        figure.set_size_inches(4,4)
    elif imgName == "32x32":
        figure.set_size_inches(5,5)
    else:
        figure.set_size_inches(10,10)
    plt.savefig(outputFile)

def visualize_value(V, outputDir, imgName):
    plt.clf()
    V /= np.max(V)
    sqr_len = np.int32(np.sqrt(len(V)))
    vis_V = np.reshape(V, (sqr_len, sqr_len))
    plt.imshow(vis_V)
    plt.colorbar()
    title = str(sqr_len) + "x" + str(sqr_len)
    plt.title(title)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    outputFile = outputDir + imgName + "Value.png"
    plt.savefig(outputFile)

def FrozenLakesRunner():
    FrozenLakeMaps = {
        "4x4": generate_random_map(4),
        "8x8": generate_random_map(8),
        "16x16": generate_random_map(16),
        "32x32": generate_random_map(32),
        "64x64": generate_random_map(64)
    }

    print('########################')
    print("### Policy Iteration ###")
    print('########################')

    info = {}
    info["PI"] = {}
    info["VI"] = {}
    info["QL"] = {}

    outputDir = './Images/PolicyIteration/FrozenLake/'
    for map_name, desc in FrozenLakeMaps.items():
        imgName = map_name
        env = FrozenLakeEnv(desc=desc, map_name=map_name)
        env.reset()

        startTime = time.time()
        policy, V, iters = testPolicyIteration(env)
        endTime = time.time()
        visualize_policy(desc, policy, outputDir, imgName)
        visualize_value(V, outputDir, imgName)
        
        runTime = endTime - startTime

        info["PI"][map_name] = [runTime, iters]

        print("Observation Space:", env.observation_space.n, "Action Space:", env.action_space.n, "Run Time:", runTime)

    print('#######################')
    print("### Value Iteration ###")
    print('#######################')

    outputDir = './Images/ValueIteration/FrozenLake/'
    for map_name, desc in FrozenLakeMaps.items():
        imgName = map_name
        env = FrozenLakeEnv(desc=desc, map_name=map_name)
        env.reset()

        startTime = time.time()
        policy, V, iters = testValueIteration(env)
        endTime = time.time()

        visualize_policy(desc, policy, outputDir, imgName)
        visualize_value(V, outputDir, imgName)
        

        runTime = endTime - startTime

        info["VI"][map_name] = [runTime, iters]

        print("Observation Space:", env.observation_space.n, "Action Space:", env.action_space.n, "Run Time:", runTime)

    print("##################")
    print("### Q-Learning ###")
    print("##################")

    outputDir = './Images/QLearning/FrozenLake/'
    for map_name, desc in FrozenLakeMaps.items():
        imgName = map_name
        env = FrozenLakeEnv(desc=desc, map_name=map_name)
        env.reset()

        startTime = time.time()
        Q, iters, visitedStates = testQLearning(env)
        endTime = time.time()

        visualize_policy(desc, Q, outputDir, imgName)

        runTime = endTime - startTime

        info["QL"][map_name] = [runTime, iters]

        plt.bar(list(visitedStates.values()), len(visitedStates.keys()))

        print("Observation Space:", env.observation_space.n, "Action Space:", env.action_space.n, "Run Time:", runTime)

    for info_key in info.keys():
        print(info_key)
        for algo_key in info[info_key].keys():
            print(algo_key)
            print(info[info_key][algo_key])
        print()


if __name__ == "__main__":
    FrozenLakesRunner()