"""
SARSA is the model free control technique of using TD(0) for policy evaluation and GLIE for policy improvement.
In GLIE, the epsilon value( the value that controls the probability with which a random action is chosen v/s the greedy
action is hyperbolically degraded as more episodes are seen. This ensures that we explore more initially and get greedy
when we pick actions as we see more episodes. We can also use a constant epsilon and
compare the 2 scenarios.

It displays:
1) The steps that need to be taken to reach the goal
2) Graph of no of steps it took to complete the episode v/s episode no
"""

import numpy as np
import environment
import random
from matplotlib import pyplot as plt

sequence = []
env = environment.GridWorld8x8()
actions = [0, 1, 2, 3]
iteration = 0

# initializing q values
q = np.zeros((env.m, env.n, len(env.actions)))

discount_factor = float(input("Enter the discount factor: "))  # gamma
update_factor = float(input("Enter the update factor: "))  # alpha

episodes = int(input("How many episodes? "))


def get_action(rand_or_greedy, m, n):
    if rand_or_greedy == 0:  # random
        return random.choice(actions)

    else:  # greedy
        return np.argmax(q[m][n])


steps = []
for episode in range(episodes):
    # defining epsilon: used to implement GLIE
    # 0 means random, 1 means greedy action will be chosen.
    # So, 0 -> epsilon 1 -> (1 - epsilon)

    epsilon = 1 / (episode + 1)

    current_state = env.starting_point

    random_or_greedy = np.random.choice(np.arange(0, 2), p=[epsilon, (1 - epsilon)])
    action = get_action(random_or_greedy, current_state[0], current_state[1])
    step_count = 0

    while True:
        reward, next_state, terminal_state = env.take_action(current_state[0], current_state[1], action)

        random_or_greedy = np.random.choice(np.arange(0, 2), p=[epsilon, (1 - epsilon)])
        action_dash = get_action(random_or_greedy, next_state[0], next_state[1])

        q[current_state[0]][current_state[1]][action] += update_factor * (reward + discount_factor *
                        q[next_state[0]][next_state[1]][action_dash] - q[current_state[0]][current_state[1]][action])

        current_state = next_state
        action = action_dash

        step_count += 1

        if terminal_state:
            steps.append(step_count)
            break

# POLICY:
print("Policy: ")
current_state = env.starting_point
while True:
    action = np.argmax(q[current_state[0]][current_state[1]])
    # print(current_state)
    if action == 0:
        print("up")
    elif action == 1:
        print("down")
    elif action == 2:
        print("right")
    else:
        print("left")

    reward, next_state, terminal_state = env.take_action(current_state[0], current_state[1], action)
    current_state = next_state

    if terminal_state:
        break

plt.plot(range(episodes), steps)
plt.xlabel("Episodes")
plt.ylabel("Steps taken to complete the episode")
plt.title("Steps taken per episode")
plt.show()
