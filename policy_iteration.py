"""
This file demonstrates Policy Iteration on the GridWorld problem
Falls under: Model-based algorithms
Are state transitions stochastic: No (deterministic upon taking an action)
Initial policy: Uniform random( other policies can be added later)
Initial Value function: All 0s
Reward function: -1 for every step taken( other reward functions can be added later). Also, the rewards are dependent
                 only on the actions, not on the states
Terminal state: Top left corner and bottom right. Other terminal states can be added by hard-coding them in the
                   following 4 places in the code:
                   TODO: 1)
                   TODO: 2)
                   TODO: 3)
                   TODO: 4)
                   Side note: Hard-coding is certainly not good design , but it's sufficient to demonstrate this simple
                   problem
"""

import numpy as np

print("Policy iteration: GridWorld : Top-left and bottom-right corners are the terminal state")

n = int(input("No of rows"))
m = int(input("No of cols"))

"""
This function is used to modify the reward function.
A simple reward function is -1 reward for any direction chosen.
Modify the values from -1 to any other value for custom rewards for the directions
"""


def get_reward_function():
    # Note: The actions are assumed to be in the format: [up, down, right, left]
    # modify the rewards if needed
    reward = list()
    reward.append(-1)  # up
    reward.append(-1)  # down
    reward.append(-1)  # right
    reward.append(-1)  # left

    reward = np.array(reward)
    return reward


"""
This function is used to calculate the expected value function
given the set of rewards for every action, reward function and a policy.
To account for the first iteration( our initial policy), the default policy is uniform random.
"""


def get_expected_value_fnc(values, reward, policy=None):
    if policy is None:  # policy is uniform random
        pdf = 0.25 * np.ones((4, ))
        expected_value = pdf * (reward + values)
        return np.sum(expected_value)


"""
This function gives the value of adjacent states for a given state
"""


def get_adjacent_indices(i, j, n, m):
    positions = list()

    # up
    positions.append([i, j]) if i == 0 else positions.append([i - 1, j])

    # down
    positions.append([i, j]) if i == (n - 1) else positions.append([i + 1, j])

    # right
    positions.append([i, j]) if j == (m - 1) else positions.append([i, j + 1])

    # left
    positions.append([i, j]) if j == 0 else positions.append([i, j - 1])

    positions = np.array(positions)
    return positions


"""
This function compares the new policy and previous policy for every state.
returns: True if there is a change in policy in at least one state
         False if the 2 policies are exactly the same
"""


def has_changed(previous_policy, current_policy):
    for x in range(len(previous_policy)):
        for y in range(len(previous_policy[0])):
            if not set(previous_policy[x][y]) == set(current_policy[x][y]):
                return True
    return False


grid_world = np.zeros((n, m))  # holds value function
rewards = get_reward_function()

action_dictionary = {0: 'up', 1: 'down', 2: 'right', 3: 'left'}
current_policy = []
for i in range(n):
    new_list = []
    for j in range(m):
        if i == 0 and j == 0:  # top-left corner of the grid
            new_list.append([5])
        elif i == (n - 1) and j == (m - 1):  # bottom-right corner of the grid
            new_list.append([5])
        # TODO: 1) add as many terminal states as you want here
        else:
            new_list.append([0, 1, 2, 3])

    current_policy.append(new_list)

new_policy = []
while True:
    new_grid_world = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:  # top-left corner of the grid
                grid_world[i][j] = 0
            elif i == (n - 1) and j == (m - 1):  # bottom-right corner of the grid
                grid_world[i][j] = 0
            # TODO: 2) add as many terminal states as you want here
            else:
                positions = get_adjacent_indices(i, j, n, m)

                values = np.array([grid_world[positions[0][0]][positions[0][1]], grid_world[positions[1][0]][positions[1][1]],
                                   grid_world[positions[2][0]][positions[2][1]], grid_world[positions[3][0]][positions[3][1]]])

                new_grid_world[i][j] = get_expected_value_fnc(values, rewards)

                # now our value function is ready, let's act greedily based on that and form a policy
                # the initial policy is: go in any of the directions( though this is not explicitly mentioned)
                # with iterations, we become more specific with which actions put us in a "better" state

    new_policy = []
    for i in range(n):
        new_list = []
        for j in range(m):
            positions = get_adjacent_indices(i, j, n, m)

            new_values = np.array(
                [new_grid_world[positions[0][0]][positions[0][1]], new_grid_world[positions[1][0]][positions[1][1]],
                 new_grid_world[positions[2][0]][positions[2][1]], new_grid_world[positions[3][0]][positions[3][1]]])

            # print("new values from new gridworld:", new_values, i, j)
            max_pos = np.argwhere(new_values == np.amax(new_values)).flatten()
            # print("max value from new gridworld:", max_pos, i, j)

            if i == 0 and j == 0:  # top-left corner of the grid
                new_list.append([5])  # terminal states append 5
            elif i == (n - 1) and j == (m - 1):  # bottom-right corner of the grid
                new_list.append([5])  # terminal states append 5
            # TODO: 3) add as many terminal states as you want here
            else:
                g = []
                for t in range(max_pos.shape[0]):
                    g.append(max_pos[t])
                new_list.append(g)

        new_policy.append(new_list)

    # we stop evaluating the value function when we narrow down to one action for every state
    # beyond this iteration, there is no point in improving the value function as the policy has converged.
    # OR IN A GENERIC CASE
    # when we see that the policy remains the same over 2 consecutive iterations.

    # print("current_policy:", current_policy)
    # print("new_policy:", new_policy)

    if not has_changed(current_policy, new_policy):
        break

    grid_world = new_grid_world
    current_policy = new_policy

    # print(grid_world)
    # print("-----------")

print("Grid world policy function: ")
print(grid_world)

print("\n Policy: ")
# POLICY
policy = []
for i in range(n):
    row_list = []
    for j in range(m):
        positions = get_adjacent_indices(i, j, n, m)

        values = np.array([grid_world[positions[0][0]][positions[0][1]], grid_world[positions[1][0]][positions[1][1]],
                           grid_world[positions[2][0]][positions[2][1]], grid_world[positions[3][0]][positions[3][1]]])

        max_pos = np.argwhere(values == np.amax(values)).flatten()

        if i == 0 and j == 0:  # top-left
            row_list.append("X")
        elif i == (n - 1) and j == (m - 1):  # bottom-right
            row_list.append("X")
        # TODO: 4) add as many terminal states as you want here
        else:
            if max_pos.shape[0] == 1:  # policy has converged for this state
                row_list.append(action_dictionary[max_pos[0]])

            else:
                g = []
                for t in range(max_pos.shape[0]):
                    g.append(action_dictionary[max_pos[t]] + "/")
                row_list.append(g)

    policy.append(row_list)
    print(row_list)

