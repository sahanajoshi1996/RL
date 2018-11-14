"""
This file demonstrates Value Iteration on the GridWorld problem
Falls under: Model-based algorithms
Are state transitions stochastic: No (deterministic upon taking an action)
Initial policy: Uniform random( other policies can be added later)
Initial Value function: All 0s
Reward function: -1 for every step taken( other reward functions can be added later). Also, the rewards are dependent
                 only on the actions, not on the states
Terminal state(s): Top left corner and bottom right. Other terminal states can be added by harding them in the
                   following 2 places in the code:
                   TODO: 1)
                   TODO: 2)

Note how the policy we arrive at is sensitive to the convergence value and the size of the grid. As the size of the grid
increases, the convergence value needs to be made smaller to arrive at the true optimal policy. Otherwise, the
iterations end at a stage where there is more than one optimal action for some state( one of them might not actually be
optimal, its just that we converged, rather, stopped evaluating too early to realize that it wasn't an optimal action
for that state)
"""

import numpy as np

print("Value iteration: GridWorld : Top-left and bottom-right corners are the terminal state")

n = int(input("No of rows"))
m = int(input("No of cols"))

threshold = float(input("Threshold for value function convergence"))

"""
This function is used to modify the reward function.
A simple reward function is -1 reward for any direction chosen.
Modify the values from -1 to any other value for custom rewards for the directions
"""


def get_reward_function():
    # Note: The actions are assumed to be in the format: [up, down, right, left]
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


def get_optimal_value_fnc(values, reward, policy=None):
    if policy is None:  # policy is uniform random
        pdf = 0.25 * np.ones((4, ))
        expected_value = pdf * (reward + values)
        return np.max(expected_value)  # the optimal value function for this state is the max. of all the values of
        #  states we might end up in from this state


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


grid_world = np.zeros((n, m))  # holds value function
rewards = get_reward_function()

# evaluate the value function ( 1st iteration)
for i in range(n):
    for j in range(m):
        if i == 0 and j == 0:  # top-left corner of the grid
            grid_world[i][j] = 0

        else:
            positions = get_adjacent_indices(i, j, n, m)

            values = np.array(
                [grid_world[positions[0][0]][positions[0][1]], grid_world[positions[1][0]][positions[1][1]],
                 grid_world[positions[2][0]][positions[2][1]], grid_world[positions[3][0]][positions[3][1]]])

            grid_world[i][j] = get_optimal_value_fnc(values, rewards)

new_grid_world = np.zeros((n, m))

diff = grid_world - new_grid_world
mse = np.sqrt(np.sum(np.square(diff)))

while mse > threshold:
    new_grid_world = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:  # top-left corner of the grid
                grid_world[i][j] = 0
            elif i == (n - 1) and j == (m - 1):  # bottom-right corner of the grid
                grid_world[i][j] = 0
            # TODO: 1) add as many terminal states as you want here
            else:
                positions = get_adjacent_indices(i, j, n, m)

                values = np.array([grid_world[positions[0][0]][positions[0][1]], grid_world[positions[1][0]][positions[1][1]],
                                   grid_world[positions[2][0]][positions[2][1]], grid_world[positions[3][0]][positions[3][1]]])

                new_grid_world[i][j] = get_optimal_value_fnc(values, rewards)

    mse = np.sqrt(np.sum(np.square(grid_world - new_grid_world)))
    grid_world = new_grid_world

    # now our value function is ready, let's act greedily based on that and form a policy
    # the initial policy is: go in any of the directions( though this is not explicitly mentioned)
    # with iterations, we become more specific with which actions put us in a "better" state
    # we stop evaluating the value function when we narrow down to one action for every state
    # beyond this iteration, there is no point in improving the value function as the policy has converged.

    # print(grid_world)

print("Value function has converged \n")

print("Policy: ")
action_dictionary = {0: 'up', 1: 'down', 2: 'right', 3: 'left'}
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
        # TODO: 2) add as many terminal states as you want here
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

# print(policy)

print("\n\nDo note that for some states, the policy might not have converged due to a high threshold, or there might"
      " be more than one optimal action for that state")
