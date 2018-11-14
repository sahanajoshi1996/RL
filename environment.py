"""
This file emulates the environment behavior for different scenarios.
The grid is an m x m grid, m is either 8 or 4. Certain columns of this grid world push the agent up by some offset
upon entering this column. The agent needs to learn this and move accordingly.
"""
import numpy as np


class GridWorld8x8:
    def __init__(self):
        self.m = 8
        self.n = 8
        self.grid = np.zeros((self.m, self.n))  # making grid
        self.starting_point = [3, 0]  # defining starting state
        self.terminal_state = [4, 5]  # defining terminal state

        # defining windy behavior
        self.push_up = {0: 0, 1: 1, 2: 1, 3: 2, 4: 1, 5: 2, 6: 0, 7: 0}

        # defining actions
        self.actions = {0: "up", 1: "down", 2: "right", 3: "left"}

    """given are the current state(m, n) and the 
    action taken(0, 1, 2, 3) to return: reward and next state and is_terminal"""
    def take_action(self, m, n, action):
        # reward: if the terminal state is reached, return 100.
        # for all other states, return 0
        # we can modify this reward function
        reward = 0

        # next state:
        next_state = []
        if action == 0:  # up
            next_state.append(max(0, m - 1))
            next_state.append(n)

            if next_state[0] == self.terminal_state[0] and next_state[1] == self.terminal_state[1]:
                reward = 100
                return reward, next_state, True

            else:
                return reward, next_state, False

        elif action == 1:  # down
            next_state.append(min(self.m - 1, m + 1))
            next_state.append(n)

            if next_state[0] == self.terminal_state[0] and next_state[1] == self.terminal_state[1]:
                reward = 100
                return reward, next_state, True

            else:
                return reward, next_state, False

        elif action == 2:  # right : this action is affected by the windy nature
            next_state.append(max(0, m - self.push_up[min(self.n - 1, n + 1)]))
            next_state.append(min(self.n - 1, n + 1))

            if next_state[0] == self.terminal_state[0] and next_state[1] == self.terminal_state[1]:
                reward = 100
                return reward, next_state, True

            else:
                return reward, next_state, False

        elif action == 3:  # left : this action is affected by the windy nature
            next_state.append(max(0, m - self.push_up[max(0, n - 1)]))
            next_state.append(max(0, n - 1))

            if next_state[0] == self.terminal_state[0] and next_state[1] == self.terminal_state[1]:
                reward = 100
                return reward, next_state, True

            else:
                return reward, next_state, False


class GridWorld4x4:
    def __init__(self):
        self.m = 4
        self.n = 4
        self.grid = np.zeros((self.m, self.n))  # making grid
        self.starting_point = [1, 0]  # defining starting state
        self.terminal_state = [2, 2]  # defining terminal state

        # defining windy behavior
        self.push_up = {0: 0, 1: 1, 2: 1, 3: 0}

        # defining actions
        self.actions = {0: "up", 1: "down", 2: "right", 3: "left"}

    """given are the current state(m, n) and the 
    action taken(0, 1, 2, 3) to return: reward and next state and is_terminal
    """
    def take_action(self, m, n, action):
        # reward: if the terminal state is reached, return 100.
        # for all other states, return 0
        reward = 0

        # next state:
        next_state = []
        if action == 0:  # up
            next_state.append(max(0, m - 1))
            next_state.append(n)

            if next_state[0] == self.terminal_state[0] and next_state[1] == self.terminal_state[1]:
                reward = 100
                return reward, next_state, True

            else:
                return reward, next_state, False

        elif action == 1:  # down
            next_state.append(min(self.m - 1, m + 1))
            next_state.append(n)

            if next_state[0] == self.terminal_state[0] and next_state[1] == self.terminal_state[1]:
                reward = 100
                return reward, next_state, True

            else:
                return reward, next_state, False

        elif action == 2:  # right : this action is affected by the windy nature
            next_state.append(max(0, m - self.push_up[min(self.n - 1, n + 1)]))
            next_state.append(min(self.n - 1, n + 1))

            if next_state[0] == self.terminal_state[0] and next_state[1] == self.terminal_state[1]:
                reward = 100
                return reward, next_state, True

            else:
                return reward, next_state, False

        elif action == 3:  # left : this action is affected by the windy nature
            next_state.append(max(0, m - self.push_up[max(0, n - 1)]))
            next_state.append(max(0, n - 1))

            if next_state[0] == self.terminal_state[0] and next_state[1] == self.terminal_state[1]:
                reward = 100
                return reward, next_state, True

            else:
                return reward, next_state, False


"""
This is the classic Tic-Tac-Toe game.
The agent( the one to be trained) plays "X". The opponent plays "O".
The reward function is simple( rather naive): 100 if the agent wins and -100 if the opponent wins, 0 for every 
other case
"""


class TicTacToe:
    def __init__(self):
        self.tic_tac_toe = np.zeros(shape=(3, 3))
        self.available_slots = [[i, j] for i in range(3) for j in range(3)]

    def play(self, i, j, item):
        for t in range(len(self.available_slots)):
            if self.available_slots[t][0] == i and self.available_slots[t][1] == j:
                del(self.available_slots[t])
                break

        if item == 'O':  # opponent's play
            self.tic_tac_toe[i][j] = 1

            return self.is_game_over(i, j, item)

        else:  # agent's play
            self.tic_tac_toe[i][j] = 4

            return self.is_game_over(i, j, item)

    def get_available_slots(self):
        return self.available_slots

    def is_game_over(self, i, j, item):
        # row
        if self.tic_tac_toe[i][j] == self.tic_tac_toe[i][(j + 1) % 3] == self.tic_tac_toe[i][(j + 2) % 3]:
            return True, 100 if item == 'X' else -100

        # column
        if self.tic_tac_toe[i][j] == self.tic_tac_toe[(i + 1) % 3][j] == self.tic_tac_toe[(i + 2) % 3][j]:
            return True, 100 if item == 'X' else -100

        # diagonals
        if self.tic_tac_toe[0][0] == self.tic_tac_toe[1][1] == self.tic_tac_toe[2][2] != 0:
            return True, 100 if item == 'X' else -100

        if self.tic_tac_toe[0][2] == self.tic_tac_toe[1][1] == self.tic_tac_toe[2][0] != 0:
            return True, 100 if item == 'X' else -100

        # if all slots are done
        if len(self.available_slots) == 0:
            return True, 0

        return False, 0

    def clear_game(self):
        self.tic_tac_toe = np.zeros(shape=(3, 3))
        self.available_slots = [[i, j] for i in range(3) for j in range(3)]


"""
"""