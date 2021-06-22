#from __future__ import print_function

import random

import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()



shapes = {
    'T': [(0, 0), (-1, 0), (1, 0), (0, -1)],
    'J': [(0, 0), (-1, 0), (0, -1), (0, -2)],
    'L': [(0, 0), (1, 0), (0, -1), (0, -2)],
    'Z': [(0, 0), (-1, 0), (0, -1), (1, -1)],
    'S': [(0, 0), (-1, -1), (0, -1), (1, 0)],
    'I': [(0, 0), (0, -1), (0, -2), (0, -3)],
    'O': [(0, 0), (0, -1), (-1, 0), (-1, -1)],
}
shape_names = ['T', 'J', 'L', 'Z', 'S', 'I', 'O']


def rotated(shape, cclk=False):
    if cclk:
        return [(-j, i) for i, j in shape]
    else:
        return [(j, -i) for i, j in shape]


def is_occupied(shape, anchor, board):
    for i, j in shape:
        x, y = anchor[0] + i, anchor[1] + j
        # if y < 0:
        #     continue
        # Fix python indexing that hurts use here really hard.
        y = max(y, 0)
        if x < 0 or x >= board.shape[0] or y >= board.shape[1] or board[x, y]:
            return True
    return False


def left(shape, anchor, board):
    new_anchor = (anchor[0] - 1, anchor[1])
    return (shape, anchor) if is_occupied(shape, new_anchor, board) else (shape, new_anchor)


def right(shape, anchor, board):
    new_anchor = (anchor[0] + 1, anchor[1])
    return (shape, anchor) if is_occupied(shape, new_anchor, board) else (shape, new_anchor)


def soft_drop(shape, anchor, board):
    new_anchor = (anchor[0], anchor[1] + 1)
    return (shape, anchor) if is_occupied(shape, new_anchor, board) else (shape, new_anchor)


def hard_drop(shape, anchor, board):
    while True:
        _, anchor_new = soft_drop(shape, anchor, board)
        if anchor_new == anchor:
            return shape, anchor_new
        anchor = anchor_new


def rotate_left(shape, anchor, board):
    new_shape = rotated(shape, cclk=False)
    return (shape, anchor) if is_occupied(new_shape, anchor, board) else (new_shape, anchor)


def rotate_right(shape, anchor, board):
    new_shape = rotated(shape, cclk=True)
    return (shape, anchor) if is_occupied(new_shape, anchor, board) else (new_shape, anchor)


def idle(shape, anchor, board):
    return (shape, anchor)

ROTATION_ACTION_COUNT = 4

def split_action(action: int):
    rotation = action % ROTATION_ACTION_COUNT
    column = action // ROTATION_ACTION_COUNT
    return rotation, column

class TetrisEngine(py_environment.PyEnvironment):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.board = np.zeros(shape=(width, height), dtype=np.float)
        # We have 4 rotations and width many columns where a tetromino can be placed.
        # Thus a one dimensional action space goes from 0 to (4 * width) - 1
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=(4 * width) - 1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(5,), dtype=np.int32, minimum=[0, 0, 0, 0, 0],
            maximum=[(width*height)+1, (height * (width-1))+1, height + 1, int(width * height/2) + 1, 7], name='observation')
            
        self._state = np.zeros(5)

        # actions are triggered by letters
        self.value_action_map = {
            0: left,
            1: right,
            2: hard_drop,
            3: soft_drop,
            4: rotate_left,
            5: rotate_right,
            6: idle,
        }
        self.action_value_map = dict(
            [(j, i) for i, j in self.value_action_map.items()])
        self.nb_actions = len(self.value_action_map)

        # for running the engine
        self.time = -1
        self.score = -1
        self.anchor = None
        self.shape = None
        self.tetromino = None
        self.n_deaths = 0
        self.total_reward = 0

        # used for generating shapes
        self._shape_counts = [0] * len(shapes)

        # clear after initializing
        self.clear()

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _choose_shape(self):
        maxm = max(self._shape_counts)
        m = [5 + maxm - x for x in self._shape_counts]
        r = random.randint(1, sum(m))
        for i, n in enumerate(m):
            r -= n
            if r <= 0:
                self._shape_counts[i] += 1
                self.tetromino = i
                return shapes[shape_names[i]]

    def _new_piece(self):
        # Place randomly on x-axis with 2 tiles padding
        #x = int((self.width/2+1) * np.random.rand(1,1)[0,0]) + 2
        self.anchor = (int(self.width / 2), 0)
        #self.anchor = (x, 0)
        self.shape = self._choose_shape()

    def _has_dropped(self):
        return is_occupied(self.shape, (self.anchor[0], self.anchor[1] + 1), self.board)

    def _clear_lines(self):
        can_clear = [np.all(self.board[:, i]) for i in range(self.height)]
        new_board = np.zeros_like(self.board)
        j = self.height - 1
        for i in range(self.height - 1, -1, -1):
            if not can_clear[i]:
                new_board[:, j] = self.board[:, i]
                j -= 1
        self.score += sum(can_clear)
        self.board = new_board

        return sum(can_clear)

    def count_valid_actions(self):
        valid_action_sum = 0

        for value, fn in self.value_action_map.items():
            # If they're equal, it is not a valid action
            if fn(self.shape, self.anchor, self.board) != (self.shape, self.anchor):
                valid_action_sum += 1

        return valid_action_sum

    def _step(self, action):
        # rotation: 0 - don't rotate, 1..3 - rotate n times left
        # first rotate to the correct orientation
        rotation, target_column = split_action(action)
        if rotation == 1:
            self.shape, self.anchor = rotate_left(
                self.shape, self.anchor, self.board)
        elif rotation == 2:
            self.shape, self.anchor = rotate_left(
                self.shape, self.anchor, self.board)
            self.shape, self.anchor = rotate_left(
                self.shape, self.anchor, self.board)
        elif rotation == 3:
            self.shape, self.anchor = rotate_right(
                self.shape, self.anchor, self.board)
        # Then move to the desired column
        diff = self.anchor[0] - target_column
        if diff < 0:
            for _ in range(abs(diff)):
                self.shape, self.anchor = right(
                    self.shape, self.anchor, self.board)
        elif diff > 0:
            for _ in range(diff):
                self.shape, self.anchor = left(
                    self.shape, self.anchor, self.board)
        # Hopefully this position is valid :D
        reward, done = self.step2(2)
        self._state = self.get_all_features()
        self.total_reward += reward
        if done:
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward, discount=1.0)
        # return features, reward, done, {}

    # def multi_step(self, actions):
    #     full_reward = 0
    #     done = False
    #     for action in actions:
    #         _, reward, done = self.step(action)
    #         full_reward += reward
    #     # Do the hard drop
    #     state, reward, done = self.step(2)
    #     full_reward += reward
    #     return state, full_reward, done

    def step2(self, action):
        self.anchor = (int(self.anchor[0]), int(self.anchor[1]))
        self.shape, self.anchor = self.value_action_map[action](
            self.shape, self.anchor, self.board)
        # Don't soft drop automatically
        # self.shape, self.anchor = soft_drop(self.shape, self.anchor, self.board)

        # Update time and reward
        self.time += 1
        # What a useless call :O
        # reward = self.count_valid_actions()
        #reward = random.randint(0, 0)
        reward = 1

        done = False
        if self._has_dropped():
            self._set_piece(True)
            reward += 100 * self._clear_lines()
            if np.any(self.board[:, 0]):
                self.clear()
                self.n_deaths += 1
                done = True
                reward = -100
                self.reset()
            else:
                self._new_piece()

        # self._set_piece(True)
        # state = np.copy(self.board)
        # self._set_piece(False)
        return reward, done

    def _reset(self):
        self.clear()
        return ts.restart(self._state)


    def clear(self):
        self.time = 0
        self.score = 0
        self.total_reward = 0
        self._new_piece()
        self.board = np.zeros_like(self.board)
        features = self.get_all_features()
        self._state = features
        return features

    ###### Feature Section #######
    def get_all_features(self):
        features = np.zeros(5, dtype=np.int32)
        self._set_piece(False)
        features[0] = self.get_aggregated_height()
        features[1] = self.get_bumpiness()
        features[2] = self.completed_lines()
        features[3] = self.get_hole_count()
        features[4] = self.tetromino
        # self._set_piece(True)
        return features

    def get_bumpiness(self):
        bumpiness = 0
        heights = np.zeros(self.width)
        for column in range(self.width):
            for row in range(self.height):
                if self.board[column, row]:
                    heights[column] = self.height - row
                    break
        for column in range(self.width - 1):
            bumpiness += abs(heights[column] - heights[column + 1])
        return bumpiness

    def completed_lines(self):
        return np.count_nonzero(np.array([np.all(self.board[:, i]) for i in range(self.height)]))

    def get_aggregated_height(self):
        aggregated_height = 0
        for column in range(self.width):
            for row in range(self.height):
                if self.board[column, row]:
                    aggregated_height += self.height - row
                    break
        return aggregated_height

    def get_hole_count(self):
        holes = 0
        for column in range(self.width):
            found_occupied_piece = False
            for row in range(self.height):
                if self.board[column, row] and not found_occupied_piece:
                    found_occupied_piece = True
                elif found_occupied_piece and not self.board[column, row]:
                    holes += 1
        return holes

    def _set_piece(self, on=False):
        for x_in_shape_space, y_in_shape_space in self.shape:
            x_in_board_space, y_in_board_space = x_in_shape_space + \
                self.anchor[0], y_in_shape_space + self.anchor[1]
            if x_in_board_space < self.width and x_in_board_space >= 0 and y_in_board_space < self.height and y_in_board_space >= 0:
                self.board[x_in_board_space, y_in_board_space] = on

    def __repr__(self):
        self._set_piece(True)
        s = 'o' + '-' * self.width + 'o\n'
        s += '\n'.join(['|' + ''.join(['X' if j else ' ' for j in i]
                                      ) + '|' for i in self.board.T])
        s += '\no' + '-' * self.width + 'o\n'
        self._set_piece(False)
        s += 'Current Reward: ' + str(self.total_reward) + "\n"
        return s
