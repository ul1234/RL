#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pprint

class Env(object):
    def __init__(self, rows, cols, X_pos, target_pos):
        self.state = np.zeros((rows, cols))
        self.valid_pos = [(row, col) for row in range(rows) for col in range(cols)]
        self.action_space = ['down', 'up', 'left', 'right']
        self.X_pos = X_pos if isinstance(X_pos, list) else [X_pos]
        self.target_pos = target_pos if isinstance(target_pos, list) else [target_pos]
        self.assist_init = []
        self.reset()

    def reset(self):
        self.cur_pos = (0, 0)
        self.assist_pos = self.assist_init[:]

    def set_assist_pos(self, assist_pos):
        self.assist_init = assist_pos if isinstance(assist_pos, list) else [assist_pos]
        self.reset()

    def _is_pos(self, pos, pos_list):
        return pos in pos_list

    def get_action_space(self):
        return self.action_space

    def observation(self):
        return self.cur_pos

    def interact(self, action):
        if action == 'down':
            pos = (self.cur_pos[0] + 1, self.cur_pos[1])
        elif action == 'up':
            pos = (self.cur_pos[0] - 1, self.cur_pos[1])
        elif action == 'left':
            pos = (self.cur_pos[0], self.cur_pos[1] - 1)
        elif action == 'right':
            pos = (self.cur_pos[0], self.cur_pos[1] + 1)
        reward = -1
        done = False
        win = False
        if self._is_pos(pos, self.valid_pos):
            if self._is_pos(pos, self.X_pos):
                reward = -10
                done = True
            elif self._is_pos(pos, self.target_pos):
                reward = 10
                done = True
                win = True
            elif self._is_pos(pos, self.assist_pos):
                self.assist_pos.remove(pos)
                #print('touch assist pos')
                reward = 1
            self.cur_pos = pos
        observation = self.cur_pos
        return (observation, reward, done, win)

    def render(self):
        pass

class RL(object):
    def __init__(self, env, greedy = 0.1, gamma = 0.9, learning_rate = 0.1):
        # q_table:  {state1: {action1: value, action2: value}, state2: {..}}
        self.q_table = {}
        self.env = env
        self.action_space = env.get_action_space()
        self.greedy = greedy
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.max_steps = 100

    def choose_action(self, state):
        explore = np.random.rand() < self.greedy
        np.random.shuffle(self.action_space)
        if explore:
            action = self.action_space[0]
        else:
            action_q_value_pair = list(self.q_table[state].items())
            action_idx = np.argmax([q for action, q in action_q_value_pair])
            action = action_q_value_pair[action_idx][0]
        return action

    def check_q_table(self, state):
        if not state in self.q_table: self.q_table[state] = dict([(action, 0) for action in self.action_space])
        
    def get_max_q_value(self, state):
        self.check_q_table(state)
        return max([q for action, q in self.q_table[state].items()])
        
    def update_q_value(self, state, action, q_value):
        self.q_table[state][action] = q_value
        
    def run_epoch(self, epoch = 0, log = False):
        env.reset()
        self.cur_state = env.observation()
        self.check_q_table(self.cur_state)
        done = False
        accum_reward = 0
        num_steps = 0
        action_history = []
        while not done and num_steps < self.max_steps:
            action = self.choose_action(self.cur_state)
            q_value = self.q_table[self.cur_state][action]
            action_history.append(action)
            state, reward, done, win = env.interact(action)
            q_target = reward + self.gamma * self.get_max_q_value(state)
            q_value = self.learning_rate * q_target + (1-self.learning_rate) * q_value
            self.update_q_value(self.cur_state, action, q_value)
            self.cur_state = state
            accum_reward += reward
            num_steps += 1
        if log:
            print('epoch: {} -- win: {}, num_steps: {}, accum rewards: {}, action: {}'.format(
                epoch, win, num_steps, accum_reward, action_history))
        else:
            print('epoch: {} -- win: {}, num_steps: {}, accum rewards: {}'.format(
                epoch, win, num_steps, accum_reward))

    def train(self, epoches = 30):
        for epoch in range(epoches):
            self.run_epoch(epoch = epoch)
        #pprint.pprint(self.q_table)

    def run(self):
        print('\n\nstart to follow optimal policy...\n')
        self.greedy = 0
        self.run_epoch(log = True)

            
if __name__ == "__main__":
    #env = Env(3, 4, X_pos = [(0, 1), (1, 1)], target_pos = (1, 2))
    env = Env(6, 6, X_pos = [(0, 2), (1, 3), (2, 4), (3, 1), (3, 4), (4, 3)], target_pos = (0, 3))
    #env = Env(6, 6, X_pos = [(0, 2), (1, 3), (3, 1), (3, 4), (4, 3)], target_pos = (0, 3))
    #env.set_assist_pos([(5, 3), (4,5)])
    rl = RL(env, greedy = 0.1, gamma = 0.9, learning_rate = 0.5)
    rl.train(50)
    rl.run()
    #input('press any key to continue...')

