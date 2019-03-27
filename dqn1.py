#!/usr/bin/python
# -*- coding: utf-8 -*-

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import namedtuple
import random
import time, pprint


class QNet(nn.Module):
    def __init__(self, observation_space, num_actions = 1):
        super().__init__()
        num_observations = np.prod(observation_space.shape)
        in_neurons = num_actions + num_observations
        self.fc1 = nn.Linear(in_neurons, 30)
        self.fc2 = nn.Linear(30, 1)   # output Q value

    def forward(self, x):
        x = torch.from_numpy(x.astype(np.float32))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class RecordBuffer(object):
    def __init__(self, buffer_size = 1000):
        self.record = namedtuple('record', 'observation action next_observation reward')
        self.buffer_size = buffer_size
        self.buffer_write_index = 0
        self.buffer = []

    def save(self, *record_elements):
        # transition: observation, action -> observation(t+1), reward
        record = self.record(*record_elements)
        if not self.is_full():
            self.buffer.append(record)
        else:
            self.buffer[self.buffer_write_index] = record
            self.buffer_write_index = (self.buffer_write_index + 1) % self.buffer_size

    def sample(self, batch_size):
        assert batch_size < len(self.buffer), 'invalid batch_size'
        records = random.sample(self.buffer, batch_size)
        return records

    def is_full(self):
        return len(self.buffer) >= self.buffer_size
        
    def show(self):
        pprint.pprint(self.buffer)


class DQN(object):
    def __init__(self, game, gamma = 0.99, greedy_max = 0.9):
        self.env = gym.make(game)
        self.env = self.env.unwrapped
        print('action space:', self.env.action_space)
        print('observation space:', self.env.observation_space)
        self.actions = list(range(self.env.action_space.n))
        self.qnet_train = QNet(self.env.observation_space)
        self.qnet_target = QNet(self.env.observation_space)
        self.gamma = gamma
        self.greedy_max = greedy_max
        self.greedy = 0
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.qnet_train.parameters(), lr = 0.01)
        self.record_buffer = RecordBuffer(buffer_size = 1000)

    def train(self, epoches):
        print('start to train...')
        train_start = False
        num_updates = 0
        for epoch in range(epoches):
            actions_history = []
            observation = self.env.reset()
            done = False
            num_steps = 0
            rewards = 0
            while not done:
                action = self.choose_action(observation)
                actions_history.append(action)
                next_observation, reward, done, info = self.env.step(action)
                x, x_dot, theta, theta_dot = next_observation
                r1 = (self.env.x_threshold - abs(x))/self.env.x_threshold - 0.8
                r2 = (self.env.theta_threshold_radians - abs(theta))/self.env.theta_threshold_radians - 0.5
                reward = r1 + r2

                self.record_buffer.save(observation, action, next_observation, reward)

                if self.record_buffer.is_full():
                    if not train_start:
                        train_start = True
                        print('start to train...')
                    records = self.record_buffer.sample(batch_size = 64)
                    q_eval_x = np.array([np.concatenate((ob, np.array([act]))) for ob, act, next, reward in records])
                    num_actions = len(self.actions)
                    q_target_x = np.array([np.concatenate((np.tile(next, (num_actions, 1)), np.array(self.actions).reshape(-1,1)), axis = 1)
                        for ob, act, next, reward in records])
                    rewards_target = np.array([reward for ob, act, next, reward in records]).astype(np.float32)

                    # Q learning
                    # Q(state, action) = reward + gamma * max(Q(state+1, action_space))
                    q_eval = self.qnet_train(q_eval_x)
                    with torch.no_grad():
                        q_target = torch.max(self.qnet_target(q_target_x).view(-1, 2)) * self.gamma + torch.from_numpy(rewards_target)

                    loss = self.criterion(q_eval, q_target)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    num_updates += 1
                    self.greedy = self.greedy + 0.01 if self.greedy < self.greedy_max else self.greedy_max
                num_steps += 1
                rewards += reward
                observation = next_observation
            print('epoch {}: steps {}, rewards: {}, actions: {}'.format(epoch, num_steps, rewards, actions_history))
            if num_updates > 50:
                self.qnet_target.load_state_dict(self.qnet_train.state_dict())
                print('update target QNet')
                num_updates = 0

    def choose_action(self, observation, optimal = False):
        explore = np.random.rand() > self.greedy and not optimal
        if explore:
            action = self.env.action_space.sample()
        else:
            num_actions = len(self.actions)
            q_target_x = np.concatenate((np.tile(observation, (num_actions, 1)), np.array(self.actions).reshape(-1,1)), axis = 1)
            action = self.actions[torch.argmax(self.qnet_train(q_target_x)).item()]
        return action

    def run(self, epoches):
        print('start to act by the optimal policy...')
        for epoch in range(epoches):
            observation = self.env.reset()
            actions_history = []
            done = False
            num_steps = 0
            rewards = 0
            while not done:
                #self.env.render()
                action = self.choose_action(observation, optimal = True)
                actions_history.append(action)
                observation, reward, done, info = self.env.step(action)
                num_steps += 1
                rewards += reward
                time.sleep(0.1)
            print('epoch {}: steps {}, rewards: {}, actions: {}'.format(epoch, num_steps, rewards, actions_history))
        self.env.close()

    def save(self, filename = 'qnet_save.txt'):
        torch.save(self.qnet_train.state_dict(), filename)

    def load(self, filename = 'qnet_save.txt'):
        self.qnet_train.load_state_dict(torch.load(filename))


                

if __name__ == '__main__':
    dqn = DQN('CartPole-v0')
    dqn.train(epoches = 1000)
    dqn.save()
    #dqn.load()
    dqn.record_buffer.show()
    dqn.run(epoches = 10)
    print('done')
