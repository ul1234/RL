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
    def __init__(self, num_observations, num_actions):
        super().__init__()
        self.fc1 = nn.Linear(num_observations, 64)
        self.fc2 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class RecordBuffer(object):
    def __init__(self, buffer_size):
        self.record = namedtuple('record', 'observation action next_observation reward')
        self.buffer_size = buffer_size
        self.buffer_write_index = 0
        self.buffer = []

    def save(self, *record_elements):
        # transition: observation, action -> observation(t+1), reward
        record = self.record(*map(np.float32, record_elements))
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
        num_actions = self.env.action_space.n
        num_observations = np.prod(self.env.observation_space.shape)
        self.actions = list(range(num_actions))
        self.qnet_train = QNet(num_observations, num_actions)
        self.qnet_target = QNet(num_observations, num_actions)
        self.gamma = gamma
        self.greedy_max = greedy_max
        self.greedy = 0
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.qnet_train.parameters(), lr = 0.01)
        self.record_buffer = RecordBuffer(buffer_size = 2000)

    def train(self, epoches):
        print('start to train...')
        train_start = False
        num_updates = 0
        for epoch in range(epoches):
            actions_history = []
            loss_history = []
            observation = self.env.reset()
            done = False
            num_steps = 0
            rewards = 0
            while not done:
                #self.env.render()
                action = self.choose_action(observation)
                actions_history.append(action)
                next_observation, reward, done, info = self.env.step(action)
                # change reward
                #x, x_dot, theta, theta_dot = next_observation
                #r1 = (self.env.x_threshold - abs(x))/self.env.x_threshold - 0.8
                #r2 = (self.env.theta_threshold_radians - abs(theta))/self.env.theta_threshold_radians - 0.5
                #reward = r1 + r2

                position, velocity = next_observation
                # the higher the better
                reward = abs(position - (-0.5))     # r in [0, 1]

                self.record_buffer.save(observation, action, next_observation, reward)
                observation = next_observation
                
                if self.record_buffer.is_full():
                    if not train_start:
                        train_start = True
                        self.qnet_target.load_state_dict(self.qnet_train.state_dict())
                        print('start to train...')
                    if num_updates > 100:
                        self.qnet_target.load_state_dict(self.qnet_train.state_dict())
                        #print('update target QNet')
                        num_updates = 0
                    num_updates += 1
                    records = self.record_buffer.sample(batch_size = 32)
                    batch_observation, batch_action, batch_next_observation, batch_reward = map(torch.tensor, zip(*records))
                    # Q learning
                    # Q(state, action) = reward + gamma * max(Q(state+1, action_space))
                    q_eval = self.qnet_train(batch_observation).gather(1, batch_action.reshape(-1, 1).long())
                    #with torch.no_grad():
                    q_target_ = self.qnet_target(batch_next_observation).detach()
                    q_target = (torch.max(q_target_, 1)[0] * self.gamma + batch_reward).view(-1, 1)

                    loss = self.criterion(q_eval, q_target)
                    loss_history.append(loss.item())
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.greedy = self.greedy + 0.001 if self.greedy < self.greedy_max else self.greedy_max
                num_steps += 1
                rewards += reward
            #print('epoch {}: steps {}, greedy: {}, rewards: {}, actions: {}'.format(epoch, num_steps, self.greedy, rewards, actions_history))
            #print('epoch {}: steps {}, rewards: {}, loss: {}'.format(epoch, num_steps, rewards, loss_history))
            print('epoch {}: steps {}, rewards: {}'.format(epoch, num_steps, rewards))

    def choose_action(self, observation, optimal = False):
        explore = np.random.rand() > self.greedy and not optimal
        if explore:
            action = self.env.action_space.sample()
        else:
            action = self.actions[torch.argmax(self.qnet_train(torch.tensor(observation).float())).item()]
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
                self.env.render()
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
    #dqn = DQN('CartPole-v0')
    dqn = DQN('MountainCar-v0')
    
    dqn.train(epoches = 1000)
    dqn.save()
    #dqn.load()
    #dqn.record_buffer.show()
    dqn.run(epoches = 1)
    print('done')
