#!/usr/bin/python
# -*- coding: utf-8 -*-

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class QNet(nn.Module):
    def __init__(self, observation_space, num_actions = 1):
        super().__init__()
        num_observations = np.prod(observation_space.shape)
        in_neurons = num_actions + num_observations
        self.fc1 = nn.Linear(in_neurons, 30)
        self.fc2 = nn.Linear(30, 1)   # output Q value

    def forward(self, observation, actions):
        num_actions = len(actions) if isinstance(actions, list) else 1
        x = np.concatenate((np.tile(np.array(observation), (num_actions, 1)), np.array(actions).reshape(-1,1)), axis = 1)
        x = torch.from_numpy(x.astype(np.float32))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DQN(object):
    def __init__(self, game, gamma = 0.9, greedy = 0.1):
        self.env = gym.make(game)
        print('action space:', self.env.action_space)
        print('observation space:', self.env.observation_space)
        self.actions = list(range(self.env.action_space.n))
        self.qnet_train = QNet(self.env.observation_space)
        self.qnet_target = QNet(self.env.observation_space)
        self.gamma = gamma
        self.greedy = greedy
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.qnet_train.parameters(), lr = 0.01)

    def train(self, epoches):
        print('start to train...')
        num_updates = 0
        for epoch in range(epoches):
            observation = self.env.reset()
            done = False
            num_steps = 0
            rewards = 0
            while not done:
                action = self.choose_action(observation)
                q_eval = self.qnet_train(observation, action)
                observation, reward, done, info = self.env.step(action)
                with torch.no_grad():
                    q_target = torch.max(self.qnet_target(observation, self.actions))
                # Q learning
                # Q(state, action) = reward + gamma * max(Q(state+1, action_space))
                loss = self.criterion(q_eval, q_target * self.gamma + reward)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                num_updates += 1
                num_steps += 1
                rewards += reward
            print('epoch {}: steps {}, rewards: {}'.format(epoch, num_steps, rewards))
            if num_updates > 10:
                self.qnet_target.load_state_dict(self.qnet_train.state_dict())
                print('update target QNet')
                num_updates = 0

    def choose_action(self, observation):
        explore = np.random.rand() < self.greedy
        if explore:
            action = self.env.action_space.sample()
        else:
            action = self.actions[torch.argmax(self.qnet_train(observation, self.actions)).item()]
        return action

    def run(self, epoches):
        self.greedy = 0
        print('start to act by the optimal policy...')
        for epoch in range(epoches):
            observation = self.env.reset()
            done = False
            num_updates = 0
            while not done:
                action = self.choose_action(observation)
                observation, reward, done, info = self.env.step(action)
            print('epoch {}: steps {}, rewards: {}'.format(epoch, num_steps, rewards))


                

if __name__ == '__main__':
    dqn = DQN('CartPole-v0')
    dqn.train(epoches = 1000)
    dqn.run(epoches = 10)
    print('done')
