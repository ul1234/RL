#!/usr/bin/python
# -*- coding: utf-8 -*-

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from collections import namedtuple
import random
import time, pprint
import matplotlib.pyplot as plt

####### policy gradient ##############

class Linear(object):
    def __init__(self, num_weights):
        self.num_weights = num_weights

    def random_init(self):
        self.weights = np.random.randn(self.num_weights+1, 1)  # bias

    def random_hill(self, rate = 0.1):
        self.weights += rate * np.random.randn(self.num_weights+1, 1)  # bias

    def forward(self, data_in):
        data_out = np.dot(np.append(data_in, 1), self.weights)
        return data_out

    def keep(self):
        self.keep_weights = self.weights

    def restore(self):
        self.weights = self.keep_weights

    def __call__(self, data_in):
        return self.forward(data_in)

class Agent(object):
    def __init__(self, observation_space, action_space):
        self.action_space = action_space
        self.actions = list(range(action_space.n))
        num_observations = np.prod(observation_space.shape)

        self.linear = Linear(num_observations)

        self.hill_climbing = True
        self.num_learns = 0
        self.rewards = 0

    def new_episode(self, optimal = False):
        if optimal:
            self.linear.restore()
        else:
            self.linear.random_init()

    def choose_action(self, observation):
        output = self.linear(observation)
        action = 0 if output > 0 else 1
        return action

    def learn(self, rewards):
        if rewards > self.rewards:
            self.rewards = rewards
            self.linear.keep()
            self.num_learns += 1

    def save(self, filename = 'qnet_save.txt'):
        print(self.linear.keep_weights)

    def load(self, filename = 'qnet_save.txt'):
        pass


class Game(object):
    def __init__(self, game_name):
        self.env = gym.make(game_name)
        #self.env.seed(1)
        print('action space:', self.env.action_space)
        print('observation space:', self.env.observation_space)
        self.agent = Agent(self.env.observation_space, self.env.action_space)
        self.resolved = getattr(self, 'resolved_{}'.format(game_name.split('-')[0]))

    def resolved_CartPole(self, scores):
        if len(scores) >= 100 and np.mean(scores[-100:]) >= 195.0:
            print('Solved after {} episodes'.format(len(scores)-100))
            return True
        return False

    def run_one_episode(self, optimal = False, render = False):
        observation = self.env.reset()
        done = False
        num_steps = 0
        rewards = 0
        while not done:
            if render: self.env.render()
            action = self.agent.choose_action(observation)
            next_observation, reward, done, _ = self.env.step(action)
            observation = next_observation
            num_steps += 1
            rewards += reward
        return rewards, num_steps

    def run(self, episodes, render = False):
        optimal = False
        scores = []
        for episode in range(episodes):
            self.agent.new_episode(optimal)
            rewards, num_steps = self.run_one_episode(render)
            self.agent.learn(rewards)
            if rewards == 200: optimal = True
            print('episode {}: steps {}, rewards: {}'.format(episode, num_steps, rewards))
            scores.append(rewards)
            if self.resolved(scores): break
        plt.plot(scores)
        plt.show()


        
if __name__ == '__main__':
    game = Game('CartPole-v0')
    #game = Game('MountainCar-v0')

    game.run(episodes = 1000)
    game.agent.save()
    #game.agent.load()
    #game.replay_buffer.show()
    #game.run(episodes = 1, optimal = True, render = True)
    game.env.close()
    print('done')
