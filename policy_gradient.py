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


class PolicyNet(nn.Module):
    def __init__(self, num_observations, num_actions):
        super().__init__()
        self.fc1 = nn.Linear(num_observations, 64)
        self.fc2 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))
        return x


class Normalizer(object):
    def __init__(self, replay_buffer):
        self.replay_buffer = replay_buffer

    def calc_mean_std(self):
        observation = np.array(self.replay_buffer.buffer)[:, [0, 2]].reshape(-1, 1)
        self.observation_mean = observation.mean()
        self.observation_std = observation.std()

    def norm_observation(self, observation):
        return (observation - self.observation_mean) / self.observation_std

    def norm_records(self, records):
        mean, std = self.observation_mean, self.observation_std
        for index, record in enumerate(records):
            observation, action, next_observation, reward, done = record
            records[index] = self.replay_buffer.record((observation - mean) / std, action, (next_observation - mean) / std, reward, done)

class TrajectoryBuffer(object):
    def __init__(self, gamma):
        self.transition_buffer = TransitionBuffer(gamma)
        self.buffer = []

    def store(self, observation, action, reward, done):
        self.transition_buffer.store(observation, action, reward, done)
        if done:
            self.buffer.append(self.transition_buffer)
            self.transition_buffer = TransitionBuffer(gamma)
        
class TransitionBuffer(object):
    def __init__(self, gamma):
        self.gamma = gamma
        self.buffer = None

    def store(self, observation, action, reward, done):
        # transition: observation, action, reward, future_rewards
        transition = np.array(observation, action, reward, 0).astype(np.float32)[np.newaxis, :]
        if self.buffer is None:
            self.buffer = transition
        else:
            self.buffer = np.concatenate((self.buffer, transition), axis = 0)
        self.buffer[:, -1] += (self.gamma ** (np.arange(len(self.buffer), 0, -1)-1) * reward)[:, np.newaxis]
        if done: self.total_rewards = self.buffer.sum(axis = 0)[-2]

class Agent(object):
    def __init__(self, observation_space, action_space):
        # hyper parameters
        self.lr = 0.01
        self.gamma = 0.99
        self.batch_size = 32

        self.action_space = action_space
        self.actions = list(range(action_space.n))
        num_observations = np.prod(observation_space.shape)
        self.policy_net = PolicyNet(num_observations, action_space.n)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr = self.lr)
        self.trajectory_buffer = TrajectoryBuffer(self.gamma)

        self.num_learns = 0

    def new_episode(self):
        if self.noisy_dqn:
            self.policy_net.noisy_weights()
        #self.scheduler.step()
        self.loss_history = []

    def memorize(self, *transition):
        # observation, action, next_observation, reward, done
        self.replay_buffer.save(*transition)
        if self.replay_buffer.is_full():
            if self.num_learns == 0:
                print('start to learn...')
                if self.normalize_observation: self.normalizer.calc_mean_std()
            self.learn()

    def choose_action(self, observation, optimal = False):
        explore = not optimal and (self.num_learns == 0 or (np.random.rand() < self.epsilon and not self.noisy_dqn))
        actions_prob = self.policy_net(torch.tensor(observation).float())
        if optimal:
            action = self.actions[torch.argmax(actions_prob).item()]
        else:
            n = np.random.rand()
            for i, action in enumerate(self.actions):
                if n < actions_prob[i]: break
                n -= actions_prob[i]
        return action

    def update_target_net(self):
        if self.num_learns % self.num_learns_to_update_target == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            #print('update target net')
        self.num_learns += 1

    def learn(self):
        self.update_target_net()
        records = self.replay_buffer.sample(self.batch_size, self.multi_steps_dqn)
        if self.normalize_observation: self.normalizer.norm_records(records)
        not_done_records = [r[:-1] for r in records if r[-1] == 0]
        done_records = [r[:-1] for r in records if r[-1] != 0]
        batch_observations, batch_actions, _, _ = map(torch.tensor, zip(*(not_done_records+done_records)))
        _, _, batch_next_observations, batch_rewards = map(torch.tensor, zip(*not_done_records))
        if done_records: _, _, _, batch_done_rewards = map(torch.tensor, zip(*done_records))
        # Q learning
        # Q(state, action) = reward + gamma * max(Q(state+1, action_space))
        q_eval = self.policy_net(batch_observations).gather(1, batch_actions.view(-1, 1).long())
        with torch.no_grad():
            if self.double_dqn:
                actions = torch.max(self.policy_net(batch_next_observations), 1)[1]
                q_target_ = self.target_net(batch_next_observations).gather(1, actions.view(-1, 1).long())
            else:
                q_target_ = torch.max(self.target_net(batch_next_observations), 1)[0].view(-1, 1)
            q_target = q_target_ * self.gamma + batch_rewards.view(-1, 1)
            if done_records: q_target = torch.cat((q_target, batch_done_rewards.view(-1, 1)), 0)
        assert q_eval.shape == q_target.shape, 'invalid shape'
        loss = self.criterion(q_eval, q_target)
        self.loss_history.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def save(self, filename = 'qnet_save.txt'):
        torch.save(self.policy_net.state_dict(), filename)

    def load(self, filename = 'qnet_save.txt'):
        self.policy_net.load_state_dict(torch.load(filename))

class Game(object):
    def __init__(self, game_name):
        self.env = gym.make(game_name)
        self.env.seed(1)
        print('action space:', self.env.action_space)
        print('observation space:', self.env.observation_space)
        self.agent = Agent(self.env.observation_space, self.env.action_space)
        #self.reward_shaping = getattr(self, 'reward_shaping_{}'.format(game_name.split('-')[0]))
        self.resolved = getattr(self, 'resolved_{}'.format(game_name.split('-')[0]))

    def reward_shaping_CartPole(self, next_observation, reward):
        x, x_dot, theta, theta_dot = next_observation
        r1 = (self.env.unwrapped.x_threshold - abs(x))/self.env.unwrapped.x_threshold - 0.8
        r2 = (self.env.unwrapped.theta_threshold_radians - abs(theta))/self.env.unwrapped.theta_threshold_radians - 0.5
        reward = r1 + r2
        return reward

    def reward_shaping_MountainCar(self, next_observation, reward):
        position, velocity = next_observation
        # the higher the better
        reward = abs(position - (-0.5))     # r in [0, 1]
        return reward

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
        
    def run(self, episodes, optimal = False, render = False):
        print('start to run (optimal: {})...'.format(optimal))
        scores = []
        for episode in range(episodes):
            actions_history = []
            observation = self.env.reset()
            self.agent.new_episode()
            done = False
            num_steps = 0
            rewards = 0
            while not done:
                if render: self.env.render()
                action = self.agent.choose_action(observation, optimal = optimal)
                actions_history.append(action)
                next_observation, reward, done, _ = self.env.step(action)
                shaping_reward = reward if not hasattr(self, 'reward_shaping') else self.reward_shaping(next_observation, reward)
                if not optimal: self.agent.memorize(observation, action, next_observation, shaping_reward, done)
                observation = next_observation
                num_steps += 1
                rewards += reward
            #print('episode {}: steps {}, rewards: {}, loss: {}'.format(episode, num_steps, rewards, self.agent.loss_history))
            #print('episode {}: steps {}, rewards: {}, actions: {}'.format(episode, num_steps, rewards, actions_history))
            print('episode {}: steps {}, rewards: {}'.format(episode, num_steps, rewards))
            scores.append(rewards)
            if self.resolved(scores): break
            #if episode % 10 == 0:
            #    plt.plot(self.agent.loss_history)
            #    plt.show()
            #if episode > 300: optimal = True
            #if rewards == 200: render = True
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
