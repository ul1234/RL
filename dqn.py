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


class QNet(nn.Module):
    def __init__(self, num_observations, num_actions):
        super().__init__()
        self.fc1 = nn.Linear(num_observations, 64)
        self.fc2 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.record = namedtuple('record', 'observation action next_observation reward done')
        self.buffer_size = buffer_size
        self.buffer_write_index = 0
        self.buffer = []

    def save(self, *record_elements):
        # transition: observation, action -> observation(t+1), reward, done
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

class Agent(object):
    def __init__(self, observation_space, action_space):
        # hyper parameters
        self.lr = 0.01
        self.lr_decay = 0.99
        self.gamma = 0.99
        self.epsilon_min = 0.01
        self.epsilon = 1
        self.epsilon_decay = 0.99
        self.batch_size = 32
        self.replay_buffer_size = 1000
        self.num_learns_to_update_target = 20

        self.action_space = action_space
        self.actions = list(range(action_space.n))
        num_observations = np.prod(observation_space.shape)
        self.eval_net = QNet(num_observations, action_space.n)
        self.target_net = QNet(num_observations, action_space.n)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr = self.lr)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size = 100, gamma = 0.3)
        self.replay_buffer = ReplayBuffer(buffer_size = self.replay_buffer_size)

        self.num_learns = 0
        self.double_dqn = True

    def new_episode(self):
        self.scheduler.step()
        self.loss_history = []

    def memorize(self, *transition):
        # observation, action, next_observation, reward, done
        self.replay_buffer.save(*transition)
        if self.replay_buffer.is_full():
            if self.num_learns == 0: print('start to learn...')
            self.learn()

    def choose_action(self, observation, optimal = False):
        explore = np.random.rand() < self.epsilon and not optimal
        if explore:
            action = self.action_space.sample()
        else:
            action = self.actions[torch.argmax(self.eval_net(torch.tensor(observation).float())).item()]
        return action

    def update_target_net(self):
        if self.num_learns % self.num_learns_to_update_target == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            #print('update target net')
        self.num_learns += 1

    def learn(self):
        self.update_target_net()
        records = self.replay_buffer.sample(self.batch_size)
        not_done_records = [r[:-1] for r in records if r[-1] == 0]
        done_records = [r[:-1] for r in records if r[-1] != 0]
        batch_observations, batch_actions, _, _ = map(torch.tensor, zip(*(not_done_records+done_records)))
        _, _, batch_next_observations, batch_rewards = map(torch.tensor, zip(*not_done_records))
        if done_records: _, _, _, batch_done_rewards = map(torch.tensor, zip(*done_records))
        # Q learning
        # Q(state, action) = reward + gamma * max(Q(state+1, action_space))
        q_eval = self.eval_net(batch_observations).gather(1, batch_actions.view(-1, 1).long())
        with torch.no_grad():
            if self.double_dqn:
                actions = torch.max(self.eval_net(batch_next_observations), 1)[1]
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
        torch.save(self.eval_net.state_dict(), filename)

    def load(self, filename = 'qnet_save.txt'):
        self.eval_net.load_state_dict(torch.load(filename))

class Game(object):
    def __init__(self, game_name):
        self.env = gym.make(game_name)
        self.env.seed(1)
        print('action space:', self.env.action_space)
        print('observation space:', self.env.observation_space)
        self.agent = Agent(self.env.observation_space, self.env.action_space)
        #self.reward_shaping = getattr(self, 'reward_shaping_{}'.format(game_name.split('-')[0]))
        self.resolved = getattr(self, 'resolved_{}'.format(game_name.split('-')[0]))

    def reward_shaping_CartPole(self, next_observation, reward, done):
        #if not done:
        x, x_dot, theta, theta_dot = next_observation
        r1 = (self.env.x_threshold - abs(x))/self.env.x_threshold - 0.8
        r2 = (self.env.theta_threshold_radians - abs(theta))/self.env.theta_threshold_radians - 0.5
        reward = r1 + r2
        return reward

    def reward_shaping_MountainCar(self, next_observation, reward, done):
        #if not done:
        position, velocity = next_observation
        # the higher the better
        reward = abs(position - (-0.5))     # r in [0, 1]
        return reward

    def resolved_CartPole(self, scores):
        if len(scores) >= 100 and np.mean(scores[-100:]) >= 195.0:
            print('Solved after {} episodes'.format(len(scores)-100))
            return True
        return False

    def run(self, episodes, optimal = False):
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
                #self.env.render()
                action = self.agent.choose_action(observation, optimal = optimal)
                actions_history.append(action)
                next_observation, reward, done, _ = self.env.step(action)
                shaping_reward = reward if not hasattr(self, 'reward_shaping') else self.reward_shaping(next_observation, reward, done)
                if not optimal: self.agent.memorize(observation, action, next_observation, shaping_reward, done)
                observation = next_observation
                num_steps += 1
                rewards += reward
            #print('episode {}: steps {}, rewards: {}, loss: {}'.format(episode, num_steps, rewards, self.agent.loss_history))
            print('episode {}: steps {}, rewards: {}'.format(episode, num_steps, rewards))
            scores.append(rewards)
            if self.resolved(scores): break
            if episode > 300: optimal = True
        plt.plot(scores)
        plt.show()

        
if __name__ == '__main__':
    game = Game('CartPole-v0')
    #game = Game('MountainCar-v0')

    game.run(episodes = 1000)
    game.agent.save()
    #game.agent.load()
    #game.replay_buffer.show()
    #game.run(episodes = 1, optimal = True)
    game.env.close()
    print('done')
