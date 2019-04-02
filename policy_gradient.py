#!/usr/bin/python
# -*- coding: utf-8 -*-

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time, pprint
import matplotlib.pyplot as plt


class PolicyNet(nn.Module):
    def __init__(self, num_observations, num_actions):
        super().__init__()
        self.num_observations = num_observations
        self.fc1 = nn.Linear(num_observations, 64)
        self.fc2 = nn.Linear(64, num_actions)

    def forward(self, x):
        batch_size, num_input = x.size()
        assert num_input == self.num_observations, 'invalid num_input'
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))
        return x

class Buffer(object):
    def __getitem__(self, key):
        return self.buffer.__getitem__(key)
        
    def __iter__(self):
        yield from self.buffer
        
class TrajectoryBuffer(Buffer):
    def __init__(self, buffer_size, gamma):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.reset()

    def store(self, observation, action, reward, done):
        self.transition_buffer.store(observation, action, reward, done)
        buffer_full = False
        if done:
            self.buffer.append(self.transition_buffer)
            self.transition_buffer = TransitionBuffer(self.gamma)
            if len(self.buffer) == self.buffer_size: buffer_full = True
        return buffer_full

    def reset(self):
        self.transition_buffer = TransitionBuffer(self.gamma)
        self.buffer = []

class TransitionBuffer(Buffer):
    def __init__(self, gamma):
        self.gamma = gamma
        self.buffer = None

    def store(self, observation, action, reward, done):
        # transition: observation, action, reward, future_rewards
        transition = np.concatenate((observation, np.array([action, reward, 0]))).astype(np.float32)[np.newaxis, :]
        if self.buffer is None:
            self.buffer = transition
        else:
            self.buffer = np.concatenate((self.buffer, transition), axis = 0)
        self.buffer[:, -1] += (self.gamma ** (np.arange(len(self.buffer), 0, -1)-1) * reward)
        if done: self.total_rewards = self.buffer.sum(axis = 0)[-2]

class Agent(object):
    def __init__(self, observation_space, action_space):
        # hyper parameters
        self.lr = 0.01
        self.gamma = 0.95
        self.batch_size = 8
        self.vanilla_policy_gradient = False
        self.future_rewards_policy_gradient = True
        self.actor_critic = False

        self.action_space = action_space
        self.actions = list(range(action_space.n))
        num_observations = np.prod(observation_space.shape)
        self.policy_net = PolicyNet(num_observations, action_space.n)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr = self.lr)
        self.trajectory_buffer = TrajectoryBuffer(self.batch_size, self.gamma)

        self.num_learns = 0

    def new_episode(self):
        self.loss_history = []

    def memorize(self, *transition):
        # observation, action, reward, done
        buffer_full = self.trajectory_buffer.store(*transition)
        if buffer_full:
            self.learn()
            self.trajectory_buffer.reset()

    def choose_action(self, observation, optimal = False):
        actions_prob = self.policy_net(torch.tensor(observation[np.newaxis, :]).float())
        if optimal:
            action = self.actions[torch.argmax(actions_prob).item()]
        else:
            n = np.random.rand()
            for i, action in enumerate(self.actions):
                if n < actions_prob[0,i].item(): break
                n -= actions_prob[0,i].item()
        return action

    def learn(self):
        if self.num_learns == 0: print('start to learn...')
        # loss function
        # L(theta) = -sum( (sum(future_rewards(t)) - b) * log(P(a(t)|s(t);theta))) )
        loss = 0
        for trajectory in self.trajectory_buffer:
            batch_observations, batch_actions, _, batch_future_rewards = map(torch.tensor, [trajectory[:, 0:-3], trajectory[:, -3], trajectory[:, -2], trajectory[:, -1]])
            
            if self.vanilla_policy_gradient:
                rewards = trajectory.total_rewards
            elif self.future_rewards_policy_gradient:
                rewards = batch_future_rewards
            elif self.actor_critic:
                pass
            prob = self.policy_net(batch_observations).gather(1, batch_actions.view(-1, 1).long())
            log_prob = torch.log(prob)
            loss += -torch.sum(rewards * log_prob)
        self.loss_history.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.num_learns += 1

    def save(self, filename = 'policy_net.txt'):
        torch.save(self.policy_net.state_dict(), filename)

    def load(self, filename = 'policy_net.txt'):
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
            action = self.agent.choose_action(observation, optimal)
            next_observation, reward, done, _ = self.env.step(action)
            shaping_reward = reward if not hasattr(self, 'reward_shaping') else self.reward_shaping(next_observation, reward)
            if not optimal: self.agent.memorize(observation, action, shaping_reward, done)
            observation = next_observation
            num_steps += 1
            rewards += reward
        return rewards, num_steps

    def run(self, episodes, optimal = False, render = False):
        print('start to run (optimal: {})...'.format(optimal))
        scores = []
        for episode in range(episodes):
            self.agent.new_episode()
            rewards, num_steps = self.run_one_episode(optimal, render)
            print('episode {}: steps {}, rewards: {}, num_learns: {}'.format(episode, num_steps, rewards, self.agent.num_learns))
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
