#!/usr/bin/python
# -*- coding: utf-8 -*-

import gym
import numpy as np
import random
import time, pprint
import matplotlib.pyplot as plt


class QTable(object):
	def __init__(self, n_observation, n_action):
		self.n_observation = n_observation
		self.n_action = n_action
		self.table = np.zeros((n_observation, n_action))

	def __getitem__(self, key):
        return self.table.__getitem__(key)


class Policy(object):
	pass

class EpsilonGreedy(Policy):
	def __init__(self, q_table, epsilon = 0.1):
		self.q_table = q_table
		self.epsilon = epsilon

	def act(self, observation):
		explore = np.random.rand() < self.epsilon
        if explore:
            action = np.random.randint(self.q_table.n_action)
        else:
            action = np.argmax(self.q_table[observation, :])
        return action

class Buffer(object):
    def __getitem__(self, key):
        return self.buffer.__getitem__(key)

    def __iter__(self):
        yield from self.buffer

class TrajectoryBuffer(Buffer):
    def __init__(self, gamma, buffer_size = 1):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.reset()

    def store(self, *transition):
        self.transition_buffer.store(*transition)
        buffer_full = False
        done = transition[-1]
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

    def store(self, observation, action, next_observation, reward, done):
        # transition: observation, next_observation, action, reward, future_rewards
        transition = np.concatenate((observation, next_observation, np.array([action, reward, 0]))).astype(np.float32)[np.newaxis, :]
        if self.buffer is None:
            self.buffer = transition
        else:
            self.buffer = np.concatenate((self.buffer, transition), axis = 0)
        # update the future rewards
        self.buffer[:, -1] += (self.gamma ** (np.arange(len(self.buffer), 0, -1)-1) * reward)
        if done: self.total_rewards = self.buffer.sum(axis = 0)[-2]

class Learning(object):
	def step(self):
		if self.num_learns == 0:
			print('start to learn...')
		self.num_learns += 1

class MonteCarlo(Learning):
	def __init__(self, alpha, gamma, q_table):
		self.alpha = alpha
		self.gamma = gamma
		self.q_table = q_table
		self.trajectory = TrajectoryBuffer(self.gamma)

	def step(self, *transition):
        # observation, action, next_observation, reward, done
		is_full = self.trajectory.store(*transition)
		if is_full:
			super().step()
			# Q(s,a) = Q(s,a) + alpha * (Gt - Q(s, a))
			for observation, next_observation, action, reward, future_rewards in self.trajectory[0]:
				self.q_table[observation, action] += self.alpha * (future_rewards - self.q_table[observation, action])
			self.trajectory.reset()


class Sarsa(Learning):
	pass


class Agent(object):
    def __init__(self, n_observation, n_action):
        # hyper parameters
		self.epsilon = 0.1
		self.alpha = 0.3
		self.gamma = 0.97

        self.actions = list(range(n_action))
        self.n_observation = n_observation
		self.n_action = n_action
		self.q_table = QTable(n_observation, n_action)
		self.learn = MonteCarlo(self.alpha, self.gamma, self.q_table)
		self.policy = EpsilonGreedy(self.q_table, self.epsilon)

        self.num_learns = 0

    def new_episode(self):
        self.loss_history = []

    def step(self, *transition):
        # observation, action, next_observation, reward, done
		self.learn.step(*transition)

    def act(self, observation, optimal = False):
        action = self.policy.act(observation)
		return action

    def save(self, filename = 'q_table.txt'):
        pass

    def load(self, filename = 'q_table.txt'):
        pass

class Game(object):
    def __init__(self, game_name):
        self.env = gym.make(game_name)
        #self.env.seed(1)
        print('action space:', self.env.action_space)
        print('observation space:', self.env.observation_space)
        self.agent = Agent(self.env.observation_space.n, self.env.action_space.n)
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

	def resolved_Taxi(self, scores):
		return False

	def run_one_episode(self, optimal = False, render = False):
        observation = self.env.reset()
        done = False
        num_steps = 0
        shaping_rewards = 0
        rewards = 0
        while not done:
            if render: self.env.render()
            action = self.agent.act(observation, optimal)
            next_observation, reward, done, _ = self.env.step(action)
            shaping_reward = reward if not hasattr(self, 'reward_shaping') else self.reward_shaping(observation, next_observation, reward)
            if not optimal: self.agent.step(observation, action, next_observation, shaping_reward, done)
            observation = next_observation
            num_steps += 1
            rewards += reward
            shaping_rewards += shaping_reward
        return rewards, shaping_rewards, num_steps

	def run(self, episodes, optimal = False, render = False):
        print('start to run (optimal: {})...'.format(optimal))
        shaping_scores = []
        scores = []
        shaping_scores = []
        for episode in range(episodes):
            self.agent.new_episode()
            rewards, shaping_rewards, num_steps = self.run_one_episode(optimal, render)
            print('episode {}: steps {}, rewards: {}, shaping_rewards: {}, num_learns: {}'.format(episode, num_steps, rewards, shaping_rewards, self.agent.num_learns))
            scores.append(rewards)
            shaping_scores.append(shaping_rewards)
            if self.resolved(scores): break
            #if episode % 10 == 0:
            #    plt.plot(self.agent.loss_history)
            #    plt.show()
            #if episode > 300: optimal = True
            #if rewards == 200: render = True
        plt.plot(scores)
        plt.plot(shaping_scores, 'r--')
        plt.show()
        pprint.pprint(self.agent.debug_rewards)



if __name__ == '__main__':
    #game = Game('CartPole-v0')
    #game = Game('MountainCar-v0')
	game = Game('Taxi-v2')

    game.run(episodes = 1000)
    #game.agent.save()
    #game.agent.load()
    #game.replay_buffer.show()
    #game.run(episodes = 1, optimal = True, render = True)
    game.env.close()
    print('done')
