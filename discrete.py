#!/usr/bin/python
# -*- coding: utf-8 -*-

import gym
import numpy as np
import random
import time, pprint
import matplotlib.pyplot as plt


class Buffer(object):
    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self.buffer.__getitem__(tuple(map(lambda k: int(k) if isinstance(k, np.float32) else k, key)))
        else:
            return self.buffer.__getitem__(int(key) if isinstance(key, np.float32) else key)

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            self.buffer.__setitem__(tuple(map(lambda k: int(k) if isinstance(k, np.float32) else k, key)), value)
        else:
            self.buffer.__setitem__(int(key) if isinstance(key, np.float32) else key, value)

    def __iter__(self):
        yield from self.buffer

    def __str__(self):
        return self.buffer.__str__()

class QTable(Buffer):
    def __init__(self, n_observation, n_action):
        self.n_observation = n_observation
        self.n_action = n_action
        self.buffer = np.zeros((n_observation, n_action))

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
            q_actions = self.q_table[observation, :]
            action = np.random.choice(np.flatnonzero(q_actions == q_actions.max()))
        return action

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
        if isinstance(observation, np.ndarray):
            transition = np.concatenate((observation, next_observation, np.array([action, reward, 0]))).astype(np.float32)[np.newaxis, :]
        else:
            transition = np.array([observation, next_observation, action, reward, 0]).astype(np.float32)[np.newaxis, :]
        if self.buffer is None:
            self.buffer = transition
        else:
            self.buffer = np.concatenate((self.buffer, transition), axis = 0)
        # update the future rewards
        self.buffer[:, -1] += (self.gamma ** (np.arange(len(self.buffer), 0, -1)-1) * reward)
        if done: self.total_rewards = self.buffer.sum(axis = 0)[-2]

class Learning(object):
    def __init__(self):
        self.num_learns = 0

    def step(self):
        if self.num_learns == 0:
            print('start to learn...')
        self.num_learns += 1

class MonteCarlo(Learning):
    def __init__(self, alpha, gamma, q_table):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = q_table
        self.trajectory = TrajectoryBuffer(self.gamma)

    def step(self, *transition):
        observation, action, next_observation, reward, done, next_action = transition
        is_full = self.trajectory.store(observation, action, next_observation, reward, done)
        if is_full:
            super().step()
            # Q(s,a) = Q(s,a) + alpha * (Gt - Q(s, a))
            for observation, next_observation, action, reward, future_rewards in self.trajectory[0]:
                self.q_table[observation, action] += self.alpha * (future_rewards - self.q_table[observation, action])
            self.trajectory.reset()

class Sarsa(Learning):
    def __init__(self, alpha, gamma, q_table):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = q_table

    def step(self, *transition):
        observation, action, next_observation, reward, done, next_action = transition        
        super().step()
        # Q(s,a) = Q(s,a) + alpha * (R(t) + gamma * Q(s', a') - Q(s, a))
        delta = reward + self.gamma * self.q_table[next_observation, next_action] - self.q_table[observation, action]
        self.q_table[observation, action] += self.alpha * delta


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
        #self.learn = MonteCarlo(self.alpha, self.gamma, self.q_table)
        self.learn = Sarsa(self.alpha, self.gamma, self.q_table)
        self.policy = EpsilonGreedy(self.q_table, self.epsilon)

    def new_episode(self):
        self.loss_history = []
        #if self.learn.num_learns == 50:
        #   self.epsilon = 0.1

    def step(self, *transition):
        # observation, action, next_observation, reward, done, next_action
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
        self.resolved = getattr(self, 'resolved_{}'.format(game_name.split('-')[0]))

    def resolved_Taxi(self, scores):
        return False

    def run_one_episode(self, optimal = False, render = False):
        observation = self.env.reset()
        done = False
        num_steps = 0
        shaping_rewards = 0
        rewards = 0
        action = self.agent.act(observation)
        while not done:
            if render: self.env.render()
            next_observation, reward, done, _ = self.env.step(action)
            #if reward < 0: done = False
            shaping_reward = reward if not hasattr(self, 'reward_shaping') else self.reward_shaping(observation, next_observation, reward)
            next_action = self.agent.act(next_observation)
            self.agent.step(observation, action, next_observation, reward, done, next_action)
            observation = next_observation
            action = next_action
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
            print('episode {}: steps {}, rewards: {}, shaping_rewards: {}, num_learns: {}'.format(episode, num_steps, rewards, shaping_rewards, self.agent.learn.num_learns))
            scores.append(rewards)
            shaping_scores.append(shaping_rewards)
            if self.resolved(scores): break
            #if episode % 10 == 0:
            #    plt.plot(self.agent.loss_history)
            #    plt.show()
            #if episode > 300: optimal = True
            #if rewards == 200: render = True
        plt.plot(scores)
        #plt.plot(shaping_scores, 'r--')
        plt.show()
        pprint.pprint(str(self.agent.q_table))



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
