#!/usr/bin/python
# -*- coding: utf-8 -*-

import gym
import numpy as np
import random
import time, pprint, copy
import matplotlib.pyplot as plt
from collections import deque
from itertools import count


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

    def __repr__(self):
        return self.buffer.__repr__()

class QTable(Buffer):
    def __init__(self, n_observation, n_action):
        self.n_observation = n_observation
        self.n_action = n_action
        self.buffer = np.zeros((n_observation, n_action))

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


class Policy(object):
    pass

class EpsilonGreedy(Policy):
    def __init__(self, q_table, epsilon = 0.1):
        self.q_table = q_table
        self.set(epsilon)

    def set(self, epsilon):
        self.epsilon = epsilon

    def act(self, observation):
        explore = np.random.rand() < self.epsilon
        if explore:
            action = np.random.randint(self.q_table.n_action)
        else:
            q_actions = self.q_table[observation, :]
            action = np.random.choice(np.flatnonzero(q_actions == q_actions.max()))
        return action

    def action_prob(self, observation):
        q_actions = self.q_table[observation, :]
        prob = np.ones_like(q_actions) * self.epsilon / q_actions.size
        max_index = np.flatnonzero(q_actions == q_actions.max())
        prob[max_index] += (1 - self.epsilon) / max_index.size
        return prob


class Learning(object):
    def __init__(self, alpha, gamma, q_table):
        self.set(alpha, gamma)
        self.q_table = q_table
        self.num_learns = 0

    def set(self, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma

    def step(self):
        if self.num_learns == 0: print('start to learn...')
        self.num_learns += 1

class MonteCarlo(Learning):
    def __init__(self, alpha, gamma, q_table):
        super(MonteCarlo, self).__init__(alpha, gamma, q_table)
        self.trajectory = TrajectoryBuffer(self.gamma)

    def step(self, *transition):
        observation, action, next_observation, reward, done, next_action = transition
        is_full = self.trajectory.store(observation, action, next_observation, reward, done)
        if is_full:
            super(MonteCarlo, self).step()
            # Q(s,a) = Q(s,a) + alpha * (Gt - Q(s, a))
            for observation, next_observation, action, reward, future_rewards in self.trajectory[0]:
                self.q_table[observation, action] += self.alpha * (future_rewards - self.q_table[observation, action])
            self.trajectory.reset()

class TD(Learning):
    def step(self, *transition):
        observation, action, next_observation, reward, done, next_action = transition
        super(TD, self).step()
        # Q(s,a) = Q(s,a) + alpha * (R(t) + gamma * Q(s', a') - Q(s, a))
        self.q_table[observation, action] += self.alpha * self.delta(*transition)

class Sarsa(TD):
    def delta(self, *transition):
        observation, action, next_observation, reward, done, next_action = transition
        # delta = R(t) + gamma * Q(s', a') - Q(s, a)
        delta = reward + self.gamma * self.q_table[next_observation, next_action] - self.q_table[observation, action]
        return delta

class ExpectedSarsa(TD):
    def __init__(self, alpha, gamma, q_table, policy):
        super(ExpectedSarsa, self).__init__(alpha, gamma, q_table)
        self.policy = policy

    def delta(self, *transition):
        observation, action, next_observation, reward, done, next_action = transition
        # V(s') = sum(P(a'|s') * Q(s', a'))
        # delta = R(t) + gamma * V(s') - Q(s, a)
        value = np.sum(self.policy.action_prob(next_observation) * self.q_table[next_observation, :])
        delta = reward + self.gamma * value - self.q_table[observation, action]
        return delta

class SarsaLambda(Sarsa):
    def __init__(self, alpha, gamma, q_table, lmda):
        super(SarsaLambda, self).__init__(alpha, gamma, q_table)
        self.lmda = lmda
        self.eligibility_traces = copy.deepcopy(q_table)

    def step(self, *transition):
        observation, action, next_observation, reward, done, next_action = transition
        super(TD, self).step()
        self.eligibility_traces[observation, action] += 1
        self.q_table.buffer += self.alpha * self.delta(*transition) * self.eligibility_traces.buffer
        self.eligibility_traces.buffer *= self.gamma * self.lmda

class ExpectedSarsaLambda(SarsaLambda):
    def __init__(self, alpha, gamma, q_table, lmda, policy):
        super(ExpectedSarsaLambda, self).__init__(alpha, gamma, q_table, lmda)
        self.expected_sarsa = ExpectedSarsa(alpha, gamma, q_table, policy)

    def delta(self, *transition):
        return self.expected_sarsa.delta(*transition)

class QLearning(TD):
    def delta(self, *transition):
        observation, action, next_observation, reward, done, next_action = transition
        # delta = R(t) + gamma * max(Q(s', a')) - Q(s, a)
        delta = reward + self.gamma * np.max(self.q_table[next_observation, :]) - self.q_table[observation, action]
        return delta

class QLearningLambda(SarsaLambda):
    def __init__(self, alpha, gamma, q_table, lmda):
        super(QLearningLambda, self).__init__(alpha, gamma, q_table, lmda)
        self.q_learning = QLearning(alpha, gamma, q_table)

    def delta(self, *transition):
        return self.q_learning.delta(*transition)


class Agent(object):
    _ID = count(0)
    LEARN_METHODS = ['MC', 'Sarsa', 'ExpectedSarsa', 'SarsaLambda', 'ExpectedSarsaLambda', 'QLearning', 'QLearningLambda']

    def __init__(self, n_observation, n_action, learn_method = 'QLearning'):
        # hyper parameters
        self.epsilon = 0.5
        self.alpha = 0.3
        self.gamma = 0.9
        self.lmda = 0.5

        self.actions = list(range(n_action))
        self.n_observation = n_observation
        self.n_action = n_action
        self.q_table = QTable(n_observation, n_action)
        self.policy = EpsilonGreedy(self.q_table, self.epsilon)

        learn_methods = {'MC': MonteCarlo(self.alpha, self.gamma, self.q_table),
                         'Sarsa': Sarsa(self.alpha, self.gamma, self.q_table),
                         'ExpectedSarsa': ExpectedSarsa(self.alpha, self.gamma, self.q_table, self.policy),
                         'SarsaLambda': SarsaLambda(self.alpha, self.gamma, self.q_table, self.lmda),
                         'ExpectedSarsaLambda': ExpectedSarsaLambda(self.alpha, self.gamma, self.q_table, self.lmda, self.policy),
                         'QLearning': QLearning(self.alpha, self.gamma, self.q_table),
                         'QLearningLambda': QLearningLambda(self.alpha, self.gamma, self.q_table, self.lmda)}
        self.learn = learn_methods[learn_method]
        self.id = next(self._ID)
        self.name = 'Agent {}: {}/{}'.format(self.id, self.learn.__class__.__name__, self.policy.__class__.__name__)

    def new_episode(self):
        if self.learn.num_learns % 1000 == 1:
           self.epsilon *= 0.95
           self.alpha *= 0.95
           self.policy.set(self.epsilon)
           self.learn.set(self.alpha, self.gamma)

    def step(self, *transition):
        self.learn.step(*transition)

    def act(self, observation, optimal = False):
        action = self.policy.act(observation)
        return action

    def __repr__(self):
        return '[{}] Epsilon: {}, Alpha: {}, Gamma: {}, Lambda: {}'.format(self.name, self.epsilon, self.alpha, self.gamma ,self.lmda)


class Game(object):
    def __init__(self, game_name):
        self.env = gym.make(game_name)
        self.env.seed(1)
        print('action space:', self.env.action_space)
        print('observation space:', self.env.observation_space)
        self.reward_shaping_enable = hasattr(self, 'reward_shaping')
        self.num_avg_history = 1

    def run_one_episode(self, agent, render = False):
        observation = self.env.reset()
        done = False
        num_steps = 0
        rewards = shaping_rewards = 0
        action = agent.act(observation)
        while not done:
            if render: self.env.render()
            next_observation, reward, done, info = self.env.step(action)
            shaping_reward = reward if not self.reward_shaping_enable else self.reward_shaping(observation, next_observation, reward)
            next_action = agent.act(next_observation)
            agent.step(observation, action, next_observation, reward, done, next_action)
            observation, action = next_observation, next_action
            num_steps += 1
            rewards += reward
            shaping_rewards += shaping_reward
        return rewards, shaping_rewards, num_steps

    def run_episodes(self, agent, episodes, render = False):
        rewards_history, shaping_rewards_history, num_steps_history = [], [], []
        for episode in range(episodes):
            agent.new_episode()
            rewards, shaping_rewards, num_steps = self.run_one_episode(agent, render)
            rewards_history.append(rewards)
            shaping_rewards_history.append(shaping_rewards)
            num_steps_history.append(num_steps)
            if episode % 1000 == 0:
                print('[{}] episode {}: steps {}, rewards: {}, shaping_rewards: {}, num_learns: {}'.format(agent.name, episode, num_steps, rewards, shaping_rewards, agent.learn.num_learns))
            if self.resolved(rewards, episodes): break
        print('\n')
        return self.avg_history(rewards_history, shaping_rewards_history, num_steps_history)

    def avg_history(self, *history):
        if self.num_avg_history <= 1:
            return history
        else:
            filter = np.ones(self.num_avg_history)
            return tuple(np.convolve(h, filter, 'valid') / self.num_avg_history for h in history)

    def resolved(self, rewards, episodes):
        return False

    def __del__(self):
        self.env.close()

class Taxi(Game):
    def __init__(self):
        super(Taxi, self).__init__('Taxi-v2')
        self.num_avg_history = 100
        self.agents = {}
        for learn_method in Agent.LEARN_METHODS:
            self.agents[learn_method] = Agent(self.env.observation_space.n, self.env.action_space.n, learn_method)

    def run(self, episodes):
        rewards_history, num_steps_history = {}, {}
        for key, agent in self.agents.items():
            rewards_history[key], _, num_steps_history[key] = self.run_episodes(agent, episodes)
        plt.figure()
        colors = ['b--', 'r--', 'g--', 'k--', 'm--', 'y--', 'c--']
        for index, (key, rewards) in enumerate(rewards_history.items()):
            plt.plot(rewards, colors[index], label = key)
            print(self.agents[key])
            print('Best Avg Rewards: {} in {}/{} episodes'.format(np.max(rewards), np.argmax(rewards), episodes))
        plt.legend()
        plt.show()


if __name__ == '__main__':
    game = Taxi()
    game.run(episodes = 200000)

    ## The output is:
    
    ##[Agent 0: MonteCarlo/EpsilonGreedy] Epsilon: 1.7526333124414336e-05, Alpha: 1.0515799874648609e-05, Gamma: 0.9, Lambda: 0.5
    ##Best Avg Rewards: 9.04 in 153024/200000 episodes
    ##[Agent 1: Sarsa/EpsilonGreedy] Epsilon: 4.888987831561423e-05, Alpha: 2.933392698936855e-05, Gamma: 0.9, Lambda: 0.5
    ##Best Avg Rewards: 9.27 in 149215/200000 episodes
    ##[Agent 2: ExpectedSarsa/EpsilonGreedy] Epsilon: 3.783005587731689e-05, Alpha: 2.2698033526390145e-05, Gamma: 0.9, Lambda: 0.5
    ##Best Avg Rewards: 9.33 in 194497/200000 episodes
    ##[Agent 3: SarsaLambda/EpsilonGreedy] Epsilon: 1.049366305098405e-05, Alpha: 6.296197830590433e-06, Gamma: 0.9, Lambda: 0.5
    ##Best Avg Rewards: 9.3 in 158916/200000 episodes
    ##[Agent 4: ExpectedSarsaLambda/EpsilonGreedy] Epsilon: 2.151773927384147e-05, Alpha: 1.2910643564304892e-05, Gamma: 0.9, Lambda: 0.5
    ##Best Avg Rewards: 9.38 in 120169/200000 episodes
    ##[Agent 5: QLearning/EpsilonGreedy] Epsilon: 5.3868388518253144e-06, Alpha: 3.232103311095192e-06, Gamma: 0.9, Lambda: 0.5
    ##Best Avg Rewards: 9.42 in 117248/200000 episodes
    ##[Agent 6: QLearningLambda/EpsilonGreedy] Epsilon: 2.641813893244094e-05, Alpha: 1.5850883359464575e-05, Gamma: 0.9, Lambda: 0.5
    ##Best Avg Rewards: 9.43 in 185391/200000 episodes


