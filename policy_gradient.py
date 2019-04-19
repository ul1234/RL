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


# add observation with timestep t
# policy gradient must have positive rewards

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
        x = F.softmax(self.fc2(x), dim = 1)
        return x

class ValueNet(nn.Module):
    def __init__(self, num_observations):
        super().__init__()
        self.num_observations = num_observations
        self.fc1 = nn.Linear(num_observations, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        batch_size, num_input = x.size()
        assert num_input == self.num_observations, 'invalid num_input'
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
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

class Agent(object):
    def __init__(self, observation_space, action_space):
        # hyper parameters
        self.lr = 0.01
        self.gamma = 0.97
        self.batch_size = 1 #8
        self.reward_bias = 0
        # 'vanilla', 'future_rewards', 'actor_critic_mc', 'actor_critic_td'
        self.policy_gradient_method = 'actor_critic_mc'
        self.add_timestep_to_observation = True

        self.need_value_net = self.policy_gradient_method in ['actor_critic_mc', 'actor_critic_td']
        self.action_space = action_space
        self.actions = list(range(action_space.n))
        self.num_observations = np.prod(observation_space.shape)
        if self.add_timestep_to_observation: self.num_observations += 1
        self.policy_net = PolicyNet(self.num_observations, action_space.n)
        self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr = self.lr)

        if self.need_value_net:
            self.value_net = ValueNet(self.num_observations)
            self.optimizer_value = optim.Adam(self.value_net.parameters(), lr = self.lr)
            self.criterion = nn.MSELoss()
            self.batch_size = 1     # the batch size should be smaller when use actor critic
        self.trajectory_buffer = TrajectoryBuffer(self.batch_size, self.gamma)

        self.num_learns = 0
        self.debug_rewards = []

    def new_episode(self):
        self.loss_history = []
        self.loss_history_value = []

    def memorize(self, *transition):
        # observation, action, next_observation, reward, done
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
        loss = 0
        loss_value = 0
        for trajectory in self.trajectory_buffer:
            # transition: observation, next_observation, action, reward, future_rewards
            batch_observations, batch_next_observations, batch_actions, batch_rewards, batch_future_rewards = \
                map(torch.tensor, [trajectory[:, 0:self.num_observations], trajectory[:, self.num_observations:2*self.num_observations], trajectory[:, -3], trajectory[:, -2], trajectory[:, -1]])

            batch_actions, batch_rewards, batch_future_rewards = map(lambda x: x.view(-1, 1), [batch_actions, batch_rewards, batch_future_rewards])

            # 'vanilla', 'future_rewards', 'actor_critic_mc', 'actor_critic_td'
            if self.policy_gradient_method == 'vanilla':
                # loss function
                # L(theta) = -sum( reward(trajectory) - b ) * log(P(a(t)|s(t);theta)) )
                rewards = trajectory.total_rewards - self.reward_bias
            elif self.policy_gradient_method == 'future_rewards':
                # L(theta) = -sum( ( sum(future_rewards(t)) - b ) * log(P(a(t)|s(t);theta)) )
                rewards = batch_future_rewards - self.reward_bias
            elif self.policy_gradient_method == 'actor_critic_mc':
                # Monte carlo based actor critic
                # L(theta) = -sum( ( sum(future_rewards(t)) - V_pi(s(t)) ) * log(P(a(t)|s(t);theta)) )
                state_value = self.value_net(batch_observations)
                rewards = batch_future_rewards - state_value.detach()
                # loss function of value net
                # L = MSE(sum(future_rewards(t)) - V_pi(s(t)))
                loss_value += self.criterion(batch_future_rewards, state_value)
            elif self.policy_gradient_method == 'actor_critic_td':
                # Temporal Difference based actor critic
                # L(theta) = -sum( ( r(t) + V_pi(s(t+1)) - V_pi(s(t)) ) * log(P(a(t)|s(t);theta)) )
                state_value = self.value_net(batch_observations)
                next_state_value = self.value_net(batch_next_observations)
                rewards = batch_rewards + next_state_value.detach() - state_value.detach()
                # loss function of value net
                # L = MSE( r(t) + V_pi(s(t+1)) - V_pi(s(t)) )
                loss_value += self.criterion(batch_rewards + next_state_value, state_value)
            self.debug_rewards.append(rewards)

            prob = self.policy_net(batch_observations).gather(1, batch_actions.long())
            log_prob = torch.log(prob)
            loss += -torch.sum(rewards * log_prob)
        if self.need_value_net:
            self.loss_history_value.append(loss_value.item())
            self.optimizer_value.zero_grad()
            loss_value.backward()
            self.optimizer_value.step()
        self.loss_history.append(loss.item())
        self.optimizer_policy.zero_grad()
        loss.backward()
        self.optimizer_policy.step()
        self.num_learns += 1

    def save(self, filename = 'policy_net.txt'):
        torch.save(self.policy_net.state_dict(), filename)

    def load(self, filename = 'policy_net.txt'):
        self.policy_net.load_state_dict(torch.load(filename))

class Game(object):
    def __init__(self, game_name):
        self.env = gym.make(game_name)
        #self.env.seed(1)
        print('action space:', self.env.action_space)
        print('observation space:', self.env.observation_space)
        self.agent = Agent(self.env.observation_space, self.env.action_space)
        self.reward_shaping = getattr(self, 'reward_shaping_{}'.format(game_name.split('-')[0]))
        self.resolved = getattr(self, 'resolved_{}'.format(game_name.split('-')[0]))

    def reward_shaping_CartPole(self, observation, next_observation, reward):
        x, x_dot, theta, theta_dot = next_observation
        r1 = (self.env.unwrapped.x_threshold - abs(x))/self.env.unwrapped.x_threshold - 0.8
        r2 = (self.env.unwrapped.theta_threshold_radians - abs(theta))/self.env.unwrapped.theta_threshold_radians - 0.5
        reward = r1 + r2
        return reward

    def reward_shaping_MountainCar(self, observation, next_observation, reward):
        position = observation[0]
        next_position, next_velocity = next_observation
        # the higher the better
        #reward = abs(position - (-0.5))     # r in [0, 1]
        #if position > -0.2: reward = 1
        if next_position > position: reward += 1
        if not hasattr(self, 'max_position'): self.max_position = next_position
        if next_position > self.max_position:
            reward += 2
            self.max_position = next_position
            print('Max position reached: {}'.format(next_position))
        return reward

    def resolved_CartPole(self, scores):
        if len(scores) >= 100 and np.mean(scores[-100:]) >= 195.0:
            print('Solved after {} episodes'.format(len(scores)-100))
            return True
        return False

    def resolved_MountainCar(self, scores):
        if len(scores) >= 10 and np.mean(scores[-10:]) >= -100.0:
            print('Solved after {} episodes'.format(len(scores)-10))
            return True
        return False

    def run_one_episode(self, optimal = False, render = False):
        observation = self.env.reset()
        done = False
        num_steps = 0
        shaping_rewards = 0
        rewards = 0
        if self.agent.add_timestep_to_observation: observation = np.append(observation, num_steps)
        while not done:
            if render: self.env.render()
            action = self.agent.choose_action(observation, optimal)
            next_observation, reward, done, _ = self.env.step(action)
            shaping_reward = reward if not hasattr(self, 'reward_shaping') else self.reward_shaping(observation, next_observation, reward)
            if self.agent.add_timestep_to_observation: next_observation = np.append(next_observation, num_steps+1)
            if not optimal: self.agent.memorize(observation, action, next_observation, shaping_reward, done)
            observation = next_observation
            num_steps += 1
            rewards += reward
            shaping_rewards += shaping_reward
        return rewards, shaping_rewards, num_steps

    def run(self, episodes, optimal = False, render = False):
        print('start to run (optimal: {})...'.format(optimal))
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
    game = Game('MountainCar-v0')

    game.run(episodes = 1000)
    game.agent.save()
    #game.agent.load()
    #game.replay_buffer.show()
    #game.run(episodes = 1, optimal = True, render = True)
    game.env.close()
    print('done')
