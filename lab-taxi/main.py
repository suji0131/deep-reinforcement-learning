from agent import Agent
from monitor import interact
import gym
import numpy as np

env = gym.make('Taxi-v2')
no_of_actions = env.action_space.n
agent = Agent(no_of_actions=no_of_actions)
avg_rewards, best_avg_reward = interact(env, agent)