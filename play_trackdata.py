#!/usr/bin/env python3

# Try our best to configure all random seeds to be able to reproduce
# results. There are still some issues within the rewards
SEED=789

import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)

import os
import gym
import json

# BipedalWalker-v2
#
# https://github.com/openai/gym/blob/master/gym/envs/box2d/bipedal_walker.py
#
# action vector is for the 4 links/joints (2 per leg) with values between -1 to 1
#
# state:
#    hull: angle speed, angular velocity, horizontal speed, vertical speed
#    joints: position and angular speed
#    legs contact with ground
#    10 lidar rangefinder measurements (to help to deal with the hardcore version)
#
# Using ARS, from the paper: https://arxiv.org/abs/1803.07055
#
# Other implementations to be considered:
# https://github.com/colinskow/move37/tree/master/ars
# https://github.com/modestyachts/ARS
# https://github.com/alexis-jacq/numpy_ARS
# 

def run_episode(cfg, weights, render = False):
	# Reset the gym environ
	state = env.reset()
	env.seed(cfg['seed'])

	# Initialize episode variables
	episode_reward = 0
	state_running_mean = np.zeros(state_size)
	state_running_mean_diff = np.zeros(state_size)
	state_variance = np.zeros(state_size)

	# Run the episode steps
	for step_idx in range(cfg['episode_steps']):
		# As per section 3.2 of the paper, we normalize the states
		# to put equal weight on each component of the state. Specially
		# important since we don't know the ranges of values (nor the
		# meaning) of the state components

		# First deal with the mean and mean difference
		mean = state_running_mean + (state - state_running_mean) / (step_idx + 1)
		state_running_mean_diff += (state - state_running_mean) * (state - mean)
		state_running_mean = mean

		# Calculate the variance and standard deviation
		state_variance = (state_running_mean_diff / (step_idx+1)).clip(min=1e-2)
		state_std = np.sqrt(state_variance)

		# Now we can normalize
		state = (state - state_running_mean) / state_std

		# Perform the action and update the rewards
		action = weights.dot(state)
		state, reward, done, info = env.step(action)

		# Clip reward between [-1,1]
		reward = max(min(reward, 1), -1)
		episode_reward += reward

		if render:
			env.render()
		if done:
			break

	return episode_reward


with open("trackdata.json","r") as inf:
	trackdata = json.load(inf)
	cfg = trackdata['cfg']
	env = gym.make(cfg['env_name'])
	state_size = env.observation_space.shape[0]
	action_size = env.action_space.shape[0]
	weights = np.asarray(trackdata['episode'][-1]['weights'])
	reward = run_episode(cfg, weights, True)
	print("Reward:",reward)

exit(0)
