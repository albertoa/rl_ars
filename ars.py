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

# Initialize

cfg = {
	'seed': SEED
	,'env_name': 'BipedalWalker-v2'
	,'num_episodes': 1000
	,'episode_steps': 2000
	,'noise_amplitude': 0.04
	,'num_deltas': 16
	,'num_deltas_per_update': 8
	,'learning_rate': 0.1
}

trackdata = {
	'cfg': cfg
	,'episode': []
}

env = gym.make(cfg['env_name'])
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

reward = None
weights = np.zeros((action_size,state_size))

# Train
for i in range(cfg['num_episodes']):
	# Figure out the noise/deltas to evaluate per update
	noise = cfg['noise_amplitude'] * np.random.random_sample((cfg['num_deltas'],action_size,state_size))
	episode_rewards_p = []
	episode_rewards_n = []
	episode_rewards = []
	episode_reward_diffs = []
	episode_weights = []

	# Evaluate each noise set in both directions: positive and negative
	for n in noise:
		weights_p = weights + n
		r_p = run_episode(cfg,weights_p)
		episode_rewards_p.append(r_p)

		weights_n = weights - n
		r_n = run_episode(cfg,weights_n)
		episode_rewards_n.append(r_n)

		# As per section 3.3 of the paper, we will use the best performing
		# weights based on the noise, and for each noise set we use the
		# best performing direction
		if r_p > r_n:
			episode_rewards.append(r_p)
			episode_weights.append(weights_p)
		else:
			episode_rewards.append(r_n)
			episode_weights.append(weights_n)

		episode_reward_diffs.append(r_p - r_n)

	# Use the best rewards to sort which noise entries we use
	sort_idx = np.argsort(episode_rewards)[::-1][:cfg['num_deltas_per_update']]

	# Now update the weights according to section 3.3 of the paper, which
	# explains that the noise is weighted by the difference of the positive
	# and negative direction rewards
	update_sum = np.zeros(weights.shape)
	update_rewards = []
	for idx in sort_idx:
		update_sum += episode_reward_diffs[idx] * noise[idx]
		update_rewards.append(episode_rewards_p[idx])
		update_rewards.append(episode_rewards_n[idx])

	# Figure out the standard deviation of the rewards
	update_rewards_std = np.array(update_rewards).std()

	weights += cfg['learning_rate'] * update_sum / (cfg['num_deltas_per_update'] * update_rewards_std)
	reward = run_episode(cfg, weights)

	print("episode", str(i+1), "reward", reward)

	trackdata['episode'].append({
		'reward': reward
		,'weights': weights.tolist()
	})


with open("trackdata.json","w") as outf:
	json.dump(trackdata, outf)
	outf.write('\n')

exit(0)
