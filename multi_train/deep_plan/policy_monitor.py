import os
import numpy as np
import time
import math

import tensorflow as tf
from env_instance_wrapper import EnvInstanceWrapper
from model_factory import ModelFactory
import helper

class PolicyMonitor(object):
	def __init__(self, envs, network, domain, instances, summary_writer, model_factory,network_copy):
		self.domain = domain
		self.instances = instances
		self.envs = envs
		self.network = network
		self.summary_writer = summary_writer
		self.env_instance_wrapper = EnvInstanceWrapper(envs)
		self.model_factory = model_factory
		self.network_copy = network_copy
		self.eval_step = 1

	def copy_params(self):
		ModelFactory.copy_params(self.network_copy.trainable_variables, self.network.trainable_variables)

	def eval_once(self, num_episodes=5, save_model=True, get_random=False, plot_graph=False, file_name=None,
	              verbose=False, test_envs=None, expert=None, meta_logging=False):

		verbose = False
		start = time.time()
		mean_total_rewards = []
		std_total_rewards = []
		std_error_rewards = []
		mean_episode_lengths = []
		previous_action = 0
		total_rewards = []
		image_name = None
		if plot_graph:
			file_name = os.path.abspath(os.path.join(file_name, "graphs"))

		if test_envs is not None:
			self.envs = test_envs

		for i in range(len(self.envs)):
			if verbose:
				print("env = %d" % (i))
			rewards_i = []
			episode_lengths_i = []

			#  Update file name for plotting
			if plot_graph:
				dir_name = os.path.abspath(os.path.join(file_name, "{}".format(self.envs[i].instance)))
				os.makedirs(dir_name, exist_ok=True)

			state_action_cache = {}  # Caches actions so that a forward pass is not done each time
			print("Num episodes: ", num_episodes)
			for j in range(num_episodes):
				initial_state, done = self.envs[i].reset()
				state = initial_state
				episode_reward = 0.0
				episode_length = 0
				print("----------------------------\n\n") if verbose else None
				while not done:
					state_repr = self.envs[i].get_state_repr(state)

					#  Update file name for plotting
					if plot_graph:
						image_name = os.path.abspath(os.path.join(dir_name, "{}.png".format(episode_length)))

					if not get_random:
						state_str = np.array_str(state)
						if state_str not in state_action_cache:
							# If state not in cache, forward pass, and save
							action = tf.argmax(tf.reshape(
								self.network_copy.policy_prediction([state], i, self.env_instance_wrapper,
								                                    plot_graph=plot_graph,
								                                    file_name=image_name, action_taken=previous_action,
								                                    training=False), [-1])).numpy()
							state_action_cache[state_str] = action
						else:
							action = state_action_cache[state_str]
					else:
						action = np.random.randint(0, self.envs[i].get_num_actions())

					# Update previous action for plotting
					previous_action = action

					if verbose:
						print("Episode: ", episode_length)
						print("State:\n ", state_repr)
						print("Action: ", str(action))
						print("---\n")
						print(self.env_instance_wrapper.get_num_to_action(i))

					next_state, reward, done, _ = self.envs[i].test_step(action)
					episode_reward += reward
					episode_length += 1
					state = next_state


				rewards_i.append(episode_reward)
				episode_lengths_i.append(episode_length)
			mean_total_reward = np.mean(rewards_i)
			mean_episode_length = np.mean(episode_lengths_i)
			std_total_reward = np.std(rewards_i)
			mean_total_rewards.append(mean_total_reward)
			mean_episode_lengths.append(mean_episode_length)
			std_total_rewards.append(std_total_reward)
			std_error_rewards.append(std_total_reward / math.sqrt(num_episodes))
			# Total rewards have shape num_instances X num_episodes
			total_rewards.append(rewards_i)
			print("Instance:", i, "Reward_i:", mean_total_reward)

		end = time.time() - start

		if save_model:
			self.model_factory.save_ckpt()

		str_to_print = ",".join([str(mr) for mr in mean_total_rewards]) + ",,"
		str_to_print += ",".join([str(me) for me in mean_episode_lengths]) + ",,"
		str_to_print += "updates seen = " + str(self.model_factory.total_num_updates) + ",,"
		str_to_print += "episodes seen = " + str(self.model_factory.total_episodes) + ",,"
		str_to_print += "degree finished seen = " + str(self.model_factory.num_complete) + ",,"
		str_to_print += "examples seen = " + str(self.model_factory.total_examples) + "\n"
		if meta_logging:
			helper.write_content(helper.meta_logging_file, str_to_print)

		print("\n==============")
		print("Total updates so far = " + str(self.model_factory.total_num_updates))
		print("Total examples seen so far = " + str(self.model_factory.total_examples))
		print("Total episodes = " + str(self.model_factory.total_episodes))
		print("Degree finished = " + str(self.model_factory.num_complete))
		print("Average length in completion = " + str(self.model_factory.avg_steps))
		print("mean_total_rewards = " + str(mean_total_rewards))
		print("mean_episode_lengths = " + str(mean_episode_lengths))
		print("std_total_rewards = " + str(std_total_rewards))
		print("std_total_rewards = " + str(std_error_rewards))
		print("==============")
		print(str_to_print)
		return mean_total_rewards, mean_episode_lengths, end, total_rewards
