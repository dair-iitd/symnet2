import collections
import numpy as np
import tensorflow as tf
import time

from model_factory import ModelFactory
from env_instance_wrapper import EnvInstanceWrapper
import numpy as np

import my_config as my_config

debug = False
mask_weights = False
use_ep_greedy = False
epsilon = 0.1


scheduled_training = False
scheduled_train_list = my_config.scheduled_train_list

Transition = collections.namedtuple("Transition", ["instance", "state", "action", "reward", "next_state", "done", 'action_onehot'])

class Worker(object):
	def __init__(self, worker_id, envs, global_network, domain, instances,
		model_factory, lock, eval_every,policy_monitor=None,discount_factor=0.99, 
		summary_writer=None, max_global_steps=None):
		self.worker_id = worker_id
		self.domain = domain
		self.instances = instances
		self.model_factory = model_factory
		self.discount_factor = discount_factor
		self.max_global_steps = max_global_steps
		self.global_network = global_network
		self.global_counter = model_factory.global_counter
		self.local_network = model_factory.create_network()
		self.summary_writer = summary_writer
		self.envs = envs
		self.current_instance = 0
		self.state = None
		self.env_instance_wrapper = EnvInstanceWrapper(envs)
		self.local_network.init_network(self.env_instance_wrapper,0)
		self.lock = lock
		self.eval_every = eval_every
		self.policy_monitor = policy_monitor

		self.schedule = []
		print(len(self.instances), len(scheduled_train_list))
		# assert len(self.instances) == len(scheduled_train_list)
		for i in range(len(self.instances)):
			for j in range(scheduled_train_list[i]):
				self.schedule.append(i)

		self.len_schedule = len(self.schedule)
		self.schedule_pos = 0

		# ! Only for academic advising
		self.degree_completed = False
		self.steps_taken = 0

		# For reward prediction
		self.action_noop = 0

		if self.policy_monitor is not None:
			self.global_network.init_network(self.env_instance_wrapper, 0)
			self.policy_monitor.network_copy.init_network(self.env_instance_wrapper, 0)
			self.policy_monitor.copy_params()
		self.copy_params()

		# To train using a dataset
		self.save_transitions = self.train_from_dataset = False

		self.masks = [None, None]
		self.masks = [None, None]

	def copy_params(self, acquire_lock=True):
		if not acquire_lock:
			ModelFactory.copy_params(self.local_network.trainable_variables, self.global_network.trainable_variables)
			return
		self.lock.acquire()
		ModelFactory.copy_params(self.local_network.trainable_variables, self.global_network.trainable_variables)
		self.lock.release()

	def evaluate(self, num_episodes=5, save_model=True, get_random = False, plot_graph=False, file_name=None, test_envs=None):
		self.policy_monitor.copy_params()
		_,_,eval_time,total_rewards = self.policy_monitor.eval_once(num_episodes, save_model, get_random, plot_graph, file_name,test_envs=test_envs)
		return total_rewards, eval_time
