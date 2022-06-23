import tensorflow as tf
import numpy as np
import pandas as pd
import random
import my_config
import copy
import helper

class SupervisedDataset(tf.keras.Model):
	def __init__(self, instance_list, env_instance_wrapper, dataset_folder, batch_size, num_episodes=None):
		super(SupervisedDataset, self).__init__()

		self.instance_list = [int(i) for i in instance_list]
		self.env_instance_wrapper = env_instance_wrapper
		self.dataset_folder = dataset_folder
		self.batch_size = batch_size
		self.num_episodes = num_episodes

		self.dataset = {}
		for i in range(len(self.instance_list)):
			self.load_dataset(i, self.instance_list[i])

		self.instance_order = self.instance_list.copy()
		random.shuffle(self.instance_order)

	def load_dataset(self, instance_index, instance):
		domain = self.env_instance_wrapper.envs[instance_index].instance_parser.domain
		action_dict = self.env_instance_wrapper.envs[instance_index].instance_parser.action_to_num
		action_dict['noop()'] = 0
		f = "%s/%s/%d.csv" % (self.dataset_folder, domain, instance)
		# f = self.root_folder + "/" + domain + "/"+str(instance)+".csv"

		nrows = None if self.num_episodes is None else self.num_episodes*40
		df = pd.read_csv(f, delimiter=":", header=None, nrows=nrows)

		# df = df.drop_duplicates([1,2])
		instances = np.array(df[0], dtype="float32")

		states = []
		for e in df[1]:
			a = np.array(e.split(","), dtype="float32")
			# print(a)
			states.append(a)
		states = np.stack(states)

		actions = np.expand_dims(np.array(df[2].apply(lambda x: action_dict[x]), dtype="float32"), axis=-1)
		rewards = np.expand_dims(np.array(df[3], dtype="float32"), -1)

		new_states, new_actions = [], []
		for i, s in enumerate(states):
			done = False
			for j, s_ in enumerate(new_states):
				if np.array_equal(s, s_):
					done = True
					break
			
			if done:
				continue 

			actions_freq = {}
			for j, a in enumerate(actions):
				if np.array_equal(states[j], s):
					actions_freq[a[0]] = actions_freq.get(a[0], 0) + 1
			
			if 0 in actions_freq and len(actions_freq) == 1:
				new_states.append(states[i])
				new_actions.append([0])
			else:
				if 0 in actions_freq:
					del actions_freq[0]
				most_common_action = None
				for ac in actions_freq:
					if actions_freq[ac] > actions_freq.get(most_common_action, 0):
						most_common_action = ac

				new_states.append(states[i])
				new_actions.append([most_common_action])
				
		
		new_states, new_actions = np.array(new_states), np.array(new_actions)
		print(len(states), "----After removing noop---->", len(new_states))
		states, actions = copy.deepcopy(new_states), copy.deepcopy(new_actions)
		# print(states, "\n", actions, "\n", action_dict)

		# env = helper.make_envs(my_config.train_instance)[instance_index]
		# for i in range(len(states)):
		# 	print(env.instance_parser.num_to_action)
		# 	if actions[i][0] == 0:
		# 		print(states[i], env.get_state_repr(states[i]) ," : ", "noop()")
		# 	else:
		# 		print(states[i], env.get_state_repr(states[i]) ," : ", env.instance_parser.num_to_action[int(actions[i][0])])
		# 	print("\n")

		combined_dataset = np.hstack([states, actions])
		np.random.shuffle(combined_dataset)
		states = combined_dataset[:, :-1]
		states = [states[i] for i in range(states.shape[0])]
		actions = combined_dataset[:, -1]

		# one hot actions
		actions = np.eye(len(action_dict))[actions.astype(np.int32)]
		self.dataset[instance] = (instance_index, states, tf.convert_to_tensor(actions))
