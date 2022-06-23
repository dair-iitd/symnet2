from helper import get_adj_mat_from_list,transform_mat
import numpy as np
import tensorflow as tf
import my_config

class EnvInstanceWrapper:
	def __init__(self, envs):
		self.envs = envs
		self.fluent_feature_dims = None
		self.nonfluent_feature_dims = None
		self.adjacency_mat = [None] * len(self.envs)
		self.extended_adjacency_mat = [None] * len(self.envs)
		self.nf_features = [None] * len(self.envs)
		self.action_details = [None] * len(self.envs)
		self.extended_action_details = [None] * len(self.envs)
		self.num_action_nodes = [None] * len(self.envs)
		self.fill_env_meta_data()  # Fetch meta data and nf_features for all envs
		self.modify_adj()  # Expand nbrhood and add bias for all envs

		# For Structural Features
		self.struct_features_all = None
		self.num_nodes = [self.adjacency_mat[i][0].shape[0] for i in range(len(envs))]
		self.nx_graphs = []     # List of nx graphs of each env
		self.out_degree = []    # List of list of out_degrees of each node in each env.
		self.in_degree = []     # List of list of in_degrees of each node in each env.
		self.bet_cent = []      # List of list of between_degrees of each node in each env.
		self.dist_leaves = []   # List of list of mean_distance from leaves of each node in each env.


	def fill_env_meta_data(self):
		for i in range(len(self.envs)):
			self.fluent_feature_dims, self.nonfluent_feature_dims = self.envs[i].get_feature_dims()
			self.nf_features[i] = self.envs[i].get_nf_features()
			self.action_details[i] = self.envs[i].get_action_details()
			self.extended_action_details[i] = self.envs[i].get_extended_action_details()
			self.num_action_nodes[i] = self.envs[i].get_num_action_nodes()

	def modify_adj(self):
		for i in range(len(self.envs)):
			adjacency_list = self.envs[i].get_adjacency_list()
			self.adjacency_mat[i] = [get_adj_mat_from_list(aj) for aj in adjacency_list]
			# extended_adjacency_list = self.envs[i].get_extended_adjacency_list()
			# self.extended_adjacency_mat[i] = [get_adj_mat_from_list(aj) for aj in extended_adjacency_list]

	def get_processed_adj_mat(self, i, batch_size):
		return np.array([[self.adjacency_mat[i][j] for batch in range(batch_size)] for j in range(len(self.adjacency_mat[i]))]).astype(np.float32)

	def get_processed_extended_adj_mat(self,i,batch_size):
		return np.array([[self.extended_adjacency_mat[i][j] for batch in range(batch_size)] for j in range(len(self.extended_adjacency_mat[i]))]).astype(np.float32)

	def get_processed_input(self, states, i,nf=True):
		def state2feature(state):
			feature_arr = self.envs[i].get_fluent_features(state)
			if nf == True:
				feature_arr = np.hstack((feature_arr, self.nf_features[i]))
			
			if my_config.use_type_encoding:
				feature_arr = np.hstack((feature_arr, self.envs[i].instance_parser.type_encoding))
			return feature_arr
		features = np.array(list(map(state2feature, states))).astype(np.float32)
		return features

	def get_processed_graph_input(self, states, i):
		def state2feature(state):
			feature_arr = self.envs[i].get_graph_fluent_features(state)
			return feature_arr
		features = np.array(list(map(state2feature, states))).astype(np.float32)
		return features

	def get_action_details(self,instance):
		return self.action_details[instance]

	def get_extended_action_details(self,instance):
		return self.extended_action_details[instance]

	def get_parsed_state(self, states, instance,extended=False):
		adj_preprocessed_mat = self.get_processed_adj_mat(instance,len(states))
		# if extended == True:
		# 	adj_preprocessed_mat = self.get_processed_extended_adj_mat(instance,len(states))
		input_features_preprocessed = self.get_processed_input(states, instance)
		graph_input_features_preprocessed = self.get_processed_graph_input(states, instance)
		return adj_preprocessed_mat, input_features_preprocessed, graph_input_features_preprocessed

	def get_attr(self,instance,batch_size,name):
		return [self.envs[instance].get_attr(name) for _ in range(batch_size)]

	def get_action_one_hot(self, instance, action):
		"""
		:param action: [batch, 1]
		:return: [batch, one_hot_dim]
		"""
		# Note: To get the one hot encoding of an action if
		one_hot = tf.one_hot(action, len(self.action_details[instance]))
		return one_hot

		# manual computation
		# batch_size = action.shape[0]
		# action_details = self.action_details[instance]
		# one_hot = tf.zeros([batch_size, len(action_details)])
		# for i in range(batch_size):
		# 	action_id = action_details[action[i].numpy()[0]][0]
		# 	one_hot[i][action_id] = 1
		# return one_hot

	def get_expected_step(self, instance, states, action_var, num_samples=50):
		def state2expectedStep(state):
			return self.envs[instance].get_expected_step(state, action_var, num_samples)

		next_step = list(map(state2expectedStep, states))
		next_states = np.array([e[0] for e in next_step]).astype(np.float32)
		next_rewards = np.array([e[1] for e in next_step]).astype(np.float32)
		return next_states, next_rewards

	def get_expected_step_cpt(self, instance, states, action_var, num_samples=50):
		def state2expectedStep(state):
			return self.envs[instance].get_expected_step_cpt(state, action_var)

		next_step = list(map(state2expectedStep, states))

		next_states = np.array([e[0] for e in next_step]).astype(np.float32)
		next_rewards = np.array([e[1] for e in next_step]).astype(np.float32)

		return next_states, next_rewards

	def get_expected_parsed_state(self, states, action_var, instance, use_sampling=False, num_samples=50):
		if use_sampling:
			next_states, next_rewards = self.get_expected_step(instance, states, action_var, num_samples)
		else: #USE CPT Table from mdp file
			next_states, next_rewards = self.get_expected_step_cpt(instance, states, action_var, num_samples)

		# adj_preprocessed_mat = self.get_processed_adj_mat(instance, len(next_statesstates))
		# if extended == True:
		# 	adj_preprocessed_mat = self.get_processed_extended_adj_mat(instance, len(next_states))
		input_features_preprocessed = self.get_processed_input(next_states, instance)
		graph_input_features_preprocessed = self.get_processed_graph_input(next_states, instance)

		return input_features_preprocessed, graph_input_features_preprocessed, next_rewards
		# return adj_preprocessed_mat, input_features_preprocessed, graph_input_features_preprocessed
    
	def get_node_dict(self, instance):
		return self.envs[instance].get_node_dict()
	
	def get_num_to_action(self, instance):
		return self.envs[instance].get_num_to_action()
    
	def get_max_reward(self, instance):
		return self.envs[instance].get_max_reward()
	
	def get_min_reward(self, instance):
		return self.envs[instance].get_min_reward()

	def get_rrnet_steps(self, instance):
		return len(self.envs[instance].instance_parser.objects_in_instance_file_gnd)
		# if my_config.domain == 'graph_path':
		# 	keys = self.get_node_dict(instance)
		# 	x_max, y_max = 0, 0
		# 	for k in keys:
		# 		if "," in k:
		# 			continue
		# 		else:
		# 			x, y = int(k.split("a")[1]), int(k.split("a")[2])
		# 			x_max = max(x, x_max)
		# 			y_max = max(y, y_max)
		#
		# 	return 2*(x_max + y_max - 2)
		#
		# elif my_config.domain == 'navigation':
		# 	keys = self.get_node_dict(instance)
		# 	x_max, y_max = 0, 0
		# 	for k in keys:
		# 		if "," in k:
		# 			continue
		# 		else:
		# 			if k[0] == "x":
		# 				x_max = max(x_max, int(k[1:]))
		# 			elif k[0] == "y":
		# 				y_max = max(y_max, int(k[1:]))
		#
		# 	return 2*(x_max + y_max - 2)
		# elif my_config.domain == 'graph_path_direction':
		# 	keys = self.get_node_dict(instance)
		# 	x_max, y_max = 0, 0
		# 	for k in keys:
		# 		if "," in k:
		# 			continue
		# 		else:
		# 			x, y = int(k.split("a")[1]), int(k.split("a")[2])
		# 			x_max = max(x, x_max)
		# 			y_max = max(y, y_max)
		#
		# 	return 2*(x_max + y_max - 2)
		#