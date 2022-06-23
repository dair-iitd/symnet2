import tensorflow as tf
from symnet2.unet import GATConvLayer
from symnet2.action_decoder import ActionDecoder
import numpy as np

use_self_loops_in_all_adj = True
remove_attn = False

symnet_params = {"channels", "filter_size", "attn_heads", "dropout_rate", "activation", "conv_type"}

class SymNet2(tf.keras.Model):
	def __init__(self, general_params, se_params, ad_params, ge_params, tm_params):

		super(SymNet2, self).__init__()

		self.general_params = general_params
		self.se_params = se_params
		self.ad_params = ad_params
		self.ge_params = ge_params
		self.tm_params = tm_params
		self.tfm_params = self.ge_params["tfm_params"]

		self.out_deg = self.general_params["add_out_deg"]
		self.in_deg = self.general_params["add_in_deg"]
		self.bet_cen = self.general_params["add_bet_cen"]
		self.dist_leaves = self.general_params["add_dist_leaves"]
		self.use_bidir_edges = self.general_params["use_bidir_edges"]
		self.make_grid = self.general_params["make_grid"]

		self.se_type = self.se_params["type"]
		self.se_count = self.se_params["num_se"]

		self.se_list = self.get_state_encoder(self.se_params["num_se"])

		self.final_node_embedder = tf.keras.layers.Dense(units=self.se_params["out_dim"], activation=self.se_params["activation"])

		# Global Embeddings
		self.ge_type = self.ge_params["type"]
		
		self.num_ge = 1

		# Action Decoders
		self.num_action_dim = self.se_params["out_dim"]
		self.action_decoders = self.get_action_decoder()

	def get_ckpt_parts(self):
		ckpt_parts = {}
		ckpt_parts["se_list"] = self.se_list
		ckpt_parts["final_node_embedder"] = self.final_node_embedder
		ckpt_parts["action_decoders"] = self.action_decoders
		
		return ckpt_parts

	def init_network(self, env_wrapper, instance):
		initial_state, _ = env_wrapper.envs[instance].reset()  # Initial state
		self.policy_prediction(states=[initial_state], instance=instance, env_wrapper=env_wrapper)

	def get_state_encoder(self, num_se):
		se_list = []
		for _ in range(num_se):
			se_list.append(GATConvLayer(**dict((k, self.se_params[k]) for k in symnet_params)))
		return se_list
	

	def get_action_decoder(self):
		action_decoders = []
		for _ in range(self.ad_params["num_action_templates"]):
			action_decoders.append(ActionDecoder(self.ad_params))
		return action_decoders

	
	def get_parsed_state(self, states, instance, env_wrapper):
		adjacency_matrix, node_features, graph_features = env_wrapper.get_parsed_state(states, instance)
		action_details = env_wrapper.get_action_details(instance)

		if self.use_bidir_edges:
			adjacency_matrix = adjacency_matrix + tf.transpose(adjacency_matrix, perm=[0, 1, 3, 2])
			adjacency_matrix = tf.clip_by_value(adjacency_matrix, 0, 1)
		
		return adjacency_matrix, node_features, graph_features, action_details


	def policy_prediction(self, states, instance, env_wrapper, sample=False, action=None, plot_graph=False, file_name=None, action_taken=None, training=True):
		adjacency_matrix, node_features, graph_features, action_details = self.get_parsed_state(states, instance, env_wrapper)
		adjacency_matrix = np.transpose(adjacency_matrix, [0, 1, 3, 2])
		batch_size = node_features.shape[0]
		num_nodes = node_features.shape[1]

		# 1. Get all state and global embedding outputs
		se_embed_l = []
		for i, se in enumerate(self.se_list):
			res = se(node_features, adjacency_matrix[i], use_self_loops_in_all_adj, remove_attn)
			se_embed_l.append(res)

		# 2. Get final node and global embedding by passing all outputs through a dense layer,
		node_embed = tf.concat(se_embed_l, axis=-1)
		final_node_embedding = self.final_node_embedder(node_embed)
		global_embed = tf.reshape(tf.concat([tf.reduce_max(final_node_embedding, axis=1), graph_features], axis=1), [batch_size, -1])
		
		# 3. Pass resulting embedding from action decoders
		action_scores = [0 for i in range(len(action_details))]  # Score of each action
		action_affects = env_wrapper.envs[instance].instance_parser.action_affects
		remove_dbn = env_wrapper.envs[instance].instance_parser.remove_dbn
		for i in range(len(action_details)):
			action_template = action_details[i][0]
			input_nodes = list(action_details[i][1])
			arg_nodes = action_details[i][2]
			global_embed_temp = global_embed

			if len(arg_nodes) == 0:  # Unparametrized action
				action_scores[i] = self.action_decoders[action_template]([global_embed_temp, None, training])
			else:
				if len(input_nodes) > 0:
					temp_embedding_list = [  # Select embeddings of nodes used
						tf.reshape(final_node_embedding[:, inp, :], [batch_size, self.num_action_dim]) for inp in input_nodes]
					node_state_embedding_concat = tf.concat(temp_embedding_list, axis=1)  # Concat embeddings of all affected nodes
					node_state_embedding_reshape = tf.reshape(node_state_embedding_concat, [batch_size, len(input_nodes), self.num_action_dim])
					node_state_embedding_pooled = tf.reshape(tf.reduce_max(node_state_embedding_reshape, axis=1), [batch_size, self.num_action_dim])  # Max Pool
					arg_embedding_list = [tf.reshape(final_node_embedding[:, inp, :], [batch_size, self.num_action_dim]) for inp in arg_nodes]
					node_state_embedding_pooled = tf.concat(arg_embedding_list + [node_state_embedding_pooled], axis=1)
					action_scores[i] = self.action_decoders[action_template]([node_state_embedding_pooled, global_embed_temp, training])
				else:
					if remove_dbn:
						arg_embedding_list = [tf.reshape(final_node_embedding[:, inp, :], [batch_size, self.num_action_dim]) for inp in arg_nodes]
						action_scores[i] = self.action_decoders[action_template]([tf.concat(arg_embedding_list, axis=1), global_embed_temp, training])
					else:
						gnd_action_affects = action_affects[action_template]
						# Wildfire case; Treat as NOOP
						if gnd_action_affects:
							action_template = action_details[0][0]
							action_scores[i] = self.action_decoders[action_template]([global_embed_temp, None, training])
						else:
							arg_embedding_list = [tf.reshape(final_node_embedding[:, inp, :], [batch_size, self.num_action_dim]) for inp in arg_nodes]
							action_scores[i] = self.action_decoders[action_template]([tf.concat(arg_embedding_list, axis= 1), global_embed_temp, training])
		action_scores = tf.concat(action_scores, axis=-1)

		if sample:
			logits = tf.nn.log_softmax(action_scores)
			return tf.random.categorical(logits=logits, num_samples=batch_size, dtype=tf.int32)  # Return sampled actions
		else:
			probs = tf.nn.softmax(action_scores)  # Expected shape is (batch_size,num_actions)
			return probs