import tensorflow as tf
import my_config

# ==================================================================
# Unet decoder params
# ==================================================================

activation = "lrelu"
activation_fn = tf.nn.leaky_relu

general_params = {"use_bidir_edges" : False,
                  "use_rev_graph" : False,
                  "add_out_deg" : True,
                  "add_in_deg" : True,
                  "add_bet_cen" : True,
                  "add_dist_leaves" : True,

                  "make_grid" : False
                  }

se_params = {"type" : "symnet",
	             "channels" : 6,              # Channels in each GAT (num_gat_features in symnet)
				 "filter_size" : 1,            # Filter size in each GAT (num of
				 "attn_heads" : 8,             # Num of attn heads
				 "out_dim" : 20,                         # out_dim of se (num_features in symnet0
				 "out_filters" : 1,                      # Keeping same as filter_size_l
				 "add_self_loops" : True,
				 "dropout_rate" : 0,
				 "activation" : activation_fn,
				 "num_se" : None,
				 "conv_type" : "GAT"  # GAT | GNNCNNStyle
	             }

ad_params = {"type" : "symnet",
			"activation" : activation,
			"decoder_dim" : 20,
			"num_action_templates" : None,      # To be set internally
             "use_ge" : True,
			"dropout_rate": 0,
            }

ge_params = {"type" : "node_max",               # unet_and_node_max | node_max | unet
             "use_multiple" : False
             }


tm_params = {"type" : "symnet"}

tfm_params = {"num_blocks" : 1,
              "embed_dim" : 20,
              "channels" : 10,
              "attn_heads" : 3,
              "transformer_ff_units" : 10,
              "dropout_rate": 0,
              "activation": activation_fn
              }

ge_params["tfm_params"] = tfm_params
se_params["global_embed_units"] = None