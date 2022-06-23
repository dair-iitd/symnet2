config_file = __file__
# ==============================================
# General settings for flags
# ==============================================
exp_description = "prost"
domain = "academic_advising"                      # 3
model_dir =  ""
network_type = "symnet2" # "symnet"

num_validation_episodes = 200
num_testing_episodes = 200

train_instance = ""
test_instance = ""

scheduled_train_list = [1 for i in range(len(train_instance.split(",")))]

dataset_folder = "/home/cse/dual/cs5180404/deep-rl-transfer7-utils/datasets/IPPC_2014"
batch_size = 40
epochs = 501
ckpt_freq = 50

# Settings: To run New merged Model where we add edge type and another adj for DBN edges and remove action based adj:
add_separate_adj = False
remove_dbn = False
add_edge_type = True
merged_model = True

gpuid = "0"
use_cpu_only = True
lr = 0.003
t_max = 20                              # Number of steps to take in an instance while training
max_global_steps = None
use_pretrained = True
discount_factor = 0.99                  # Only for training
reset = False
eval_every = 600                       # in seconds
parallelism = 4

# new dev
grad_clip_value = 5.0
use_type_encoding = True