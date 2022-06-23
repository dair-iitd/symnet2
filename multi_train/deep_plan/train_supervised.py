import sys
import os
import multiprocessing
import my_config
from env_instance_wrapper import EnvInstanceWrapper
import time
import argparse

lock = multiprocessing.Lock()

curr_dir_path = os.path.dirname(os.path.realpath(__file__))
gym_path = os.path.abspath(os.path.join(curr_dir_path, "../.."))
if gym_path not in sys.path:
    sys.path = [gym_path] + sys.path
parser_path = os.path.abspath(os.path.join(curr_dir_path, "../../utils"))
if parser_path not in sys.path:
    sys.path = [parser_path] + sys.path
network_path = os.path.abspath(os.path.join(curr_dir_path,"networks"))
if network_path not in sys.path:
    sys.path = [network_path] + sys.path

import tensorflow as tf
from policy_monitor import PolicyMonitor
import helper
from model_factory import ModelFactory
from supervised_dataset import SupervisedDataset

# @tf.function
def train_step(network, x, y, instance, env_instance_wrapper, loss_fn, optimizer, grad_clip_value):
    with tf.GradientTape() as policynet_tape:
        policynet_pred = network.policy_prediction(x, instance, env_instance_wrapper)
        policynet_loss = loss_fn(y, policynet_pred)
    grads_of_policynet = policynet_tape.gradient(policynet_loss, network.trainable_variables)
    grads_of_policynet = tf.clip_by_global_norm(grads_of_policynet, grad_clip_value)

    optimizer.apply_gradients(zip(grads_of_policynet[0], network.trainable_variables))
    return policynet_loss

def train():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.keras.backend.set_floatx('float64')

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    if my_config.gpuid is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = my_config.gpuid
    if my_config.use_cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    train_instances, N_train_instances, test_instances, N_test_instances, instances = helper.get_instance_names()
    envs_ = helper.make_envs(instances)
    num_nodes_list, num_valid_actions_list, num_graph_fluent_list, num_adjacency_list = helper.get_env_metadata(envs_)
    MODEL_DIR, CHECKPOINT_DIR, train_summary_path, val_summary_path = helper.get_model_dir()
    NUM_WORKERS = 1
    train_summary_writer = None
    val_summary_writer = None

    policynet_optim = tf.keras.optimizers.Adam(lr=my_config.lr)

    args = helper.create_modelfactory_args(policynet_optim=policynet_optim)
    helper.add_network_args(args, envs_[0],MODEL_DIR)

    model_factory = ModelFactory(args)
    env_instance_wrapper = EnvInstanceWrapper(envs_[:N_train_instances])
    
    network = model_factory.create_network()
    network.init_network(env_instance_wrapper, 0)

    # Create policy_monitor
    network_copy = model_factory.create_network()
    pe = PolicyMonitor(
        envs=helper.make_envs(instances),
        network=network,
        domain=my_config.domain,
        instances=instances,
        summary_writer=val_summary_writer,
        model_factory=model_factory,
        network_copy=network_copy)

    network.init_network(env_instance_wrapper, 0)
    pe.network_copy.init_network(env_instance_wrapper, 0)
    pe.copy_params()

    for e in envs_:
        e.close()

    # Create CheckpointManager
    # ckpt_parts = network.get_ckpt_parts()
    ckpt_parts = {}
    ckpt_parts["network"] = network
    ckpt_parts["policynet_optim"] = policynet_optim
    ckpt = tf.train.Checkpoint(**ckpt_parts)
    ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_DIR, 2000)
    model_factory.set_ckpt_metadata(ckpt, ckpt_manager)

    if my_config.use_pretrained:
        model_factory.load_ckpt(ckpt_num=None)

    # SUPERVISED TRAINING STARTS
    # Training dataset
    dataset_folder = my_config.dataset_folder
    batch_size = my_config.batch_size
    dataset_ob = SupervisedDataset(train_instances, env_instance_wrapper, dataset_folder, batch_size, num_episodes=None)
    # Loss Function
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    grad_clip_value = model_factory.grad_clip_value

    for epoch in range(my_config.epochs):
        print("\n\n------Start of epoch %d" % (epoch,))
        start_time = time.time()

        for ins in dataset_ob.instance_order:
            print("Instance: %d" % (ins,))
            instance, states, actions = dataset_ob.dataset[ins]
            total_size = len(states)
            cur_loc= 0
            while cur_loc < total_size:
                if cur_loc + batch_size < total_size:
                    x = states[cur_loc: cur_loc + batch_size]
                    y = actions[cur_loc: cur_loc + batch_size]
                    cur_loc += batch_size
                else:
                    x = states[cur_loc:]
                    y = actions[cur_loc:]
                    cur_loc = total_size+1

                loss_value = train_step(network, x, y, instance, env_instance_wrapper, loss_fn, model_factory.policynet_optim, grad_clip_value)
            print("\tTraining loss (for last batch) after seeing instance %d: %.4f" % (ins, float(loss_value)))
            print("Seen so far: %d samples" % (total_size))

        # Validation
        if epoch%my_config.ckpt_freq==0:
            pe.copy_params()
            _, _, eval_time, total_rewards = pe.eval_once(meta_logging=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", help="name of the domain")
    parser.add_argument("--model_dir", help="directory to save model")
    parser.add_argument("--train_instances", help="instances to train on")
    parser.add_argument("--val_instances", help="instances to validate on")
    args = parser.parse_args()

    my_config.domain = args.domain
    my_config.model_dir = args.model_dir
    my_config.train_instance = args.train_instances
    my_config.test_instance = args.val_instances
    
    print("Domain: ", my_config.domain)
    print("Model dir: ", my_config.model_dir)
    train()