import sys
import os
import threading
import multiprocessing
import my_config
import csv
from time import time
import multiprocessing
import pandas as pd
import numpy as np
from env_instance_wrapper import EnvInstanceWrapper
import argparse

lock = threading.Lock()

curr_dir_path = os.path.dirname(os.path.realpath(__file__))
gym_path = os.path.abspath(os.path.join(curr_dir_path, "../.."))
if gym_path not in sys.path:
    sys.path = [gym_path] + sys.path
parser_path = os.path.abspath(os.path.join(curr_dir_path, "../../utils"))
if parser_path not in sys.path:
    sys.path = [parser_path] + sys.path
network_path = os.path.abspath(os.path.join(curr_dir_path, "networks"))
if network_path not in sys.path:
    sys.path = [network_path] + sys.path

from policy_monitor import PolicyMonitor
from worker_testing import Worker
import helper
from model_factory import ModelFactory

def create_workers(NUM_WORKERS, network, instances, N_train_instances, train_summary_writer, val_summary_writer, model_factory):
    network_copy = model_factory.create_network()
    pe = PolicyMonitor(
        envs=helper.make_envs(instances),
        network=network,
        domain=my_config.domain,
        instances=instances,
        summary_writer=val_summary_writer,
        model_factory=model_factory,
        network_copy=network_copy)
    workers = []
    for worker_id in range(NUM_WORKERS):
        worker_summary_writer = None
        policy_monitor = None
        if worker_id == 0:
            worker_summary_writer = train_summary_writer
            policy_monitor = pe
        worker = Worker(
            worker_id=worker_id,
            envs=helper.make_envs(instances[:N_train_instances]),
            global_network=network,
            domain=my_config.domain,
            instances=instances[:N_train_instances],
            model_factory=model_factory,
            lock=lock,
            eval_every=my_config.eval_every,
            policy_monitor=policy_monitor,
            discount_factor=my_config.discount_factor,
            summary_writer=worker_summary_writer,
            max_global_steps=my_config.max_global_steps)
        workers.append(worker)
    return workers


# Multiprocessig function for validation
def validate(num_episodes, csv_file, index, num_threads, start_index, end_index):
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.keras.backend.set_floatx('float64')

    train_instances, N_train_instances, test_instances, N_test_instances, instances = helper.get_instance_names()

    envs_ = helper.make_envs(instances)
    num_nodes_list, num_valid_actions_list, num_graph_fluent_list, num_adjacency_list = helper.get_env_metadata(envs_)
    MODEL_DIR, CHECKPOINT_DIR, train_summary_path, val_summary_path = helper.get_test_model_dir()

    NUM_WORKERS = 1

    if my_config.gpuid is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = my_config.gpuid
    if my_config.use_cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    train_summary_writer = None
    val_summary_writer = None

    policynet_optim = tf.keras.optimizers.RMSprop(lr=my_config.lr, rho=0.99, momentum=0.0, epsilon=1e-6)
    
    env_instance_wrapper_all = EnvInstanceWrapper(envs_)
    args = helper.create_modelfactory_args(policynet_optim=policynet_optim, instances=instances, env_instance_wrapper=env_instance_wrapper_all)
    helper.add_network_args(args, envs_[0], MODEL_DIR, copy_config=False)

    model_factory = ModelFactory(args)
    network = model_factory.create_network()
    env_instance_wrapper = EnvInstanceWrapper([envs_[0]])

    network.init_network(env_instance_wrapper, 0)
    for e in envs_:
        e.close()
    
    
    # Create CheckpointManager
    ckpt_parts = {}
    ckpt_parts["network"] = network
    ckpt_parts["policynet_optim"] = policynet_optim
    ckpt = tf.train.Checkpoint(**ckpt_parts)
    ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_DIR, 2000)
    model_factory.set_ckpt_metadata(ckpt, ckpt_manager)
    csv_file = os.path.abspath(os.path.join(csv_file, "{}.csv".format(index)))

    num_models = len(ckpt_manager.checkpoints)
    models_per_thread = int(num_models / num_threads)
    # Check if starting from beginning or continuing
    try:
        reader = pd.read_csv(csv_file)

        if reader.empty:
            raise FileNotFoundError

        # Number of models already evaluated
        num_done = len(reader) - 1
        best_model = reader["Best Model"][num_done]
        best_rewards = reader["Best Mean Reward"][num_done]
        current_model = max(reader["Current Model"][num_done] + 1, start_index)

    except:
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                ["Current Model", "Current Reward", "Current std", "Best Model", "Best Mean Reward", "Eval time",
                 "Val instance", "Rewards(per episode)"])
        best_model = -1
        best_rewards = -np.inf
        current_model = start_index

    workers = create_workers(NUM_WORKERS, network, test_instances, N_train_instances, train_summary_writer, val_summary_writer, model_factory)


    for i in range(current_model, end_index + 1):
        model_factory.load_ckpt(str(i))
        policy_monitor = workers[0]

        start_time = time()
        total_rewards, eval_time = policy_monitor.evaluate(num_episodes, save_model=False)
        end_time = time()
        print("Validation time:", end_time - start_time)
        # Calculate mean and variance
        mean_total_rewards = np.mean(total_rewards, axis=-1)
        std = np.std(total_rewards, axis=-1)

        if mean_total_rewards[0] > best_rewards:
            best_rewards = mean_total_rewards[0]
            best_model = i

        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i, mean_total_rewards[0], std / (num_episodes ** 0.5), best_model, best_rewards, eval_time,
                             test_instances, total_rewards])

        print("==========================")
        print("Current Process: ", index)
        print("Current model: ", i)
        print("Current Rewards: ", mean_total_rewards[0])
        print("Standard deviation:", std / (num_episodes ** 0.5))
        print("Best model:", best_model)
        print("Best rewards:", best_rewards)
        print("Eval time:", eval_time)
        print("==========================")

    print("===========Best Model==========\n", best_model, best_rewards)


def run_validation():
    train_instances, N_train_instances, test_instances, N_test_instances, instances = helper.get_instance_names()
    # Control variables
    is_human_eval = False
    get_random_policy = False
    plot_graph = False
    num_episodes = my_config.num_validation_episodes

    # Csv file path
    csv_file = os.path.abspath(os.path.join(my_config.trained_model_path, "validation_{}_{}".format(test_instances[0], num_episodes)))
    os.makedirs(csv_file, exist_ok=True)
    # Master process control
    CHECKPOINT_DIR = os.path.abspath(os.path.join(my_config.trained_model_path, "checkpoints"))
    num_threads = my_config.num_threads
    models_completed = 0

    while True:
        # total number of models
        f = open(os.path.join(CHECKPOINT_DIR, "checkpoint"))
        line = f.readlines()[0]
        num_models = int(line.split("-")[1].strip().strip("\""))

        models_per_thread = int(np.ceil((num_models - models_completed) / num_threads))
        print("Num models:", num_models, "Models per thread:", models_per_thread)
        if models_completed >= num_models:
            break

        # Spawn processes
        processes = []
        for i in range(num_threads):
            # Deciding the start index and the end index
            start_index = i * models_per_thread + 1 + models_completed
            if i == num_threads - 1:
                end_index = num_models
            else:
                end_index = start_index + models_per_thread - 1
            p1 = multiprocessing.Process(target=validate, args=(num_episodes, csv_file, i, num_threads, int(start_index), int(end_index)))
            processes.append(p1)
            p1.start()

        for p in processes:
            p.join()

        models_completed = num_models

    lst = []
    best_rewards = -np.inf
    best_model = -1
    variance = 0
    # Writing the final file
    for i in range(num_threads):
        file = os.path.abspath(os.path.join(csv_file, "{}.csv".format(i)))

        reader = pd.read_csv(file)
        num_done = len(reader) - 1
        if reader["Best Mean Reward"][num_done] > best_rewards:
            best_model = reader["Best Model"][num_done]
            best_rewards = reader["Best Mean Reward"][num_done]

        lst.append(reader)

    df = pd.DataFrame({"Current Model": [best_model],
                        "Current Reward": [best_rewards],
                        "Current std": [0],
                        "Best Model": [best_model],
                        "Best Mean Reward": [best_rewards],
                        "Eval time": [0],
                        "Val instance": [test_instances[0]],
                        "Rewards(per episode)": [[0] * num_episodes]})

    # Append final best rewards
    lst.append(df)
    frame = pd.concat(lst, ignore_index=True)
    file = os.path.abspath(os.path.join(csv_file, "final.csv"))
    frame.to_csv(file, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", help="name of the domain")
    parser.add_argument("--trained_model_path", help="directory to save model")
    parser.add_argument("--num_threads", help="number of processes in parallel to validate all ckpts", type=int)
    parser.add_argument("--val_instances", help="instances to validate on, enter comma sep values")
    args = parser.parse_args()
    
    my_config.domain = args.domain
    my_config.trained_model_path = args.trained_model_path
    my_config.num_threads = args.num_threads
    my_config.train_instance = args.val_instances
    my_config.test_instance = args.val_instances
    my_config.scheduled_train_list = [1]
    run_validation()
    
