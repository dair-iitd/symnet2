import argparse
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
from multiprocessing import Manager

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



def evaluate(test_instance, num_episodes, process_index, output_dict):
    is_human_eval = False
    get_random_policy = False
    plot_graph = False
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.keras.backend.set_floatx('float64')

    envs_ = helper.make_envs([test_instance])
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
    args = helper.create_modelfactory_args(policynet_optim=policynet_optim, instances=[test_instance], env_instance_wrapper=env_instance_wrapper_all)
    helper.add_network_args(args, envs_[0], MODEL_DIR, copy_config=False)

    model_factory = ModelFactory(args)
    network = model_factory.create_network()
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

    workers = create_workers(NUM_WORKERS, network, [test_instance],
                             1, train_summary_writer, val_summary_writer, model_factory)
    if not get_random_policy and not is_human_eval:
        model_factory.load_ckpt(my_config.exact_checkpoint)

    policy_monitor = workers[0]
    
    if is_human_eval:
        policy_monitor.evaluate_human(num_episodes)
        return

    print("In process:", process_index)
    total_rewards, _ = policy_monitor.evaluate(num_episodes=num_episodes,
                                               save_model=False, get_random=get_random_policy,
                                               plot_graph=plot_graph, file_name=my_config.trained_model_path)

    sys.stdout = sys.__stdout__
    print("Rewards:", total_rewards)
    output_dict[process_index] = total_rewards
   

def test():
    train_instances, N_train_instances, test_instances, N_test_instances, instances = helper.get_instance_names()
    print(test_instances)
    # Control variables
    get_random_policy = False
    num_episodes = my_config.num_testing_episodes
    
    # print(f"Launching {num_threads} processes")
    manager = Manager()
    rewards_all_instances = []
    output_dict = manager.dict()

    with multiprocessing.Pool(my_config.num_threads) as pool:
        print(pool._processes)
        results = pool.starmap(evaluate, [(inst, num_episodes, i, output_dict) for i, inst in enumerate(test_instances)])


    print(output_dict, len(test_instances))
    for i in range(len(test_instances)):
        rewards_all_instances.append(output_dict[i])

    
    total_rewards = np.array(rewards_all_instances)
    
    if not get_random_policy:
        csv_file = os.path.abspath(os.path.join(my_config.trained_model_path, "testing_{}.csv".format(num_episodes)))
    else:
        csv_file = os.path.abspath(
            "/scratch/cse/dual/cs5170419/results/random/testing_random_{}_{}.csv".format(my_config.domain,
                                                                                        num_episodes))
    
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Instance", "Mean Rewards", "Standard deviation", "Rewards(per episode)"])

    for i in range(len(test_instances)):
        current_rewards = total_rewards[i]
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([test_instances[i], np.mean(current_rewards), np.std(current_rewards) / (num_episodes ** 0.5),
                            current_rewards])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", help="name of the domain")
    parser.add_argument("--trained_model_path", help="directory to save model")
    parser.add_argument("--num_threads", help="number of processes in parallel to test all instances", type=int)
    parser.add_argument("--test_instances", help="instances to test on, enter comma sep values")
    args = parser.parse_args()

    my_config.domain = args.domain
    my_config.trained_model_path = args.trained_model_path
    my_config.train_instance = ""
    my_config.test_instance = args.test_instances
    my_config.scheduled_train_list = [1 for i in range(len(my_config.train_instance.split(",")))]
    my_config.num_threads = args.num_threads

    # This selects the best checkpoint from final.csv in the validation folder
    final_csv = os.path.join(my_config.trained_model_path, 'validation_4_200/final.csv')
    df = pd.read_csv(final_csv)
    my_config.exact_checkpoint = str(df.iloc[df.shape[0] - 1][0])
    print("Exact checkpoint:", my_config.exact_checkpoint)
    test()
