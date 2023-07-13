### The code for the successor version, Symnet 3.0, published in [UAI 2023](https://openreview.net/forum?id=sWzgZUmJich), is available at [SymNet 3.0](https://github.com/dair-iitd/symnet3)

<hr />

## SymNet 2.0

Repo contains the code for UAI 2022 paper "**SymNet 2.0: Effectively handling Non-Fluents and Actions in Generalized Neural Policies for RDDL Relational MDPs**".

<hr />

### Libraries required:

1. python3  
2. tensorflow=2.0  
3. unittest  
4. multiprocessing  
5. threading
6. shutil
7. better_exceptions
8. pickle
9. networkx 
10. scipy

Note that we also use [Spektral](https://github.com/danielegrattarola/spektral/), however, there is no need to install it as we have incorporated that in the repo.

<hr/>

### How to run an experiment:

1. Generate trajectories using PROST as
   1. Follow the instructions mentioned at https://github.com/prost-planner/prost. to setup PROST 2014.
   2. Run PROST using below command and save the logs generated by PROST to a file $RESULT_FILE.
   ```commandline
   ./prost.py $INSTANCE "[Prost -s 1 -se [IPC2014]]" > $RESULT_FILE
   ```

2. Generate a dataset from a raw trajectory: 
```commandline
python dataset_builder.py --domain DOMAIN --start_instance START_INSTANCE --num_instances NUM_INSTANCES --prost_log PROST_LOG --save_folder SAVE_FOLDER
```
For example, assume that PROST_LOG contains files 1.result, 2.result...10.result of domain academic_advising. 
To generate the dataset for all these files, run 
```commandline
python dataset_builder.py --domain academic_advising --start_instance 1 --num_instances 10 --prost_log PROST_LOG --save_folder SAVE_FOLDER
```


3. Set hyperparameters of the experiment as explained at the end of this readme.
4. Run these commands from the main folder once:
```commandline
for i in {1..10}
do
cp -r ./rddl/lib ./gym/envs/rddl/rddl
cp ./gym/envs/rddl/rddl/lib/clibxx.so ./gym/envs/rddl/rddl/lib/clibxx$i.so
cp ./rddl/lib/clibxx.so ./rddl/lib/clibxx$i.so
done
```

5. To train SymNet2.0 on a domain run below command.
```commandline
python ./multi_train/deep_plan/train_supervised.py --domain DOMAIN --model_dir MODEL_DIR --train_instances TRAIN_INSTANCES --val_instances VAL_INSTANCES
``` 
Example, to train on academic_advising instances 1,2,3, and validate on instance 4, run:
```commandline
python ./multi_train/deep_plan/train_supervised.py --domain academic_advising --model_dir MODEL_DIR --train_instances "1,2,3" --val_instance "4"
``` 

6. To validate all models:
```commandline
python ./multi_train/deep_plan/validate.py --domain DOMAIN --trained_model_path PATH --num_threads NUM_THREADS --val_instances VAL_INSTANCES
```

Example, to validate on instance 4 of academic_advising using 6 threads, 
```commandline
python ./multi_train/deep_plan/validate.py --domain academic_advising --trained_model_path PATH --num_threads 6 --val_instances "4"
```

7. To test the best model:
```commandline
python ./multi_train/deep_plan/test.py --domain DOMAIN --trained_model_path PATH --num_threads NUM_THREADS --test_instances TEST_INSTANCES
```

Example, to test on instance 1,2,3,4,5 of academic_advising using 2 threads, 
```commandline
python ./multi_train/deep_plan/test.py --domain academic_advising --trained_model_path PATH --num_threads 2 --val_instances "1,2,3,4,5"
```
<br/>
<hr/>

### Hyper Parameters explained:

You can set the general hyperparameters in two files: 
For my_config:
1. domains: academic_advising, crossing_traffic, game_of_life, navigation, skill_teaching, sysadmin, tamarisk, traffic, wildfire
2. num_validation_episodes: Number of episodes to validate on
3. num_testing_episodes: Number of episodes to test on
4. dataset_folder: Folder storing the datasets generated using scripts mentioned above
5. batch_size: Batch size for supervised training
6. epochs: Number of epochs to train for
7. ckpt_freq: Frequency of saving checkpoint
8. lr: Learning rate

[//]: # (8. t_max: Number of steps in each episode before applying an A3C update)
[//]: # (grad_clip_value: Value at which gradient is clipped.)
<br/>
<hr/>

### Code Organization:

1. ./gym contains the code for setting up the RDDL domains and instances in ATARI gym based framework (used for generating the trajectories during validation and testing).
2. ./rddl contains files (RDDL, parsed files and dbbn files) for various domains and instances used.
3. ./multi_train/deep_plan contains the main code.

For the predecessor version refer to [SymNet 1.0](https://github.com/dair-iitd/symnet)
