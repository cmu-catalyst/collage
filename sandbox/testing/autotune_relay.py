# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Auto-tuning a Convolutional Network for NVIDIA GPU
==================================================
**Author**: `Lianmin Zheng <https://github.com/merrymercy>`_, `Eddie Yan <https://github.com/eqy/>`_

Auto-tuning for specific devices and workloads is critical for getting the
best performance. This is a tutorial on how to tune a whole convolutional
network for NVIDIA GPU.

The operator implementation for NVIDIA GPU in TVM is written in template form.
The template has many tunable knobs (tile factor, unrolling, etc).
We will tune all convolution and depthwise convolution operators
in the neural network. After tuning, we produce a log file which stores
the best knob values for all required operators. When the TVM compiler compiles
these operators, it will query this log file to get the best knob values.

We also released pre-tuned parameters for some NVIDIA GPUs. You can go to
`NVIDIA GPU Benchmark <https://github.com/apache/tvm/wiki/Benchmark#nvidia-gpu>`_
to see the results.

Note that this tutorial will not run on Windows or recent versions of macOS. To
get it to run, you will need to wrap the body of this tutorial in a :code:`if
__name__ == "__main__":` block.
"""

######################################################################
# Install dependencies
# --------------------
# To use the autotvm package in tvm, we need to install some extra dependencies.
# (change "3" to "2" if you use python2):
#
# .. code-block:: bash
#
#   pip3 install --user psutil xgboost tornado cloudpickle
#
# To make TVM run faster during tuning, it is recommended to use cython
# as FFI of tvm. In the root directory of tvm, execute:
#
# .. code-block:: bash
#
#   pip3 install --user cython
#   sudo make cython3
#
# Now return to python code. Import packages.

import os
import argparse

import numpy as np

import tvm
from tvm import relay, autotvm, auto_scheduler
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib import graph_executor
from tvm.relay.transform.backend_operator.target import OPT_LEVEL
from tvm.relay.transform.utility.json_logger import *
import time
from tvm.relay.transform.utility.debug_helper import *
# import tvm.contrib.graph_runtime as runtime

# import tensorflow as tf
# import tvm.relay.testing.tf as tf_testing
import sys

# from ..workloads.onnx_workloads import get_network_from_onnx
from workloads.torch_workloads import get_network_from_torch

# try:
#     tf_compat_v1 = tf.compat.v1
# except ImportError:
#     tf_compat_v1 = tf

####################################################################
#
# .. note:: How to set tuning options
#
#   In general, the default value provided here works well.
#
#   If you have large time budget, you can set :code:`n_trial`, :code:`early_stopping` larger,
#   which makes the tuning runs longer.
#
#   If you have multiple devices, you can use all of them for measurement to
#   accelerate the tuning process. (see the 'Scale up measurement` section below).
#

def set_module_input(module, shape_dict, args):
    for input_name, input_shape in shape_dict.items():
        module.set_input(input_name, tvm.nd.array((np.random.uniform(size=input_shape)).astype(args.dtype)))


###################################################################
# Begin Tuning
# ------------
# Now we can extract tuning tasks from the network and begin tuning.
# Here, we provide a simple utility function to tune a list of tasks.
# This function is just an initial implementation which tunes them in sequential order.
# We will introduce a more sophisticated tuning scheduler in the future.

# You can skip the implementation of this function for this tutorial.
def tune_autotvm_tasks(
    tasks,
    measure_option,
    network,
    tuner="xgb",
    n_trial=1000,
    early_stopping=None,
    log_filename="tuning.log",
    use_transfer_learning=True,
):
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == "random":
            tuner_obj = RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        print(f"Task name: {tsk.name} / # of trials: {tsk_trial} ({n_trial} vs. {len(tsk.config_space)})")
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)

########################################################################
# Finally, we launch tuning jobs and evaluate the end-to-end performance.


def tune_and_evaluate_autotvm(tuning_opt, args):
    # extract workloads from relay program
    print("Extract tasks...")
    mod, params, shape_dict, _ = get_network_from_torch(args.network,
                                                        batch_size=args.batch_size)

    print(f"Target Host: {args.target_host}")
    tasks = autotvm.task.extract_from_program(
        mod["main"], target=args.target, target_host=args.target_host, params=params,
        #mod["main"], target=target, target_host=target_host, params=params, ops=(relay.op.get("nn.conv2d"),)
    )

    # run tuning tasks
    print("Tuning...")
    tune_autotvm_tasks(tasks, **tuning_opt)

    # Add n_tasks to dictionary for logging
    tuning_opt["n_tasks"] = len(tasks)

    # compile kernels with history best records
    with autotvm.apply_history_best(args.log_file):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=OPT_LEVEL.get()):
            lib = relay.build_module.build(mod, target=args.target, params=params)

        # load parameters
        dev = tvm.device(str(args.target), 0)
        module = graph_executor.GraphModule(lib["default"](dev))

        # Set the input of the module
        set_module_input(module, shape_dict, args)

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", dev, number=1, repeat=600)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print(
            "Mean inference time (std dev): %.2f ms (%.2f ms)"
            % (np.mean(prof_res), np.std(prof_res))
        )

        
def tune_auto_scheduler_tasks(
    tasks, 
    task_weights,
    measure_option,
    network,
    tuner="xgb",
    n_trial=1000,
    early_stopping=None,
    log_filename="tuning.log",
    use_transfer_learning=True,):

    # create tmp log file
    tmp_log_file = f"{log_filename}.{network}.tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    # min_repeat_ms >= 300ms is recommended
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10)

    # 900 * len(tasks) is recommended (20000 for ResNet)
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=n_trial,
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(tmp_log_file)],
    )

    tuner.tune(tune_option)

    # pick best records to a cache file
    os.system(f"python3 -m tvm.auto_scheduler.measure_record --mode distill -i {tmp_log_file}")
    os.system(f"cat {tmp_log_file}.best.json >> {log_filename}")

    # Move redundant files to tmp dir
    this_code_path = os.path.dirname(os.path.abspath(__file__))
    os.system(f"mv {tmp_log_file}.best.json {this_code_path}/tmp/")
    os.system(f"mv {tmp_log_file} {this_code_path}/tmp/")
    # os.remove(f'{tmp_log_file}.best.json')
    # os.remove(tmp_log_file)

def tune_and_evaluate_auto_scheduler(tuning_opt, args):
    # extract workloads from relay program
    print("Extract tasks...")
    mod, params, shape_dict, _ = get_network_from_torch(args.network,
                                                        batch_size=args.batch_size)

    print(f"Target Host: {args.target_host}")
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params=params, target=args.target)#, target_host=args.target_host)
    
    # run tuning tasks
    print("Tuning...")
    tuning_opt["n_trial"] = len(tasks)*900
    tune_auto_scheduler_tasks(tasks, task_weights, **tuning_opt)

    # Add n_tasks to dictionary for logging
    tuning_opt["n_tasks"] = len(tasks)

    # compile kernels with history best records
    with auto_scheduler.ApplyHistoryBest(args.log_file):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=OPT_LEVEL.get(), config={"relay.backend.use_auto_scheduler": True}):
            lib = relay.build(mod, target=args.target, params=params)
        
        dev = tvm.device(str(args.target), 0)
        module = graph_executor.GraphModule(lib["default"](dev))

        # Set the input of the module
        set_module_input(module, shape_dict, args)

        # Evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", dev, repeat=3, min_repeat_ms=500)
        prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res)))


def args_checker(args, parser):
    is_missing_arg = not args.tuner
    is_missing_arg |= not args.target
    is_missing_arg |= not args.target_host
    is_missing_arg |= not args.network
    is_missing_arg |= not args.log_file
    is_missing_arg |= not args.dtype
    is_missing_arg |= not args.batch_size
    is_missing_arg |= not args.hw

    if is_missing_arg:
        parser.error('Make sure you input all arguments')

    if args.tuner not in ['autotvm', 'autoscheduler']:
        parser.error('Wrong tuner input')

    if args.target == 'cuda':
        args.target = tvm.target.cuda()
    else:
        parser.error('Unsupported target')

    this_code_path = os.path.dirname(os.path.abspath(__file__))
    args.log_file = f"{this_code_path}/../../python/tvm/relay/transform/logs/{args.log_file}_{args.hw}.json"

def get_args():
    parser = argparse.ArgumentParser()
    # Default type is string for argparse
    parser.add_argument("-tu", "--tuner", help="tuner (autotvm or autoscheduler)")
    parser.add_argument("-t", "--target", help="target device")
    parser.add_argument("-th", "--target-host", help="host device for autotuning tasks")
    parser.add_argument("-n", "--network", help="name of a neural network")
    parser.add_argument("-l", "--log-file", help="log file path")
    parser.add_argument("-dt", "--dtype", help="data type")
    parser.add_argument("-bs", "--batch-size", type=int, help="batch size")
    parser.add_argument("-hw", "--hw", help="target hardware")
    args = parser.parse_args()

    args_checker(args, parser)
    return args

if __name__ == "__main__":
    ###########################################
    # Set Tuning Options
    # ------------------
    # Before tuning, we apply some configurations.

    #### DEVICE CONFIG ####
    #host = "jetson-1"
    #port = 9090

    #remote = rpc.connect(host, port)


    # Check if we have all important args
    args = get_args()

    # Setup automatic logging feature (to file)
    setup_logging(task_name="autotvm_tuning", net_name=args.network, hw_name=args.hw)
    print(args)
    # target = tvm.target.cuda()
    # #target_host = 'llvm -mtriple=aarch64-linux-gnu'
    # target_host = 'llvm'
    # 
    # #### TUNING OPTION ####
    # # network = "bert"
    # network = "resnet-50"
    # # log_file = "dp_tuning.log"
    # log_file = "dp_tuning_ansor.log"
    # dtype = "float32"

    tuning_option = {
        "log_filename": args.log_file,
        "network": args.network,
        "tuner": "xgb",
        # "n_trial": 10,  # Debug: Note that if this is too small, AutoTVM can't find valid schedules.
        "n_trial": 2000, # This is for AutoTVM. AutoScheduler dynamically adjusts before autotuning based on n_tasks
        # "n_trial": 4000,  # This is for AutoTVM. AutoScheduler dynamically adjusts before autotuning based on n_tasks
        "early_stopping": 600, # This only applies to AutoTVM now

        #"measure_option": autotvm.measure_option(
        #    builder=autotvm.LocalBuilder(timeout=10),
        #    runner=autotvm.RPCRunner(
        #        "jetson-1",  # change the device key to your key
        #        "tracker",
        #        9191,
        #        number=20,
        #        repeat=3,
        #        timeout=10,
        #        min_repeat_ms=150,
        #    ),
        #),

        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(timeout=10),
            runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
        ),
    }

    # We do not run the tuning in our webpage server since it takes too long.
    # Uncomment the following line to run it by yourself.

    start_time = time.time()
    if args.tuner == 'autotvm':
        tune_and_evaluate_autotvm(tuning_option, args)
    else:
        # N_Trial is going to be decided in the autotuning function
        tune_and_evaluate_auto_scheduler(tuning_option, args)

    # Dump the tuning information into JSON file
    search_time = time.time() - start_time
    tuning_option["opt_level"] = OPT_LEVEL.get()
    dump_autotvm_tuning_info(tuning_option, search_time, args.hw)

######################################################################
# Sample Output
# -------------
# The tuning needs to compile many programs and extract feature from them.
# So a high performance CPU is recommended. One sample output is listed below.
# It takes about 4 hours to get the following output on a 32T AMD Ryzen Threadripper.
# The tuning target is NVIDIA 1080 Ti.
# (You can see some errors during compilation. If the tuning is not stuck, it is okay.)
#
# .. code-block:: bash
#
#    Extract tasks...
#    Tuning...
#    [Task  1/12]  Current/Best:  541.83/3570.66 GFLOPS | Progress: (960/2000) | 1001.31 s Done.
#    [Task  2/12]  Current/Best:    0.56/ 803.33 GFLOPS | Progress: (704/2000) | 608.08 s Done.
#    [Task  3/12]  Current/Best:  103.69/1141.25 GFLOPS | Progress: (768/2000) | 702.13 s Done.
#    [Task  4/12]  Current/Best: 2905.03/3925.15 GFLOPS | Progress: (864/2000) | 745.94 sterminate called without an active exception
#    [Task  4/12]  Current/Best: 2789.36/3925.15 GFLOPS | Progress: (1056/2000) | 929.40 s Done.
#    [Task  5/12]  Current/Best:   89.06/1076.24 GFLOPS | Progress: (704/2000) | 601.73 s Done.
#    [Task  6/12]  Current/Best:   40.39/2129.02 GFLOPS | Progress: (1088/2000) | 1125.76 s Done.
#    [Task  7/12]  Current/Best: 4090.53/5007.02 GFLOPS | Progress: (800/2000) | 903.90 s Done.
#    [Task  8/12]  Current/Best:    4.78/1272.28 GFLOPS | Progress: (768/2000) | 749.14 s Done.
#    [Task  9/12]  Current/Best: 1391.45/2325.08 GFLOPS | Progress: (992/2000) | 1084.87 s Done.
#    [Task 10/12]  Current/Best: 1995.44/2383.59 GFLOPS | Progress: (864/2000) | 862.60 s Done.
#    [Task 11/12]  Current/Best: 4093.94/4899.80 GFLOPS | Progress: (224/2000) | 240.92 sterminate called without an active exception
#    [Task 11/12]  Current/Best: 3487.98/4909.91 GFLOPS | Progress: (480/2000) | 534.96 sterminate called without an active exception
#    [Task 11/12]  Current/Best: 4636.84/4912.17 GFLOPS | Progress: (1184/2000) | 1381.16 sterminate called without an active exception
#    [Task 11/12]  Current/Best:   50.12/4912.17 GFLOPS | Progress: (1344/2000) | 1602.81 s Done.
#    [Task 12/12]  Current/Best: 3581.31/4286.30 GFLOPS | Progress: (736/2000) | 943.52 s Done.
#    Compile...
#    Evaluate inference time cost...
#    Mean inference time (std dev): 1.07 ms (0.05 ms)
#
# As a reference baseline, the time cost of MXNet + TensorRT on resnet-18 is 1.30ms. So we are a little faster.

######################################################################
#
# .. note:: **Experiencing Difficulties?**
#
#   The auto tuning module is error-prone. If you always see " 0.00/ 0.00 GFLOPS",
#   then there must be something wrong.
#
#   First, make sure you set the correct configuration of your device.
#   Then, you can print debug information by adding these lines in the beginning
#   of the script. It will print every measurement result, where you can find useful
#   error messages.
#
#   .. code-block:: python
#
#      import logging
#      logging.getLogger('autotvm').setLevel(logging.DEBUG)
#
#   Finally, always feel free to ask our community for help on https://discuss.tvm.apache.org


#################################################################
# Scale up measurement by using multiple devices
# ----------------------------------------------
# .. _tutorials-autotvm-rpc-tracker:
#
# If you have multiple devices, you can use all of them for measurement.
# TVM uses the RPC Tracker to manage distributed devices.
# The RPC Tracker is a centralized controller node. We can register all devices to
# the tracker. For example, if we have 10 GPU cards, we can register all of them
# to the tracker, and run 10 measurements in parallel, accelerating the tuning process.
#
# To start an RPC tracker, run this command on the host machine. The tracker is
# required during the whole tuning process, so we need to open a new terminal for
# this command:
#
# .. code-block:: bash
#
#   python -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190
#
# The expected output is
#
# .. code-block:: bash
#
#   INFO:RPCTracker:bind to 0.0.0.0:9190
#
# Then open another new terminal for the RPC server. We need to start one server
# for each dedicated device. We use a string key to distinguish the types of devices.
# You can pick a name you like.
# (Note: For rocm backend, there are some internal errors with the compiler,
# we need to add `--no-fork` to the argument list.)
#
# .. code-block:: bash
#
#     python -m tvm.exec.rpc_server --tracker=0.0.0.0:9190 --key=1080ti
#
# After registering devices, we can confirm it by querying rpc_tracker
#
# .. code-block:: bash
#
#   python -m tvm.exec.query_rpc_tracker --host=0.0.0.0 --port=9190
#
# For example, if we have four 1080ti, two titanx and one gfx900, the output can be
#
# .. code-block:: bash
#
#    Queue Status
#    ----------------------------------
#    key          total  free  pending
#    ----------------------------------
#    1080ti       4      4     0
#    titanx       2      2     0
#    gfx900       1      1     0
#    ----------------------------------
#
# Finally, we need to change the tuning option to use RPCRunner. Use the code below
# to replace the corresponding part above.
"""
tuning_option = {
    "log_filename": log_file,
    "tuner": "xgb",
    "n_trial": 2000,
    "early_stopping": 600,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=10),
        runner=autotvm.RPCRunner(
            "1080ti",  # change the device key to your key
            "0.0.0.0",
            9190,
            number=20,
            repeat=3,
            timeout=4,
            min_repeat_ms=150,
        ),
    ), 
}
"""
