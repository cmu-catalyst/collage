from workloads.torch_workloads import get_network_from_torch
import numpy as np
import collage
import tvm
import logging
from tvm.contrib import graph_executor as runtime

# Remove warnings for clean output
import warnings
warnings.filterwarnings('ignore')

# [NOTE]
# * Available networks: bert_full, dcgan, nasneta, resnet50_3d, resnext50_32x4d, yolov3, mobilenet_v2
# * Collage supports following backends by default:
#      NVIDIDIA GPUs - TVM, TensorRT, cuBLAS, cuDNN
#      Intel CPUs    - TVM, MKL, DNNL
# * For the best performance, TVM operators should be tuned with AutoTVM beforehand.
# * Collage offers two optimizers: "op-level", "two-level"
#   Since two-level optimization takes long time (~30min), op-level optimizer is configured by default in this demo.

# Define Collage workload
workload = {
#    "optimizer": "op-level",
    "optimizer": "two-level",
    "backends": ["autotvm", "cudnn", "cublas", "tensorrt"],
    "network_name": "resnext50_32x4d",
    "target": "cuda",
    "batch_size": 1,
    "ev-budget": 0.5,
}

# Default logging level. Skip messages during optimization
logging.basicConfig(level=logging.ERROR)

# Enable logging to monitor optimization progress e.g., operator matching, profiling...
# logging.basicConfig(level=logging.INFO)

def measure_perf(lib, workload):
    # Create workload
    dev = tvm.device(workload["target"], 0)
    module = runtime.GraphModule(lib["default"](dev))

    # Setup execution
    for input_name, input_shape in workload["shape_dict"].items():
        input_data = np.random.uniform(-1, 1, size=input_shape).astype("float32")
        module.set_input(input_name, input_data)

    # Measure performance
    ftimer = module.module.time_evaluator("run", dev, number=10, repeat=20)
    perfs = np.array(ftimer().results) * 1000
    return np.mean(perfs), np.std(perfs)

def setup_workload(workload):
    network_name, batch_size, target = \
          workload["network_name"], workload["batch_size"], workload["target"]

    mod, params, shape_dict, _ = get_network_from_torch(network_name, batch_size)
    # Since Collage utilizes tvm as its codegen, we need to pass the following info for tvm codegen.
    workload["mod"] = mod
    workload["params"] = params
    workload["shape_dict"] = shape_dict

def run_workload(workload, collage_mod):
    setup_workload(workload)

    # Invoke collage optimizer
    lib = collage_mod.optimize_backend_placement(**workload)

    # For two-level optimizer, we have the measurement logged for the best placement during the optimization
    if workload["optimizer"] == "two-level":
        with open('plots/e2e_perf_two_level.log') as f:
            lines = f.readlines()[-1] # Consider only last line correspoding to the most recent eval
            lines = lines.split(",")
            collage_mean_perf, collage_std_perf = float(lines[0]), float(lines[1]) # The last one is the most recently measured one
    else:
        collage_mean_perf, collage_std_perf = measure_perf(lib, workload)

    print(f"[ End-to-End Performance Evaluation ]")
    print(f"# Network: {workload['network_name']}, Collage optimizer: {workload['optimizer']}")
    print(f"  * Collage Performance (ms) (mean, std) = ({collage_mean_perf:.4f}+-{collage_std_perf:.4f})")

def setup_two_level_log():
    # Delete outdated log file for e2e perf of two-level optimizer
    if workload["optimizer"] == "two-level":
        import os
        two_level_log_path = "plots/e2e_perf_two_level.log"
        if os.path.exists(two_level_log_path):
            os.remove(two_level_log_path)

if __name__ == "__main__":
    # Operator cost will be logged at "operator_cost.log" by default.
    # If you want to start from scratch, delete previous log file for operator cost.
    # Since it is unreadable, users can dump human-readable form by passing 'dump_readable_cost_log = True'
    collage_mod = collage.Module(op_cost_log_path = "operator_cost.log", dump_readable_cost_log = False)
    print(f"Default Collage backends: {collage_mod.get_registered_backends()}\n")

    # Override the default tuning log
    # If you don't have tuning log, generate one by running 'autotune_tvm_ops.py'
    collage_mod.update_backend_tuning_log("autotvm", "autotvm_tuning_log_rtx2070.json")
    setup_two_level_log()

    import time
    start_time = time.time()
    networks = ['bert_full', 'dcgan', 'nasneta', 'resnet50_3d', 'resnext50_32x4d']
    for nn in networks:
        nn_start_time = time.time()
        workload['network_name'] = nn
        setup_workload(workload)
        run_workload(workload, collage_mod)
        el_time = int(time.time() - nn_start_time)
        print(f"Elapsed time for {nn}: {el_time}s\n")
    el_time = int(time.time() - start_time)
    print(f"Total elapsed time: {el_time}s")


