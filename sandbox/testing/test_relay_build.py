import tvm
from tvm import relay
from tvm.contrib import graph_executor as runtime

from tvm.relay.transform.backend_operator.utils import is_function_node
from tvm.relay.transform.backend_operator.target import measure, NUM_MEASUREMENTS_PER_REPEAT, NUM_REPEATS, AUTOSCH_LOG
from tvm.relay.transform.backend_operator.target import OPT_LEVEL
from tvm.relay.transform.optimizer.custom_fusion_pass import *
from tvm.relay.transform.optimizer.optimizer_utils import *
from tvm import autotvm, auto_scheduler

from measure_end_to_end import verify_network_output
from workloads.onnx_workloads import get_network_from_onnx
from workloads.torch_workloads import *
from workloads.relay_workloads import get_network_from_relay

import numpy as np
import argparse

from workloads.workloads import WORKLOADS_DIC

def build_network(net, params, mode, net_name, hw_name):
    assert is_function_node(net)
    assert CustomFusionPass.has_value(mode)

#    net = net.with_attr("CustomFusionPass", mode)
#    net = net.with_attr(NETWORK_FUNC_ATTR, net_name)
#    net = net.with_attr(HW_FUNC_ATTR, hw_name)

#    with autotvm.apply_history_best(get_autotvm_log_path(hw_name)):
    with tvm.transform.PassContext(opt_level=OPT_LEVEL.get()):
    # with tvm.transform.PassContext(opt_level=OPT_LEVEL.get(), trace=print_ir):
        lib = relay.build(net, "cuda", params=params)

    return lib

def measure_network(lib, target_str, shape_dict):
    # Create workload
    dev = tvm.device(target_str, 0)
    module = runtime.GraphModule(lib["default"](dev))

    # Setup execution
    for input_name, input_shape in shape_dict.items():
        input_data = np.random.uniform(-1, 1, size=input_shape).astype("float32")
        module.set_input(input_name, input_data)

    ftimer = module.module.time_evaluator("run", dev, number=NUM_MEASUREMENTS_PER_REPEAT, repeat=NUM_REPEATS)

    return measure(ftimer, is_net=False)

def build_network_tensorrt(mod, params):
    from tvm.relay.op.contrib.tensorrt import partition_for_tensorrt
    mod, config = partition_for_tensorrt(mod, params)

    # print("-"*50)
    # print("Expr" + "-"*30)
    # print(repr(mod["main"]))
    # print("config" + "-"*30)
    # print(config)
    # visualize_network(mod["main"], "o3_resnet_tensorrt")
    target = "cuda"
    with tvm.transform.PassContext(opt_level=OPT_LEVEL.get(), config={'relay.ext.tensorrt.options': config}):
        lib = relay.build(mod, target=target, params=params)
        printe("Built done")

    lib.export_library('compiled.so')

    dev = tvm.gpu(0)
    loaded_lib = tvm.runtime.load_module('compiled.so')
    module = tvm.contrib.graph_executor.GraphModule(loaded_lib['default'](dev))

def args_checker(args, parser):
    is_missing_arg = not args.network
    is_missing_arg |= not args.hw
    # is_missing_arg |= not args.target
    # is_missing_arg |= not args.dtype
    # is_missing_arg |= not args.batch_size

    if is_missing_arg:
        parser.error('Make sure you input all arguments')

def get_args():
    parser = argparse.ArgumentParser()
    # Default type is string for argparse
    parser.add_argument("-n", "--network", help="name of a neural network")
    parser.add_argument("-hw", "--hw", help="target hardware")
    # parser.add_argument("-t", "--target", help="target device")
    # parser.add_argument("-dt", "--dtype", help="data type")
    # parser.add_argument("-bs", "--batch-size", type=int, help="batch size")
    args = parser.parse_args()

    args_checker(args, parser)
    return args

if __name__ == "__main__":
    args = get_args()

    # NasNet-A only works for opt_level 2 (not 3 due to the avgpool2d issue)
    # if args.network == "nasneta":
    #     OPT_LEVEL.set(2)

    mod, params, shape_dict, _ = get_network_from_torch(args.network, 1)
    # mod, params, shape_dict, _ = crop_network_from_torch(args.network, 1, 25)
    # mod, params = get_network_from_relay(args.network, 1)
    # printe(repr(mod["main"]))
    # build_network_tensorrt(mod, params)
    lib = build_network(mod["main"], params, CustomFusionPass.TWO_LEVEL_OPT, args.network, args.hw)
    # lib = build_network(mod["main"], params, CustomFusionPass.DP, args.network)
    # lib = build_network(mod["main"], params, CustomFusionPass.USER_DEFINED_FUSION, args.network)
    print(f"We successfully built the {args.network} on {args.hw}")

    # Verify if the network output is same after our optimization
    # verify_network_output(mod["main"], params, 'cuda', shape_dict)


    # Verify if the network can be measured
    # For Conv2d and conv2d+relu
    # inference_time = measure_network(lib, "cuda", {"data": [1, 3, 224, 224]})

    # For (conv2d+relu)x2
    # inference_time = measure_network(lib, "cuda", {"data": [1, 64, 56, 56]})

    # For networks from torch
    # inference_time = measure_network(lib, "cuda", WORKLOADS_DIC[args.network])
    # print(f"Inference time: {inference_time}")
