from tvm import relay
import tvm
from tvm.relay.transform.backend_operator.utils import is_function_node
from tvm.relay.transform.backend_operator.target import *
from tvm.relay.transform.optimizer.custom_fusion_pass import CustomFusionPass
from workloads.torch_workloads import get_network_from_torch
from tvm.contrib import graph_executor as runtime
import numpy as np
import argparse
from tvm import autotvm, auto_scheduler

tvm.autotvm.measure.measure_methods.set_cuda_target_arch("sm_72")

def measure_end_to_end_perf_tensorrt(mod, params, target_str, shape_dict, is_ours):
    from tvm.relay.op.contrib.tensorrt import partition_for_tensorrt
    mod, config = partition_for_tensorrt(mod, params)

    with tvm.transform.PassContext(opt_level=OPT_LEVEL.get(), config={'relay.ext.tensorrt.options': config}):
        lib = relay.build(mod, target=target_str, params=params)

    lib.export_library('compiled.so')

    dev = tvm.gpu(0)
    loaded_lib = tvm.runtime.load_module('compiled.so')
    module = tvm.contrib.graph_executor.GraphModule(loaded_lib['default'](dev))

    # Setup execution
    for input_name, input_shape in shape_dict.items():
        input_data = np.random.uniform(-1, 1, size=input_shape).astype("float32")
        module.set_input(input_name, input_data)

    ftimer = module.module.time_evaluator("run", dev, number=NUM_MEASUREMENTS_PER_REPEAT_E2E, repeat=NUM_REPEATS_E2E)

    return measure(ftimer, is_net=False)

def measure_end_to_end_perf_autotvm(net, params, target_str, shape_dict, is_ours, net_name, hw_name):
    assert is_function_node(net)
    if is_ours:
        # net = net.with_attr("CustomFusionPass", CustomFusionPass.DP)
        # net = net.with_attr("CustomFusionPass", CustomFusionPass.USER_DEFINED_FUSION)
        net = net.with_attr("CustomFusionPass", CustomFusionPass.TWO_LEVEL_OPT)
        net = net.with_attr(NETWORK_FUNC_ATTR, net_name)
        net = net.with_attr(HW_FUNC_ATTR, hw_name)


    with autotvm.apply_history_best(get_autotvm_log_path(hw_name)):
        with tvm.transform.PassContext(opt_level=OPT_LEVEL.get()):
            lib = relay.build(net, target_str, params=params)
        print(f"We successfully built the network")
        # Create workload
        dev = tvm.device(target_str, 0)
        module = runtime.GraphModule(lib["default"](dev))

        # Setup execution
        for input_name, input_shape in shape_dict.items():
            input_data = np.random.uniform(-1, 1, size=input_shape).astype("float32")
            module.set_input(input_name, input_data)


        ftimer = module.module.time_evaluator("run", dev, number=NUM_MEASUREMENTS_PER_REPEAT_E2E, repeat=NUM_REPEATS_E2E)

    return measure(ftimer, is_net=False)

def measure_end_to_end_perf_autosch(net, params, target_str, shape_dict, is_ours):
    assert is_function_node(net)
    if is_ours:
        net = net.with_attr("CustomFusionPass", CustomFusionPass.DP)

    with auto_scheduler.ApplyHistoryBest(AUTOSCH_LOG):
        with tvm.transform.PassContext(opt_level=OPT_LEVEL.get()):
            lib = relay.build(net, target_str, params=params)

    # Create workload
    dev = tvm.device(target_str, 0)
    module = runtime.GraphModule(lib["default"](dev))

    # Setup execution
    for input_name, input_shape in shape_dict.items():
        input_data = np.random.uniform(-1, 1, size=input_shape).astype("float32")
        module.set_input(input_name, input_data)

    ftimer = module.module.time_evaluator("run", dev, number=NUM_MEASUREMENTS_PER_REPEAT_E2E, repeat=NUM_REPEATS_E2E)

    return measure(ftimer, is_net=False)


def verify_network_output(net, params, target_str, shape_dict, hw_name):
    assert is_function_node(net)

    # Create same input data for two networks
    name_to_data = {}
    for input_name, input_shape in shape_dict.items():
        input_data = np.random.uniform(-1, 1, size=input_shape).astype("float32")
        name_to_data[input_name] = input_data

    # Run original TVM (or AutoTVM)
    with autotvm.apply_history_best(get_autotvm_log_path(hw_name)):
        with tvm.transform.PassContext(opt_level=OPT_LEVEL.get()):
            lib = relay.build(net, "cuda", params=params)

    # Create workload
    dev = tvm.device(target_str, 0)
    module = runtime.GraphModule(lib["default"](dev))

    # Setup execution
    for input_name, input_data in name_to_data.items():
        module.set_input(input_name, input_data)

    module.run()
    out_tvm = module.get_output(0).asnumpy()

    # Run ours
    net = net.with_attr("CustomFusionPass", CustomFusionPass.DP)
    with autotvm.apply_history_best(get_autotvm_log_path(hw_name)):
        with tvm.transform.PassContext(opt_level=OPT_LEVEL.get()):
            lib = relay.build(net, "cuda", params=params)

    # Create workload
    dev = tvm.device(target_str, 0)
    module = runtime.GraphModule(lib["default"](dev))

    # Setup execution
    for input_name, input_data in name_to_data.items():
        module.set_input(input_name, input_data)

    module.run()
    out_ours = module.get_output(0).asnumpy()

    TOL = 1e-01
    print("First 10 outputs")
    print(f"TVM    : {out_tvm.flatten()[:10]}")
    # print(f"AutoTVM: {out_tvm.flatten()[:10]}")
    print(f"Ours   : {out_ours.flatten()[:10]}")
    assert np.allclose(out_tvm, out_ours, rtol=TOL, atol=TOL)

    print(f"Passed the verification of output test")
    print(f"Worst diffence : {np.abs((out_ours - out_tvm)).max():.4f}")

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

    # We can't test this because this network include batch norm.
    mod, params, shape_dict, _ = get_network_from_torch(args.network, 1)

    # mean_perf, std_perf = measure_end_to_end_perf_autotvm(mod["main"], params, 'cuda', shape_dict,
    #                                                       True, args.network, args.hw)
    # print(f"[{args.network}] Performance of Ours on {args.hw} (mean, std) = ({mean_perf:.4f}+-{std_perf:.4f})")

    # mean_perf, std_perf = measure_end_to_end_perf_tensorrt(mod, params, 'cuda', shape_dict, False)
    # print(f"[{args.network}] Performance of TensorRT on {args.hw} (mean, std) = ({mean_perf:.4f}+-{std_perf:.4f})")

    mean_perf, std_perf = measure_end_to_end_perf_autotvm(mod["main"], params, 'cuda', shape_dict,
                                                           False, args.network, args.hw)
    print(f"[{args.network}] Performance of AutoTVM on {args.hw} (mean, std) = ({mean_perf:.4f}+-{std_perf:.4f})")
    #
    # mean_perf, std_perf = measure_end_to_end_perf_autotvm(mod["main"], params, 'cuda -libs=cudnn', shape_dict,
    #                                                       False, args.network, args.hw)
    # print(f"[{args.network}] Performance of AutoTVM+CuDNN on {args.hw} (mean, std) = ({mean_perf:.4f}+-{std_perf:.4f})")

    # mean_perf, std_perf = measure_end_to_end_perf_autosch(mod["main"], params, 'cuda', shape_dict, False)
    # print(f"[AutoSCH] Performance of {args.network} (mean, std) = ({mean_perf:.4f}+-{std_perf:.4f})")


    # verify_network_output(mod["main"], params, 'cuda', shape_dict, args.hw)
