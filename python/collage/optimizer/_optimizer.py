import tvm._ffi
import tvm.driver
from collage.pattern_manager.pattern_registry import PatternRegistry
from collage.utils import get_function_body, is_constant_node
from .comp_graph import ComputationGraph
from .comp_graph_optimizer import (
                    CompGraphOptimizer,
                    AssignBackendExprVisitor,
                )
from .evolutionary_searcher import EvolutionarySearcher
from .ext_compiler_op_annotator import ExtCompilerOpAnnotator
from tvm.relay.op.contrib.tensorrt import prune_tensorrt_subgraphs
from tvm.relay import transform

import logging
from collage.interface import CollageContext
from .evolutionary_searcher_state import (MatchToOpGroupTranslator, OpStateToMatchTranslator)
from .op_match_logger import OpMatchLogger, OpMatchReader
from collage.backend import BackendKind
from collage.analysis import visualize_backend_placement

def setup_pattern_registry(hw_name):
    pattern_registry = PatternRegistry.get(hw_name)
    return pattern_registry

@tvm._ffi.register_func("collage.optimizer.print_attr_args")
def print_attr_args(expr):
    logger.info(f"attr: {get_attr_vals(expr)}")

@tvm._ffi.register_func("collage.optimizer.visualize_network_debug")
def visualize_network_debug(relay_expr, name):
    net_name = 'default'
    if relay_expr.attrs is not None and "Network" in dict(relay_expr.attrs):
        net_name = dict(relay_expr.attrs)["Network"]
        visualize_network(relay_expr, f"{net_name}_{name}")
        logger.info("[Done] Debug visualization")

def apply_tensorrt_op(mod):
    logging.info("Applying TensorRT op pass")
    from tvm.relay.op.contrib.tensorrt import RemoveDropoutPass
    # Get best op match info
    fn_body = mod["main"].body
    # Annotating expression
    target_str = "tensorrt"

    # Do merge and partition pass
    use_implicit_batch = False
    remove_no_mac_subgraphs = False
    max_workspace_size = 1 << 30
    version = None

    config = {
        "use_implicit_batch": use_implicit_batch,
        "max_workspace_size": max_workspace_size,
        "remove_no_mac_subgraphs": remove_no_mac_subgraphs,
    }

    if version:
        assert isinstance(version, tuple) and len(version) == 3
        config["tensorrt_version"] = version
    else:
        linked_version = tuple(tvm.get_global_func("relay.op.get_tensorrt_version")())
        if not linked_version:
            logger.warning(
                "TVM was not built against TensorRT and no version was provided to "
                "partition_for_tensorrt. Defaulting to 6.0.1"
            )
            linked_version = (6, 0, 1)
        config["tensorrt_version"] = linked_version

    # Warning(@Soo): I assume this is only useful when folding constant
    seq = tvm.transform.Sequential(
        [
            transform.InferType(),
            #RemoveDropoutPass(),
            #transform.RemoveUnusedFunctions(),
            #transform.ConvertLayout(
            #    {
            #        "nn.conv2d": ["NCHW", "default"],
            #        "nn.conv3d": ["NCDHW", "default"],
            #        "nn.conv2d_transpose": ["NCHW", "default"],
            #    }
            #),
            transform.FoldConstant(),
            transform.AnnotateTarget("tensorrt"),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
            transform.InferType(),
        ]
    )

    # Do prune_tensorrt_subgraphs
    with tvm.transform.PassContext(opt_level=3, config={"relay.ext.tensorrt.options": config}):
        mod = seq(mod)
    return mod

def apply_dnnl_op(mod):
    opt_pass = tvm.transform.Sequential(
        [
            transform.InferType(),
            transform.SimplifyInference(),
            transform.FoldConstant(),
            transform.FoldScaleAxis(),
            transform.AnnotateTarget("dnnl"),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
            transform.InferType(),
        ]
    )

    with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        mod = opt_pass(mod)

    return mod

@tvm._ffi.register_func("collage.optimizer.apply_external_compiler_op")
def apply_external_compiler_op(mod):
    target = mod["main"].attrs["BuildTarget"]
    if "cuda" in target:
        mod = apply_tensorrt_op(mod)
    elif "llvm" in target:
        mod = apply_dnnl_op(mod)
    else:
        Exception(f"Unexpected HW for external compiler op pass: {hw_name}")

    return mod
    # return mod, config

@tvm._ffi.register_func("collage.optimizer.get_user_fusion")
def get_user_fusion(relay_expr):
    logging.info("User-defined fusion")
    relay_expr = get_function_body(relay_expr)
    match_path = CollageContext.graph_level_tmp_file
    opt_match = OpMatchReader().read(relay_expr, match_path)

@tvm._ffi.register_func("collage.optimizer.visualize_backend_placement")
def run_backend_placement_visualization(relay_expr):
    logging.info("Visualize backend placement")
    relay_expr = get_function_body(relay_expr)
    match_path = CollageContext.input_placement_log_file
    opt_match = OpMatchReader().read(relay_expr, match_path)
    visualize_backend_placement(relay_expr, CollageContext.placement_vis_file)

def get_backends(func_expr, backend_registry):
    assert("BackendList" in func_expr.attrs)
    backend_list_str = func_expr.attrs["BackendList"]
    backend_str_list = backend_list_str.split(",")
    backends = [backend_registry[b] for b in backend_str_list]

    return backends

def get_backend_names(backends):
    return [ b.name for b in backends ]

def run_op_level_opt(func_expr):
    target = func_expr.attrs["BuildTarget"]
    pattern_registry = CollageContext.pattern_registry
    backend_registry = pattern_registry.backend_registry
    given_backends = get_backends(func_expr, backend_registry)
    relay_expr = get_function_body(func_expr)

    logging.info(f"[Op-Level: DP] Computation graph generation...")
    comp_graph = ComputationGraph(relay_expr)
    n_relay_nodes = comp_graph.n_relay_nodes
    logging.info(f"# of relay nodes in comp graph: {n_relay_nodes}")

    # Optimizing graph

    assert(pattern_registry is not None)
    optimizer = CompGraphOptimizer(pattern_registry, given_backends)

    """
    Warning(@Soo): Note that current DP optimizer does not work for patterns with more than one root.
    For example, Conv     Conv (Two parallel convolution) case can't be handled
                   \       /
                      ReLU
    Following lines need to be modified to afford more than two roots
    - pat.get_relay_pattern().match(f_expr)

    This is because of inherent limitation of Relay pattern and
    the discrepancy between what Relay pattern supports and how TVM fusion strategy works.
    We can come back to this later if this is critical to performance, which is unlikely for now given networks we have.
    """
    optimized_match = optimizer.optimize(comp_graph, target)

    logging.info("[Op-Level: DP] It finished optimizing comp graph and assigning backend ops to Relay Expr (backend attr)")

    # Save fisrt layer best results
    OpMatchLogger().save(relay_expr, optimized_match, log_path=CollageContext.op_level_placement_log)
    return optimized_match, relay_expr, pattern_registry, n_relay_nodes


@tvm._ffi.register_func("collage.optimizer.run_two_level_opt")
def run_two_level_opt(relay_expr):
    """
    Two-level optimization pass
    First level is for backend operators from TVM and CuDNN, which are not external compilers.
    Second level is for external compilers such as TensorRT.
    We have two separate levels because
    1) First level remove need to enumerate all possible extenral compiler patterns
    2) First level provide us with optimal solutions for backend operators.
    Thus, for each backend operators, second level only needs to consider between
    TensorRT operators and chosen optimal backend operators from first level.

    Parameters
    ----------
    relay_expr : tvm.relay.expr
        Relay IR for computation graph

    Returns
    -------
    matched_relay_expr : tvm.relay.expr
        The result matching between backend operators and Relay operators
    """

    # It is a function if you get it from last pass of Relay build
    logging.info("[Python side] Run two-level optimization")

    # op-level optimization: DP with all backends but external compilers, e.g., TensorRT
    func_expr = relay_expr
    optimized_match, relay_expr, pattern_registry, n_relay_nodes = run_op_level_opt(relay_expr)
    logging.info("[Python side] Op-level optimization is done")

    net_name, batch_size = func_expr.attrs["Network"], int(func_expr.attrs["BatchSize"])
    build_target = func_expr.attrs["BuildTarget"]
    logging.info(f"Network name, batch_size: {net_name}, {batch_size}")

    # subgraph-level optimization for external compilers
    # Translate optimized_match into OpState class that can be used for evolutionary search
    group_id_to_exprs_anno = MatchToOpGroupTranslator().translate(relay_expr, optimized_match)

    given_backends = get_backends(func_expr, CollageContext.pattern_registry.backend_registry)
    graph_backend = None
    for backend in given_backends:
        if backend.kind == BackendKind.GRAPH_LEVEL:
            assert(graph_backend is None, "Current tvm build only supports TensorRT and DNNL")
            graph_backend = backend.name

    # Prepare OpStateToMatchTranslator
    # This translates Op State to optimized_match with the following two dictionaries
    op_state_to_match_translator = OpStateToMatchTranslator(optimized_match, group_id_to_exprs_anno, backend_name = graph_backend)

    # Run evolutionary search
    n_ops_after_first_level = len(group_id_to_exprs_anno.keys())
    logging.info(f"# of matched operators from first level : {n_ops_after_first_level}")

    # Unnecessary ops include external compiler ops chosen from first level or
    # Ops that shouldn't be assigned to external compiler such as Tuple or TupleGetItem
    n_ops = len(op_state_to_match_translator.state_id_to_group_id)
    logging.info(f"# of matched operators from first level after excluding unnecessary ops: {n_ops}")

    # On the second level, Consider only ops that are not assigned to TensorRT
    # Extract ops that are not assigned to TensorRT

    # Save it for user-defined fusion pass to measure end-to-end perf
    OpMatchLogger().save(relay_expr, optimized_match, log_path=CollageContext.graph_level_tmp_file)

    # n_ops for each network (it may vary depending on trials)
    # Search space size: 2^n_ops
    # ResNet: 169 -> 65 -> 19
    # ResNext: 169 -> 79
    # Nasrnn: 311 -> 97
    # NasNet: 683 -> 312
    # BERT: 169 -> 96

    # Evolutionary search hyperparameter info
    # Example: pop_size * max_iter (=1) roughly takes 2~4 secs
    # References: ResNet50, 10 * 56 (560) takes 1559.51 s (2.78 secs per pop size per iteration)
    # References: ResNext50, 20 * 100 (2000) takes 4474 s (2.27 secs per pop size per iteration)

    # 100 * 200 (20000) leads to out of memory issues. We attribute this to large population issue of deap lib
    # Note that some of individuals may not be measured in each generation if they are measured anytime earlier
    # visualize_network(relay_expr, "o3_mobilenet_v2_after_match")
    # cx_prob = 0.8, mut_prob = 0.5, resnet50: 2.512

    # pop_size=30,   max_iter=100000) # when n_hours == 3
    # pop_size=50,   max_iter=100000) # when n_hours == 6

    if n_ops > 0:
        ev_searcher = EvolutionarySearcher(
                                            op_state_to_match_translator,
                                            relay_expr,
                                            net_name,
                                            build_target,
                                            batch_size=batch_size,
                                            n_ops=n_ops,
                                            match_path=CollageContext.graph_level_tmp_file,
                                            pop_size=CollageContext.evolutionary_search_pop_size,
                                            max_iter=CollageContext.evolutionary_search_max_iter
                                          )
        second_opt_match = ev_searcher.search(rnd_seed=64, n_hours = CollageContext.evolutionary_search_budget)
    else:
        second_opt_match = optimized_match
        logger.info("No need for subgraph optimization because either 1) op optimization pass only chose Ext compiler ops"
               + " or 2) External compiler can't support ops that are not assigned to external compilers")

    # Update backend information to corresponding best match
    second_opt_match = OpMatchReader().read(relay_expr, CollageContext.graph_level_placement_log)

    return second_opt_match

@tvm._ffi.register_func("collage.optimizer.run_dp")
def run_dp(relay_expr):
    run_op_level_opt(relay_expr)

@tvm._ffi.register_func("collage.optimizer.assign_backend_for_op_measurement")
def assign_backend_for_op_measurement(relay_expr):
    backend_pattern_name = relay_expr.attrs["BackendOP"]
    assert isinstance(backend_pattern_name, str)

    relay_expr = get_function_body(relay_expr)
    AssignBackendExprVisitor().assign(relay_expr, backend_pattern_name)

    # logger.info(repr(relay_expr))
    # sys.exit(0)
