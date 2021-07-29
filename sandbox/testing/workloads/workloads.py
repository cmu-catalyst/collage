from tvm import relay
import tvm.relay.testing as testing
from tvm.relay.transform.backend_operator.utils import *

# Value is shape dict
WORKLOADS_DIC = {
    "resnet_block" : {"input0": [1, 64, 56, 56]},
    "resnet50" : {"input0": [1, 64, 56, 56]},
    "resnext50_32x4d" : {"input0": [1, 64, 56, 56]},
    "nasneta" : {"input0": [1, 64, 56, 56]},
    # NasRNN always have some errors during autotuning operators with AutoTVM
    # "nasrnn": {'x.1': [1, 512]},
    # "nasrnn": {'x.1': [1, 1024]},
    # "nasrnn": {'x.1': [1, 2048]},
    "nasrnn": {'x.1': [1, 2560]},
    # "nasrnn": {'x.1': [1, 512], 'x.2': [1, 512], 'x.3': [1, 512], 'x.4': [1, 512], 'x': [1, 512]},
    "bert": {"input0": [64, 1024]},
    "resnet50_3d": {"input0": [1, 64, 3, 56, 56]},
    "mobilenet_v2": {"input0": [1, 32, 224, 224]},
    # "mobilenet_v2": {"input0": [1, 32, 56, 56]},
}

def create_relay_workload(expr):
    inputs = relay.analysis.free_vars(expr)
    expr_func = relay.Function(inputs, expr)
    mod, params = testing.create_workload(expr_func)

    return mod, params

def crop_expr_by_post_dfs_order(expr, post_dfs_order):
    return ExprCropper(post_dfs_order).crop(expr)

class ExprCropper:
    def __init__(self, target_post_dfs_order):
        self._memo = {}
        self._post_dfs_order = -1
        self._target_post_dfs_order = target_post_dfs_order
        self.target_expr = None

    def crop(self, expr):
        self._memo = {}
        self.target_expr = None
        self.visit_expr(expr)

        return self.target_expr

    # Visit Relay expressions in post-order
    def visit_expr(self, expr):
        if hash(expr) in self._memo:
            return
        else:
            # memorize this visit to prevent it from visiting twice
            self._memo[hash(expr)] = True

        # We assume that child class at least have methods for these
        if is_constant_node(expr):
            self.visit_expr_const(expr)
            node_type = "Const"
        elif is_var_node(expr):
            self.visit_expr_var(expr)
            node_type = "Var"
        elif is_tuplegetitem_node(expr):
            self.visit_expr_tuplegetitem(expr)
            node_type = "TupleGetItem"
        elif is_call_node(expr):
            self.visit_expr_call(expr)
            node_type = expr.op
        elif is_function_node(expr):
            self.visit_expr_func(expr)
            node_type = "Function"
        elif is_tuple_node(expr):
            self.visit_expr_tuple(expr)
            node_type = "Tuple"
        else:
            raise Exception(f"Unexpected expression type, {type(expr)}")

        self._post_dfs_order += 1
        if self._post_dfs_order == self._target_post_dfs_order:
            self.target_expr = expr

    def visit_expr_const(self, expr):
        pass

    def visit_expr_var(self, expr):
        pass

    def visit_expr_tuple(self, expr):
        for arg in expr.fields:
            self.visit_expr(arg)

    def visit_expr_tuplegetitem(self, expr):
        self.visit_expr(expr.tuple_value)

    def visit_expr_call(self, expr):
        op, args, attrs, type_args, span = expr.op, expr.args, expr.attrs, expr.type_args, expr.span

        for arg in args:
            self.visit_expr(arg)

    def visit_expr_func(self, expr):
        params, body, ret_type, type_params = expr.params, expr.body, expr.ret_type, expr.type_params
        self.visit_expr(body)
