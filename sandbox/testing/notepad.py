from tvm.relay.dataflow_pattern import *
from tvm import relay
import logging
import sys

class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, level):
       self.logger = logger
       self.level = level
       self.linebuf = ''

    def write(self, buf):
       for line in buf.rstrip().splitlines():
          self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass

logging.basicConfig(filename='example.log', level=logging.WARNING, format='%(asctime)s:[%(levelname)s] %(message)s')
log = logging.getLogger('logger')
sys.stdout = StreamToLogger(log,logging.WARNING)
sys.stderr = StreamToLogger(log,logging.CRITICAL)
print('Test to standard out')
raise Exception('Test to standard error')

# logging.debug('This message should go to the log file')
# logging.info('So should this')
# logging.warning('And this, too')
# raise ValueError("What if it happens")
# logging.error('And non-ASCII stuff, too, like Øresund and Malmö')


# dshape = (1, 16, 64, 64)
# x = relay.var("x", shape=dshape)
# pooled = relay.nn.max_pool2d(x, pool_size=(2, 2), strides=(2, 2), padding=(0, 0))
# upsampled = relay.nn.upsampling(pooled, scale_h=2, scale_w=2, layout="NCHW")
# out = relay.Tuple((upsampled, x))
# print(out)
# print(out.fields)
# data = relay.var("data", relay.TensorType((1, 64, 56, 56), "float32"))
# conv_weight = relay.var("2_weight", relay.TensorType((64, 64, 1, 1), "float32"))
# conv2d = relay.nn.conv2d(
#     data=data, weight=conv_weight, kernel_size=(1, 1), channels=64, padding=(0, 0)
# )
#
# is_tuple_get_item()
# print(repr(add))
# print(pat)
# print(match)


# from tvm.relay.transform.backend_operator.op_type import OpType
# for op_type in OpType:
#     print(op_type)

# from subprocess import Popen, PIPE, STDOUT, DEVNULL
# import time
#
# start_time = time.time()
# cmd = ['python3',  'tmp_measure_network.py', "nasrnn", "cuda"]
# p = Popen(cmd, stdout=DEVNULL, stderr=PIPE)
# p.wait()
# out, err = p.communicate()
# res = err.decode("utf-8").partition("##result:")
# assert(len(res)==3)
# numbers = res[2].split()
# mean_perf, std_perf = float(numbers[0]), float(numbers[1])
# print(f"time elapsed: {time.time()-start_time}")
#
# # import tvm
# # from tvm import relay
# #
# # def _traverse_expr(node, node_dict):
# #     if node in node_dict:
# #         return
# #     # if isinstance(node, relay.op.op.Op):
# #     #    return
# #     if isinstance(node, tvm.ir.op.Op):
# #         return
# #
# #     # print("{} : {}".format(node, type(node)))
# #     node_dict[node] = len(node_dict)
# #     print(node.backend)
# #
# # data = relay.var("data", shape=(10, 10))
# # expr = relay.nn.relu(data)
# # relay.analysis.update_backend(expr, "wow")
# # relay.analysis.update_backend(data, "wow2")
# # node_dict = {}
# # relay.analysis.post_order_visit(expr, lambda node: _traverse_expr(node, node_dict))