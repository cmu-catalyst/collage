import torch
import argparse
import numpy as np
import time 
import torch.nn as nn
import numpy
import tvm.contrib.graph_executor as runtime
import torch
import tvm
from tvm import relay
import torch
import argparse
import onnx
import onnxruntime
import torchvision.models as models
import torch.autograd.profiler as profiler
import tvm.relay.op
from tqdm import tqdm 
from tvm import relay
import tvm
from tvm import te
import numpy as np
import tvm.contrib.graph_executor as runtime
from tvm.relay import testing
from torch.nn.parameter import Parameter

torch.backends.cudnn.benchmark = True

# NasRNN always have some errors during autotuning operators with AutoTVM
HIDDEN_SIZE = 512
# HIDDEN_SIZE = 1024
# HIDDEN_SIZE = 2048
# HIDDEN_SIZE = 2560

# 4096 causes error for end-to-end TensorRT measurement
# HIDDEN_SIZE = 4096
LENGTH = 5
NAME = "nasrnn"

class CombineCell(nn.Module):
    def __init__(self):
        super(CombineCell, self).__init__()
        self.w1 = nn.parameter.Parameter(data = torch.randn(HIDDEN_SIZE, HIDDEN_SIZE))
        self.w2 = nn.parameter.Parameter(data = torch.randn(HIDDEN_SIZE, HIDDEN_SIZE))

    def forward(self, x, h):
        w1 = torch.matmul(x, self.w1)
        w2 = torch.matmul(h, self.w2) 
        return torch.add(torch.nn.functional.relu(w1), torch.nn.functional.relu(w2))

class NASCell(nn.Module):
    def __init__(self):
        super(NASCell, self).__init__()
        self.nodes = nn.ModuleList([CombineCell() for i in range(8)])

    def forward(self, input, x):
        t = [self.nodes[i](x, input) for i in range(8)]      
        midt = []
        midt.append(torch.add(torch.nn.functional.relu(t[0]), torch.sigmoid(t[3])))
        midt.append(torch.add(torch.sigmoid(t[1]), torch.tanh(t[2])))
        midt.append(torch.mul(torch.sigmoid(t[4]), torch.tanh(t[5])))
        midt.append(torch.mul(torch.sigmoid(t[6]), torch.nn.functional.relu(t[7])))
        midt.append(torch.add(torch.sigmoid(midt[1]), torch.tanh(midt[2])))
        midt.append(torch.mul(torch.tanh(midt[0]), torch.tanh(midt[3])))
        midt.append(torch.mul(torch.tanh(midt[4]), torch.tanh(midt[5])))
        return torch.tanh(midt[6])

class NASRNN(nn.Module):
    def __init__(self, is_gpu=True):
        super(NASRNN, self).__init__()
        self.cells = nn.ModuleList([NASCell() for i in range(LENGTH)])
        if is_gpu:
            self.state_init = (torch.randn(1, HIDDEN_SIZE)).cuda()
            self.b = (torch.randn(1, HIDDEN_SIZE)).cuda()
            self.c = (torch.randn(1, HIDDEN_SIZE)).cuda()
            self.d = (torch.randn(1, HIDDEN_SIZE)).cuda()
            self.e = (torch.randn(1, HIDDEN_SIZE)).cuda()
        else:
            self.state_init = (torch.randn(1, HIDDEN_SIZE))
            self.b = (torch.randn(1, HIDDEN_SIZE))
            self.c = (torch.randn(1, HIDDEN_SIZE))
            self.d = (torch.randn(1, HIDDEN_SIZE))
            self.e = (torch.randn(1, HIDDEN_SIZE))

    def forward(self, a):
        state = self.state_init
        state = self.cells[0](state, a)
        state = self.cells[1](state, self.b)
        state = self.cells[2](state, self.c)
        state = self.cells[3](state, self.d)
        state = self.cells[4](state, self.e)

        return state

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", help="How many iterations to average for timing (default 5000)", type=int, default=5000)
    parser.add_argument("--discard_iter", help="How many iterations to not time during warm up (default 1000)", type=int, default=1000)
    args = parser.parse_args()

    model = NASRNN().cuda()

    x = torch.randn(1,HIDDEN_SIZE).cuda()


    from torch2trt import torch2trt
    import time
    model_trt = torch2trt(model, [x])

    times = []
    for i in tqdm(range(args.discard_iter + args.iterations)):

        torch.cuda.current_stream().synchronize()
        t0 = time.time()
        model_trt(x)
        torch.cuda.current_stream().synchronize()
        t1 = time.time()
        times.append(1000.0 * (t1 - t0))

    total = 0
    for i in range(args.discard_iter, len(times)):
        total += times[i]
    avg = total / (args.iterations)
    print("TensorRT: Average inference time of the last " + str(args.iterations) + " iterations: " + str(avg) + " ms")

    times = []
    for i in range(args.discard_iter + args.iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        model(x)
        end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()

        times.append(start.elapsed_time(end))

    total = 0

    out = model(x)
    print([a.size() for a in out])

    for i in range(args.discard_iter, len(times)):
        total += times[i]
    avg = total / (args.iterations)
    print("Average inference time of the last " + str(args.iterations) + " iterations: " + str(avg) + " ms")

    model.eval()

    with torch.no_grad():
        traced_model = torch.jit.trace(model, x)
        traced_model.eval()

    # Save tne model
    # scripted_model = torch.jit.trace(model.cpu(), x.cpu()).eval()
    torch.jit.save(traced_model, f'models/{NAME}.pth')

    traced_model.eval()
    for p in traced_model.parameters():
        p.requires_grad_(False)

    print(traced_model.graph)
    shape_list = [(i.debugName(), i.type().sizes()) for i in  list(traced_model.graph.inputs())[1:]]

    print(shape_list)

    mod, params = relay.frontend.pytorch.from_pytorch(traced_model,
                            shape_list, default_dtype="float32")

    with open(f"models/{NAME}.txt", "w") as text_file:
        text_file.write(mod.astext(show_meta_data=True))


    input_names = [x[0] for x in shape_list]
    output_names = [ "output0" ]

    inputs = x
    model.eval()

    with torch.no_grad():
        out_torch = model(x).cpu().detach().numpy()

    torch.onnx.export(model, tuple(inputs),
                      f"models/{NAME}.onnx", verbose=False,
                      export_params=True,
                      do_constant_folding=False,
                      input_names=input_names, output_names=output_names,
                      training = torch.onnx.TrainingMode.TRAINING,
                      opset_version=12)
    onnx_model = onnx.load(f"models/{NAME}.onnx")

    sess = onnxruntime.InferenceSession(f"models/{NAME}.onnx")
    out_onnx = sess.run(output_names, {k:v.cpu().numpy() for k,v in zip(input_names, inputs)})[0]

    shape_dict = {l[0]: l[1] for l in shape_list}
    mod2, params2 = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)


    with open(f"models/{NAME}_onnx.txt", "w") as text_file:
        text_file.write(mod2.astext(show_meta_data=True))

    # Bulid the subgraph
    ctx = tvm.device("cuda", 0)

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target="cuda", target_host="llvm", params=params)

    with tvm.transform.PassContext(opt_level=3):
        lib2 = relay.build(mod2, target="cuda", target_host="llvm", params=params2)

    m = runtime.GraphModule(lib["default"](ctx))
    m2 = runtime.GraphModule(lib2["default"](ctx))

    # Set inputs
    for k,v in zip(input_names, inputs):
        m.set_input(k, tvm.nd.array(v.cpu().numpy().astype(np.float32)))
        m2.set_input(k, tvm.nd.array(v.cpu().numpy().astype(np.float32)))

    # Measure performance
    ftimer = m.module.time_evaluator("run", ctx, number=100, repeat=3)
    prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    perf = np.mean(prof_res)
    print("%.5f ms" % (perf))

    ftimer = m2.module.time_evaluator("run", ctx, number=100, repeat=3)
    prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    perf = np.mean(prof_res)
    print("%.5f ms" % (perf))

    m.run()
    out = m.get_output(0)
    out_tvm = out.asnumpy()

    m2.run()
    out = m2.get_output(0)
    out_tvm2 = out.asnumpy()

    print(out_tvm[0,:10])
    print(out_tvm2[0,:10])
    print(out_torch[0,:10])
    print(out_onnx[0,:10])
    TOL = 2e-01
    assert np.allclose(out_onnx, out_torch, rtol=TOL, atol=TOL)
    assert np.allclose(out_onnx, out_tvm, rtol=TOL, atol=TOL)
    assert np.allclose(out_torch, out_tvm, rtol=TOL, atol=TOL)
    assert np.allclose(out_onnx, out_tvm2, rtol=TOL, atol=TOL)
    assert np.allclose(out_torch, out_tvm2, rtol=TOL, atol=TOL)
