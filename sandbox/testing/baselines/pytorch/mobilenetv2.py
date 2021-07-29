import torch.nn as nn
import torch
import math


def conv(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        #nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        #nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                #nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                #nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                #nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                #nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                #nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return self.conv(x) + x
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = []
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenet_v2():
    model = MobileNetV2(width_mult=1)
    return model

import torch
import argparse
import onnx
import onnxruntime
import torch.autograd.profiler as profiler
import tvm.relay.op
from tqdm import tqdm 
from tvm import relay
import tvm
from tvm import te
import numpy as np
import tvm.contrib.graph_executor as runtime
from tvm.relay import testing
from torchvision.models import resnet

torch.backends.cudnn.benchmark = True

NAME = 'mobilenet_v2'
# W = 56
W = 224
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", help="How many iterations to average for timing", type=int, default=500)
    parser.add_argument("--discard_iter", help="How many iterations to not time during warm up", type=int, default=100)
    args = parser.parse_args()

    model = mobilenet_v2().cuda()
    model.eval()
    inputs = torch.randn(1, 32, W, W).cuda()

    from torch2trt import torch2trt
    import time
    model_trt = torch2trt(model, [inputs])

    times = []
    for i in tqdm(range(args.discard_iter + args.iterations)):

        torch.cuda.current_stream().synchronize()
        t0 = time.time()
        model_trt(inputs)
        torch.cuda.current_stream().synchronize()
        t1 = time.time()
        times.append(1000.0 * (t1 - t0))

    total = 0
    for i in range(args.discard_iter, len(times)):
        total += times[i]
    avg = total / (args.iterations)
    print("TensorRT: Average inference time of the last " + str(args.iterations) + " iterations: " + str(avg) + " ms")

    print(model(inputs).size())

    times = []
    with torch.no_grad():
        for i in tqdm(range(args.discard_iter + args.iterations)):
            
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            model(inputs)
            end.record()

            # Waits for everything to finish running
            torch.cuda.synchronize()

            times.append(start.elapsed_time(end))

    total = 0
    for i in range(args.discard_iter, len(times)):
        total += times[i]
    avg = total / (args.iterations)
    print("Average inference time of the last " + str(args.iterations) + " iterations: " + str(avg) + " ms")

    input_shape = [1, 32, W, W]
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model.cpu(), input_data).eval()

    torch.jit.save(scripted_model, f'models/{NAME}.pth')

    input_name = "input0"
    shape_list = [(input_name, input_shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

    #print("Relay module function:\n", mod.astext(show_meta_data=True))

    with open(f"models/{NAME}.txt", "w") as text_file:
        text_file.write(mod.astext(show_meta_data=True))

    input_names = [ "input0" ]
    output_names = [ "output0" ]

    model.eval()

    with torch.no_grad():
        out_torch = model(inputs.cpu()).cpu().detach().numpy()

    torch.onnx.export(scripted_model, input_data, 
                    f"models/{NAME}.onnx", verbose=False, 
                    export_params=True,
                    do_constant_folding=False,
                    input_names=input_names, output_names=output_names, 
                    training = torch.onnx.TrainingMode.TRAINING,
                    # example_outputs=torch.rand((1, 1280, 4, 4)),
                    opset_version=12)
    onnx_model = onnx.load(f"models/{NAME}.onnx")

    sess = onnxruntime.InferenceSession(f"models/{NAME}.onnx")
    out_onnx = sess.run(["output0"], {"input0": inputs.cpu().numpy()})[0]

    input_name = "input0"
    shape_dict = {input_name: input_shape}
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
    # Set inputs
    m.set_input(input_name, tvm.nd.array(inputs.cpu().numpy().astype(np.float32)))

    m2 = runtime.GraphModule(lib2["default"](ctx))
    # Set inputs
    m2.set_input(input_name, tvm.nd.array(inputs.cpu().numpy().astype(np.float32)))

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

    print(out_tvm[0,:10,0,0])
    print(out_tvm2[0,:10,0,0])
    print(out_torch[0,:10,0,0])
    print(out_onnx[0,:10,0,0])
    TOL = 1e-01
    assert np.allclose(out_onnx, out_torch, rtol=TOL, atol=TOL)
    assert np.allclose(out_onnx, out_tvm, rtol=TOL, atol=TOL)
    assert np.allclose(out_torch, out_tvm, rtol=TOL, atol=TOL)
    assert np.allclose(out_onnx, out_tvm2, rtol=TOL, atol=TOL)
    assert np.allclose(out_torch, out_tvm2, rtol=TOL, atol=TOL)

    print(np.abs((out_torch - out_tvm)).max())
