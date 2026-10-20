import sys, numpy as np, torch, tensorrt as trt
mode = sys.argv[1]  # conv | attn
class Net(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.c1=torch.nn.Conv2d(3,32,3,2,1); s.c2=torch.nn.Conv2d(32,64,3,2,1)
    def forward(s,x):
        x=torch.nn.functional.silu(s.c2(torch.nn.functional.silu(s.c1(x))))
        if mode=="attn":
            b,c,h,w=x.shape; q=x.flatten(2)  # (b,c,hw)
            a=(q.transpose(1,2)@q).softmax(-1); x=x+(q@a.transpose(1,2)).view(b,c,h,w)
        return x.sigmoid()
torch.onnx.export(Net().eval(), torch.randn(1,3,64,64), f"t_{mode}.onnx", opset_version=18, dynamo=False)
L=trt.Logger(trt.Logger.WARNING); B=trt.Builder(L); net=B.create_network(0); p=trt.OnnxParser(net,L)
assert p.parse(open(f"t_{mode}.onnx","rb").read())
cfg=B.create_builder_config(); cfg.set_flag(trt.BuilderFlag.INT8); cfg.set_flag(trt.BuilderFlag.FP16)
class C(trt.IInt8MinMaxCalibrator):
    def __init__(s): super().__init__(); s.n=0; s.buf=torch.rand(1,3,64,64,device="cuda")
    def get_batch_size(s): return 1
    def get_batch(s,names):
        s.n+=1; return [int(s.buf.data_ptr())] if s.n<=8 else None
    def read_calibration_cache(s): return None
    def write_calibration_cache(s,c): pass
cfg.int8_calibrator=C()
e=B.build_serialized_network(net,cfg); print(mode, "OK" if e is not None else "FAILED")
