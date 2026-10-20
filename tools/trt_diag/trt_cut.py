import sys, torch, tensorrt as trt
ONNX = "/home/sathvik_lokesh/Hanuman/nvidia-target/bdd-yolo-robustness/runs/B_s0/weights/last.onnx"
cut = int(sys.argv[1])
L = trt.Logger(trt.Logger.ERROR); B = trt.Builder(L); net = B.create_network(0); p = trt.OnnxParser(net, L)
assert p.parse(open(ONNX, "rb").read())
if cut >= 0:
    last = max(i for i in range(net.num_layers) if f"/model.{cut}/" in net.get_layer(i).name)
    t = net.get_layer(last).get_output(0)
    for i in range(net.num_outputs - 1, -1, -1): net.unmark_output(net.get_output(i))
    net.mark_output(t)
    print("cut", cut, "at", net.get_layer(last).name, end=" ")
cfg = B.create_builder_config(); cfg.set_flag(trt.BuilderFlag.INT8); cfg.set_flag(trt.BuilderFlag.FP16)
class C(trt.IInt8MinMaxCalibrator):
    def __init__(s): super().__init__(); s.n = 0; s.buf = torch.rand(1, 3, 640, 640, device="cuda")
    def get_batch_size(s): return 1
    def get_batch(s, names):
        s.n += 1; return [int(s.buf.data_ptr())] if s.n <= 4 else None
    def read_calibration_cache(s): return None
    def write_calibration_cache(s, c): pass
cfg.int8_calibrator = C()
print("OK" if B.build_serialized_network(net, cfg) is not None else "FAILED", flush=True)
