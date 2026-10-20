import sys, re, torch, tensorrt as trt
ONNX = "/home/sathvik_lokesh/Hanuman/nvidia-target/bdd-yolo-robustness/runs/B_s0/weights/last.onnx"
pat = re.compile(sys.argv[1])  # regex on conv layer names to pin to FP16 (with their act)
L = trt.Logger(trt.Logger.ERROR); B = trt.Builder(L); net = B.create_network(0); p = trt.OnnxParser(net, L)
assert p.parse(open(ONNX, "rb").read())
pinned = []
for i in range(net.num_layers):
    ly = net.get_layer(i)
    if pat.search(ly.name):
        ly.precision = trt.float16
        for j in range(ly.num_outputs): ly.set_output_type(j, trt.float16)
        pinned.append(ly.name)
conv = sum(net.get_layer(i).type == trt.LayerType.CONVOLUTION for i in range(net.num_layers))
cfg = B.create_builder_config(); cfg.set_flag(trt.BuilderFlag.INT8); cfg.set_flag(trt.BuilderFlag.FP16)
cfg.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
class C(trt.IInt8MinMaxCalibrator):
    def __init__(s): super().__init__(); s.n = 0; s.buf = torch.rand(1, 3, 640, 640, device="cuda")
    def get_batch_size(s): return 1
    def get_batch(s, names):
        s.n += 1; return [int(s.buf.data_ptr())] if s.n <= 4 else None
    def read_calibration_cache(s): return None
    def write_calibration_cache(s, c): pass
cfg.int8_calibrator = C()
npc = sum("conv/Conv" in n for n in pinned)
print(f"pinned {len(pinned)} layers ({npc}/{conv} convs)", "OK" if B.build_serialized_network(net, cfg) is not None else "FAILED", flush=True)
