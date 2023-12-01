trtexec --workspace=8192 --onnx=./models/gpt2.onnx --saveEngine=./models/gpt2.trt
trtexec --workspace=8192 --onnx=./models/bert.bs1.onnx --saveEngine=./models/bert.bs1.trt
trtexec --workspace=8192 --onnx=./models/vit.bs1.onnx --saveEngine=./models/vit.bs1.trt
trtexec --workspace=8192 --onnx=./models/sardrn.bs1.onnx --saveEngine=./models/sardrn.bs1.trt
trtexec --workspace=8192 --onnx=./models/shufflenet.bs1.onnx --saveEngine=./models/shufflenet.bs1.trt
trtexec --workspace=8192 --onnx=./models/rednet50.bs1.onnx --saveEngine=./models/rednet50.bs1.trt

trtexec --workspace=8192 --onnx=./models/bert.bs16.onnx --saveEngine=./models/bert.bs16.trt
trtexec --workspace=8192 --onnx=./models/vit.bs16.onnx --saveEngine=./models/vit.bs16.trt
trtexec --workspace=8192 --onnx=./models/sardrn.bs16.onnx --saveEngine=./models/sardrn.bs16.trt
trtexec --workspace=8192 --onnx=./models/shufflenet.bs16.onnx --saveEngine=./models/shufflenet.bs16.trt
trtexec --workspace=8192 --onnx=./models/rednet50.bs16.onnx --saveEngine=./models/rednet50.bs16.trt

trtexec --loadEngine=./models/gpt2.trt
trtexec --loadEngine=./models/bert.bs1.trt
trtexec --loadEngine=./models/vit.bs1.trt
trtexec --loadEngine=./models/sardrn.bs1.trt
trtexec --loadEngine=./models/shufflenet.bs1.trt
trtexec --loadEngine=./models/rednet50.bs1.trt

trtexec --loadEngine=./models/bert.bs16.trt
trtexec --loadEngine=./models/vit.bs16.trt
trtexec --loadEngine=./models/sardrn.bs16.trt
trtexec --loadEngine=./models/shufflenet.bs16.trt
trtexec --loadEngine=./models/rednet50.bs16.trt
