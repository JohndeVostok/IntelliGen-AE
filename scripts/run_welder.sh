mkdir ./models/gpt2
cp ./models/gpt2.onnx ./models/gpt2/model.onnx
mkdir ./models/bert.bs1
cp ./models/bert.bs1.onnx ./models/bert.bs1/model.onnx
mkdir ./models/vit.bs1
cp ./models/vit.bs1.onnx ./models/vit.bs1/.onnx
mkdir ./models/sardrn.bs1
cp ./models/sardrn.bs1.onnx ./models/sardrn.bs1/model.onnx
mkdir ./models/shufflenet.bs1
cp ./models/shufflenet.bs1.onnx ./models/shufflenet.bs1/.model.onnx
mkdir ./models/rednet50.bs1
cp ./models/rednet50.bs1.onnx ./models/rednet50.bs1/.model.onnx

mkdir ./models/bert.bs16
cp ./models/bert.bs16.onnx ./models/bert.bs16/.model.onnx
mkdir ./models/vit.bs16
cp ./models/vit.bs16.onnx ./models/vit.bs16/.model.onnx
mkdir ./models/sardrn.bs16
cp ./models/sardrn.bs16.onnx ./models/sardrn.bs16/.model.onnx
mkdir ./models/shufflenet.bs16
cp ./models/shufflenet.bs16.onnx ./models/shufflenet.bs16/.model.onnx
mkdir ./models/rednet50.bs1
cp ./models/rednet50.bs1.onnx ./models/rednet50.bs1/.model.onnx

python tune_welder.py ./models/gpt2 --topk 20 --arch V100
python tune_welder.py ./models/bert.bs1 --topk 20 --arch V100 --skip_dot
python tune_welder.py ./models/vit.bs1 --topk 20 --arch V100 --skip_dot
python tune_welder.py ./models/sardrn.bs1 --topk 20 --arch V100
python tune_welder.py ./models/shufflenet.bs1 --topk 20 --arch V100
python tune_welder.py ./models/rednet50.bs1 --topk 20 --arch V100 --skip_dot

python tune_welder.py ./models/bert.bs16 --topk 20 --arch V100 --skip_dot
python tune_welder.py ./models/vit.bs16 --topk 20 --arch V100 --skip_dot
python tune_welder.py ./models/sardrn.bs16 --topk 20 --arch V100 --skip_dot
python tune_welder.py ./models/shufflenet.bs16 --topk 20 --arch V100 --skip_dot
python tune_welder.py ./models/rednet50.bs16 --topk 20 --arch V100 --skip_dot

./models/bert.bs1/nnfusion_rt/cuda_codegen/build/main_test
./models/gpt2/nnfusion_rt/cuda_codegen/build/main_test
./models/vit.bs1/nnfusion_rt/cuda_codegen/build/main_test
./models/sardrn.bs1/nnfusion_rt/cuda_codegen/build/main_test
./models/shufflenet.bs1/nnfusion_rt/cuda_codegen/build/main_test
./models/rednet50.bs1/nnfusion_rt/cuda_codegen/build/main_test

./models/bert.bs16/nnfusion_rt/cuda_codegen/build/main_test
./models/vit.bs16/nnfusion_rt/cuda_codegen/build/main_test
./models/sardrn.bs16/nnfusion_rt/cuda_codegen/build/main_test
./models/shufflenet.bs16/nnfusion_rt/cuda_codegen/build/main_test
./models/rednet50.bs16/nnfusion_rt/cuda_codegen/build/main_test