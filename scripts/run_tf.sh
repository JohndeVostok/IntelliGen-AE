python ./scripts/onnx2pb.py ./models/gpt2.onnx ./models/gpt2.pb
python ./scripts/onnx2pb.py ./models/bert.bs1.onnx ./models/bert.bs1.pb
python ./scripts/onnx2pb.py ./models/vit.bs1.onnx ./models/vit.bs1.pb
python ./scripts/onnx2pb.py ./models/sardrn.bs1.onnx ./models/sardrn.bs1.pb
python ./scripts/onnx2pb.py ./models/shufflenet.bs1.onnx ./models/shufflenet.bs1.pb
python ./scripts/onnx2pb.py ./models/rednet50.bs1.onnx ./models/rednet50.bs1.pb

python ./scripts/onnx2pb.py ./models/bert.bs16.onnx ./models/bert.bs16.pb
python ./scripts/onnx2pb.py ./models/vit.bs16.onnx ./models/vit.bs16.pb
python ./scripts/onnx2pb.py ./models/sardrn.bs16.onnx ./models/sardrn.bs16.pb
python ./scripts/onnx2pb.py ./models/shufflenet.bs16.onnx ./models/shufflenet.bs16.pb
python ./scripts/onnx2pb.py ./models/rednet50.bs16.onnx ./models/rednet50.bs16.pb

echo "RUN TF"
python ./scripts/run_tf.py ./models/gpt2.pb 128 1 768 False
python ./scripts/run_tf.py ./models/bert.bs1.pb 1 512 768 False
python ./scripts/run_tf.py ./models/vit.bs1.pb 1 64 768 False
python ./scripts/run_tf.py ./models/sardrn.bs1.pb 1 512 512 False
python ./scripts/run_tf.py ./models/shufflenet.bs1.pb 1 3 224 224 False
python ./scripts/run_tf.py ./models/rednet50.bs1.pb 1 3 224 224 False

python ./scripts/run_tf.py ./models/bert.bs16.pb 16 512 768 False
python ./scripts/run_tf.py ./models/vit.bs16.pb 16 64 768 False
python ./scripts/run_tf.py ./models/sardrn.bs16.pb 16 512 512 False
python ./scripts/run_tf.py ./models/shufflenet.bs16.pb 16 3 224 224 False
python ./scripts/run_tf.py ./models/rednet50.bs16.pb 16 3 224 224 False

echo "RUN TF-XLA"
python ./scripts/run_tf.py ./models/gpt2.pb 128 1 768 True
python ./scripts/run_tf.py ./models/bert.bs1.pb 1 512 768 True
python ./scripts/run_tf.py ./models/vit.bs1.pb 1 64 768 True
python ./scripts/run_tf.py ./models/sardrn.bs1.pb 1 512 512 True
python ./scripts/run_tf.py ./models/shufflenet.bs1.pb 1 3 224 224 True
python ./scripts/run_tf.py ./models/rednet50.bs1.pb 1 3 224 224 True

python ./scripts/run_tf.py ./models/bert.bs16.pb 16 512 768 True
python ./scripts/run_tf.py ./models/vit.bs16.pb 16 64 768 True
python ./scripts/run_tf.py ./models/sardrn.bs16.pb 16 512 512 True
python ./scripts/run_tf.py ./models/shufflenet.bs16.pb 16 3 224 224 True
python ./scripts/run_tf.py ./models/rednet50.bs16.pb 16 3 224 224 True

