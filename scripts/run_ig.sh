python ./scripts/convert_onnx.py ./models/gpt2.onnx
python ./scripts/convert_onnx.py ./models/bert.bs1.onnx
python ./scripts/convert_onnx.py ./models/vit.bs1.onnx
python ./scripts/convert_onnx.py ./models/sardrn.bs1.onnx
python ./scripts/convert_onnx.py ./models/shufflenet.bs1.onnx
python ./scripts/convert_onnx.py ./models/rednet50.bs1.onnx

python ./scripts/convert_onnx.py ./models/bert.bs16.onnx
python ./scripts/convert_onnx.py ./models/vit.bs16.onnx
python ./scripts/convert_onnx.py ./models/sardrn.bs16.onnx
python ./scripts/convert_onnx.py ./models/shufflenet.bs16.onnx
python ./scripts/convert_onnx.py ./models/rednet50.bs16.onnx

./ig/test_onnx ./models/gpt2.graph
./ig/test_onnx ./models/bert.bs1.graph
./ig/test_onnx ./models/vit.bs1.graph
./ig/test_onnx ./models/sardrn.bs1.graph
./ig/test_onnx ./models/shufflenet.bs1.graph
./ig/test_onnx ./models/rednet50.bs1.graph

./ig/test_onnx ./models/bert.bs16.graph
./ig/test_onnx ./models/vit.bs16.graph
./ig/test_onnx ./models/sardrn.bs16.graph
./ig/test_onnx ./models/shufflenet.bs16.graph
./ig/test_onnx ./models/rednet50.bs16.graph
