# IntelliGen-AE

This repo is for AE of IntelliGen. Our paper discusses a proprietary system. The source code is currently unable to be publicly available for now, so we use binary executable for supplemental materials. Moreover, the experimental setup for our study requires specific hardware configurations, including some newest domain-specific accelerators (Cambricon MLU). Additionally, some baselines in our experiment necessitate unique modifications tailored to this hardware. So we strongly recommend conducting reproducibility experiments on our in-house cluster. This approach would ensure the system is runnable and the data are consistent with the details presented in the paper.

step1: build onnx model, eval PyTorch and TorchInductor

```
./scripts/run_pytorch.sh
```

step2: build pb model, eval TensorFlow and TensorFlow-XLA

```
./scripts/run_tf.sh
```

step3: eval TensorRT

```
./scripts/run_trt.sh
```

step4: eval Welder

```
./scripts/run_welder.sh
```

step5: build graph model, eval IntelliGen

```
./scripts/run_ig.sh
```
