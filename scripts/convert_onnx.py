import onnx
import numpy as np
import sys


def getNpType(typeId):
    map = {1: np.float32, 7: np.int64, 9: np.bool8}
    igAssert(typeId in map)
    return (map[typeId])


def parseMatMul(node):
    tensorFlag[node.input[0]] = True
    tensorFlag[node.input[1]] = True
    tensorFlag[node.output[0]] = True
    buf = " ".join(["MATMUL", node.input[0],
                   node.input[1], node.output[0], str(0), str(0)])
    return buf


def parseGemm(node, ops):
    tensorFlag[node.input[0]] = True
    tensorFlag[node.input[1]] = True
    tensorFlag[node.input[2]] = True
    tensorFlag[node.output[0]] = True
    buffer = node.name + "/buffer"
    tensorShape[buffer] = tensorShape[node.output[0]]
    tensorFlag[buffer] = True
    attrs = node.attribute
    alpha = 1.0
    beta = 1.0
    transA = 0
    transB = 0
    for attr in attrs:
        if attr.name == "alpha":
            alpha = attr.f
        elif attr.name == "beta":
            beta = attr.f
        elif attr.name == "transA":
            transA = attr.i
        elif attr.name == "transB":
            transB = attr.i
    igAssert(alpha == 1.0 and beta == 1.0)

    ops.append(" ".join(["MATMUL", node.input[0],
                         node.input[1], buffer, str(transA), str(transB)]))
    ops.append(" ".join(["ADD", node.input[2], buffer, node.output[0]]))


def parseConv(node, ops):
    attrs = node.attribute
    igAssert(attrs[0].name == "dilations" and len(attrs[0].ints)
             == 2 and attrs[0].ints[0] == attrs[0].ints[1])
    dilation = attrs[0].ints[0]
    igAssert(attrs[1].name == "group")
    group = attrs[1].i
    igAssert(attrs[2].name == "kernel_shape" and len(attrs[2].ints)
             == 2 and attrs[2].ints[0] == attrs[2].ints[1])
    kernel_shape = attrs[2].ints[0]
    igAssert(attrs[3].name == "pads" and len(attrs[3].ints) == 4 and attrs[3].ints[0] == attrs[3].ints[1]
             and attrs[3].ints[0] == attrs[3].ints[2] and attrs[3].ints[0] == attrs[3].ints[3])
    pad = attrs[3].ints[0]
    igAssert(attrs[4].name == "strides" and len(attrs[4].ints)
             == 2 and attrs[4].ints[0] == attrs[4].ints[1])
    stride = attrs[4].ints[0]

    tensorFlag[node.input[0]] = True
    tensorFlag[node.input[1]] = True
    tensorFlag[node.input[2]] = True
    tensorFlag[node.output[0]] = True
    ops.append(" ".join(["CONV", node.input[0],
                         node.input[1], node.input[2], node.output[0], str(kernel_shape), str(pad), str(stride), str(dilation), str(group)]))


def parseMaxPool(node, ops):
    attrs = node.attribute
    for attr in attrs:
        if attr.name == "ceil_mode":
            igAssert(attr.i == 0)
        if attr.name == "kernel_shape":
            igAssert(len(attr.ints) == 2 and attr.ints[0] == attr.ints[1])
            kernel_shape = attr.ints[0]
        if attr.name == "pads":
            igAssert(len(attr.ints) == 4 and attr.ints[0] == attr.ints[1]
                     and attr.ints[0] == attr.ints[2] and attr.ints[0] == attr.ints[3])
            pad = attr.ints[0]
        if attr.name == "strides":
            igAssert(len(attr.ints) == 2 and attr.ints[0] == attr.ints[1])
            stride = attrs[2].ints[0]

    tensorFlag[node.input[0]] = True
    tensorFlag[node.output[0]] = True
    ops.append(" ".join(["MAXPOOL", node.input[0],
                         node.output[0], str(kernel_shape), str(pad), str(stride)]))


def parseGlobalAveragePool(node, ops):
    tensorFlag[node.input[0]] = True
    tensorFlag[node.output[0]] = True
    ops.append(" ".join(["REDUCEMEAN", node.input[0],
                         node.output[0], "2", "2", "3"]))


def parseBinary(node):
    tensorFlag[node.input[0]] = True
    tensorFlag[node.output[0]] = True
    if node.input[1] in tensorData and len(tensorShape[node.input[1]]) == 0:
        value = tensorData[node.input[1]][0]
        buf = " ".join([node.op_type.upper() + "C",
                       node.input[0], node.output[0], str(value)])
    else:
        tensorFlag[node.input[1]] = True
        buf = " ".join([node.op_type.upper(), node.input[0],
                        node.input[1], node.output[0]])
    return buf


def parseUnary(node):
    tensorFlag[node.input[0]] = True
    tensorFlag[node.output[0]] = True
    buf = " ".join([node.op_type.upper(),
                    node.input[0], node.output[0]])
    return buf


def parseReduce(node):
    tensorFlag[node.input[0]] = True
    tensorFlag[node.output[0]] = True
    attr = node.attribute[0]
    igAssert(attr.name == "axis" or attr.name == "axes")
    lenDim = len(tensorShape[node.input[0]])
    if (attr.name == "axis"):
        axes = [(attr.i + lenDim) % lenDim]
    else:
        axes = [(x + lenDim) % lenDim for x in attr.ints]
    bufv = [node.op_type.upper(),
            node.input[0], node.output[0], str(len(axes))]
    for x in axes:
        bufv.append(str(x))
    buf = " ".join(bufv)
    return buf


def parseReshape(node):
    tensorFlag[node.input[0]] = True
    tensorFlag[node.output[0]] = True

    size = 1
    for x in tensorShape[node.input[0]]:
        size *= x
    idx = -1
    shape = [0 for x in list(tensorData[node.input[1]])]
    for i in range(len(list(tensorData[node.input[1]]))):
        if (list(tensorData[node.input[1]])[i] == -1):
            igAssert(idx == -1)
            idx = i
        else:
            shape[i] = list(tensorData[node.input[1]])[i]
            size = size / shape[i]
    if (idx != -1):
        shape[idx] = size

    igAssert(shape == tensorShape[node.output[0]])
    buf = " ".join(["RESHAPE",
                   node.input[0], node.output[0]])
    return buf


def parseFlatten(node):
    tensorFlag[node.input[0]] = True
    tensorFlag[node.output[0]] = True

    sizeA = 1
    for x in tensorShape[node.input[0]]:
        sizeA *= x
    sizeB = 1
    for x in tensorShape[node.output[0]]:
        sizeB *= x
    igAssert(sizeA == sizeB)
    buf = " ".join(["RESHAPE",
                   node.input[0], node.output[0]])
    return buf


def parseTranspose(node):
    tensorFlag[node.input[0]] = True
    tensorFlag[node.output[0]] = True
    attr = node.attribute[0]
    igAssert(attr.name == "perm")
    perm = list(attr.ints)
    bufv = ["TRANSPOSE",
            node.input[0], node.output[0], str(len(perm))]
    for x in perm:
        bufv.append(str(x))
    buf = " ".join(bufv)
    return buf


def parseGather(node):
    tensorFlag[node.input[0]] = True
    tensorFlag[node.output[0]] = True
    attr = node.attribute[0]
    igAssert(attr.name == "axis" and attr.i == 0)
    igAssert(tensorData[node.input[1]][0] == 0)
    buf = " ".join(["RESHAPE",
                   node.input[0], node.output[0]])
    return buf


def parseSplit(node, ops):
    attr = node.attribute[0]
    igAssert(attr.name == "axis")
    axis = attr.i
    tensorFlag[node.input[0]] = True
    for i in range(len(node.output)):
        tensorFlag[node.output[i]] = True

    listBuf = ["SPLIT", node.input[0]]
    for i in range(len(node.output)):
        listBuf.append(node.output[i])
    listBuf.append(str(axis))
    buf = " ".join(listBuf)

    ops.append(buf)


def parseConcat(node, ops):
    attr = node.attribute[0]
    igAssert(attr.name == "axis")
    axis = attr.i
    tensorFlag[node.input[0]] = True
    tensorFlag[node.input[1]] = True
    tensorFlag[node.output[0]] = True

    buf = " ".join(["CONCAT", node.input[0], node.input[1],
                   node.output[0], str(axis)])
    ops.append(buf)


def parseWhere(node, ops):
    tensorFlag[node.input[0]] = True
    tensorFlag[node.output[0]] = True

    cond = np.array(tensorData[node.input[0]])
    for x in cond:
        igAssert(x == True)

    igAssert(tensorShape[node.input[1]] == tensorShape[node.output[0]])
    ops.append(" ".join(["RESHAPE",
                         node.input[1], node.output[0]]))


if __name__ == "__main__":
    model_filename = sys.argv[-1]
    graph_filename = model_filename.replace(".onnx", ".graph")
    model = onnx.load(model_filename)

    tensorShape = {}
    tensorData = {}
    tensorFlag = {}

    # Get input shapes.
    for tensor in model.graph.input:
        igAssert(tensor.name not in tensorShape)
        shape = [x.dim_value for x in tensor.type.tensor_type.shape.dim]
        tensorShape[tensor.name] = shape

    # Get output shapes.
    for tensor in model.graph.output:
        igAssert(tensor.name not in tensorShape)
        shape = [x.dim_value for x in tensor.type.tensor_type.shape.dim]
        tensorShape[tensor.name] = shape

    # Get weight shapes.
    for tensor in model.graph.initializer:
        igAssert(tensor.name not in tensorShape)
        params = np.frombuffer(
            tensor.raw_data, dtype=getNpType(tensor.data_type))
        tensorShape[tensor.name] = tensor.dims
        tensorData[tensor.name] = params

    # Tensor_inference
    for tensor in model.graph.value_info:
        igAssert(tensor.name not in tensorShape)
        shape = [x.dim_value for x in tensor.type.tensor_type.shape.dim]
        tensorShape[tensor.name] = shape

    ops = []
    for node in model.graph.node:
        if node.op_type == "MatMul":
            ops.append(parseMatMul(node))
        elif node.op_type == "Gemm":
            parseGemm(node, ops)
        elif node.op_type == "Conv":
            parseConv(node, ops)
        elif node.op_type == "MaxPool":
            pass
            # parseMaxPool(node, ops)
        elif node.op_type == "GlobalAveragePool":
            parseGlobalAveragePool(node, ops)
        elif node.op_type == "Sqrt" or node.op_type == "Erf" or node.op_type == "Tanh" or node.op_type == "Relu" or node.op_type == "Sigmoid":
            ops.append(parseUnary(node))
        elif node.op_type == "Add" or node.op_type == "Sub" or node.op_type == "Mul" or node.op_type == "Div" or node.op_type == "Pow":
            ops.append(parseBinary(node))
        elif node.op_type == "ReduceMean" or node.op_type == "Softmax":
            ops.append(parseReduce(node))
        elif node.op_type == "Reshape":
            ops.append(parseReshape(node))
        elif node.op_type == "Flatten":
            ops.append(parseFlatten(node))
        elif node.op_type == "Transpose":
            ops.append(parseTranspose(node))
        elif node.op_type == "Gather":
            # ops.append(parseGather(node))
            pass
        elif node.op_type == "Split":
            parseSplit(node, ops)
        elif node.op_type == "Concat":
            parseConcat(node, ops)
        elif node.op_type == "Where":
            parseWhere(node, ops)
        else:
            print(node.op_type)
            # igAssert (False)

    tensors = []
    for name in tensorFlag:
        bufv = [name, str(len(tensorShape[name]))]
        for x in tensorShape[name]:
            bufv.append(str(x))
        buf = " ".join(bufv)
        tensors.append(buf)

    lines = []
    lines.append(str(len(tensors)) + "\n")
    for x in tensors:
        lines.append(x + "\n")
    lines.append(str(len(ops)) + "\n")
    for x in ops:
        lines.append(x + "\n")
    with open(graph_filename, "w") as f:
        f.writelines(lines)
