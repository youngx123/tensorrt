# -*- coding: utf-8 -*-
# @Author : youngx
# @Time : 10:59  2022-04-06
import glob
import sys

sys.path.append(r"D:\Inference\TensorRT-8.2.0.6\lib")
import imageio
import cv2
import os
import numpy as np
import pycuda
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import torch
import torchvision


print("tensorrt version :", trt.__version__)


def Inference(engine, input_data, input_device, out_data, out_device, stream, repeat_time=1):
    """
    inference
    :param engine: trt engine
    :param input_data: input data
    :param input_device: input data device number on cuda
    :param out_data: output data  placeholder
    :param out_device: output data device number on cuda
    :param stream:
    :param repeat_time:
    :return:
    """
    # load random data to page-locked buffer
    assert len(input_data) == len(input_device)
    for i_data, d_mem in zip(input_data, input_device):
        cuda.memcpy_htod_async(d_mem, i_data, stream)

    for _ in range(repeat_time):
        # # 将输入数据放到对应开辟的gpu 上, htod
        for o_data, o_device in zip(out_data, out_device):
            cuda.memcpy_htod_async(o_device, o_data, stream)

        # # 输入和输出数据的内存号
        bindings = [int(x) for x in input_device] + [int(x) for x in out_device]
        engine.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

        # # 将网络结果 由开辟的GPU 放到CPU上 dtoh
        for o_data, o_device in zip(out_data, out_device):
            cuda.memcpy_dtoh_async(o_data, o_device, stream)
        stream.synchronize()


def build_engin(onxpath, engine_file_path):
    """
    model load and convert
    :param onxpath: onnx model path
    :param engine_file_path: if save trt model converted from onnx
    :return:
    """
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if os.path.exists(engine_file_path):
        print("load trt parse")
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_file_path, "rb") as f:
            engin = runtime.deserialize_cuda_engine(f.read())
            return engin
    else:
        print("load onnx parse")
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(EXPLICIT_BATCH)
        config = builder.create_builder_config()
        parser = trt.OnnxParser(network, TRT_LOGGER)
        runtime = trt.Runtime(TRT_LOGGER)

        config.max_workspace_size = 1 << 32  # 256MiB
        builder.max_batch_size = 2

        print('Loading ONNX file from path {}...'.format(onxpath))

        with open(onxpath, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
        # # #M2
        # if not parser.parse_from_file(str(onxpath)):
        #     raise RuntimeError(f'failed to load ONNX file: {onxpath}')

        input_shape = [network.get_input(i).shape for i in range(network.num_inputs)]
        outputs_shape = [network.get_output(i).shape for i in range(network.num_outputs)]

        print('Completed parsing of ONNX file')
        print('Building an engine from file {}; this may take a while...'.format(onxpath))
        plan = builder.build_serialized_network(network, config)
        print("Completed creating Engine")
        assert plan, "build serialized network error"

        # # convert trt file
        if engine_file_path:
            with open(engine_file_path, "wb") as f:
                f.write(plan)

        # # return engine
        engine = runtime.deserialize_cuda_engine(plan)

        return engine


def build_engin2(onxpath, engine_file_path, half=True):
    """
    model load and convert
    :param onxpath: onnx model path
    :param engine_file_path: if save trt model converted from onnx
    :return:
    """
    logger = trt.Logger(trt.Logger.INFO)
    if os.path.exists(engine_file_path):
        print("load trt parse")
        runtime = trt.Runtime(logger)
        with open(engine_file_path, "rb") as f:
            engin = runtime.deserialize_cuda_engine(f.read())
            return engin
    else:
        print("load onnx parse")
        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 32
        builder.max_batch_size = 2

        flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, logger)

        # #M1
        if not parser.parse_from_file(str(onxpath)):
            raise RuntimeError(f'failed to load ONNX file: {onxpath}')

        # #M2
        # with open(onxpath, 'rb') as model:
        #     if not parser.parse(model.read()):
        #         for error in range(parser.num_errors):
        #             print(parser.get_error(error))
        #         return None

        inputs = [[i, network.get_input(i)] for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        for s in inputs:
            print(s[0], s[1].shape)

        for s in outputs:
            print(s.shape)

        half &= builder.platform_has_fast_fp16
        if half:
            config.set_flag(trt.BuilderFlag.FP16)
            # builder.fp16_mode = True

        engine = builder.build_engine(network, config)
        if engine_file_path:
            with open(engine_file_path, 'wb') as t:
                t.write(engine.serialize())

        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine.serialize())
        # for names in engine:
        #     print(names)

        return engine


def allocate_buffers(engine, bathsize=1):
    """
    allocate cuda buffer
    :param engine:
    :return:
    """
    inputs_data = list()
    inputs_mem = list()
    output_data = list()
    output_mem = list()
    bindings = list()
    stream = cuda.Stream()

    nodeList = ["images", "343", "390", "437", "output"]
    for binding in nodeList:
        bindingShape = engine.get_binding_shape(binding)
        # print(bindingShape)
        size = 1
        for i in range(len(bindingShape)):
            size *= bindingShape[i] * bathsize  # engine.max_batch_size
        # print(size)
        # size = trt.volume(bindingShape) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        # bindings 记录的是输入和输出在GPU 开辟的内存
        bindings.append(int(device_mem))

        # 记录开辟的输入和输出数据的大小和内存
        if engine.binding_is_input(binding):
            inputs_data.append(host_mem)
            inputs_mem.append(device_mem)
        else:
            output_data.append(host_mem)
            output_mem.append(device_mem)
    return inputs_data, inputs_mem, output_data, output_mem, stream


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def None_Max_Suppression(data, conf_thred=0.001, iou_thres=0.6):
    data = data[-1][0]
    selcect = data[:, 4] > conf_thred

    data = data[selcect]

    box = xywh2xyxy(data[:, :4])

    # conf = obj_conf * cls_conf
    data[:, 5:] = data[:, 5:] * data[:, 4].reshape(-1, 1)

    # Detections matrix n x 6 (xyxy, conf, cls)
    if True:
        i, j = (data[:, 5:] > conf_thred).nonzero()
        x = np.concatenate((box[i], data[i, j + 5, None], j[:, None]), 1)
    else:
        # best class only
        conf, j = data[:, 5:].max(1, keepdim=True)
        x = np.concatenate((box, conf, j), 1)[conf.view(-1) > conf_thres]

    n = x.shape[0]  # number of boxes

    c = x[:, 5:6] * (0 if False else 0)  # classes
    boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores

    boxes = torch.from_numpy(boxes).float()
    scores = torch.from_numpy(scores).float()
    i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
    return x[i]


def SingleThread(test_image, outShape, engine, context):
    inputs_data, inputs_mem, output_data, output_mem, stream = allocate_buffers(engine)

    # s1 = time.time()
    # print(test_image)
    image = imageio.imread(test_image)
    image = cv2.resize(image, (640, 640))
    data = image / 255.0
    data = np.ascontiguousarray(data.transpose(2, 0, 1))
    data = np.array(data[None, :, :, :], dtype=np.float32, order='C')
    # data = np.concatenate((data, data), 0)
    inputs_data[0] = data

    # # 进行推理
    Inference(engine=context, input_data=inputs_data, input_device=inputs_mem,
              out_data=output_data, out_device=output_mem,
              stream=stream)

    output_data = [output_data[-1]]
    for idx, pairs in enumerate(zip(output_data, [outShape[-1]])):
        data, size_shape = pairs
        totalsize = 1
        for i in list(size_shape):
            totalsize = totalsize * i
        assert totalsize == len(data), "shape error !!!"
        output_data[idx] = data.reshape(size_shape)

    # s2 = time.time()
    result = None_Max_Suppression(output_data)
    # result =
    # draw bbox on images
    # result = result[0]
    # print(result.shape)
    result = result.reshape(-1, 6)
    if len(result) > 0:

        box = result[:, :4]
        conf = result[:, 4]
        cls = result[:, :5]

        conf_gt = conf > 0.2
        conf = conf[conf_gt]
        cls = cls[conf_gt]
        box = box[conf_gt]
        for idx, item in enumerate(box):
            cls_item = cls[idx]
            cv2.rectangle(image, (int(item[0]), int(item[1])), (int(item[2]), int(item[3])), (255, 0, 0), 2)

        # import matplotlib.pyplot as plt
        # plt.imshow(image)
        # plt.show()
        # print(s2 - s1)


def main(onxpath, engine_file_path=""):
    engine = build_engin2(onxpath, engine_file_path)
    context = engine.create_execution_context()

    outShape = [
        (1, 3, 80, 80, 85),
        (1, 3, 40, 40, 85),
        (1, 3, 20, 20, 85),
        (1, 25200, 85),
    ]
    import time

    fileNames = glob.glob("./images/*.jpg")

    t0 = time.time()
    for test_image in fileNames:
        SingleThread(test_image, outShape, engine, context)
    te1 = time.time()
    print("{0} images total use time {1}, {2} fps".format(len(fileNames), te1 - t0, len(fileNames) / (te1 - t0)))
    # from multiprocessing import Pool
    # p = Pool(1)
    # for test_image in fileNames[ :2]:
    #     p.apply_async(SingleThread, args=(test_image, outShape, engine, context,))
    #
    # p.close()
    # p.join()

    te2 = time.time()
    # print("{0} images total use time {1}, {2} fps".format(len(fileNames), te2 - te1, len(fileNames) / (te2 - te1)))


if __name__ == '__main__':
    onnx_path = "yolov5s.onnx"
    # onnx_path = "./convert_model/multi_class_classification_sim.onnx"
    trt_path = onnx_path.replace(".onnx", "33.trt")
    main(onnx_path, trt_path)
