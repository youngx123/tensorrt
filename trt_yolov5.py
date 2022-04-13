# -*- coding: utf-8 -*-
# @Author : youngx
# @Time : 15:05  2022-04-11
import glob
import imageio
import cv2
import os
import numpy as np
import torch
import torchvision
from trtInference import TRT


def xywh2xyxy(x):
    """
    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    """
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


def SingleThread(test_image, outShape, TRT, engine, context):
    inputs_data, inputs_mem, output_data, output_mem, stream = TRT.allocate_buffers(engine)

    image = imageio.imread(test_image)
    image = cv2.resize(image, (640, 640))
    data = image / 255.0
    data = np.ascontiguousarray(data.transpose(2, 0, 1))
    data = np.array(data[None, :, :, :], dtype=np.float32, order='C')
    inputs_data[0] = data

    # # 进行推理
    TRT.Inference(engine=context, input_data=inputs_data, input_device=inputs_mem,
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
    # draw bbox on images
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
    trtModel = TRT(onxpath, engine_file_path)
    engine = trtModel.build_engin()
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
        SingleThread(test_image, outShape, trtModel, engine, context)
    te1 = time.time()
    print("{0} images total use time {1}, {2} fps".format(len(fileNames), te1 - t0, len(fileNames) / (te1 - t0)))
    # from multiprocessing import Pool
    # p = Pool(1)
    # for test_image in fileNames[ :2]:
    #     p.apply_async(SingleThread, args=(test_image, outShape, engine, context,))
    # p.close()
    # p.join()


if __name__ == '__main__':
    onnx_path = "yolov5s.onnx"
    # onnx_path = "./convert_model/multi_class_classification_sim.onnx"
    trt_path = onnx_path.replace(".onnx", "33.trt")
    main(onnx_path, trt_path)
