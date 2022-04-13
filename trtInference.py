# -*- coding: utf-8 -*-
# @Author : youngx
# @Time : 14:57  2022-04-11
import os
import pycuda
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

print("tensorrt version :", trt.__version__)


class TRT():
    def __init__(self, onxPath, enginePath):
        self.onxPath = onxPath
        self.enginePath = enginePath

    def Inference(self, engine, input_data, input_device, out_data, out_device, stream, repeat_time=1):
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

    def build_engin(self, half=True):
        """
        model load and convert
        :param onxpath: onnx model path
        :param engine_file_path: if save trt model converted from onnx
        :return:
        """
        logger = trt.Logger(trt.Logger.INFO)
        if os.path.exists(self.enginePath):
            print("load trt parse")
            runtime = trt.Runtime(logger)
            with open(self.enginePath, "rb") as f:
                engin = runtime.deserialize_cuda_engine(f.read())
                return engin
        else:
            print("load onnx parse")
            builder = trt.Builder(logger)
            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 32
            builder.max_batch_size = 1

            flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            network = builder.create_network(flag)
            parser = trt.OnnxParser(network, logger)

            # # #M1
            # if not parser.parse_from_file(str(self.onxPath)):
            #     raise RuntimeError(f'failed to load ONNX file: {self.onxPath}')

            #M2
            with open(self.onxPath, 'rb') as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None

            inputs = [[i, network.get_input(i)] for i in range(network.num_inputs)]
            outputs = [network.get_output(i) for i in range(network.num_outputs)]
            for s in inputs:
                print(s[0], s[1].shape)

            for s in outputs:
                print(s.shape)

            # half &= builder.platform_has_fast_fp16
            # if half:
            #     config.set_flag(trt.BuilderFlag.FP16)
                # builder.fp16_mode = True

            # engine = builder.build_engine(network, config)  # # engine.serialize()
            engine = builder.build_serialized_network(network, config)  # # engine
            if engine is not None:
                if self.enginePath:
                    with open(self.enginePath, 'wb') as t:
                        t.write(engine)

                runtime = trt.Runtime(logger)
                engine = runtime.deserialize_cuda_engine(engine)

                return engine
            else:
                print("engine none error")
                exit(0)

    def allocate_buffers(self, engine, bathsize=1):
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
            size = 1
            for i in range(len(bindingShape)):
                size *= bindingShape[i] * bathsize  # engine.max_batch_size
            # size = trt.volume(bindingShape) * engine.max_batch_size  # # trt.volume meat error in my env
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
