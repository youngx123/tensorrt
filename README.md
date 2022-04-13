使用 tensorrt 对模型进行推理测试

主要以 yolov5和BiSeNet两个网络结构为主进行测试。

显卡型号为： GTX2070,  CUDA版本为：10.2，  cudnn版本为8.2，  tensorrt版本为：8.2.0.6


| model name                 | fps      |
|----------------------------|----------|
|  yolov5s                   | 38       |



1. 遇到 Python版本onnx 模型无法转换保存为序列文件，但是使用`trtexec.exe`可以正确转换的。并且Python加载其转换模型可能会出错，c++可以加载进行推理。
2. `trtexec.exe`在对onnx模型转化过程中，会标出无法转换的层，若转化成功会对转化的模型进行测试已检查模型是否正确。

利用`trtexec.exe`转换的 `BiSeNet` 模型, python加载序列文件错误 

[BiSeNet.onnx](https://drive.google.com/file/d/1cR8zQXnJqHE0Hl-4EVcPB1AlZYITTxme/view?usp=sharing)

[BiSeNet.trt](https://drive.google.com/file/d/1WDEzlSXtGDrIyBCnuwwJUeQqPV6f_Id7/view?usp=sharing)


利用`Python`转换的 `yolov5s` 模型 

[yolov5s.onnx](https://drive.google.com/file/d/1rLahPd8NIvIhWRaZGLm4gcASSxEM4gjb/view?usp=sharing)

[yolov5s.pt](https://drive.google.com/file/d/1BrVUXOQuvScYrLypP9i7m0UUng3qa7I4/view?usp=sharing)

模型项目:
>https://github.com/mrgloom/awesome-semantic-segmentation

>https://github.com/ultralytics/yolov5

>https://github.com/dbolya/yolact
