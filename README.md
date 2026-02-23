# 1. Phase1目标: 单摄像头 + 单线程

Pipeline:
RTSP -> Decode -> Preprocess -> ONNX -> Postprocess -> Visualize

问题:
高清RTSP: /onvif/profile1/ ---200ms/frame
低清RTSP: /onvif/profile2/ ---20ms/frame
推理阻塞: CPU                    ---300ms/frame
延迟叠加: 2-3 FPS

# 2. 主干框架

## 主干代码：
```cpp
while(running){
	read_frame();
	preprocessing();
	infer();
	postprocessing();
	visualize();
	}```
