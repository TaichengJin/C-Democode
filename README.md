# 1. Phase1目标: 单摄像头 + 单线程

Pipeline:
RTSP -> Decode -> Preprocess -> ONNX -> Postprocess -> Visualize

问题:\
## Performance Analysis (CPU Version)

| Component        | Latency        |
|------------------|---------------|
| RTSP (HD)       | 200 ms/frame |
| RTSP (Substream)| 20 ms/frame  |
| Inference (CPU) | 300 ms/frame |
| End-to-End FPS  | 2–3 FPS      |

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
