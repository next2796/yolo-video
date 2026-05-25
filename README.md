# 视觉大模型实时视频处理

基于 YOLOv8 人体姿态检测和视觉大模型实时处理视频流的项目。

## 功能特性

- **YOLOv8 人体姿态检测**：实时检测 17 个人体关键点
- **骨骼可视化**：绿色线条连接关键点，黄色圆点标记位置
- **视觉大模型分析**：使用 Qwen3.5-4B 视觉模型分析视频内容
- **RTSP 推流**：通过 FFmpeg 推流，支持 VLC 实时观看

## 文件说明

| 文件 | 说明 |
|------|------|
| `vision_pose_ai.py` | 完整功能版本（姿态检测 + 视觉大模型 + 推流） |
| `vision_video.py` | 仅姿态检测版本（已注释 AI 分析功能） |
| `test.py` | 原始测试代码 |

## 环境要求

- Python 3.8+
- OpenCV
- Ultralytics (YOLOv8)
- LangChain & LangChain-Community
- FFmpeg

## 安装依赖

```bash
pip install opencv-python ultralytics langchain langchain-community
```

## 使用方法

### 1. 启动 RTSP 服务器

确保有可用的 RTSP 服务器（如 EasyDarwin）：
```
rtsp://127.0.0.1:25544/output
```

### 2. 运行程序

```bash
python vision_pose_ai.py
```

### 3. 使用 VLC 观看

打开 VLC → 媒体 → 打开网络串流 → 输入地址：
```
rtsp://127.0.0.1:25544/output
```

## 配置说明

在 `vision_pose_ai.py` 中可以修改以下参数：

```python
# 视频文件
video_path = "2.mp4"

# RTSP 推流地址
rtsp_url = "rtsp://127.0.0.1:25544/output"

# 分析间隔（帧数）
frame_interval = 30

# API 配置
chat_model = ChatOpenAI(
    openai_api_key="your-api-key",
    base_url="https://api.siliconflow.cn/v1",
    model="Qwen/Qwen3.5-4B",
    temperature=0.7,
    max_tokens=512,
)
```

## 输出效果

- **视频流**：实时显示带骨骼标注的视频
- **终端输出**：
  - 帧号
  - 姿态检测信息（检测到的人数和关键点数量）
  - AI 分析结果
- **VLC 观看**：支持局域网内任何设备观看

## 按键操作

- 按 `q` 键退出程序

## 注意事项

1. 确保 FFmpeg 已安装并添加到系统 PATH
2. 确保 RTSP 服务器正常运行
3. 确保 API Key 有效
4. 根据网络状况调整 `frame_interval` 参数