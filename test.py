# import cv2
# import subprocess
# import numpy as np
# # EasyDarwin的RTSP流地址
# rtsp_input_url = "rtsp://127.0.0.1:25544/input"
# # 新的RTSP发布地址 (你需要一个RTSP服务器来接收这个流，例如你也可以再运行一个EasyDarwin实例)
# rtsp_output_url = "rtsp://127.0.0.1:25544/output" # 假设你希望发布到这个地址和端口
# # FFmpeg命令
# ffmpeg_cmd = [
#     'ffmpeg',
#     '-y', # 覆盖已存在的文件（如果需要）
#     '-f', 'rawvideo',
#     '-pix_fmt', 'gray', # 指定输入像素格式为灰度
#     '-s', f'{1280}x{720}', # 指定输入视频尺寸 (需要根据你的实际视频尺寸调整)
#     '-i', '-', # 从标准输入读取数据
#     '-c:v', 'libx264',
#     '-preset', 'ultrafast',
#     '-tune', 'zerolatency',
#     '-crf', '28', # 设置视频质量 (可以调整)
#     '-threads', '4', # 尝试使用多线程
#     '-f', 'rtsp',
# rtsp_output_url
# ]
# # 打开视频流
# cap = cv2.VideoCapture(rtsp_input_url)
# if not cap.isOpened():
#     print(f"无法打开视频流：{rtsp_input_url}")
#     exit()
# # 启动FFmpeg进程
# process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("无法读取帧，退出")
#         break
#     # 将彩色帧转换为灰度帧
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # 将灰度帧转换为字节数据并写入FFmpeg的stdin
#     try:
#         process.stdin.write(gray_frame.tobytes())
#     except BrokenPipeError:
#         print("FFmpeg进程已关闭")
#         break
# # 显示处理后的帧 (可选，仅用于本地查看)
# # cv2.imshow("Processed Frame", gray_frame)
# # if cv2.waitKey(1) & 0xFF == ord('q'):
# # break
# # 清理资源
# cap.release()
# cv2.destroyAllWindows()
# if process.poll() is None:
#     process.stdin.close()
#     process.wait()
























from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
import base64
# ===== 配置API =====
chat_model = ChatOpenAI(
    openai_api_key="sk-oxqdtzxhfhxgielecvkxgadvjugtopldfygbksqpsgdeuxea", # 替换为你的APIKey
    base_url="https://api.siliconflow.cn/v1", # SiliconFlow API 地址
    model="Qwen/Qwen3.5-4B", # 指定模型
    temperature=0.7, # 可选：控制生成多样性
    max_tokens=1024, # 可选：最大生成长度
)
# ===== 图片处理 =====
def image_to_base64(image_path: str) -> str:
    """将图片转换为 base64 编码（兼容 SiliconFlow API 格式）"""
    with open(image_path, "rb") as f:
        return f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode('utf-8')}"
# ===== 构造多模态消息 =====
def create_multimodal_prompt(question: str, image_path: str) -> list:
    """构造包含图片和文本的输入消息"""
    return [HumanMessage(
        content=[
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": image_to_base64(image_path)}}
        ]
    )]
# ===== 主函数 =====
if __name__ == "__main__":
    # 用户输入
    img_path = "4.mp4" # 替换为你的图片路径
    question = "描述这张图片的内容。" # 你的问题
    # 构造输入
    messages = create_multimodal_prompt(question, img_path)
    # 调用API
    try:
        response = chat_model.invoke(messages)
        print("AI 回答：", response.content)
    except Exception as e:
        print(f"API 调用失败: {str(e)}")

