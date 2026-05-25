import cv2
import subprocess
import base64
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
from ultralytics import YOLO

chat_model = ChatOpenAI(
    openai_api_key="sk-oxqdtzxhfhxgielecvkxgadvjugtopldfygbksqpsgdeuxea",
    base_url="https://api.siliconflow.cn/v1",
    model="Qwen/Qwen3.5-4B",
    temperature=0.7,
    max_tokens=512,
)

yolo_model = YOLO('yolov8n-pose.pt')

def frame_to_base64(frame, quality=60):
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

def analyze_frame(frame, question):
    try:
        frame_base64 = frame_to_base64(frame)
        messages = [HumanMessage(content=[
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": frame_base64}}
        ])]
        response = chat_model.invoke(messages)
        return response.content
    except Exception as e:
        return f"分析失败: {str(e)}"

def detect_pose(frame):
    results = yolo_model(frame)
    pose_info = []
    
    POSE_CONNECTIONS = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
    ]
    
    for result in results:
        if result.keypoints is not None:
            keypoints = result.keypoints.xy.cpu().numpy()
            confidences = result.keypoints.conf.cpu().numpy()
            
            for person_idx, (keypoint_set, conf_set) in enumerate(zip(keypoints, confidences)):
                person_info = f"Person {person_idx + 1}"
                visible_keypoints = 0
                
                for kp_idx, (x, y) in enumerate(keypoint_set):
                    if conf_set[kp_idx] > 0.5:
                        visible_keypoints += 1
                        cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 255), -1)
                
                for connection in POSE_CONNECTIONS:
                    idx1, idx2 = connection
                    if conf_set[idx1] > 0.5 and conf_set[idx2] > 0.5:
                        x1, y1 = keypoint_set[idx1]
                        x2, y2 = keypoint_set[idx2]
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                pose_info.append(f"{person_info} ({visible_keypoints} keypoints)")
    
    return frame, pose_info

def process_video_and_stream(video_path, rtsp_url, frame_interval=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"视频: {width}x{height} @ {fps} FPS")
    print(f"推流地址: {rtsp_url}")

    ffmpeg_cmd = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{width}x{height}',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-tune', 'zerolatency',
        '-crf', '23',
        '-f', 'rtsp',
        '-rtsp_transport', 'tcp',
        rtsp_url
    ]

    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    frame_count = 0
    question = "描述图片中的场景和人物动作"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("视频读取完毕")
                break

            frame_count += 1

            frame, poses = detect_pose(frame)
            
            if frame_count % frame_interval == 0:
                print(f"\n--- 分析第 {frame_count} 帧 ---")
                print(f"姿态检测: {', '.join(poses)}")
                
                result = analyze_frame(frame, question)
                print(f"AI分析: {result[:100]}...")

                cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, result[:50], (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            process.stdin.write(frame.tobytes())

    except BrokenPipeError:
        print("FFmpeg进程中断")
    except Exception as e:
        print(f"错误: {str(e)}")
    finally:
        cap.release()
        if process.poll() is None:
            process.stdin.close()
            process.wait()
        print("处理完成")

if __name__ == "__main__":
    video_path = "4.mp4"
    rtsp_url = "rtsp://127.0.0.1:25544/output"
    
    process_video_and_stream(video_path, rtsp_url, frame_interval=30)