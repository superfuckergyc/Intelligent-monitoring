import cv2
import os
import numpy as np


def extract_frames_from_video(video_path, output_dir, fps=20):
    """
    从单个视频中提取每秒指定帧数的图像

    参数:
    video_path: 视频文件路径
    output_dir: 输出目录
    fps: 每秒提取的帧数
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        return 0

    # 获取视频信息
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(video_fps / fps))  # 确保至少每帧提取一次

    video_name = os.path.basename(video_path)
    print(f"\n处理视频: {video_name}")
    print(f"视频帧率: {video_fps:.1f}, 提取帧率: {fps}")

    # 提取帧
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 每frame_interval帧保存一次
        if frame_count % frame_interval == 0:
            # 保存为图像
            img_path = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(img_path, frame)
            saved_count += 1

        frame_count += 1

        # 打印进度
        if frame_count % 100 == 0:
            print(f"进度: {frame_count}/{total_frames} 帧", end='\r')

    cap.release()
    print(f"完成! 提取了 {saved_count} 帧")
    return saved_count


def process_video_folder(input_folder, output_base_dir, fps=20):
    """
    处理文件夹中的所有视频文件

    参数:
    input_folder: 包含视频文件的输入文件夹
    output_base_dir: 输出基础目录
    fps: 每秒提取的帧数
    """
    # 支持的视频格式
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']

    # 确保输出目录存在
    os.makedirs(output_base_dir, exist_ok=True)

    # 获取所有视频文件
    video_files = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(root, file))

    if not video_files:
        print(f"错误: 在 {input_folder} 中未找到视频文件")
        return

    print(f"在 {input_folder} 中找到 {len(video_files)} 个视频文件")

    total_frames = 0

    # 处理每个视频文件
    for video_path in video_files:
        # 创建针对该视频的输出子目录
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(output_base_dir, video_name)

        # 提取帧
        frames_count = extract_frames_from_video(video_path, output_dir, fps)
        total_frames += frames_count

    print(f"\n所有视频处理完成! 共提取 {total_frames} 帧图像")
    print(f"输出目录: {output_base_dir}")


if __name__ == "__main__":
    # 设置路径
    input_folder = r"C:\workspace\Intelligent-monitoring\utils"  # 视频文件夹路径
    output_base_dir = r"C:\workspace\Intelligent-monitoring\results\frames"  # 基础输出目录

    # 处理整个文件夹
    process_video_folder(input_folder, output_base_dir, fps=20)