import os
import cv2
import logging
from typing import List, Tuple, Optional, Union


class VideoImageLoader:
    """视频/图像加载子模块：负责读取视频文件或图像序列，输出统一格式的帧数据"""

    def __init__(self, source_path: str, is_video: bool = True,
                 gray_scale: bool = True, fps: int = 25):
        """
        初始化加载器

        Args:
            source_path: 视频文件路径或图像序列所在目录
            is_video: 是否为视频文件（否则为图像序列）
            gray_scale: 是否转换为灰度图
            fps: 视频帧率（仅在读取视频时使用）
        """
        self.source_path = source_path
        self.is_video = is_video
        self.gray_scale = gray_scale
        self.fps = fps
        self.current_frame_idx = 0
        self.total_frames = 0
        self.cap = None  # 视频捕获对象
        self.image_files = []  # 图像文件列表

        # 初始化日志
        self.logger = logging.getLogger(__name__)

        # 验证输入路径
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"源路径不存在: {source_path}")

        # 根据类型初始化
        if is_video:
            self._init_video_loader()
        else:
            self._init_image_sequence_loader()

    def _init_video_loader(self):
        """初始化视频加载器"""
        self.cap = cv2.VideoCapture(self.source_path)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频文件: {self.source_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) if self.fps <= 0 else self.fps
        self.logger.info(f"视频加载成功: {self.source_path}, 总帧数: {self.total_frames}, FPS: {self.fps}")

    def _init_image_sequence_loader(self):
        """初始化图像序列加载器"""
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        self.image_files = sorted(
            [os.path.join(self.source_path, f) for f in os.listdir(self.source_path)
             if os.path.isfile(os.path.join(self.source_path, f))
             and os.path.splitext(f)[1].lower() in valid_extensions]
        )

        if not self.image_files:
            raise ValueError(f"未找到有效图像文件在目录: {self.source_path}")

        self.total_frames = len(self.image_files)
        self.logger.info(f"图像序列加载成功: {self.source_path}, 图像数量: {self.total_frames}")

    def get_next_frame(self) -> Optional[Tuple[int, Union[cv2.Mat, None]]]:
        """
        获取下一帧图像

        Returns:
            元组 (帧索引, 帧图像)，如果没有更多帧则返回 (None, None)
        """
        if self.current_frame_idx >= self.total_frames:
            return None, None

        frame = None
        if self.is_video:
            ret, frame = self.cap.read()
            if not ret:
                self.logger.warning(f"读取视频帧失败，索引: {self.current_frame_idx}")
                return None, None
        else:
            try:
                frame = cv2.imread(self.image_files[self.current_frame_idx])
            except Exception as e:
                self.logger.error(f"读取图像失败: {self.image_files[self.current_frame_idx]}, 错误: {str(e)}")
                return None, None

        # 转换为灰度图
        if self.gray_scale and frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame_idx = self.current_frame_idx
        self.current_frame_idx += 1
        return frame_idx, frame

    def reset(self):
        """重置加载器到初始状态"""
        if self.is_video and self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_frame_idx = 0
        self.logger.info("加载器已重置")

    def release(self):
        """释放资源"""
        if self.is_video and self.cap:
            self.cap.release()
        self.logger.info("加载器资源已释放")

    def get_total_frames(self) -> int:
        """获取总帧数"""
        return self.total_frames

    def get_fps(self) -> float:
        """获取帧率"""
        return self.fps


# 模块测试代码
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 测试视频加载
    video_path = "path/to/your/video.mp4"  # 替换为实际视频路径
    try:
        video_loader = VideoImageLoader(video_path, is_video=True, gray_scale=True)

        print(f"总帧数: {video_loader.get_total_frames()}")
        print(f"帧率: {video_loader.get_fps()}")

        # 读取前10帧
        for _ in range(10):
            frame_idx, frame = video_loader.get_next_frame()
            if frame is None:
                break
            print(f"读取帧索引: {frame_idx}, 帧形状: {frame.shape}")

            # 显示帧 (取消注释以可视化)
            # cv2.imshow('Frame', frame)
            # cv2.waitKey(100)  # 等待100ms

        video_loader.release()
    except Exception as e:
        print(f"视频加载测试失败: {str(e)}")

    # 测试图像序列加载
    image_dir = "path/to/your/images/"  # 替换为实际图像目录
    try:
        image_loader = VideoImageLoader(image_dir, is_video=False, gray_scale=True)

        print(f"总图像数: {image_loader.get_total_frames()}")

        # 读取前5张图像
        for _ in range(5):
            frame_idx, frame = image_loader.get_next_frame()
            if frame is None:
                break
            print(f"读取图像索引: {frame_idx}, 图像形状: {frame.shape}")

        image_loader.release()
    except Exception as e:
        print(f"图像序列加载测试失败: {str(e)}")