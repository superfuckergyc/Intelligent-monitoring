import cv2
import logging
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt


class TargetRegionDetector:
    """目标区域识别子模块：实现目标区域的二值化与形态学优化（对应文献1.1节）"""

    def __init__(self,
                 morph_kernel_size: Tuple[int, int] = (5, 5),  # 形态学操作核大小
                 binary_threshold: float = 0.5):  # 二值化阈值系数（相对值）
        """
        初始化目标区域识别参数

        Args:
            morph_kernel_size: 形态学操作（腐蚀/膨胀）的核大小
            binary_threshold: 二值化阈值相对系数（用于调整前景判断灵敏度）
        """
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, morph_kernel_size
        )
        self.binary_threshold = binary_threshold
        self.logger = logging.getLogger(__name__)
        self.logger.info("目标区域识别模块初始化完成")

    def binarize(self, foreground_mask: np.ndarray, background: np.ndarray) -> np.ndarray:
        """
        实现文献公式（5）：对前景掩码进行二值化处理

        Args:
            foreground_mask: 前景掩码（来自背景建模模块的输出）
            background: 当前背景图像（用于动态调整阈值）

        Returns:
            二值化图像F(t)（0为背景，1为目标区域）
        """
        # 文献公式（5）：F(t) = 1 if N_l(x,y) < Y0 else 0
        # 此处基于前景掩码和背景动态阈值实现
        if foreground_mask.shape != background.shape:
            raise ValueError(f"掩码与背景尺寸不匹配: {foreground_mask.shape} vs {background.shape}")

        # 动态计算阈值Y0（基于背景灰度均值的比例）
        background_mean = np.mean(background)
        Y0 = background_mean * self.binary_threshold

        # 二值化：前景掩码中像素值超过阈值的为目标区域
        _, binary = cv2.threshold(
            foreground_mask, Y0, 1, cv2.THRESH_BINARY
        )
        return binary.astype(np.uint8)

    def morphologcial_processing(self, binary_image: np.ndarray) -> np.ndarray:
        """
        形态学操作优化目标区域轮廓（文献1.1节末尾）

        Args:
            binary_image: 二值化图像

        Returns:
            优化后的二值化图像（轮廓更清晰）
        """
        # 先腐蚀去除噪声，再膨胀恢复目标区域
        eroded = cv2.erode(binary_image, self.morph_kernel, iterations=1)
        dilated = cv2.dilate(eroded, self.morph_kernel, iterations=1)
        return dilated

    def detect(self, foreground_mask: np.ndarray, background: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        完整目标区域识别流程：二值化→形态学优化

        Args:
            foreground_mask: 前景掩码（背景建模模块输出）
            background: 当前背景图像

        Returns:
            原始二值化图像、优化后的目标区域图像
        """
        # 1. 二值化处理（公式5）
        binary_raw = self.binarize(foreground_mask, background)

        # 2. 形态学优化
        target_region = self.morphologcial_processing(binary_raw)

        # 统计目标区域占比
        target_ratio = np.sum(target_region) / (target_region.size + 1e-8)
        self.logger.debug(f"目标区域占比: {target_ratio:.2%}")

        return binary_raw, target_region

    def visualize(self,
                  original_frame: np.ndarray,
                  target_region: np.ndarray,
                  save_path: Optional[str] = None) -> None:
        """
        可视化目标区域识别结果

        Args:
            original_frame: 原始帧图像
            target_region: 优化后的目标区域图像
            save_path: 保存路径（None则直接显示）
        """
        # 创建彩色叠加图像
        if len(original_frame.shape) == 2:
            original_rgb = cv2.cvtColor(original_frame, cv2.COLOR_GRAY2RGB)
        else:
            original_rgb = original_frame.copy()

        # 在原始图像上标记目标区域（红色）
        target_rgb = np.zeros_like(original_rgb)
        target_rgb[target_region == 1] = [255, 0, 0]  # 红色标记目标区域
        overlay = cv2.addWeighted(original_rgb, 0.7, target_rgb, 0.3, 0)

        # 显示或保存
        plt.figure(figsize=(10, 6))
        plt.subplot(121)
        plt.imshow(original_frame, cmap='gray')
        plt.title("原始帧")
        plt.axis('off')

        plt.subplot(122)
        plt.imshow(overlay)
        plt.title("目标区域标记")
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            self.logger.info(f"识别结果已保存至: {save_path}")
        else:
            plt.show()

        plt.close()


# 模块测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 1. 导入依赖模块
    from data_loader import VideoImageLoader
    from background_model import BackgroundModel

    # 2. 初始化组件
    video_path = "path/to/your/video.mp4"  # 替换为实际视频路径
    loader = VideoImageLoader(video_path, is_video=True, gray_scale=True)
    bg_model = BackgroundModel()
    detector = TargetRegionDetector()

    try:
        # 3. 读取前两帧初始化背景
        _, frame1 = loader.get_next_frame()
        _, frame2 = loader.get_next_frame()
        bg_model.initialize(frame1, frame2)

        # 4. 处理第3帧并识别目标区域
        frame_idx, frame = loader.get_next_frame()
        if frame is not None:
            # 背景更新
            diff_img, fg_mask = bg_model.update_background(frame)
            background = bg_model.get_background()

            # 目标区域识别
            binary_raw, target_region = detector.detect(fg_mask, background)

            # 可视化
            detector.visualize(frame, target_region, f"target_region_{frame_idx}.png")
            print(f"目标区域识别完成，结果已保存")

    except Exception as e:
        print(f"测试失败: {str(e)}")
    finally:
        loader.release()
        bg_model.reset()