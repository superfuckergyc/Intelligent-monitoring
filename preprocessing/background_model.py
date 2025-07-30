import os
import cv2
import logging
import numpy as np
from typing import Tuple, Optional, List


class BackgroundModel:
    """初始背景建模子模块：实现背景模型的初始化与动态更新（对应文献公式1-4）"""

    def __init__(self,
                 init_weight_s: float = 0.5,  # 初始背景加权因子s（s + n = 1）
                 update_rate_alpha: float = 0.8,  # 背景更新速率α（1≥α≥0）
                 threshold_theta: float = 0.3,  # 差分阈值系数θ
                 gray_scale: bool = True):  # 是否为灰度图（与VideoImageLoader保持一致）
        """
        初始化背景模型参数

        Args:
            init_weight_s: 公式（1）中前两帧的加权因子s（n = 1 - s）
            update_rate_alpha: 公式（4）中的背景更新速率α（值越大更新越慢）
            threshold_theta: 公式（2）中的阈值系数θ
            gray_scale: 图像是否为灰度图（影响差分计算维度）
        """
        self.init_weight_s = init_weight_s
        self.init_weight_n = 1.0 - init_weight_s  # 公式（1）中的n
        self.update_rate_alpha = update_rate_alpha
        self.threshold_theta = threshold_theta
        self.gray_scale = gray_scale

        # 背景模型变量
        self.initial_background = None  # 初始背景N0(x,y)（公式1）
        self.current_background = None  # 当前帧背景Nl(x,y)
        self.frame_shape = None  # 图像尺寸（height, width, channels）
        self.is_initialized = False  # 背景是否已初始化

        # 日志配置
        self.logger = logging.getLogger(__name__)
        self.logger.info("背景建模模块初始化完成")

    def initialize(self, first_frame: np.ndarray, second_frame: np.ndarray) -> None:
        """
        根据前两帧初始化背景模型（文献公式1）

        Args:
            first_frame: 第1帧图像（I_{l-2}）
            second_frame: 第2帧图像（I_{l-1}）
        """
        # 验证输入帧尺寸一致性
        if first_frame.shape != second_frame.shape:
            raise ValueError(f"前两帧尺寸不一致: {first_frame.shape} vs {second_frame.shape}")
        self.frame_shape = first_frame.shape
        self.logger.debug(f"图像尺寸: {self.frame_shape}，灰度图: {self.gray_scale}")

        # 公式（1）：N0(x,y) = s*I_{l-2} + n*I_{l-1}
        self.initial_background = (self.init_weight_s * first_frame +
                                   self.init_weight_n * second_frame).astype(np.uint8)
        self.current_background = self.initial_background.copy()  # 初始当前背景等于初始背景
        self.is_initialized = True
        self.logger.info("初始背景模型构建完成")

    def _compute_difference(self, frame: np.ndarray) -> np.ndarray:
        """
        计算当前帧与背景的差分图像（文献公式2）

        Args:
            frame: 当前帧图像I_l(x,y)

        Returns:
            差分图像F_l(x,y)
        """
        if not self.is_initialized:
            raise RuntimeError("背景模型未初始化，请先调用initialize方法")

        # 验证输入帧尺寸
        if frame.shape != self.frame_shape:
            raise ValueError(f"输入帧尺寸与背景不匹配: {frame.shape} vs {self.frame_shape}")

        # 公式（2）：F_l(x,y) = |I_l - [N_l - N0]| * θ * g
        # 注：文献中g为灰度值，此处简化为对差分结果按阈值系数缩放
        background_diff = self.current_background - self.initial_background
        frame_diff = np.abs(frame - background_diff).astype(np.float32)
        diff_image = frame_diff * self.threshold_theta

        # 归一化到0-255（便于后续处理）
        diff_image = np.clip(diff_image, 0, 255).astype(np.uint8)
        return diff_image

    def _update_background_block(self,
                                 frame_block: np.ndarray,
                                 background_block: np.ndarray,
                                 is_foreground: bool) -> np.ndarray:
        """
        分块更新背景（文献公式4）

        Args:
            frame_block: 当前帧的子块I_ij^l(x,y)
            background_block: 背景子块N_ij^l(x,y)
            is_foreground: 该子块是否为前景（W >= Y时为前景）

        Returns:
            更新后的背景子块
        """
        if is_foreground:
            # 前景子块：背景保持不变（公式4第三行）
            return background_block
        else:
            # 背景子块：加权更新（公式4第二行）
            updated_block = ((1 - self.update_rate_alpha) * frame_block +
                             self.update_rate_alpha * background_block).astype(np.uint8)
            return updated_block

    def update_background(self,
                          frame: np.ndarray,
                          block_size: Tuple[int, int] = (32, 32)) -> Tuple[np.ndarray, np.ndarray]:
        """
        动态更新背景模型（文献公式4），并返回差分图像与二值化前景

        Args:
            frame: 当前帧图像I_l(x,y)
            block_size: 分块尺寸（Z, E），用于子块判断

        Returns:
            差分图像F_l(x,y)、二值化前景图（1为前景，0为背景）
        """
        if not self.is_initialized:
            raise RuntimeError("背景模型未初始化，请先调用initialize方法")

        # 1. 计算差分图像
        diff_image = self._compute_difference(frame)

        # 2. 分块处理（文献1.1节：将差分图像分为m×q个子块）
        height, width = diff_image.shape[:2]
        block_h, block_w = block_size
        foreground_mask = np.zeros_like(diff_image, dtype=np.uint8)  # 前景掩码（1为前景）

        # 遍历所有子块
        for y in range(0, height, block_h):
            for x in range(0, width, block_w):
                # 子块边界（处理边缘情况）
                y_end = min(y + block_h, height)
                x_end = min(x + block_w, width)

                # 提取当前子块
                frame_block = frame[y:y_end, x:x_end]
                bg_block = self.current_background[y:y_end, x:x_end]
                diff_block = diff_image[y:y_end, x:x_end]

                # 计算子块像素和W（文献：子块内所有像素值之和）
                W = np.sum(diff_block)

                # 计算阈值Y（文献公式：Y = F_ij^l(x,y) * (Z×E×θ×g)，简化为子块像素数×平均阈值）
                block_pixels = (y_end - y) * (x_end - x)  # Z×E
                Y = block_pixels * self.threshold_theta * np.mean(diff_block) if np.mean(diff_block) > 0 else 0

                # 判断是否为前景子块
                is_foreground = W >= Y
                if is_foreground:
                    foreground_mask[y:y_end, x:x_end] = 1  # 标记为前景

                # 分块更新背景
                updated_bg_block = self._update_background_block(frame_block, bg_block, is_foreground)
                self.current_background[y:y_end, x:x_end] = updated_bg_block

        return diff_image, foreground_mask

    def get_background(self) -> np.ndarray:
        """获取当前背景图像"""
        if not self.is_initialized:
            raise RuntimeError("背景模型未初始化，请先调用initialize方法")
        return self.current_background.copy()

    def reset(self) -> None:
        """重置背景模型"""
        self.initial_background = None
        self.current_background = None
        self.frame_shape = None
        self.is_initialized = False
        self.logger.info("背景模型已重置")


# 模块测试代码
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 1. 用VideoImageLoader加载前两帧（需替换为实际视频路径）
    from data_loader import VideoImageLoader  # 导入1.1模块

    video_path = "path/to/your/video.mp4"  # 替换为实际视频路径
    try:
        # 初始化视频加载器（灰度图模式）
        loader = VideoImageLoader(video_path, is_video=True, gray_scale=True)

        # 读取前两帧用于初始化背景
        _, frame1 = loader.get_next_frame()
        _, frame2 = loader.get_next_frame()
        if frame1 is None or frame2 is None:
            raise ValueError("无法读取前两帧图像")

        # 2. 初始化背景模型
        bg_model = BackgroundModel(
            init_weight_s=0.5,
            update_rate_alpha=0.8,
            threshold_theta=0.3,
            gray_scale=True
        )
        bg_model.initialize(frame1, frame2)

        # 3. 读取后续帧并更新背景
        for i in range(5):  # 测试5帧更新
            frame_idx, frame = loader.get_next_frame()
            if frame is None:
                break

            # 更新背景并获取差分图像和前景掩码
            diff_img, fg_mask = bg_model.update_background(frame, block_size=(32, 32))

            # 保存结果用于可视化（可选）
            cv2.imwrite(f"test_diff_{frame_idx}.png", diff_img)
            cv2.imwrite(f"test_foreground_{frame_idx}.png", fg_mask * 255)  # 前景掩码二值化显示
            cv2.imwrite(f"test_background_{frame_idx}.png", bg_model.get_background())

            print(f"已处理帧 {frame_idx}，背景更新完成")

        # 释放资源
        loader.release()
        bg_model.reset()
        print("测试完成")

    except Exception as e:
        print(f"测试失败: {str(e)}")