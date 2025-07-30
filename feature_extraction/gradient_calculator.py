import cv2
import numpy as np
import logging
from typing import Tuple, List, Dict, Optional


class GradientCalculator:
    """梯度计算子模块：实现图像梯度计算与特征提取（对应文献2.2节）"""

    def __init__(self,
                 sobel_kernel_size: int = 3,  # Sobel算子核大小
                 magnitude_threshold: float = 10.0,  # 梯度幅值阈值
                 orientation_bins: int = 9,  # HOG方向直方图bin数
                 cell_size: Tuple[int, int] = (8, 8),  # HOG细胞大小
                 block_size: Tuple[int, int] = (16, 16),  # HOG块大小
                 block_stride: Tuple[int, int] = (8, 8)):  # HOG块步长
        """
        初始化梯度计算参数

        Args:
            sobel_kernel_size: Sobel算子核大小，常用3、5、7
            magnitude_threshold: 梯度幅值阈值，低于此值的梯度将被忽略
            orientation_bins: HOG特征的方向直方图bin数
            cell_size: HOG特征的细胞大小（像素）
            block_size: HOG特征的块大小（细胞）
            block_stride: HOG特征的块移动步长（细胞）
        """
        self.sobel_kernel_size = sobel_kernel_size
        self.magnitude_threshold = magnitude_threshold
        self.orientation_bins = orientation_bins
        self.cell_size = cell_size
        self.block_size = block_size
        self.block_stride = block_stride

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"梯度计算模块初始化完成 - HOG参数: "
                         f"bins={orientation_bins}, cell={cell_size}, "
                         f"block={block_size}, stride={block_stride}")

    def compute_gradient(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算图像梯度幅值和方向（对应文献公式6-7）

        Args:
            image: 输入图像（单通道灰度图）

        Returns:
            gradient_magnitude: 梯度幅值图像
            gradient_orientation: 梯度方向图像（弧度）
        """
        # 计算x和y方向的梯度
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel_size)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel_size)

        # 计算梯度幅值和方向（文献公式6-7）
        magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        orientation = np.arctan2(sobely, sobelx)  # 弧度制

        # 应用阈值过滤微弱梯度
        magnitude[magnitude < self.magnitude_threshold] = 0

        return magnitude, orientation

    def compute_hog_features(self, image: np.ndarray) -> np.ndarray:
        """
        计算图像的HOG（方向梯度直方图）特征（对应文献2.2节）

        Args:
            image: 输入图像（单通道灰度图）

        Returns:
            hog_features: HOG特征向量
        """
        # 计算梯度
        magnitude, orientation = self.compute_gradient(image)

        # 将梯度方向从弧度转换为0-180度（无符号方向）
        orientation_deg = np.degrees(orientation) % 180

        # 创建HOG描述符
        hog = cv2.HOGDescriptor(
            _winSize=(image.shape[1] // self.cell_size[1] * self.cell_size[1],
                      image.shape[0] // self.cell_size[0] * self.cell_size[0]),
            _blockSize=(self.block_size[1] * self.cell_size[1],
                        self.block_size[0] * self.cell_size[0]),
            _blockStride=(self.block_stride[1] * self.cell_size[1],
                          self.block_stride[0] * self.cell_size[0]),
            _cellSize=(self.cell_size[1] * self.cell_size[0],
                       self.cell_size[0] * self.cell_size[0]),
            _nbins=self.orientation_bins
        )

        # 计算HOG特征
        hog_features = hog.compute(image)

        return hog_features

    def compute_block_gradient(self, image_block: np.ndarray) -> Dict:
        """
        计算单个运动块的梯度特征（对应文献2.2节算法）

        Args:
            image_block: 输入的运动块图像（单通道灰度图）

        Returns:
            包含梯度特征的字典:
            {
                'magnitude': 梯度幅值图像,
                'orientation': 梯度方向图像,
                'hog_features': HOG特征向量,
                'dominant_orientation': 主导方向（度）
            }
        """
        # 计算梯度
        magnitude, orientation = self.compute_gradient(image_block)

        # 计算HOG特征
        hog_features = self.compute_hog_features(image_block)

        # 计算主导方向（对应文献中的主方向计算）
        orientation_hist = np.histogram(
            orientation.flatten(),
            bins=self.orientation_bins,
            range=(0, np.pi)
        )[0]
        dominant_orientation = np.degrees(np.argmax(orientation_hist) *
                                          (np.pi / self.orientation_bins))

        return {
            'magnitude': magnitude,
            'orientation': orientation,
            'hog_features': hog_features,
            'dominant_orientation': dominant_orientation
        }

    def visualize_gradient(self, image: np.ndarray,
                           magnitude: np.ndarray,
                           orientation: np.ndarray,
                           save_path: Optional[str] = None) -> None:
        """
        可视化梯度计算结果

        Args:
            image: 原始图像
            magnitude: 梯度幅值
            orientation: 梯度方向
            save_path: 保存路径（None则直接显示）
        """
        # 创建彩色图像用于可视化
        if len(image.shape) == 2:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = image.copy()

        # 可视化梯度方向（每隔一定像素绘制箭头）
        step = 10  # 箭头绘制间隔
        for y in range(0, magnitude.shape[0], step):
            for x in range(0, magnitude.shape[1], step):
                if magnitude[y, x] > self.magnitude_threshold:
                    # 计算箭头终点
                    angle = orientation[y, x]
                    length = magnitude[y, x] * 0.5  # 缩放箭头长度
                    end_x = int(x + length * np.cos(angle))
                    end_y = int(y + length * np.sin(angle))

                    # 确保终点在图像范围内
                    end_x = min(end_x, vis_image.shape[1] - 1)
                    end_y = min(end_y, vis_image.shape[0] - 1)

                    # 绘制箭头（颜色表示方向）
                    color = self._get_color_from_angle(angle)
                    cv2.arrowedLine(vis_image, (x, y), (end_x, end_y),
                                    color, 1, tipLength=0.3)

        # 显示或保存结果
        if save_path:
            cv2.imwrite(save_path, vis_image)
            self.logger.info(f"梯度可视化结果已保存至: {save_path}")
        else:
            cv2.imshow("Gradient Visualization", vis_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def _get_color_from_angle(self, angle: float) -> Tuple[int, int, int]:
        """根据梯度方向生成颜色（HSV转BGR）"""
        # 将角度（-π到π）映射到HSV的色调（0-180）
        hue = int(((angle + np.pi) % (2 * np.pi)) * (180 / (2 * np.pi)))
        hsv = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return tuple(int(c) for c in bgr[0][0])


# 模块测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 1. 导入依赖模块
    from preprocessing.data_loader import VideoImageLoader
    from preprocessing.background_model import BackgroundModel
    from preprocessing.target_region_detector import TargetRegionDetector
    from feature_extraction.motion_block_detector import MotionBlockDetector

    # 2. 初始化组件
    video_path = "path/to/your/video.mp4"  # 替换为实际视频路径
    loader = VideoImageLoader(video_path, is_video=True, gray_scale=True)
    bg_model = BackgroundModel()
    target_detector = TargetRegionDetector()
    motion_detector = MotionBlockDetector()
    gradient_calculator = GradientCalculator()

    try:
        # 3. 读取前两帧初始化背景
        _, frame1 = loader.get_next_frame()
        _, frame2 = loader.get_next_frame()
        bg_model.initialize(frame1, frame2)

        # 4. 读取第3帧和第4帧用于梯度计算
        _, frame3 = loader.get_next_frame()
        _, frame4 = loader.get_next_frame()

        if frame3 is not None and frame4 is not None:
            # 更新背景并获取目标区域
            diff_img, fg_mask = bg_model.update_background(frame3)
            background = bg_model.get_background()
            _, target_region = target_detector.detect(fg_mask, background)

            # 检测运动块
            motion_blocks = motion_detector.detect_motion_blocks(
                target_region, frame3, frame4
            )

            if motion_blocks:
                # 选择第一个运动块进行梯度计算
                sample_block = motion_blocks[0]['block']
                block_position = motion_blocks[0]['position']

                # 计算梯度特征
                gradient_features = gradient_calculator.compute_block_gradient(sample_block)

                # 可视化梯度
                gradient_calculator.visualize_gradient(
                    sample_block,
                    gradient_features['magnitude'],
                    gradient_features['orientation'],
                    "gradient_visualization.png"
                )

                print(f"梯度计算完成，主导方向: {gradient_features['dominant_orientation']:.1f}度，"
                      f"HOG特征维度: {gradient_features['hog_features'].shape}，结果已保存")
            else:
                print("未检测到有效运动块，无法进行梯度计算")

    except Exception as e:
        print(f"测试失败: {str(e)}")
    finally:
        loader.release()
        bg_model.reset()