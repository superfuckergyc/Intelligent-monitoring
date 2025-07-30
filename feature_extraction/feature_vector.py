import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler


class FeatureVectorGenerator:
    """特征向量生成子模块：整合多维度特征生成统一特征向量（对应文献2.3节）"""

    def __init__(self,
                 motion_weight: float = 0.4,  # 运动特征权重
                 gradient_weight: float = 0.6,  # 梯度特征权重
                 normalize: bool = True):  # 是否标准化特征
        """
        初始化特征向量生成器

        Args:
            motion_weight: 运动特征在最终向量中的权重
            gradient_weight: 梯度特征在最终向量中的权重
            normalize: 是否对特征向量进行标准化处理
        """
        self.motion_weight = motion_weight
        self.gradient_weight = gradient_weight
        self.normalize = normalize
        self.scaler = StandardScaler() if normalize else None

        # 验证权重和为1
        if abs(motion_weight + gradient_weight - 1.0) > 1e-6:
            logging.warning(f"特征权重之和不为1: motion={motion_weight}, gradient={gradient_weight}")

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"特征向量生成器初始化完成 - "
                         f"motion_weight={motion_weight}, gradient_weight={gradient_weight}, "
                         f"normalize={normalize}")

    def generate_from_motion_block(self, motion_block: Dict) -> np.ndarray:
        """
        从单个运动块生成特征向量

        Args:
            motion_block: 运动块信息字典，包含:
                'motion_pixels': 运动像素数
                'motion_intensity': 运动强度
                'position': 块位置(x, y)
                'size': 块大小(width, height)

        Returns:
            运动块特征向量
        """
        # 提取运动块特征
        motion_pixels = motion_block['motion_pixels']
        motion_intensity = motion_block['motion_intensity']
        x, y = motion_block['position']
        width, height = motion_block['size']

        # 计算位置和尺寸的相对值（归一化）
        rel_x = x / 1000.0  # 假设图像宽度不超过1000像素
        rel_y = y / 1000.0
        rel_width = width / 1000.0
        rel_height = height / 1000.0

        # 构建特征向量
        feature_vector = np.array([
            motion_pixels,
            motion_intensity,
            rel_x,
            rel_y,
            rel_width,
            rel_height
        ])

        return feature_vector

    def generate_from_gradient(self, gradient_features: Dict) -> np.ndarray:
        """
        从梯度特征生成特征向量

        Args:
            gradient_features: 梯度特征字典，包含:
                'hog_features': HOG特征向量
                'dominant_orientation': 主导方向

        Returns:
            梯度特征向量
        """
        # 提取HOG特征并展平
        hog_features = gradient_features['hog_features'].flatten()

        # 提取主导方向（归一化为0-1范围）
        dominant_orientation = gradient_features['dominant_orientation'] / 180.0

        # 添加主导方向作为额外特征
        gradient_vector = np.append(hog_features, dominant_orientation)

        return gradient_vector

    def combine_features(self,
                         motion_features: np.ndarray,
                         gradient_features: np.ndarray) -> np.ndarray:
        """
        融合运动特征和梯度特征（对应文献2.3节特征融合算法）

        Args:
            motion_features: 运动特征向量
            gradient_features: 梯度特征向量

        Returns:
            融合后的最终特征向量
        """
        # 如果需要标准化，先拟合数据
        if self.normalize and self.scaler is not None:
            combined = np.hstack((motion_features, gradient_features)).reshape(1, -1)
            self.scaler.partial_fit(combined)

            # 应用标准化
            combined = self.scaler.transform(combined).flatten()
        else:
            combined = np.hstack((motion_features, gradient_features))

        # 应用特征权重
        motion_dim = motion_features.shape[0]
        gradient_dim = gradient_features.shape[0]

        weighted_vector = np.zeros_like(combined)
        weighted_vector[:motion_dim] = combined[:motion_dim] * self.motion_weight
        weighted_vector[motion_dim:] = combined[motion_dim:] * self.gradient_weight

        return weighted_vector

    def generate_from_multiple_blocks(self,
                                      motion_blocks: List[Dict],
                                      gradient_calculator) -> np.ndarray:
        """
        从多个运动块及其梯度特征生成最终特征向量

        Args:
            motion_blocks: 运动块列表
            gradient_calculator: 梯度计算器实例，用于计算梯度特征

        Returns:
            整合所有运动块的最终特征向量
        """
        if not motion_blocks:
            raise ValueError("运动块列表为空，无法生成特征向量")

        # 为每个运动块生成特征向量并融合
        all_features = []

        for block in motion_blocks:
            # 生成运动特征
            motion_features = self.generate_from_motion_block(block)

            # 计算梯度特征
            block_image = block['block']
            gradient_features = gradient_calculator.compute_block_gradient(block_image)
            gradient_vector = self.generate_from_gradient(gradient_features)

            # 融合单个块的特征
            combined = self.combine_features(motion_features, gradient_vector)
            all_features.append(combined)

        # 整合所有块的特征（简单平均）
        if len(all_features) > 1:
            final_feature = np.mean(np.vstack(all_features), axis=0)
        else:
            final_feature = all_features[0]

        return final_feature

    def save_feature_vector(self, feature_vector: np.ndarray, path: str) -> None:
        """
        保存特征向量到文件

        Args:
            feature_vector: 特征向量
            path: 保存路径
        """
        np.save(path, feature_vector)
        self.logger.info(f"特征向量已保存至: {path}，维度: {feature_vector.shape}")

    def load_feature_vector(self, path: str) -> np.ndarray:
        """
        从文件加载特征向量

        Args:
            path: 文件路径

        Returns:
            加载的特征向量
        """
        feature_vector = np.load(path)
        self.logger.info(f"特征向量已加载，维度: {feature_vector.shape}")
        return feature_vector


# 模块测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 1. 导入依赖模块
    from preprocessing.data_loader import VideoImageLoader
    from preprocessing.background_model import BackgroundModel
    from preprocessing.target_region_detector import TargetRegionDetector
    from feature_extraction.motion_block_detector import MotionBlockDetector
    from feature_extraction.gradient_calculator import GradientCalculator

    # 2. 初始化组件
    video_path = "path/to/your/video.mp4"  # 替换为实际视频路径
    loader = VideoImageLoader(video_path, is_video=True, gray_scale=True)
    bg_model = BackgroundModel()
    target_detector = TargetRegionDetector()
    motion_detector = MotionBlockDetector()
    gradient_calculator = GradientCalculator()
    feature_generator = FeatureVectorGenerator()

    try:
        # 3. 读取前两帧初始化背景
        _, frame1 = loader.get_next_frame()
        _, frame2 = loader.get_next_frame()
        bg_model.initialize(frame1, frame2)

        # 4. 读取第3帧和第4帧用于特征提取
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
                # 生成特征向量
                feature_vector = feature_generator.generate_from_multiple_blocks(
                    motion_blocks, gradient_calculator
                )

                # 保存特征向量
                feature_generator.save_feature_vector(feature_vector, "test_feature_vector.npy")

                print(f"特征向量生成完成，维度: {feature_vector.shape}，已保存")
            else:
                print("未检测到有效运动块，无法生成特征向量")

    except Exception as e:
        print(f"测试失败: {str(e)}")
    finally:
        loader.release()
        bg_model.reset()