import cv2
import numpy as np
import torch
from torchvision import models, transforms
import logging
from typing import Tuple, List, Dict, Optional, Union


class AnomalyRegionMarker:
    """异常区域标记子模块：实现视频帧中异常行为区域的标记与可视化（对应文献4.1节）"""

    def __init__(self,
                 threshold: float = 0.7,  # 异常置信度阈值
                 color_normal: Tuple[int, int, int] = (0, 255, 0),  # 正常区域颜色（绿色）
                 color_abnormal: Tuple[int, int, int] = (0, 0, 255),  # 异常区域颜色（红色）
                 line_thickness: int = 2,  # 边界线粗细
                 opacity: float = 0.5,  # 填充透明度
                 use_gpu: bool = True,  # 是否使用GPU
                 detection_model: str = "yolov5s",  # 目标检测模型
                 feature_model: str = "resnet18"):  # 特征提取模型
        """
        初始化异常区域标记器

        Args:
            threshold: 异常置信度阈值，超过此值的区域将被标记为异常
            color_normal: 正常区域边界颜色 (B, G, R)
            color_abnormal: 异常区域边界颜色 (B, G, R)
            line_thickness: 边界线粗细
            opacity: 填充颜色的透明度
            use_gpu: 是否使用GPU加速
            detection_model: 目标检测模型名称
            feature_model: 特征提取模型名称
        """
        self.threshold = threshold
        self.color_normal = color_normal
        self.color_abnormal = color_abnormal
        self.line_thickness = line_thickness
        self.opacity = opacity
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"异常区域标记器初始化完成 - "
                         f"threshold={threshold}, opacity={opacity}, "
                         f"device={self.device.type}")

        # 初始化目标检测模型
        self.object_detector = self._init_object_detector(detection_model)

        # 初始化特征提取模型
        self.feature_extractor, self.feature_dim = self._init_feature_extractor(feature_model)

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _init_object_detector(self, model_name: str):
        """初始化目标检测模型"""
        try:
            if model_name.lower() == "yolov5s":
                # 使用YOLOv5小模型
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            elif model_name.lower() == "yolov5m":
                # 使用YOLOv5中等模型
                model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
            else:
                self.logger.warning(f"不支持的目标检测模型: {model_name}，使用YOLOv5s作为默认")
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

            model = model.to(self.device)
            model.eval()
            self.logger.info(f"目标检测模型初始化成功: {model_name}")
            return model
        except Exception as e:
            self.logger.error(f"目标检测模型初始化失败: {e}")
            return None

    def _init_feature_extractor(self, model_name: str):
        """初始化特征提取模型"""
        try:
            if model_name.lower() == "resnet18":
                model = models.resnet18(pretrained=True)
                feature_dim = 512
            elif model_name.lower() == "resnet50":
                model = models.resnet50(pretrained=True)
                feature_dim = 2048
            else:
                self.logger.warning(f"不支持的特征提取模型: {model_name}，使用ResNet18作为默认")
                model = models.resnet18(pretrained=True)
                feature_dim = 512

            # 移除最后的全连接层，只保留特征提取部分
            model = torch.nn.Sequential(*list(model.children())[:-1])
            model = model.to(self.device)
            model.eval()

            self.logger.info(f"特征提取模型初始化成功: {model_name}, 特征维度: {feature_dim}")
            return model, feature_dim
        except Exception as e:
            self.logger.error(f"特征提取模型初始化失败: {e}")
            return None, 512  # 默认返回512维特征

    def detect_objects(self, frame: np.ndarray, classes: Optional[List[str]] = None) -> List[Dict]:
        """
        检测视频帧中的目标

        Args:
            frame: 输入视频帧
            classes: 需要检测的目标类别列表，None表示检测所有类别

        Returns:
            检测到的目标列表，每个目标包含边界框和类别信息
        """
        if self.object_detector is None:
            self.logger.warning("目标检测模型未初始化，返回空结果")
            return []

        # 执行目标检测
        results = self.object_detector(frame)

        # 如果指定了类别，过滤结果
        if classes is not None:
            detections = []
            for det in results.pandas().xyxy[0].to_dict('records'):
                if det['name'] in classes:
                    detections.append(det)
        else:
            detections = results.pandas().xyxy[0].to_dict('records')

        # 转换为所需格式
        regions = []
        for i, det in enumerate(detections):
            regions.append({
                'bbox': [int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])],
                'id': i,
                'class': det['name'],
                'confidence': det['confidence']
            })

        self.logger.debug(f"检测到 {len(regions)} 个目标")
        return regions

    def extract_features(self, frame: np.ndarray, regions: List[Dict]) -> np.ndarray:
        """
        从检测到的区域中提取特征

        Args:
            frame: 输入视频帧
            regions: 检测到的目标区域

        Returns:
            特征向量数组，形状为 (n_regions, feature_dim)
        """
        if self.feature_extractor is None or len(regions) == 0:
            return np.array([])

        features = []

        for region in regions:
            x1, y1, x2, y2 = region['bbox']

            # 提取区域ROI
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                # 如果ROI为空，生成零向量
                features.append(np.zeros(self.feature_dim))
                continue

            # 预处理ROI
            try:
                roi_tensor = self.transform(roi).unsqueeze(0).to(self.device)

                # 提取特征
                with torch.no_grad():
                    roi_features = self.feature_extractor(roi_tensor)

                # 展平特征
                roi_features = roi_features.squeeze().cpu().numpy().flatten()
                features.append(roi_features)
            except Exception as e:
                self.logger.warning(f"特征提取失败: {e}，使用零向量替代")
                features.append(np.zeros(self.feature_dim))

        return np.array(features)

    def mark_regions(self,
                     frame: np.ndarray,
                     regions: List[Dict],
                     predictions: List[int],
                     confidence_scores: Optional[List[float]] = None) -> np.ndarray:
        """
        在视频帧上标记异常区域（对应文献4.1节算法）

        Args:
            frame: 原始视频帧，numpy数组，形状为(H, W, 3)
            regions: 检测到的区域列表，每个区域为字典，包含'bbox'键（边界框坐标）
            predictions: 区域的预测结果列表，1表示异常，0表示正常
            confidence_scores: 预测置信度列表（可选）

        Returns:
            标记后的视频帧
        """
        # 创建用于绘制的副本
        marked_frame = frame.copy()
        overlay = frame.copy()

        # 确保各列表长度一致
        n_regions = len(regions)
        if len(predictions) != n_regions:
            raise ValueError("预测结果数量与区域数量不匹配")

        if confidence_scores is not None and len(confidence_scores) != n_regions:
            raise ValueError("置信度分数数量与区域数量不匹配")

        # 遍历所有区域并标记
        for i, (region, pred) in enumerate(zip(regions, predictions)):
            # 获取边界框坐标 [x1, y1, x2, y2]
            bbox = region['bbox']
            x1, y1, x2, y2 = map(int, bbox)

            # 根据预测结果选择颜色
            color = self.color_abnormal if pred == 1 else self.color_normal

            # 绘制边界框
            cv2.rectangle(marked_frame, (x1, y1), (x2, y2), color, self.line_thickness)

            # 添加标签文本
            label = "Abnormal" if pred == 1 else "Normal"
            if confidence_scores is not None:
                label += f": {confidence_scores[i]:.2f}"

            # 添加类别信息
            if 'class' in region:
                label += f" ({region['class']})"

            # 计算文本位置
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_x = max(x1, 5)  # 确保文本不超出图像边界
            text_y = max(y1 - 10, text_size[1] + 5)  # 确保文本不超出图像边界

            # 绘制文本背景
            cv2.rectangle(marked_frame,
                          (text_x - 5, text_y - text_size[1] - 5),
                          (text_x + text_size[0] + 5, text_y + 5),
                          color, -1)

            # 绘制文本
            cv2.putText(marked_frame, label,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # 为异常区域添加半透明填充
            if pred == 1:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), self.color_abnormal, -1)

        # 合并叠加层，使异常区域半透明
        if np.sum(overlay - frame) != 0:  # 如果有异常区域
            marked_frame = cv2.addWeighted(overlay, self.opacity, marked_frame, 1 - self.opacity, 0)

        return marked_frame

    def create_anomaly_mask(self,
                            frame_shape: Tuple[int, int, int],
                            regions: List[Dict],
                            predictions: List[int]) -> np.ndarray:
        """
        创建异常区域掩码（对应文献4.1节辅助算法）

        Args:
            frame_shape: 视频帧形状 (H, W, C)
            regions: 检测到的区域列表
            predictions: 区域的预测结果列表

        Returns:
            异常区域掩码，二值图像，异常区域为255，正常区域为0
        """
        # 创建全黑掩码
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)

        # 确保各列表长度一致
        n_regions = len(regions)
        if len(predictions) != n_regions:
            raise ValueError("预测结果数量与区域数量不匹配")

        # 遍历所有区域并标记异常区域
        for region, pred in zip(regions, predictions):
            if pred == 1:  # 如果是异常区域
                # 获取边界框坐标
                bbox = region['bbox']
                x1, y1, x2, y2 = map(int, bbox)

                # 在掩码上填充白色矩形
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        return mask

    def calculate_anomaly_ratio(self, mask: np.ndarray) -> float:
        """
        计算异常区域占比（对应文献4.1节评估指标）

        Args:
            mask: 异常区域掩码，二值图像

        Returns:
            异常区域占比，范围从0.0到1.0
        """
        if mask.size == 0:
            return 0.0

        # 计算异常区域像素数
        anomaly_pixels = np.count_nonzero(mask)

        # 计算总像素数
        total_pixels = mask.shape[0] * mask.shape[1]

        # 计算异常区域占比
        return anomaly_pixels / total_pixels

    def visualize_regions(self,
                          frame: np.ndarray,
                          regions: List[Dict],
                          title: str = "Detected Regions",
                          wait_time: int = 0) -> None:
        """
        可视化检测到的区域（用于调试）

        Args:
            frame: 原始视频帧
            regions: 检测到的区域列表
            title: 窗口标题
            wait_time: 等待时间（毫秒），0表示无限等待
        """
        # 创建用于绘制的副本
        vis_frame = frame.copy()

        # 遍历所有区域并绘制
        for region in regions:
            # 获取边界框坐标
            bbox = region['bbox']
            x1, y1, x2, y2 = map(int, bbox)

            # 绘制边界框
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # 添加区域ID和类别
            label = f"ID: {region['id']}"
            if 'class' in region:
                label += f" ({region['class']})"

            cv2.putText(vis_frame, label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # 添加置信度
            if 'confidence' in region:
                conf_text = f"Conf: {region['confidence']:.2f}"
                cv2.putText(vis_frame, conf_text,
                            (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 显示结果
        cv2.imshow(title, vis_frame)
        cv2.waitKey(wait_time)
        cv2.destroyAllWindows()