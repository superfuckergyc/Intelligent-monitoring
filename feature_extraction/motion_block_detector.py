import cv2
import numpy as np
import logging
from typing import Tuple, List, Dict, Optional


class MotionBlockDetector:
    """有效运动块采集子模块：实现运动区域的分块检测与筛选（对应文献2.1节）"""

    def __init__(self,
                 block_size: Tuple[int, int] = (32, 32),  # 基本块大小
                 overlap_ratio: float = 0.25,  # 块重叠率
                 min_motion_threshold: int = 10,  # 最小运动像素数阈值
                 motion_intensity_threshold: float = 0.1):  # 运动强度阈值
        """
        初始化运动块检测参数

        Args:
            block_size: 分块大小(Z, E)，对应文献中的块划分
            overlap_ratio: 相邻块的重叠比例(0~1)
            min_motion_threshold: 判定为有效运动块的最小运动像素数
            motion_intensity_threshold: 判定为有效运动块的最小平均运动强度
        """
        self.block_size = block_size
        self.overlap_ratio = overlap_ratio
        self.min_motion_threshold = min_motion_threshold
        self.motion_intensity_threshold = motion_intensity_threshold

        # 计算步长（考虑重叠）
        self.step_size = (
            int(block_size[0] * (1 - overlap_ratio)),
            int(block_size[1] * (1 - overlap_ratio))
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"运动块检测器初始化完成 - 块大小: {block_size}, 步长: {self.step_size}")

    def detect_motion_blocks(self,
                             target_region: np.ndarray,
                             prev_frame: np.ndarray,
                             curr_frame: np.ndarray) -> List[Dict]:
        """
        检测并筛选有效运动块（对应文献2.1节算法）

        Args:
            target_region: 目标区域二值图像（来自1.3模块）
            prev_frame: 上一帧图像
            curr_frame: 当前帧图像

        Returns:
            有效运动块列表，每个元素包含:
            {
                'block': 块图像,
                'position': (x, y) 块左上角坐标,
                'size': (width, height) 块大小,
                'motion_pixels': 运动像素数,
                'motion_intensity': 运动强度
            }
        """
        if target_region.shape != prev_frame.shape or target_region.shape != curr_frame.shape:
            raise ValueError(f"输入图像尺寸不一致: {target_region.shape} vs {prev_frame.shape} vs {curr_frame.shape}")

        height, width = target_region.shape
        valid_blocks = []

        # 遍历所有可能的块位置
        for y in range(0, height - self.block_size[0] + 1, self.step_size[0]):
            for x in range(0, width - self.block_size[1] + 1, self.step_size[1]):
                # 提取块区域
                block_mask = target_region[y:y + self.block_size[0], x:x + self.block_size[1]]
                prev_block = prev_frame[y:y + self.block_size[0], x:x + self.block_size[1]]
                curr_block = curr_frame[y:y + self.block_size[0], x:x + self.block_size[1]]

                # 计算块内运动像素数和运动强度
                motion_pixels = np.count_nonzero(block_mask)

                if motion_pixels > 0:
                    # 计算块内运动强度（帧间差分的平均绝对值）
                    frame_diff = np.abs(curr_block.astype(np.int32) - prev_block.astype(np.int32))
                    motion_intensity = np.mean(frame_diff[block_mask > 0])

                    # 判断是否为有效运动块
                    if (motion_pixels >= self.min_motion_threshold and
                            motion_intensity >= self.motion_intensity_threshold):
                        valid_blocks.append({
                            'block': curr_block,
                            'position': (x, y),
                            'size': self.block_size,
                            'motion_pixels': motion_pixels,
                            'motion_intensity': motion_intensity
                        })

        self.logger.debug(f"检测到 {len(valid_blocks)} 个有效运动块")
        return valid_blocks

    def merge_overlapping_blocks(self,
                                 blocks: List[Dict],
                                 iou_threshold: float = 0.5) -> List[Dict]:
        """
        合并重叠度高的运动块（基于IOU准则）

        Args:
            blocks: 有效运动块列表
            iou_threshold: IOU阈值，超过此值的块将被合并

        Returns:
            合并后的运动块列表
        """
        if not blocks:
            return []

        # 按运动像素数降序排序
        blocks = sorted(blocks, key=lambda x: x['motion_pixels'], reverse=True)
        merged_blocks = [blocks[0]]

        for block in blocks[1:]:
            # 计算与已合并块的最大IOU
            max_iou = 0
            for merged in merged_blocks:
                iou = self._calculate_iou(
                    block['position'], merged['position'],
                    block['size'], merged['size']
                )
                max_iou = max(max_iou, iou)

            # 如果最大IOU小于阈值，则添加为新块
            if max_iou < iou_threshold:
                merged_blocks.append(block)

        self.logger.debug(f"合并后剩余 {len(merged_blocks)} 个运动块")
        return merged_blocks

    def _calculate_iou(self,
                       pos1: Tuple[int, int],
                       pos2: Tuple[int, int],
                       size1: Tuple[int, int],
                       size2: Tuple[int, int]) -> float:
        """计算两个块的IOU（交并比）"""
        x1, y1 = pos1
        x2, y2 = pos2
        w1, h1 = size1
        w2, h2 = size2

        # 计算交集区域
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = w1 * h1
        box2_area = w2 * h2

        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return iou

    def visualize_motion_blocks(self,
                                frame: np.ndarray,
                                blocks: List[Dict],
                                save_path: Optional[str] = None) -> None:
        """
        可视化运动块检测结果

        Args:
            frame: 原始帧图像
            blocks: 运动块列表
            save_path: 保存路径（None则直接显示）
        """
        # 创建彩色图像用于可视化
        if len(frame.shape) == 2:
            vis_image = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = frame.copy()

        # 绘制运动块边界
        for block in blocks:
            x, y = block['position']
            w, h = block['size']
            intensity = block['motion_intensity']

            # 根据运动强度调整颜色（蓝色到红色）
            color = (int(255 - intensity * 10), 0, int(intensity * 10))

            # 绘制矩形框
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)

            # 添加运动强度文本
            text = f"I: {intensity:.1f}"
            cv2.putText(vis_image, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 显示或保存结果
        if save_path:
            cv2.imwrite(save_path, vis_image)
            self.logger.info(f"运动块可视化结果已保存至: {save_path}")
        else:
            cv2.imshow("Motion Blocks", vis_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


# 模块测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 1. 导入依赖模块
    from preprocessing.data_loader import VideoImageLoader
    from preprocessing.background_model import BackgroundModel
    from preprocessing.target_region_detector import TargetRegionDetector

    # 2. 初始化组件
    video_path = "path/to/your/video.mp4"  # 替换为实际视频路径
    loader = VideoImageLoader(video_path, is_video=True, gray_scale=True)
    bg_model = BackgroundModel()
    target_detector = TargetRegionDetector()
    motion_detector = MotionBlockDetector()

    try:
        # 3. 读取前两帧初始化背景
        _, frame1 = loader.get_next_frame()
        _, frame2 = loader.get_next_frame()
        bg_model.initialize(frame1, frame2)

        # 4. 读取第3帧和第4帧用于运动块检测
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

            # 合并重叠块
            merged_blocks = motion_detector.merge_overlapping_blocks(motion_blocks)

            # 可视化
            motion_detector.visualize_motion_blocks(
                frame4, merged_blocks, "motion_blocks_result.png"
            )

            print(f"有效运动块检测完成，共检测到 {len(merged_blocks)} 个运动块，结果已保存")

    except Exception as e:
        print(f"测试失败: {str(e)}")
    finally:
        loader.release()
        bg_model.reset()