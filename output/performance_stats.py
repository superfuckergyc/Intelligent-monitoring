import time
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional


class PerformanceStats:
    """识别时间/准确率统计子模块：实现异常行为识别系统的性能指标统计（对应文献4.2节）"""

    def __init__(self):
        """初始化性能统计器"""
        self.reset()
        self.logger = logging.getLogger(__name__)
        self.logger.info("性能统计器初始化完成")

    def reset(self) -> None:
        """重置所有统计数据"""
        # 时间统计
        self.frame_processing_times = []  # 每帧处理时间
        self.feature_extraction_times = []  # 特征提取时间
        self.model_prediction_times = []  # 模型预测时间
        self.region_marking_times = []  # 区域标记时间

        # 准确率统计
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0

        # 辅助统计
        self.total_frames = 0
        self.anomaly_frames = 0

    def start_timer(self) -> float:
        """
        开始计时

        Returns:
            当前时间戳
        """
        return time.time()

    def end_timer(self, start_time: float, time_list: List[float]) -> float:
        """
        结束计时并记录时间差

        Args:
            start_time: 开始时间戳
            time_list: 存储时间差的列表

        Returns:
            时间差（毫秒）
        """
        elapsed_time = (time.time() - start_time) * 1000  # 转换为毫秒
        time_list.append(elapsed_time)
        return elapsed_time

    def record_frame_processing_time(self, start_time: float) -> float:
        """
        记录帧处理时间

        Args:
            start_time: 开始时间戳

        Returns:
            帧处理时间（毫秒）
        """
        return self.end_timer(start_time, self.frame_processing_times)

    def record_feature_extraction_time(self, start_time: float) -> float:
        """
        记录特征提取时间

        Args:
            start_time: 开始时间戳

        Returns:
            特征提取时间（毫秒）
        """
        return self.end_timer(start_time, self.feature_extraction_times)

    def record_model_prediction_time(self, start_time: float) -> float:
        """
        记录模型预测时间

        Args:
            start_time: 开始时间戳

        Returns:
            模型预测时间（毫秒）
        """
        return self.end_timer(start_time, self.model_prediction_times)

    def record_region_marking_time(self, start_time: float) -> float:
        """
        记录区域标记时间

        Args:
            start_time: 开始时间戳

        Returns:
            区域标记时间（毫秒）
        """
        return self.end_timer(start_time, self.region_marking_times)

    def update_accuracy_stats(self, ground_truth: List[int], predictions: List[int]) -> None:
        """
        更新准确率统计数据

        Args:
            ground_truth: 真实标签列表
            predictions: 预测标签列表
        """
        # 确保两个列表长度相同
        if len(ground_truth) != len(predictions):
            raise ValueError("真实标签和预测标签数量不匹配")

        # 更新帧计数
        self.total_frames += 1

        # 检查是否有异常
        if any(p == 1 for p in predictions):
            self.anomaly_frames += 1

        # 计算准确率统计
        for gt, pred in zip(ground_truth, predictions):
            if gt == 1 and pred == 1:
                self.true_positives += 1
            elif gt == 0 and pred == 1:
                self.false_positives += 1
            elif gt == 0 and pred == 0:
                self.true_negatives += 1
            elif gt == 1 and pred == 0:
                self.false_negatives += 1

    def calculate_time_stats(self) -> Dict[str, float]:
        """
        计算时间统计数据

        Returns:
            包含各项时间统计的字典
        """
        if not self.frame_processing_times:
            return {
                'avg_frame_processing_time': 0,
                'max_frame_processing_time': 0,
                'min_frame_processing_time': 0,
                'fps': 0,
                'avg_feature_extraction_time': 0,
                'avg_model_prediction_time': 0,
                'avg_region_marking_time': 0
            }

        # 计算帧处理时间统计
        avg_frame_time = np.mean(self.frame_processing_times)
        max_frame_time = np.max(self.frame_processing_times)
        min_frame_time = np.min(self.frame_processing_times)
        fps = 1000.0 / avg_frame_time  # FPS = 1000ms / 平均处理时间

        # 计算各组件平均时间
        avg_feature_time = np.mean(self.feature_extraction_times) if self.feature_extraction_times else 0
        avg_prediction_time = np.mean(self.model_prediction_times) if self.model_prediction_times else 0
        avg_marking_time = np.mean(self.region_marking_times) if self.region_marking_times else 0

        return {
            'avg_frame_processing_time': avg_frame_time,
            'max_frame_processing_time': max_frame_time,
            'min_frame_processing_time': min_frame_time,
            'fps': fps,
            'avg_feature_extraction_time': avg_feature_time,
            'avg_model_prediction_time': avg_prediction_time,
            'avg_region_marking_time': avg_marking_time
        }

    def calculate_accuracy_stats(self) -> Dict[str, float]:
        """
        计算准确率统计数据

        Returns:
            包含各项准确率统计的字典
        """
        # 避免除零错误
        total_predictions = self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        if total_predictions == 0:
            return {
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'false_positive_rate': 0,
                'false_negative_rate': 0,
                'anomaly_ratio': 0
            }

        # 计算准确率指标
        accuracy = (self.true_positives + self.true_negatives) / total_predictions

        # 避免除零错误
        precision = self.true_positives / (self.true_positives + self.false_positives) if (
                                                                                                      self.true_positives + self.false_positives) > 0 else 0
        recall = self.true_positives / (self.true_positives + self.false_negatives) if (
                                                                                                   self.true_positives + self.false_negatives) > 0 else 0

        # 计算F1分数
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # 计算误报率和漏报率
        false_positive_rate = self.false_positives / (self.false_positives + self.true_negatives) if (
                                                                                                                 self.false_positives + self.true_negatives) > 0 else 0
        false_negative_rate = self.false_negatives / (self.false_negatives + self.true_positives) if (
                                                                                                                 self.false_negatives + self.true_positives) > 0 else 0

        # 计算异常帧比例
        anomaly_ratio = self.anomaly_frames / self.total_frames if self.total_frames > 0 else 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'anomaly_ratio': anomaly_ratio
        }

    def generate_report(self) -> str:
        """
        生成性能统计报告

        Returns:
            包含所有统计信息的字符串
        """
        time_stats = self.calculate_time_stats()
        accuracy_stats = self.calculate_accuracy_stats()

        report = "=" * 50 + "\n"
        report += "异常行为识别系统性能统计报告\n"
        report += "=" * 50 + "\n\n"

        # 添加时间统计
        report += "时间性能指标:\n"
        report += f"  平均帧处理时间: {time_stats['avg_frame_processing_time']:.2f} ms\n"
        report += f"  最大帧处理时间: {time_stats['max_frame_processing_time']:.2f} ms\n"
        report += f"  最小帧处理时间: {time_stats['min_frame_processing_time']:.2f} ms\n"
        report += f"  处理帧率: {time_stats['fps']:.2f} FPS\n"
        report += f"  平均特征提取时间: {time_stats['avg_feature_extraction_time']:.2f} ms\n"
        report += f"  平均模型预测时间: {time_stats['avg_model_prediction_time']:.2f} ms\n"
        report += f"  平均区域标记时间: {time_stats['avg_region_marking_time']:.2f} ms\n\n"

        # 添加准确率统计
        report += "准确率性能指标:\n"
        report += f"  准确率: {accuracy_stats['accuracy']:.4f}\n"
        report += f"  精确率: {accuracy_stats['precision']:.4f}\n"
        report += f"  召回率: {accuracy_stats['recall']:.4f}\n"
        report += f"  F1分数: {accuracy_stats['f1_score']:.4f}\n"
        report += f"  误报率: {accuracy_stats['false_positive_rate']:.4f}\n"
        report += f"  漏报率: {accuracy_stats['false_negative_rate']:.4f}\n"
        report += f"  异常帧比例: {accuracy_stats['anomaly_ratio']:.4f}\n\n"

        # 添加计数统计
        report += "样本计数:\n"
        report += f"  总帧数: {self.total_frames}\n"
        report += f"  异常帧数: {self.anomaly_frames}\n"
        report += f"  真正例: {self.true_positives}\n"
        report += f"  假正例: {self.false_positives}\n"
        report += f"  真反例: {self.true_negatives}\n"
        report += f"  假反例: {self.false_negatives}\n"
        report += "=" * 50

        return report

    def save_report(self, file_path: str) -> None:
        """
        将性能统计报告保存到文件

        Args:
            file_path: 文件路径
        """
        report = self.generate_report()
        with open(file_path, 'w') as f:
            f.write(report)

        self.logger.info(f"性能报告已保存至: {file_path}")


# 模块测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 1. 初始化性能统计器
    stats = PerformanceStats()

    # 2. 模拟处理100帧数据
    logging.info("模拟处理100帧数据...")
    for i in range(100):
        # 模拟帧处理
        frame_start_time = stats.start_timer()

        # 模拟特征提取
        feature_start_time = stats.start_timer()
        time.sleep(np.random.uniform(0.01, 0.03))  # 10-30ms
        stats.record_feature_extraction_time(feature_start_time)

        # 模拟模型预测
        pred_start_time = stats.start_timer()
        time.sleep(np.random.uniform(0.02, 0.05))  # 20-50ms
        stats.record_model_prediction_time(pred_start_time)

        # 模拟区域标记
        mark_start_time = stats.start_timer()
        time.sleep(np.random.uniform(0.005, 0.015))  # 5-15ms
        stats.record_region_marking_time(mark_start_time)

        # 记录帧处理时间
        stats.record_frame_processing_time(frame_start_time)

        # 模拟预测结果和真实标签
        if i % 10 == 0:  # 10%的帧包含异常
            predictions = [1, 0, 0]  # 有一个异常
            ground_truth = [1, 0, 0]
        else:
            predictions = [0, 0, 0]  # 无异常
            ground_truth = [0, 0, 0]

        # 更新准确率统计
        stats.update_accuracy_stats(ground_truth, predictions)

        # 每10帧输出一次进度
        if (i + 1) % 10 == 0:
            logging.info(f"已处理 {i + 1}/100 帧")

    # 3. 添加一些误报和漏报
    logging.info("添加模拟的误报和漏报...")
    stats.update_accuracy_stats([1, 0, 0], [0, 0, 0])  # 漏报
    stats.update_accuracy_stats([0, 0, 0], [0, 1, 0])  # 误报

    # 4. 计算并打印时间统计
    time_stats = stats.calculate_time_stats()
    logging.info(f"平均帧处理时间: {time_stats['avg_frame_processing_time']:.2f} ms")
    logging.info(f"处理帧率: {time_stats['fps']:.2f} FPS")

    # 5. 计算并打印准确率统计
    accuracy_stats = stats.calculate_accuracy_stats()
    logging.info(f"准确率: {accuracy_stats['accuracy']:.4f}")
    logging.info(f"精确率: {accuracy_stats['precision']:.4f}")
    logging.info(f"召回率: {accuracy_stats['recall']:.4f}")
    logging.info(f"F1分数: {accuracy_stats['f1_score']:.4f}")

    # 6. 生成并保存报告
    report = stats.generate_report()
    logging.info("性能统计报告:\n" + report)

    # 7. 保存报告到文件
    stats.save_report("performance_report.txt")
    logging.info("性能报告已保存")