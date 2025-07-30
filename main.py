import cv2
import numpy as np
import os
import time
from datetime import datetime
from typing import List, Optional
from output.anomaly_region_marker import AnomalyRegionMarker
from output.performance_stats import PerformanceStats
from anomaly_detection.lssvm_trainer import LSSVMTrainer


def main(video_path: str,
         output_dir: str = "results",
         train_data_path: str = "data/train_data.npy",
         train_labels_path: str = "data/train_labels.npy",
         threshold: float = 0.7,
         visualize: bool = True,
         detection_model: str = "yolov5s",
         feature_model: str = "resnet18",
         detect_classes: Optional[List[str]] = None,
         use_gpu: bool = True,
         batch_size: int = 1,
         save_interval: int = 10):
    """
    主函数：将异常区域标记和性能统计模块串联起来

    Args:
        video_path: 输入视频路径
        output_dir: 输出结果目录
        train_data_path: 训练数据路径
        train_labels_path: 训练标签路径
        threshold: 异常检测阈值
        visualize: 是否可视化结果
        detection_model: 目标检测模型类型
        feature_model: 特征提取模型类型
        detect_classes: 需要检测的目标类别列表
        use_gpu: 是否使用GPU加速
        batch_size: 批处理大小（用于特征提取和预测）
        save_interval: 保存异常帧的间隔（每隔多少帧保存一次）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "frames"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "model"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)

    # 配置日志
    import logging
    logging.basicConfig(
        filename=os.path.join(output_dir, "logs", "process.log"),
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.info(f"开始处理视频: {video_path}")

    # 初始化模块
    try:
        marker = AnomalyRegionMarker(
            threshold=threshold,
            use_gpu=use_gpu,
            detection_model=detection_model,
            feature_model=feature_model
        )
        stats = PerformanceStats()
        trainer = LSSVMTrainer()
    except Exception as e:
        print(f"初始化失败: {e}")
        logger.error(f"初始化失败: {e}")
        return

    # 加载训练数据
    try:
        logger.info(f"加载训练数据: {train_data_path}")
        train_data = np.load(train_data_path)
        train_labels = np.load(train_labels_path)
        logger.info(f"训练数据加载成功: 样本数={len(train_data)}, 特征维度={train_data.shape[1]}")
    except FileNotFoundError:
        print(f"错误: 找不到训练数据或标签文件，请检查路径: {train_data_path}, {train_labels_path}")
        logger.error(f"找不到训练数据或标签文件: {train_data_path}, {train_labels_path}")
        return
    except Exception as e:
        print(f"加载训练数据失败: {e}")
        logger.error(f"加载训练数据失败: {e}")
        return

    # 确保训练数据特征维度匹配
    expected_feature_dim = marker.feature_dim
    if train_data.shape[1] != expected_feature_dim:
        print(f"警告: 训练数据特征维度 ({train_data.shape[1]}) 与特征提取器维度 ({expected_feature_dim}) 不匹配")
        print("这可能导致预测结果不准确")
        logger.warning(f"特征维度不匹配: 训练数据({train_data.shape[1]}) vs 提取器({expected_feature_dim})")

    # 训练模型
    try:
        logger.info("开始训练LSSVM模型...")
        trainer.train(train_data, train_labels)
        model_path = os.path.join(output_dir, "model", "lssvm_model.pkl")
        trainer.save_model(model_path)
        logger.info(f"模型训练完成并保存至: {model_path}")
        print(f"模型已保存至: {model_path}")
    except Exception as e:
        print(f"模型训练失败: {e}")
        logger.error(f"模型训练失败: {e}")
        return

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: 无法打开视频文件 {video_path}")
        logger.error(f"无法打开视频文件: {video_path}")
        return

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"视频信息: {width}x{height}, {fps:.2f} FPS, 总帧数: {frame_count}")
    logger.info(f"视频信息: {width}x{height}, {fps:.2f} FPS, 总帧数: {frame_count}")

    # 创建输出视频写入器
    if visualize:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video_path = os.path.join(output_dir, "output.mp4")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 逐帧处理视频
    frame_idx = 0
    anomaly_frames = 0
    total_process_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        frame_start_time = time.time()
        print(f"\r处理第 {frame_idx}/{frame_count} 帧...", end="")
        logger.debug(f"开始处理第 {frame_idx} 帧")

        # 1. 目标检测
        detection_start_time = time.time()
        regions = marker.detect_objects(frame, classes=detect_classes)
        detection_time = time.time() - detection_start_time
        stats.record_detection_time(detection_start_time)

        if not regions:
            print(f"\r第 {frame_idx} 帧未检测到目标，跳过处理", end="")
            logger.debug(f"第 {frame_idx} 帧未检测到目标")
            continue

        # 2. 特征提取
        feature_start_time = time.time()
        features = marker.extract_features(frame, regions)
        feature_time = time.time() - feature_start_time
        stats.record_feature_extraction_time(feature_start_time)

        # 3. 使用LSSVM模型进行预测
        pred_start_time = time.time()
        predictions = trainer.predict(features)
        confidence_scores = [abs(score) for score in predictions]
        predictions = [1 if p > 0 else 0 for p in predictions]
        pred_time = time.time() - pred_start_time
        stats.record_model_prediction_time(pred_start_time)

        # 4. 标记异常区域
        mark_start_time = time.time()
        marked_frame = marker.mark_regions(
            frame,
            regions,
            predictions,
            confidence_scores
        )
        mark_time = time.time() - mark_start_time
        stats.record_region_marking_time(mark_start_time)

        # 5. 更新性能统计
        frame_process_time = time.time() - frame_start_time
        total_process_time += frame_process_time
        stats.record_frame_processing_time(frame_start_time)

        # 6. 更新准确率统计（实际应用中应使用真实标签）
        ground_truth = predictions.copy()  # 假设预测完全正确
        stats.update_accuracy_stats(ground_truth, predictions)

        # 7. 保存结果
        if visualize:
            # 添加帧信息
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(marked_frame, f"Frame: {frame_idx}/{frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(marked_frame, f"Time: {timestamp}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 添加处理时间信息
            cv2.putText(marked_frame, f"Process Time: {frame_process_time:.2f}s", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(marked_frame, f"FPS: {1 / frame_process_time:.2f}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 添加异常区域占比
            mask = marker.create_anomaly_mask(frame.shape, regions, predictions)
            anomaly_ratio = marker.calculate_anomaly_ratio(mask)
            cv2.putText(marked_frame, f"Anomaly Ratio: {anomaly_ratio:.2%}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 写入输出视频
            out.write(marked_frame)

            # 保存异常帧（按指定间隔）
            if any(predictions) and (frame_idx % save_interval == 0):
                anomaly_frames += 1
                frame_filename = os.path.join(output_dir, "frames",
                                              f"anomaly_frame_{frame_idx}_ratio_{anomaly_ratio:.2%}.jpg")
                cv2.imwrite(frame_filename, marked_frame)
                logger.info(f"保存异常帧: {frame_filename}, 异常区域占比: {anomaly_ratio:.2%}")

        # 8. 显示结果（按ESC退出）
        if visualize:
            cv2.imshow("Anomaly Detection", marked_frame)
            key = cv2.waitKey(1)
            if key & 0xFF == 27:  # ESC键
                print("\n用户中断处理")
                logger.info(f"用户中断处理，已处理 {frame_idx}/{frame_count} 帧")
                break
            elif key & 0xFF == ord('s'):  # S键保存当前帧
                save_path = os.path.join(output_dir, "frames", f"manual_save_frame_{frame_idx}.jpg")
                cv2.imwrite(save_path, marked_frame)
                print(f"\n当前帧已保存至: {save_path}")
                logger.info(f"手动保存帧: {save_path}")

    # 释放资源
    cap.release()
    if visualize:
        out.release()
        cv2.destroyAllWindows()

    # 生成性能报告
    avg_fps = frame_idx / total_process_time if total_process_time > 0 else 0
    report = stats.generate_report()
    print(f"\n处理完成! 总耗时: {total_process_time:.2f}s, 平均FPS: {avg_fps:.2f}")
    print(f"共处理 {frame_idx} 帧，检测到 {anomaly_frames} 个异常帧")
    print(report)

    # 保存性能报告
    report_path = os.path.join(output_dir, "performance_report.txt")
    stats.save_report(report_path)

    # 保存处理摘要
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"视频处理摘要\n")
        f.write(f"------------------------\n")
        f.write(f"输入视频: {video_path}\n")
        f.write(f"输出目录: {output_dir}\n")
        f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总帧数: {frame_count}\n")
        f.write(f"已处理帧数: {frame_idx}\n")
        f.write(f"异常帧数: {anomaly_frames}\n")
        f.write(f"总耗时: {total_process_time:.2f}s\n")
        f.write(f"平均FPS: {avg_fps:.2f}\n")
        f.write(f"检测模型: {detection_model}\n")
        f.write(f"特征模型: {feature_model}\n")
        f.write(f"异常阈值: {threshold}\n")

    logger.info(f"处理完成! 结果已保存至: {output_dir}")
    print(f"处理摘要已保存至: {summary_path}")
    print(f"详细报告已保存至: {report_path}")


if __name__ == "__main__":
    # 示例参数
    video_path = "path/to/your/video.mp4"  # 替换为实际视频路径
    train_data_path = "data/train_data.npy"  # 替换为实际训练数据路径
    train_labels_path = "data/train_labels.npy"  # 替换为实际训练标签路径

    main(
        video_path=video_path,
        output_dir="results",
        train_data_path=train_data_path,
        train_labels_path=train_labels_path,
        threshold=0.7,
        visualize=True,
        detection_model="yolov5s",
        feature_model="resnet18",
        detect_classes=["person", "car", "truck"],  # 检测行人、汽车和卡车
        use_gpu=True,
        batch_size=1,
        save_interval=10
    )