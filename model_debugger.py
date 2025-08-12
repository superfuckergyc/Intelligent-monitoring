import numpy as np
import os
import logging
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from anomaly_detection.lssvm_trainer import LSSVMTrainer

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


class ModelDebugger:
    def __init__(self,
                 output_dir: str = "debug_results",
                 random_seed: int = 42):
        """
        模型调试工具初始化

        Args:
            output_dir: 调试结果输出目录
            random_seed: 随机种子，保证结果可复现
        """
        self.output_dir = output_dir
        self.random_seed = random_seed

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

        # 配置日志
        self.logger = logging.getLogger("ModelDebugger")
        self.logger.setLevel(logging.INFO)
        log_file = os.path.join(output_dir, "logs", f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # 初始化训练器
        self.trainer = LSSVMTrainer()
        self.best_model_path = os.path.join(output_dir, "best_model.pkl")

        # 记录最佳指标
        self.best_metrics = {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0
        }

    def load_dataset(self,
                     train_data_path: str,
                     train_labels_path: str,
                     val_data_path: str,
                     val_labels_path: str) -> tuple:
        """
        加载并验证数据集

        Returns:
            (train_data, train_labels, val_data, val_labels)
        """
        try:
            # 加载训练集
            train_data = np.load(train_data_path)
            train_labels = np.load(train_labels_path)
            self.logger.info(f"训练数据加载成功: 样本数={len(train_data)}, 特征维度={train_data.shape[1]}")
            print(f"训练数据: 样本数={len(train_data)}, 特征维度={train_data.shape[1]}")

            # 加载验证集
            val_data = np.load(val_data_path)
            val_labels = np.load(val_labels_path)
            self.logger.info(f"验证数据加载成功: 样本数={len(val_data)}, 特征维度={val_data.shape[1]}")
            print(f"验证数据: 样本数={len(val_data)}, 特征维度={val_data.shape[1]}")

            # 检查特征维度一致性
            if train_data.shape[1] != val_data.shape[1]:
                raise ValueError(f"训练集与验证集特征维度不一致: {train_data.shape[1]} vs {val_data.shape[1]}")

            # 分析类别分布
            self._analyze_class_distribution(train_labels, "训练集")
            self._analyze_class_distribution(val_labels, "验证集")

            return train_data, train_labels, val_data, val_labels

        except Exception as e:
            self.logger.error(f"数据集加载失败: {str(e)}")
            raise

    def _analyze_class_distribution(self, labels: np.ndarray, dataset_name: str):
        """分析并记录类别分布"""
        unique, counts = np.unique(labels, return_counts=True)
        distribution = dict(zip(unique, counts))

        self.logger.info(f"{dataset_name}类别分布: {distribution}")
        print(f"{dataset_name}类别分布: {distribution}")

        # 计算异常样本比例
        if 1 in distribution:
            anomaly_ratio = distribution[1] / sum(counts)
            self.logger.info(f"{dataset_name}异常样本比例: {anomaly_ratio:.2%}")
            print(f"{dataset_name}异常样本比例: {anomaly_ratio:.2%}")

            # 检查类别是否极度不平衡
            if anomaly_ratio < 0.05:  # 异常样本占比小于5%
                self.logger.warning(f"{dataset_name}存在严重类别不平衡，可能影响模型性能")
                print(f"警告: {dataset_name}存在严重类别不平衡，可能影响模型性能")

    def train_and_evaluate(self,
                           train_data: np.ndarray,
                           train_labels: np.ndarray,
                           val_data: np.ndarray,
                           val_labels: np.ndarray,
                           params: dict = None) -> dict:
        """
        训练模型并评估性能

        Args:
            params: LSSVM模型参数

        Returns:
            评估指标字典
        """
        try:
            self.logger.info("开始模型训练...")
            print("\n开始模型训练...")

            # 如果提供了参数，则设置模型参数
            if params:
                for key, value in params.items():
                    setattr(self.trainer, key, value)
                    self.logger.info(f"设置模型参数: {key} = {value}")

            # 训练模型
            self.trainer.train(train_data, train_labels)

            # 评估训练集
            train_preds = self.trainer.predict(train_data)
            train_preds = [1 if p > 0 else 0 for p in train_preds]
            train_metrics = self._calculate_metrics(train_labels, train_preds, "训练集")

            # 评估验证集
            val_preds = self.trainer.predict(val_data)
            val_preds = [1 if p > 0 else 0 for p in val_preds]
            val_metrics = self._calculate_metrics(val_labels, val_preds, "验证集")

            # 生成混淆矩阵
            self._plot_confusion_matrix(val_labels, val_preds, "验证集混淆矩阵")

            # 生成分类报告
            self._generate_classification_report(val_labels, val_preds)

            # 保存性能更好的模型
            if val_metrics['f1'] > self.best_metrics['f1']:
                self.best_metrics = val_metrics
                self.trainer.save_model(self.best_model_path)
                self.logger.info(f"保存最佳模型至 {self.best_model_path} (F1: {val_metrics['f1']:.4f})")
                print(f"\n保存最佳模型至 {self.best_model_path} (F1: {val_metrics['f1']:.4f})")

            return {
                'train': train_metrics,
                'val': val_metrics
            }

        except Exception as e:
            self.logger.error(f"模型训练或评估失败: {str(e)}")
            raise

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, dataset_name: str) -> dict:
        """计算评估指标"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        # 处理二分类情况
        average = 'binary' if len(np.unique(y_true)) == 2 else 'weighted'

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
        }

        self.logger.info(f"{dataset_name}评估指标: "
                         f"准确率={metrics['accuracy']:.4f}, "
                         f"精确率={metrics['precision']:.4f}, "
                         f"召回率={metrics['recall']:.4f}, "
                         f"F1分数={metrics['f1']:.4f}")

        print(f"\n{dataset_name}评估指标:")
        print(f"准确率: {metrics['accuracy']:.4f}")
        print(f"精确率: {metrics['precision']:.4f}")
        print(f"召回率: {metrics['recall']:.4f}")
        print(f"F1分数: {metrics['f1']:.4f}")

        return metrics

    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, title: str):
        """绘制混淆矩阵并保存"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['正常', '异常'],
                    yticklabels=['正常', '异常'])
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title(title)

        plot_path = os.path.join(self.output_dir, "plots", f"{title.replace(' ', '_')}.png")
        plt.savefig(plot_path)
        plt.close()

        self.logger.info(f"混淆矩阵已保存至 {plot_path}")

    def _generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray):
        """生成详细分类报告"""
        report = classification_report(
            y_true, y_pred,
            target_names=['正常', '异常'],
            zero_division=0
        )

        self.logger.info(f"详细分类报告:\n{report}")
        print("\n详细分类报告:")
        print(report)

        # 保存报告
        report_path = os.path.join(self.output_dir, "classification_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

    def parameter_search(self,
                         train_data: np.ndarray,
                         train_labels: np.ndarray,
                         val_data: np.ndarray,
                         val_labels: np.ndarray,
                         param_grid: dict):
        """
        简单参数搜索功能，帮助找到较优参数

        Args:
            param_grid: 参数网格，例如 {'gamma': [0.1, 1, 10], 'C': [0.1, 1, 10]}
        """
        from itertools import product

        # 生成参数组合
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        param_combinations = list(product(*param_values))

        self.logger.info(f"开始参数搜索，共 {len(param_combinations)} 种组合")
        print(f"\n开始参数搜索，共 {len(param_combinations)} 种组合...")

        results = []

        # 遍历所有参数组合
        for i, params in enumerate(param_combinations):
            param_dict = dict(zip(param_names, params))
            print(f"\n参数组合 {i + 1}/{len(param_combinations)}: {param_dict}")

            # 训练并评估
            metrics = self.train_and_evaluate(
                train_data, train_labels,
                val_data, val_labels,
                param_dict
            )

            # 记录结果
            results.append({
                'params': param_dict,
                'train_metrics': metrics['train'],
                'val_metrics': metrics['val']
            })

        # 找出最佳参数组合
        best_idx = np.argmax([r['val_metrics']['f1'] for r in results])
        best_params = results[best_idx]['params']
        best_val_metrics = results[best_idx]['val_metrics']

        self.logger.info(f"最佳参数组合: {best_params}, 验证集F1: {best_val_metrics['f1']:.4f}")
        print(f"\n最佳参数组合: {best_params}")
        print(f"最佳验证集F1分数: {best_val_metrics['f1']:.4f}")

        # 保存参数搜索结果
        results_path = os.path.join(self.output_dir, "parameter_search_results.txt")
        with open(results_path, 'w') as f:
            f.write("参数搜索结果汇总\n")
            f.write("==================\n")
            for i, result in enumerate(results):
                f.write(f"\n组合 {i + 1}: {result['params']}\n")
                f.write(f"  训练集F1: {result['train_metrics']['f1']:.4f}\n")
                f.write(f"  验证集F1: {result['val_metrics']['f1']:.4f}\n")

            f.write("\n最佳组合:\n")
            f.write(f"参数: {best_params}\n")
            f.write(f"验证集指标: {best_val_metrics}\n")


if __name__ == "__main__":
    # 数据集路径
    train_data_path = "data/train_data.npy"
    train_labels_path = "data/train_labels.npy"
    val_data_path = "data/val_data.npy"
    val_labels_path = "data/val_labels.npy"

    # 初始化调试器
    debugger = ModelDebugger(output_dir="model_debug_results")

    try:
        # 加载数据集
        train_data, train_labels, val_data, val_labels = debugger.load_dataset(
            train_data_path, train_labels_path,
            val_data_path, val_labels_path
        )

        # 选项1: 使用默认参数训练并评估
        print("\n===== 使用默认参数训练 =====")
        debugger.train_and_evaluate(train_data, train_labels, val_data, val_labels)

        # 选项2: 进行参数搜索（根据LSSVM实际参数调整）
        print("\n===== 开始参数搜索 =====")
        param_grid = {
            'gamma': [0.01, 0.1, 1, 10, 100],  # 核函数参数
            'C': [0.1, 1, 10, 100]  # 正则化参数
        }
        debugger.parameter_search(train_data, train_labels, val_data, val_labels, param_grid)

        print("\n调试完成! 最佳模型已保存，可用于主程序")

    except Exception as e:
        print(f"调试过程出错: {str(e)}")
