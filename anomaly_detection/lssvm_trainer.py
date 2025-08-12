import numpy as np
import logging
from typing import Tuple, List, Dict, Optional
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 导入3.2模块的RBF核函数实现
from anomaly_detection.rbf_kernel import RBFKernel


class LSSVMTrainer:
    """LSSVM模型训练子模块：实现最小二乘支持向量机的训练与优化（对应文献3.1节）"""

    def __init__(self,
                 kernel_type: str = 'rbf',  # 核函数类型
                 gamma: float = 1.0,  # RBF核参数
                 C: float = 100.0,  # 正则化参数（默认值调大，减少偏向性）
                 test_size: float = 0.3,  # 内部验证集比例
                 random_state: int = 42,  # 随机种子
                 optimize_gamma: bool = True,  # 是否优化gamma参数
                 class_weight: Optional[str] = 'balanced'):  # 类别权重处理
        """
        初始化LSSVM训练器（新增类别权重参数）
        """
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.C = C
        self.test_size = test_size
        self.random_state = random_state
        self.optimize_gamma = optimize_gamma
        self.class_weight = class_weight  # 新增：类别权重参数

        # 初始化模型和标准化器
        self.model = None
        self.scaler = StandardScaler()

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"LSSVM训练器初始化完成 - "
                         f"kernel={kernel_type}, gamma={gamma}, C={C}, "
                         f"class_weight={class_weight}, test_size={test_size}")

    def train(self, X: np.ndarray, y: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> None:
        """训练LSSVM模型（支持外部验证集）"""
        # 1. 标签转换：将0/1转换为LSSVM标准的-1/1
        y = np.where(y == 0, -1, 1)  # 0→-1（正常），1保持1（异常）
        if y_val is not None:
            y_val = np.where(y_val == 0, -1, 1)

        # 2. 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)

        # 3. 优化RBF核gamma参数
        if self.kernel_type == 'rbf' and self.optimize_gamma:
            rbf_kernel = RBFKernel(gamma=self.gamma)
            self.gamma = rbf_kernel.optimize_gamma(X_scaled)
            self.logger.info(f"优化后的gamma值: {self.gamma}")

        # 4. 划分内部验证集（仅当无外部验证集时）
        if X_val is None or y_val is None:
            X_train, X_val_scaled, y_train, y_val = train_test_split(
                X_scaled, y, test_size=self.test_size,
                random_state=self.random_state, stratify=y
            )
            self.logger.warning("未提供外部验证集，使用内部划分的验证集")
        else:
            X_train, y_train = X_scaled, y  # 全部训练数据用于训练

        # 5. 训练LSSVM模型（传入类别权重参数）
        self.model = LSSVMClassifier(
            kernel=self.kernel_type,
            gamma=self.gamma,
            C=self.C,
            class_weight=self.class_weight
        )
        self.model.fit(X_train, y_train)

        # 6. 在验证集上评估模型
        y_pred = self.model.predict(X_val_scaled)
        self._evaluate_model(y_val, y_pred)

        self.logger.info(f"LSSVM模型训练完成，验证集准确率: {self.model.score(X_val_scaled, y_val):.4f}")

    def grid_search(self, X: np.ndarray, y: np.ndarray,
                    param_grid: Dict,
                    X_val: Optional[np.ndarray] = None,
                    y_val: Optional[np.ndarray] = None) -> None:
        """网格搜索优化LSSVM参数（修复数据合并问题）"""
        # 标签转换：0→-1，1→1
        y = np.where(y == 0, -1, 1)
        if y_val is not None:
            y_val = np.where(y_val == 0, -1, 1)

        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None

        # 优化gamma（如启用）
        if self.kernel_type == 'rbf' and self.optimize_gamma:
            rbf_kernel = RBFKernel(gamma=self.gamma)
            self.gamma = rbf_kernel.optimize_gamma(X_scaled)
            self.logger.info(f"优化后的gamma值: {self.gamma}")
            # 移除网格中的gamma参数（已优化）
            param_grid = {k: v for k, v in param_grid.items() if k != 'gamma'}

        # 关键修复：仅用训练集做网格搜索（不合并验证集）
        grid_search = GridSearchCV(
            estimator=LSSVMClassifier(
                kernel=self.kernel_type,
                gamma=self.gamma,
                class_weight=self.class_weight
            ),
            param_grid=param_grid,
            cv=5,
            scoring='f1_macro',  # 改用F1宏平均作为评分（更关注少数类）
            n_jobs=-1,
            verbose=1
        )

        # 执行网格搜索（仅用训练集）
        grid_search.fit(X_scaled, y)

        # 更新最佳参数并重新训练
        self.C = grid_search.best_params_['C']
        self.logger.info(f"最佳参数: {grid_search.best_params_}, 最佳交叉验证F1: {grid_search.best_score_:.4f}")

        # 用最佳参数重新训练（使用全部训练数据和原始验证集）
        self.train(X, np.where(y == -1, 0, 1), X_val, np.where(y_val == -1, 0, 1) if y_val is not None else None)

    def _evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """评估模型性能（支持-1/1标签）"""
        # 转换回0/1标签用于报告（更直观）
        y_true_01 = np.where(y_true == -1, 0, 1)
        y_pred_01 = np.where(y_pred == -1, 0, 1)

        # 打印分类报告
        report = classification_report(y_true_01, y_pred_01)
        self.logger.info(f"分类报告:\n{report}")

        # 打印混淆矩阵
        cm = confusion_matrix(y_true_01, y_pred_01)
        self.logger.info(f"混淆矩阵:\n{cm}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测标签（返回0/1）"""
        if self.model is None:
            raise RuntimeError("模型未训练，请先调用train()方法")

        # 标准化特征
        X_scaled = self.scaler.transform(X)

        # 预测（模型返回-1/1，转换为0/1）
        y_pred = self.model.predict(X_scaled)
        return np.where(y_pred == -1, 0, 1)

    def save_model(self, path: str) -> None:
        """保存模型（包含所有必要参数）"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'kernel_type': self.kernel_type,
            'gamma': self.gamma,
            'C': self.C,
            'optimize_gamma': self.optimize_gamma,
            'class_weight': self.class_weight  # 新增：保存类别权重参数
        }
        joblib.dump(model_data, path)
        self.logger.info(f"模型已保存至: {path}")

    @staticmethod
    def load_model(path: str) -> 'LSSVMTrainer':
        """加载模型（兼容新旧版本）"""
        model_data = joblib.load(path)

        # 处理可能的旧版本数据
        trainer = LSSVMTrainer(
            kernel_type=model_data.get('kernel_type', 'rbf'),
            gamma=model_data.get('gamma', 1.0),
            C=model_data.get('C', 100.0),  # 匹配新默认值
            optimize_gamma=model_data.get('optimize_gamma', True),
            class_weight=model_data.get('class_weight', 'balanced')  # 新增：加载类别权重
        )
        trainer.model = model_data['model']
        trainer.scaler = model_data['scaler']

        return trainer


class LSSVMClassifier(BaseEstimator, ClassifierMixin):
    """
    自定义LSSVM分类器实现（优化版，支持类别权重）
    """

    def __init__(self,
                 kernel: str = 'rbf',
                 gamma: float = 1.0,
                 C: float = 100.0,
                 class_weight: Optional[str] = 'balanced'):  # 新增类别权重参数
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        self.class_weight = class_weight  # 类别权重（'balanced'或None）
        self.alphas = None  # 拉格朗日乘子
        self.bias = None  # 偏置项
        self.support_vectors = None  # 支持向量（训练样本）
        self.support_labels = None  # 支持向量标签（-1/1）
        self.logger = logging.getLogger(__name__)  # 新增日志实例

        # 初始化核函数
        if kernel == 'rbf':
            self.rbf_kernel = RBFKernel(gamma=gamma)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LSSVMClassifier':
        """
        训练LSSVM模型（支持类别权重，标签需为-1/1）
        """
        n_samples = X.shape[0]
        self.support_vectors = X
        self.support_labels = y  # 已转换为-1/1（正常/异常）

        # 1. 计算类别权重（解决不平衡问题）
        if self.class_weight == 'balanced':
            # 自动计算权重：与类别样本数成反比
            n_neg = np.sum(y == -1)  # 正常样本数
            n_pos = np.sum(y == 1)  # 异常样本数

            # 防止除以零（如果某类样本数为0）
            weight_neg = n_samples / (2 * n_neg) if n_neg > 0 else 1.0
            weight_pos = n_samples / (2 * n_pos) if n_pos > 0 else 1.0

            # 生成权重向量（每个样本对应自身类别的权重）
            weights = np.where(y == -1, weight_neg, weight_pos)
            self.logger.info(f"类别权重计算完成 - 正常样本权重: {weight_neg:.2f}, 异常样本权重: {weight_pos:.2f}")
        else:
            # 无权重（所有样本权重为1）
            weights = np.ones(n_samples)

        # 2. 计算核矩阵（带权重优化）
        K = self._compute_kernel_matrix(X, X)
        K_weighted = K * np.outer(weights, weights)  # 应用样本权重到核矩阵

        # 3. 构建LSSVM方程组（文献公式12，使用加权核矩阵）
        A = np.block([
            [0, y.reshape(1, -1)],
            [y.reshape(-1, 1), K_weighted + np.eye(n_samples) / self.C]  # 加权核矩阵+正则项
        ])
        b = np.hstack([0, np.ones(n_samples)])  # 右侧向量

        # 4. 求解方程组（增强稳定性）
        try:
            solution = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # 处理奇异矩阵：使用最小二乘近似
            self.logger.warning("核矩阵可能奇异，使用最小二乘求解以增强稳定性")
            solution, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        # 5. 提取参数
        self.bias = solution[0]
        self.alphas = solution[1:] * y  # 乘标签（文献公式推导结果）

        return self

    def _compute_kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """计算核矩阵（矩阵运算优化）"""
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel == 'rbf':
            return self.rbf_kernel.compute(X1, X2)  # 调用RBF核的矩阵实现
        else:
            raise ValueError(f"不支持的核函数类型: {self.kernel}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测标签（返回-1/1）"""
        if self.alphas is None or self.bias is None:
            raise RuntimeError("模型未训练，请先调用fit()方法")

        # 矩阵运算优化：批量计算核函数值
        K = self._compute_kernel_matrix(X, self.support_vectors)  # (n_samples, n_support)
        y_pred = np.dot(K, self.alphas * self.support_labels) + self.bias  # 向量运算

        return np.sign(y_pred).astype(int)  # 符号函数分类

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算准确率（支持-1/1标签）"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
