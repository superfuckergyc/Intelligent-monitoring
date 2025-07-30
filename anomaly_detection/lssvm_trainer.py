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
                 C: float = 10.0,  # 正则化参数
                 test_size: float = 0.3,  # 测试集比例
                 random_state: int = 42,  # 随机种子
                 optimize_gamma: bool = True):  # 是否优化gamma参数
        """
        初始化LSSVM训练器

        Args:
            kernel_type: 核函数类型，支持 'linear', 'rbf', 'poly'
            gamma: RBF核参数，控制核函数的宽度
            C: 正则化参数，控制模型复杂度
            test_size: 测试集占比
            random_state: 随机种子，保证结果可复现
            optimize_gamma: 是否使用中位数法优化gamma参数
        """
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.C = C
        self.test_size = test_size
        self.random_state = random_state
        self.optimize_gamma = optimize_gamma

        # 初始化模型和标准化器
        self.model = None
        self.scaler = StandardScaler()

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"LSSVM训练器初始化完成 - "
                         f"kernel={kernel_type}, gamma={gamma}, C={C}, "
                         f"test_size={test_size}, optimize_gamma={optimize_gamma}")

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        训练LSSVM模型（对应文献公式11-13）

        Args:
            X: 特征矩阵，形状为(n_samples, n_features)
            y: 标签向量，形状为(n_samples,)
        """
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)

        # 如果使用RBF核且需要优化gamma
        if self.kernel_type == 'rbf' and self.optimize_gamma:
            rbf_kernel = RBFKernel(gamma=self.gamma)
            optimized_gamma = rbf_kernel.optimize_gamma(X_scaled)
            self.gamma = optimized_gamma
            self.logger.info(f"优化后的gamma值: {optimized_gamma}")

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=self.test_size,
            random_state=self.random_state, stratify=y
        )

        # 训练LSSVM模型
        self.model = LSSVMClassifier(
            kernel=self.kernel_type,
            gamma=self.gamma,
            C=self.C
        )
        self.model.fit(X_train, y_train)

        # 在测试集上评估模型
        y_pred = self.model.predict(X_test)
        self._evaluate_model(y_test, y_pred)

        self.logger.info(f"LSSVM模型训练完成，测试集准确率: {self.model.score(X_test, y_test):.4f}")

    def grid_search(self, X: np.ndarray, y: np.ndarray,
                    param_grid: Dict) -> None:
        """
        网格搜索优化LSSVM参数

        Args:
            X: 特征矩阵
            y: 标签向量
            param_grid: 参数网格，例如:
                {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10]}
        """
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)

        # 如果使用RBF核且需要优化gamma
        if self.kernel_type == 'rbf' and self.optimize_gamma:
            rbf_kernel = RBFKernel(gamma=self.gamma)
            optimized_gamma = rbf_kernel.optimize_gamma(X_scaled)
            self.gamma = optimized_gamma
            self.logger.info(f"优化后的gamma值: {optimized_gamma}")

            # 如果网格参数中包含gamma，则移除（已通过中位数法优化）
            if 'gamma' in param_grid:
                self.logger.info("使用中位数法优化gamma，忽略网格搜索中的gamma参数")
                param_grid = {k: v for k, v in param_grid.items() if k != 'gamma'}

        # 定义网格搜索
        grid_search = GridSearchCV(
            estimator=LSSVMClassifier(kernel=self.kernel_type, gamma=self.gamma),
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=2
        )

        # 执行网格搜索
        grid_search.fit(X_scaled, y)

        # 输出最佳参数
        self.logger.info(f"最佳参数: {grid_search.best_params_}")
        self.logger.info(f"最佳得分: {grid_search.best_score_:.4f}")

        # 使用最佳参数更新模型
        self.C = grid_search.best_params_['C']

        # 用最佳参数重新训练模型
        self.train(X, y)

    def _evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        评估模型性能

        Args:
            y_true: 真实标签
            y_pred: 预测标签
        """
        # 打印分类报告
        report = classification_report(y_true, y_pred)
        self.logger.info(f"分类报告:\n{report}")

        # 打印混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        self.logger.info(f"混淆矩阵:\n{cm}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用训练好的模型进行预测

        Args:
            X: 输入特征矩阵

        Returns:
            预测标签
        """
        if self.model is None:
            raise RuntimeError("模型未训练，请先调用train()方法")

        # 标准化特征
        X_scaled = self.scaler.transform(X)

        # 预测
        return self.model.predict(X_scaled)

    def save_model(self, path: str) -> None:
        """
        保存模型和标准化器到文件

        Args:
            path: 保存路径
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'kernel_type': self.kernel_type,
            'gamma': self.gamma,
            'C': self.C,
            'optimize_gamma': self.optimize_gamma
        }
        joblib.dump(model_data, path)
        self.logger.info(f"模型已保存至: {path}")

    @staticmethod
    def load_model(path: str) -> 'LSSVMTrainer':
        """
        从文件加载模型

        Args:
            path: 文件路径

        Returns:
            加载的LSSVMTrainer实例
        """
        model_data = joblib.load(path)

        trainer = LSSVMTrainer(
            kernel_type=model_data['kernel_type'],
            gamma=model_data['gamma'],
            C=model_data['C'],
            optimize_gamma=model_data.get('optimize_gamma', True)
        )
        trainer.model = model_data['model']
        trainer.scaler = model_data['scaler']

        return trainer


class LSSVMClassifier(BaseEstimator, ClassifierMixin):
    """
    自定义LSSVM分类器实现（基于scikit-learn接口）
    """

    def __init__(self, kernel: str = 'rbf', gamma: float = 1.0, C: float = 10.0):
        """
        初始化LSSVM分类器

        Args:
            kernel: 核函数类型
            gamma: RBF核参数
            C: 正则化参数
        """
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        self.alphas = None
        self.bias = None
        self.support_vectors = None
        self.support_labels = None

        # 初始化RBF核
        if kernel == 'rbf':
            self.rbf_kernel = RBFKernel(gamma=gamma)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LSSVMClassifier':
        """
        训练LSSVM模型（对应文献公式11-13）

        Args:
            X: 训练特征矩阵
            y: 训练标签向量

        Returns:
            训练好的模型
        """
        n_samples, n_features = X.shape

        # 保存支持向量和标签
        self.support_vectors = X
        self.support_labels = y

        # 构建核矩阵
        K = self._compute_kernel_matrix(X)

        # 构建并求解LSSVM方程组（文献公式12）
        # [0  y^T] [b]   [0]
        # [y  K+I/C] [α] = [1]
        O = np.ones((n_samples, 1))
        y_col = y.reshape(-1, 1)

        # 构建增广矩阵
        A = np.zeros((n_samples + 1, n_samples + 1))
        A[0, 1:] = y_col.T
        A[1:, 0] = y_col.flatten()
        A[1:, 1:] = K + np.eye(n_samples) / self.C

        # 构建右侧向量
        b = np.zeros(n_samples + 1)
        b[1:] = np.ones(n_samples)

        # 求解方程组
        solution = np.linalg.solve(A, b)

        # 提取偏置和拉格朗日乘子
        self.bias = solution[0]
        self.alphas = solution[1:] * y

        return self

    def _compute_kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """计算核矩阵"""
        if self.kernel == 'linear':
            return np.dot(X, X.T)

        elif self.kernel == 'rbf':
            # 使用3.2模块的RBF核实现
            return self.rbf_kernel.compute(X, X)

        else:
            raise ValueError(f"不支持的核函数类型: {self.kernel}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测新样本的标签

        Args:
            X: 待预测样本特征矩阵

        Returns:
            预测标签向量
        """
        if self.alphas is None or self.bias is None:
            raise RuntimeError("模型未训练，请先调用fit()方法")

        # 计算预测值（文献公式13）
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples)

        for i in range(n_samples):
            kernel_vals = np.array([self._compute_kernel(X[i], self.support_vectors[j])
                                    for j in range(len(self.support_vectors))])
            y_pred[i] = np.sum(self.alphas * self.support_labels * kernel_vals) + self.bias

        # 将预测值转换为类别标签（>0为1，<=0为-1）
        return np.sign(y_pred).astype(int)

    def _compute_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """计算两个样本之间的核函数值"""
        if self.kernel == 'linear':
            return np.dot(x1, x2)

        elif self.kernel == 'rbf':
            # 使用3.2模块的RBF核实现
            return self.rbf_kernel.compute(x1.reshape(1, -1), x2.reshape(1, -1))[0, 0]

        else:
            raise ValueError(f"不支持的核函数类型: {self.kernel}")

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算模型准确率"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

# 模块测试代码保持不变...