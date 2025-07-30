import numpy as np
import logging
from typing import Tuple, List, Dict, Optional
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

class KKTOptimizer:
    """库恩塔克条件优化子模块：实现LSSVM的KKT条件优化（对应文献3.3节）"""

    def __init__(self,
                 C: float = 10.0,  # 正则化参数
                 kernel_type: str = 'rbf',  # 核函数类型
                 gamma: float = 1.0,  # RBF核参数
                 max_iter: int = 100,  # 最大迭代次数
                 tol: float = 1e-6):  # 收敛容差
        """
        初始化KKT条件优化器

        Args:
            C: 正则化参数
            kernel_type: 核函数类型
            gamma: RBF核参数
            max_iter: 最大迭代次数
            tol: 收敛容差
        """
        self.C = C
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"KKT条件优化器初始化完成 - "
                         f"C={C}, kernel={kernel_type}, gamma={gamma}, "
                         f"max_iter={max_iter}, tol={tol}")

    def optimize(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        基于KKT条件优化LSSVM模型参数（对应文献公式14-16）

        Args:
            X: 特征矩阵，形状为(n_samples, n_features)
            y: 标签向量，形状为(n_samples,)

        Returns:
            alphas: 拉格朗日乘子
            bias: 偏置项
        """
        n_samples, n_features = X.shape

        # 1. 计算核矩阵
        K = self._compute_kernel_matrix(X)

        # 2. 定义优化目标函数（文献公式14）
        def objective_function(alphas: np.ndarray) -> float:
            """优化目标函数：最小化结构风险"""
            return 0.5 * np.dot(alphas, np.dot(K, alphas)) - np.sum(alphas)

        # 3. 定义约束条件（文献公式15）
        constraints = [
            {
                'type': 'eq',
                'fun': lambda alphas: np.dot(alphas, y)  # y^T * α = 0
            }
        ]

        # 4. 定义边界条件（文献公式16）
        bounds = [(0, self.C) for _ in range(n_samples)]  # 0 ≤ α_i ≤ C

        # 5. 初始化拉格朗日乘子
        initial_alphas = np.zeros(n_samples)

        # 6. 使用SLSQP算法求解优化问题
        self.logger.info("开始KKT条件优化...")
        result = minimize(
            objective_function,
            initial_alphas,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.max_iter, 'disp': True}
        )

        # 7. 提取优化结果
        alphas = result.x
        bias = self._compute_bias(X, y, alphas, K)

        self.logger.info(f"KKT条件优化完成 - 迭代次数: {result.nit}, 目标函数值: {result.fun:.6f}")
        return alphas, bias

    def _compute_kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        计算核矩阵

        Args:
            X: 特征矩阵

        Returns:
            核矩阵
        """
        if self.kernel_type == 'linear':
            return np.dot(X, X.T)

        elif self.kernel_type == 'rbf':
            # 计算RBF核矩阵
            pairwise_dists = np.sum(X ** 2, axis=1).reshape(-1, 1) + np.sum(X ** 2, axis=1) - 2 * np.dot(X, X.T)
            return np.exp(-self.gamma * pairwise_dists)

        else:
            raise ValueError(f"不支持的核函数类型: {self.kernel_type}")

    def _compute_bias(self, X: np.ndarray, y: np.ndarray, alphas: np.ndarray, K: np.ndarray) -> float:
        """
        计算偏置项（文献公式17）

        Args:
            X: 特征矩阵
            y: 标签向量
            alphas: 拉格朗日乘子
            K: 核矩阵

        Returns:
            偏置项
        """
        # 寻找支持向量（0 < α_i < C）
        support_indices = np.where((alphas > self.tol) & (alphas < self.C - self.tol))[0]

        if len(support_indices) == 0:
            # 如果没有严格在边界内的支持向量，使用所有非零α_i
            support_indices = np.where(alphas > self.tol)[0]

        # 计算偏置（文献公式17）
        bias_sum = 0.0
        for i in support_indices:
            bias_sum += y[i] - np.sum(alphas * y * K[:, i])

        return bias_sum / len(support_indices)

    def compute_kkt_violation(self, X: np.ndarray, y: np.ndarray, alphas: np.ndarray, bias: float) -> float:
        """
        计算KKT条件违反程度（对应文献3.3节评估指标）

        Args:
            X: 特征矩阵
            y: 标签向量
            alphas: 拉格朗日乘子
            bias: 偏置项

        Returns:
            KKT条件违反程度
        """
        n_samples = X.shape[0]
        K = self._compute_kernel_matrix(X)

        # 计算决策函数值
        f = np.zeros(n_samples)
        for i in range(n_samples):
            f[i] = np.sum(alphas * y * K[:, i]) + bias

        # 计算KKT条件违反程度（文献公式18）
        violation = 0.0
        for i in range(n_samples):
            # 计算每个样本的KKT条件违反程度
            if alphas[i] < self.tol:  # α_i = 0
                violation += max(0, 1 - y[i] * f[i]) ** 2
            elif alphas[i] > self.C - self.tol:  # α_i = C
                violation += max(0, y[i] * f[i] - 1) ** 2
            else:  # 0 < α_i < C
                violation += (1 - y[i] * f[i]) ** 2

        return violation / n_samples


class OptimizedLSSVMClassifier(BaseEstimator, ClassifierMixin):
    """
    基于KKT条件优化的LSSVM分类器
    """

    def __init__(self,
                 C: float = 10.0,
                 kernel_type: str = 'rbf',
                 gamma: float = 1.0,
                 max_iter: int = 100,
                 tol: float = 1e-6):
        """
        初始化优化的LSSVM分类器

        Args:
            C: 正则化参数
            kernel_type: 核函数类型
            gamma: RBF核参数
            max_iter: 最大迭代次数
            tol: 收敛容差
        """
        self.C = C
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol

        self.alphas = None
        self.bias = None
        self.support_vectors = None
        self.support_labels = None

        # 初始化KKT优化器
        self.kkt_optimizer = KKTOptimizer(
            C=C,
            kernel_type=kernel_type,
            gamma=gamma,
            max_iter=max_iter,
            tol=tol
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'OptimizedLSSVMClassifier':
        """
        训练优化的LSSVM模型

        Args:
            X: 训练特征矩阵
            y: 训练标签向量

        Returns:
            训练好的模型
        """
        # 保存支持向量和标签
        self.support_vectors = X
        self.support_labels = y

        # 使用KKT条件优化模型参数
        self.alphas, self.bias = self.kkt_optimizer.optimize(X, y)

        # 计算KKT条件违反程度
        kkt_violation = self.kkt_optimizer.compute_kkt_violation(
            X, y, self.alphas, self.bias
        )
        self.logger.info(f"KKT条件违反程度: {kkt_violation:.8f}")

        return self

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

        # 计算预测值
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples)

        for i in range(n_samples):
            # 计算核函数值
            kernel_vals = np.zeros(len(self.support_vectors))
            for j in range(len(self.support_vectors)):
                if self.kernel_type == 'linear':
                    kernel_vals[j] = np.dot(X[i], self.support_vectors[j])
                elif self.kernel_type == 'rbf':
                    dist = np.linalg.norm(X[i] - self.support_vectors[j])
                    kernel_vals[j] = np.exp(-self.gamma * dist ** 2)
                else:
                    raise ValueError(f"不支持的核函数类型: {self.kernel_type}")

            # 计算预测值
            y_pred[i] = np.sum(self.alphas * self.support_labels * kernel_vals) + self.bias

        # 将预测值转换为类别标签
        return np.sign(y_pred).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算模型准确率"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


# 模块测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 1. 生成示例数据
    np.random.seed(42)  # 固定随机种子，确保结果可复现
    n_samples = 200
    X = np.vstack([
        np.random.randn(n_samples // 2, 2) + np.array([2, 2]),  # 正类
        np.random.randn(n_samples // 2, 2) - np.array([2, 2])  # 负类
    ])
    y = np.hstack([np.ones(n_samples // 2), -np.ones(n_samples // 2)])

    # 2. 初始化KKT优化器
    optimizer = KKTOptimizer(
        C=10.0,
        kernel_type='rbf',
        gamma=0.1,
        max_iter=100,
        tol=1e-6
    )

    # 3. 使用KKT条件优化模型
    logging.info("开始KKT条件优化...")
    alphas, bias = optimizer.optimize(X, y)

    # 4. 计算KKT条件违反程度
    kkt_violation = optimizer.compute_kkt_violation(X, y, alphas, bias)
    logging.info(f"KKT条件违反程度: {kkt_violation:.8f}")

    # 5. 使用优化的LSSVM分类器
    clf = OptimizedLSSVMClassifier(
        C=10.0,
        kernel_type='rbf',
        gamma=0.1
    )

    # 6. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 7. 训练模型并评估
    clf.fit(X_train, y_train)
    train_accuracy = clf.score(X_train, y_train)
    test_accuracy = clf.score(X_test, y_test)

    logging.info(f"训练集准确率: {train_accuracy:.4f}")
    logging.info(f"测试集准确率: {test_accuracy:.4f}")

    # 8. 可视化决策边界（需要matplotlib）
    try:
        import matplotlib.pyplot as plt

        # 生成网格点
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))

        # 预测网格点的类别
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # 绘制决策边界和样本点
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(X[y == 1, 0], X[y == 1, 1], c='b', marker='o', label='Positive')
        plt.scatter(X[y == -1, 0], X[y == -1, 1], c='r', marker='x', label='Negative')
        plt.title('Optimized LSSVM Decision Boundary')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.savefig('lssvm_decision_boundary.png')
        logging.info("决策边界可视化已保存")

    except ImportError:
        logging.warning("无法导入matplotlib，跳过可视化")