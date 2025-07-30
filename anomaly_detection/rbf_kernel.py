import numpy as np
import logging
from typing import Tuple, List, Dict, Optional


class RBFKernel:
    """径向基核函数实现（对应文献公式9）"""

    def __init__(self, gamma: float = 1.0):
        """
        初始化RBF核函数

        Args:
            gamma: 核函数参数，控制函数的宽度
        """
        self.gamma = gamma
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"RBF核函数初始化完成 - gamma={gamma}")

    def compute(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        计算两个输入矩阵之间的RBF核函数值（对应文献公式9）

        Args:
            X1: 输入特征矩阵1，形状为(n_samples1, n_features)
            X2: 输入特征矩阵2，形状为(n_samples2, n_features)

        Returns:
            核矩阵，形状为(n_samples1, n_samples2)
        """
        # 计算样本间的欧氏距离平方
        pairwise_dists = self._compute_squared_euclidean_distance(X1, X2)

        # 计算RBF核（文献公式9）: K(x_i, x_j) = exp(-γ||x_i - x_j||²)
        return np.exp(-self.gamma * pairwise_dists)

    def _compute_squared_euclidean_distance(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        高效计算样本间的欧氏距离平方

        Args:
            X1: 输入特征矩阵1
            X2: 输入特征矩阵2

        Returns:
            距离矩阵，形状为(n_samples1, n_samples2)
        """
        # 使用矩阵运算高效计算欧氏距离平方
        # ||x_i - x_j||² = ||x_i||² + ||x_j||² - 2<x_i, x_j>
        X1_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1)  # n_samples1 x 1
        X2_sq = np.sum(X2 ** 2, axis=1).reshape(1, -1)  # 1 x n_samples2
        dot_product = np.dot(X1, X2.T)  # n_samples1 x n_samples2

        return X1_sq + X2_sq - 2 * dot_product

    def optimize_gamma(self, X: np.ndarray, method: str = 'median') -> float:
        """
        优化gamma参数（对应文献3.2节参数优化）

        Args:
            X: 输入特征矩阵
            method: 优化方法，支持 'median'（中位数法）

        Returns:
            优化后的gamma值
        """
        if method == 'median':
            # 计算所有样本对之间的欧氏距离
            pairwise_dists = self._compute_squared_euclidean_distance(X, X)

            # 提取非对角元素（避免计算自身距离）
            off_diag_indices = np.where(~np.eye(pairwise_dists.shape[0], dtype=bool))
            distances = pairwise_dists[off_diag_indices]

            # 计算中位数
            median_distance = np.median(np.sqrt(distances))

            # 根据中位数设置gamma: γ = 1/median²
            optimized_gamma = 1.0 / (median_distance ** 2)

            self.logger.info(f"使用中位数法优化gamma: {self.gamma} → {optimized_gamma}")
            self.gamma = optimized_gamma
            return optimized_gamma

        else:
            raise ValueError(f"不支持的gamma优化方法: {method}")

    def visualize_kernel(self, X: np.ndarray, save_path: Optional[str] = None) -> None:
        """
        可视化核矩阵

        Args:
            X: 输入特征矩阵
            save_path: 保存路径（None则显示图像）
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # 计算核矩阵
            kernel_matrix = self.compute(X, X)

            # 绘制热图
            plt.figure(figsize=(10, 8))
            sns.heatmap(kernel_matrix, cmap='viridis')
            plt.title(f'RBF核矩阵 (γ={self.gamma})')
            plt.xlabel('样本索引')
            plt.ylabel('样本索引')

            if save_path:
                plt.savefig(save_path)
                self.logger.info(f"核矩阵可视化已保存至: {save_path}")
            else:
                plt.show()

        except ImportError:
            self.logger.warning("无法导入matplotlib或seaborn，跳过可视化")


# 模块测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 1. 生成示例数据
    np.random.seed(42)  # 固定随机种子，确保结果可复现
    X = np.random.randn(100, 10)  # 100个样本，每个样本10个特征

    # 2. 初始化RBF核
    rbf = RBFKernel(gamma=0.5)

    # 3. 计算核矩阵
    kernel_matrix = rbf.compute(X, X)
    logging.info(f"核矩阵形状: {kernel_matrix.shape}")

    # 4. 优化gamma参数
    optimized_gamma = rbf.optimize_gamma(X, method='median')
    logging.info(f"优化后的gamma: {optimized_gamma}")

    # 5. 使用优化后的gamma重新计算核矩阵
    optimized_kernel_matrix = rbf.compute(X, X)

    # 6. 可视化核矩阵（需要matplotlib和seaborn）
    try:
        rbf.visualize_kernel(X, "rbf_kernel_matrix.png")
        logging.info("核矩阵可视化已保存")
    except Exception as e:
        logging.warning(f"可视化失败: {str(e)}")

    # 7. 测试不同gamma值对核矩阵的影响
    test_gammas = [0.1, 1.0, 10.0]
    for gamma in test_gammas:
        rbf.gamma = gamma
        k_matrix = rbf.compute(X[:20], X[:20])  # 使用前20个样本

        # 计算核矩阵的均值和标准差，用于评估核函数的"平滑度"
        mean_k = np.mean(k_matrix)
        std_k = np.std(k_matrix)

        logging.info(f"gamma={gamma:.1f}: 核矩阵均值={mean_k:.4f}, 标准差={std_k:.4f}")