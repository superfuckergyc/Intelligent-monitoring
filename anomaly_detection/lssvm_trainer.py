import numpy as np
import logging
from typing import Dict, Optional
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 导入RBF核函数实现
from anomaly_detection.rbf_kernel import RBFKernel


class LSSVMTrainer:
    """LSSVM模型训练器"""

    def __init__(self,
                 kernel_type: str = 'rbf',
                 gamma: float = 1.0,
                 C: float = 100.0,
                 test_size: float = 0.3,
                 random_state: int = 42,
                 optimize_gamma: bool = True,
                 class_weight: Optional[str] = 'balanced'):
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.C = C
        self.test_size = test_size
        self.random_state = random_state
        self.optimize_gamma = optimize_gamma
        self.class_weight = class_weight
        self.model = None
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"LSSVM训练器初始化完成 - kernel={kernel_type}, gamma={gamma}, C={C}, "
            f"class_weight={class_weight}, test_size={test_size}"
        )

    @staticmethod
    def _to_internal_labels(y: np.ndarray) -> np.ndarray:
        """将外部0/1标签转换为内部-1/1"""
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("输入标签必须为0或1")
        return np.where(y == 0, -1, 1)

    @staticmethod
    def _to_external_labels(y: np.ndarray) -> np.ndarray:
        """将内部-1/1标签转换为外部0/1"""
        return np.where(y == -1, 0, 1)

    def train(self, X: np.ndarray, y: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> None:
        """训练LSSVM模型"""
        # 标签转换
        y_internal = self._to_internal_labels(y)
        y_val_internal = self._to_internal_labels(y_val) if y_val is not None else None

        X_train = X
        X_val_scaled = X_val if X_val is not None else None

        # gamma优化
        if self.kernel_type == 'rbf' and self.optimize_gamma and (X_val is None):
            rbf_kernel = RBFKernel(gamma=self.gamma)
            self.gamma = rbf_kernel.optimize_gamma(X_train)
            self.logger.info(f"优化后的gamma值: {self.gamma}")

        # 内部分割验证集
        if X_val is None or y_val is None:
            X_train, X_val_scaled, y_internal, y_val_internal = train_test_split(
                X_train, y_internal, test_size=self.test_size,
                random_state=self.random_state, stratify=y_internal
            )
            self.logger.warning("未提供外部验证集，使用内部划分的验证集")

        # 新增：打印训练集和验证集的特征统计值，验证尺度是否一致
        print(f"训练集特征 - 均值: {X_train.mean():.4f}, 标准差: {X_train.std():.4f}, "
                         f"最小值: {X_train.min():.4f}, 最大值: {X_train.max():.4f}")
        if X_val_scaled is not None:
            print(f"验证集特征 - 均值: {X_val_scaled.mean():.4f}, 标准差: {X_val_scaled.std():.4f}, "
                             f"最小值: {X_val_scaled.min():.4f}, 最大值: {X_val_scaled.max():.4f}")

        # 模型训练
        self.model = LSSVMClassifier(
            kernel=self.kernel_type,
            gamma=self.gamma,
            C=self.C,
            class_weight=self.class_weight
        )
        self.model.fit(X_train, y_internal)

        # 验证集评估
        y_pred_internal = self.model.predict(X_val_scaled)
        self._evaluate_model(y_val_internal, y_pred_internal)
        self.logger.info(f"LSSVM模型训练完成，验证集准确率: {self.model.score(X_val_scaled, y_val_internal):.4f}")

    def grid_search(self, X: np.ndarray, y: np.ndarray,
                    param_grid: Dict,
                    X_val: Optional[np.ndarray] = None,
                    y_val: Optional[np.ndarray] = None) -> None:
        """网格搜索优化参数"""
        y_internal = self._to_internal_labels(y)
        y_val_internal = self._to_internal_labels(y_val) if y_val is not None else None

        X_scaled = X

        # gamma优化（仅当用户未在param_grid中提供）
        if self.kernel_type == 'rbf' and self.optimize_gamma and 'gamma' not in param_grid:
            rbf_kernel = RBFKernel(gamma=self.gamma)
            self.gamma = rbf_kernel.optimize_gamma(X_scaled)
            self.logger.info(f"优化后的gamma值: {self.gamma}")

        grid_search = GridSearchCV(
            estimator=LSSVMClassifier(
                kernel=self.kernel_type,
                gamma=self.gamma,
                class_weight=self.class_weight
            ),
            param_grid=param_grid,
            cv=5,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_scaled, y_internal)

        self.C = grid_search.best_params_.get('C', self.C)
        self.gamma = grid_search.best_params_.get('gamma', self.gamma)
        self.logger.info(f"最佳参数: {grid_search.best_params_}, 最佳交叉验证F1: {grid_search.best_score_:.4f}")

        # 用最佳参数重新训练
        self.train(X, y, X_val, y_val)

    def _evaluate_model(self, y_true_internal: np.ndarray, y_pred_internal: np.ndarray) -> None:
        """性能评估"""
        y_true_01 = self._to_external_labels(y_true_internal)
        y_pred_01 = self._to_external_labels(y_pred_internal)
        report = classification_report(y_true_01, y_pred_01)
        cm = confusion_matrix(y_true_01, y_pred_01)
        self.logger.info(f"分类报告:\n{report}")
        self.logger.info(f"混淆矩阵:\n{cm}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测0/1标签"""
        if self.model is None:
            raise RuntimeError("模型未训练")
        y_pred_internal = self.model.predict(X)
        return self._to_external_labels(y_pred_internal)

    def save_model(self, path: str) -> None:
        """保存模型"""
        model_data = {
            'model': self.model,
            'kernel_type': self.kernel_type,
            'gamma': self.gamma,
            'C': self.C,
            'optimize_gamma': self.optimize_gamma,
            'class_weight': self.class_weight
        }
        joblib.dump(model_data, path)
        self.logger.info(f"模型已保存至: {path}")

    @staticmethod
    def load_model(path: str) -> 'LSSVMTrainer':
        """加载模型"""
        model_data = joblib.load(path)
        trainer = LSSVMTrainer(
            kernel_type=model_data.get('kernel_type', 'rbf'),
            gamma=model_data.get('gamma', 1.0),
            C=model_data.get('C', 100.0),
            optimize_gamma=model_data.get('optimize_gamma', True),
            class_weight=model_data.get('class_weight', 'balanced')
        )
        trainer.model = model_data['model']
        return trainer


class LSSVMClassifier(BaseEstimator, ClassifierMixin):
    """自定义LSSVM分类器"""

    def __init__(self,
                 kernel: str = 'rbf',
                 gamma: float = 1.0,
                 C: float = 100.0,
                 class_weight: Optional[str] = 'balanced'):
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        self.class_weight = class_weight
        self.alphas = None
        self.bias = None
        self.support_vectors = None
        self.support_labels = None
        self.logger = logging.getLogger(__name__)
        if kernel == 'rbf':
            self.rbf_kernel = RBFKernel(gamma=gamma)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LSSVMClassifier':
        """训练LSSVM（y需为-1/1）"""
        n_samples = X.shape[0]
        self.support_vectors = X
        self.support_labels = y

        # 1. 计算类别权重（解决不平衡问题）
        if self.class_weight == 'balanced':
            n_neg = np.sum(y == -1)  # 正常样本数
            n_pos = np.sum(y == 1)  # 异常样本数
            weight_neg = n_samples / (2 * n_neg) if n_neg > 0 else 1.0
            weight_pos = n_samples / (2 * n_pos) if n_pos > 0 else 1.0
            weights = np.where(y == -1, weight_neg, weight_pos)
            self.logger.info(f"类别权重 - 正常: {weight_neg:.2f}, 异常: {weight_pos:.2f}")
        else:
            weights = np.ones(n_samples)

        # 2. 计算核矩阵并应用权重（核心修复1：恢复权重应用）
        K = self._compute_kernel_matrix(X, X)
        K_weighted = K * np.outer(weights, weights)  # 权重矩阵外积应用到核矩阵

        # 3. 构建LSSVM方程组（核心修复2：目标向量恢复为全1）
        ridge = 1e-8  # 保持数值稳定性
        A = np.block([
            [np.zeros((1, 1)), y.reshape(1, -1)],
            [y.reshape(-1, 1), K_weighted + np.eye(n_samples) / self.C + ridge * np.eye(n_samples)]
        ])
        b = np.hstack([0, np.ones(n_samples)])  # 目标向量为全1（符合LSSVM数学定义）

        # 4. 求解方程组
        try:
            solution = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            self.logger.warning("核矩阵奇异，使用最小二乘近似")
            solution, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        self.bias = solution[0]
        self.alphas = solution[1:]   # 乘标签（保持正确）
        # 在fit方法末尾添加
        print(f"模型偏置项bias: {self.bias:.4f}")
        return self

    def _compute_kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel == 'rbf':
            return self.rbf_kernel.compute(X1, X2)
        else:
            raise ValueError(f"不支持的核函数类型: {self.kernel}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.alphas is None or self.bias is None:
            raise RuntimeError("模型未训练")

        K = self._compute_kernel_matrix(X, self.support_vectors)
        # 打印核矩阵的统计值（检查是否异常）
        print(f"核矩阵形状: {K.shape}, 均值: {K.mean():.4f}, 最小值: {K.min():.4f}, 最大值: {K.max():.4f}")

        # 打印alphas和支持向量标签的乘积
        alpha_y = self.alphas * self.support_labels
        print(f"alpha·y 均值: {alpha_y.mean():.4f}, 符号分布: 正={np.sum(alpha_y > 0)}, 负={np.sum(alpha_y < 0)}")

        # 打印决策函数中间结果
        y_pred = np.dot(K, alpha_y) + self.bias
        print(f"决策函数输出: 均值={y_pred.mean():.4f}, 全部正值={np.all(y_pred >= 0)}, 最小值={y_pred.min():.4f}")

        return np.where(y_pred >= 0, 1, -1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
