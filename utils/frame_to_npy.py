import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image


class FrameToNpyConverter:
    def __init__(self, root_dir, output_dir="npy_dataset", test_size=0.2, batch_size=32):
        """
        初始化转换器（新增批量处理参数）
        :param root_dir: 包含normal和abnormal子文件夹的根目录
        :param output_dir: 输出npy文件的目录
        :param test_size: 验证集比例
        :param batch_size: 批量处理大小，提升特征提取效率
        """
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.test_size = test_size
        self.batch_size = batch_size
        self.normal_dir = os.path.join(root_dir, "normal")  # 正常帧文件夹
        self.abnormal_dir = os.path.join(root_dir, "abnormal")  # 异常帧文件夹

        # 检查子目录是否存在
        if not os.path.isdir(self.normal_dir):
            raise FileNotFoundError(f"未找到正常帧目录: {self.normal_dir}")
        if not os.path.isdir(self.abnormal_dir):
            raise FileNotFoundError(f"未找到异常帧目录: {self.abnormal_dir}")

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 加载预训练的ResNet50作为特征提取器（不含顶层分类层）
        self.feature_extractor = ResNet50(
            weights='imagenet',
            include_top=False,
            pooling='avg'  # 全局平均池化，输出固定维度特征(2048维)
        )

    def load_and_preprocess_image(self, img_path):
        """加载图像并预处理（新增通道检查）"""
        img = image.load_img(img_path, target_size=(224, 224))  # 调整为ResNet输入尺寸
        img_array = image.img_to_array(img)

        # 检查并统一通道数为3（RGB）
        if img_array.ndim != 3 or img_array.shape[-1] not in [1, 3, 4]:
            raise ValueError(f"不支持的图像维度: {img_array.shape}，路径: {img_path}")

        # 处理单通道图像（转为3通道）
        if img_array.shape[-1] == 1:
            img_array = np.repeat(img_array, 3, axis=-1)
        # 处理4通道图像（去除alpha通道）
        elif img_array.shape[-1] == 4:
            img_array = img_array[..., :3]

        return img_array.astype(np.float32)

    def extract_features_batch(self, img_paths):
        """批量提取图像特征（提升效率）"""
        batch_imgs = []
        valid_paths = []  # 记录有效图像路径

        for img_path in img_paths:
            try:
                img_array = self.load_and_preprocess_image(img_path)
                batch_imgs.append(img_array)
                valid_paths.append(img_path)
            except Exception as e:
                print(f"[警告] 预处理失败 {os.path.basename(img_path)}: {str(e)}")

        if not batch_imgs:  # 空批量
            return np.array([]), []

        batch_imgs = np.array(batch_imgs, dtype=np.float32)
        batch_imgs = preprocess_input(batch_imgs)  # ResNet标准化
        features = self.feature_extractor.predict(batch_imgs, verbose=0)

        # 显存优化：清空中间变量
        del batch_imgs

        return features.reshape(len(features), -1), valid_paths

    def process_all_frames(self):
        """处理所有帧并生成标准化的npy文件"""
        all_img_paths = []
        all_labels = []

        # 正常帧
        normal_img_paths = [
            os.path.join(self.normal_dir, f)
            for f in os.listdir(self.normal_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))
        ]
        all_img_paths.extend(normal_img_paths)
        all_labels.extend([0] * len(normal_img_paths))

        # 异常帧
        abnormal_img_paths = [
            os.path.join(self.abnormal_dir, f)
            for f in os.listdir(self.abnormal_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))
        ]
        all_img_paths.extend(abnormal_img_paths)
        all_labels.extend([1] * len(abnormal_img_paths))

        print(f"发现正常帧: {len(normal_img_paths)} 张，异常帧: {len(abnormal_img_paths)} 张")
        if len(all_img_paths) == 0:
            raise ValueError("未找到任何图像文件，请检查输入目录")

        # 批量提取特征
        all_features = []
        valid_labels = []
        total_batches = (len(all_img_paths) + self.batch_size - 1) // self.batch_size

        print("\n开始批量提取特征...")
        for i in range(total_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, len(all_img_paths))
            batch_paths = all_img_paths[start_idx:end_idx]
            batch_labels = all_labels[start_idx:end_idx]

            batch_features, valid_paths = self.extract_features_batch(batch_paths)
            if len(batch_features) > 0:
                all_features.extend(batch_features)
                valid_labels.extend([
                    batch_labels[j] for j in range(len(batch_paths))
                    if batch_paths[j] in valid_paths
                ])

            if (i + 1) % 5 == 0 or i + 1 == total_batches:
                print(f"已完成 {i + 1}/{total_batches} 批，累计有效特征: {len(all_features)}")

        X = np.array(all_features, dtype=np.float32)
        y = np.array(valid_labels)

        if len(X) == 0:
            raise RuntimeError("未成功提取任何特征，请检查图像文件是否有效")
        if len(np.unique([len(feat) for feat in X])) != 1:
            raise RuntimeError(f"特征维度不一致，发现: {np.unique([len(feat) for feat in X])}")

        print(f"\n特征提取完成 - 样本数: {X.shape[0]}, 特征维度: {X.shape[1]}")
        print(f"正常帧: {np.sum(y == 0)}, 异常帧: {np.sum(y == 1)}")

        # 特征标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        scaler_path = os.path.join(self.output_dir, "feature_scaler.pkl")
        joblib.dump(scaler, scaler_path)
        print(f"特征标准化器已保存至: {scaler_path}")

        # 划分训练集与验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y,
            test_size=self.test_size,
            random_state=42,
            stratify=y
        )

        np.save(os.path.join(self.output_dir, "train_data.npy"), X_train)
        np.save(os.path.join(self.output_dir, "train_labels.npy"), y_train)
        np.save(os.path.join(self.output_dir, "val_data.npy"), X_val)
        np.save(os.path.join(self.output_dir, "val_labels.npy"), y_val)

        print(f"\n数据集已保存至 {self.output_dir}")
        print(f"训练集: {X_train.shape[0]} 样本, 验证集: {X_val.shape[0]} 样本")


if __name__ == "__main__":
    annotated_frames_dir = r"D:\try\data"
    output_directory = r"D:\try\npy_dataset"

    converter = FrameToNpyConverter(
        root_dir=annotated_frames_dir,
        output_dir=output_directory,
        test_size=0.2,
        batch_size=32
    )
    converter.process_all_frames()
