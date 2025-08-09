import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image

class FrameToNpyConverter:
    def __init__(self, root_dir, output_dir="npy_dataset", test_size=0.2):
        """
        初始化转换器
        :param root_dir: 包含normal和abnormal子文件夹的根目录
        :param output_dir: 输出npy文件的目录
        :param test_size: 验证集比例
        """
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.test_size = test_size
        self.normal_dir = os.path.join(root_dir, "normal")  # 正常帧文件夹
        self.abnormal_dir = os.path.join(root_dir, "abnormal")  # 异常帧文件夹
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载预训练的ResNet50作为特征提取器（不含顶层分类层）
        self.feature_extractor = ResNet50(
            weights='imagenet',
            include_top=False,
            pooling='avg'  # 全局平均池化，输出固定维度特征
        )

    def load_and_preprocess_image(self, img_path):
        """加载图像并预处理"""
        img = image.load_img(img_path, target_size=(224, 224))  # 调整为ResNet输入尺寸
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # 增加批次维度
        return preprocess_input(img_array)  # ResNet的预处理

    def extract_features(self, img_path):
        """提取图像特征并返回特征向量"""
        preprocessed_img = self.load_and_preprocess_image(img_path)
        features = self.feature_extractor.predict(preprocessed_img)
        return features.flatten()  # 转换为1D向量

    def process_all_frames(self):
        """处理所有帧并生成npy文件"""
        # 存储特征和标签
        all_features = []
        all_labels = []
        
        # 处理正常帧（标签0）
        print(f"处理正常帧...")
        for img_name in os.listdir(self.normal_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                img_path = os.path.join(self.normal_dir, img_name)
                try:
                    # 提取特征
                    features = self.extract_features(img_path)
                    all_features.append(features)
                    all_labels.append(0)  # 正常帧标签为0
                    
                    # 显示进度
                    if len(all_features) % 50 == 0:
                        print(f"已处理 {len(all_features)} 帧")
                except Exception as e:
                    print(f"处理 {img_name} 失败: {str(e)}")

        # 处理异常帧（标签1）
        print(f"\n处理异常帧...")
        abnormal_count = 0
        for img_name in os.listdir(self.abnormal_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                img_path = os.path.join(self.abnormal_dir, img_name)
                try:
                    # 提取特征
                    features = self.extract_features(img_path)
                    all_features.append(features)
                    all_labels.append(1)  # 异常帧标签为1
                    abnormal_count += 1
                    
                    # 显示进度
                    if abnormal_count % 10 == 0:  # 异常帧数量通常较少，进度间隔小一些
                        print(f"已处理 {abnormal_count} 帧")
                except Exception as e:
                    print(f"处理 {img_name} 失败: {str(e)}")

        # 转换为numpy数组
        X = np.array(all_features)
        y = np.array(all_labels)
        
        print(f"\n数据处理完成 - 总样本数: {X.shape[0]}, 特征维度: {X.shape[1]}")
        print(f"正常帧: {np.sum(y == 0)}, 异常帧: {np.sum(y == 1)}")

        # 划分训练集和验证集（保持标签分布）
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=42,
            stratify=y  # 确保训练集和验证集中正常/异常比例一致
        )

        # 保存为npy文件
        np.save(os.path.join(self.output_dir, "train_data.npy"), X_train)
        np.save(os.path.join(self.output_dir, "train_labels.npy"), y_train)
        np.save(os.path.join(self.output_dir, "val_data.npy"), X_val)
        np.save(os.path.join(self.output_dir, "val_labels.npy"), y_val)
        
        print(f"\n数据集已保存至 {self.output_dir}")
        print(f"训练集: {X_train.shape[0]} 样本, 验证集: {X_val.shape[0]} 样本")

if __name__ == "__main__":
    # 配置路径（请根据实际情况修改）
    # 注意：该目录下应包含两个子文件夹：normal（正常帧）和abnormal（异常帧）
    annotated_frames_dir = r"C:\workspace\Intelligent-monitoring\results\annotated_frames"
    output_directory = r"C:\workspace\Intelligent-monitoring\results\npy_dataset"
    
    # 创建转换器实例并处理
    converter = FrameToNpyConverter(
        root_dir=annotated_frames_dir,
        output_dir=output_directory,
        test_size=0.2  # 20%数据作为验证集
    )
    converter.process_all_frames()
