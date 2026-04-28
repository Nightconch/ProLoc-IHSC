import torch
from transformers import ViTModel, ViTImageProcessor
from PIL import Image
import os
from tqdm import tqdm  # 用于显示进度条
import numpy as np
import pandas as pd  # 用于读取CSV文件


class ViTFeatureExtractorModel:
    def __init__(self, model_name='google/vit-base-patch16-224-in21k'):
        """
        初始化ViT模型和图像处理器
        :param model_name: Hugging Face transformers中的模型名称
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ViTModel.from_pretrained(model_name)

        # 检查是否有多个GPU
        if torch.cuda.device_count() > 1:
            print(f"使用 {torch.cuda.device_count()} 个GPU进行并行计算!")
            self.model = torch.nn.DataParallel(self.model)  # 使用 DataParallel 并行计算

        # 将模型移动到GPU
        self.model = self.model.to(self.device)
        self.image_processor = ViTImageProcessor.from_pretrained(model_name)
        self.model.eval()  # 设置模型为评估模式，不会计算梯度

    def preprocess_image(self, image_path):
        """
        预处理图像，将其转换为模型可接受的输入格式
        :param image_path: 图像文件路径
        :return: 处理后的图像张量
        """
        image = Image.open(image_path)
        inputs = self.image_processor(images=image, return_tensors="pt")
        return inputs

    def extract_features(self, image_path):
        """
        提取图像的CLS特征嵌入
        :param image_path: 图像文件路径
        """
        inputs = self.preprocess_image(image_path)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}  # 将输入数据移动到GPU

        with torch.no_grad():
            outputs = self.model(**inputs)
        # 获取CLS标记的嵌入向量 (batch_size, hidden_size)
        cls_embedding = outputs.last_hidden_state.cpu().numpy()  # 将输出移回CPU并转换为numpy数组
        return cls_embedding

    def extract_image_paths_from_csv(self, csv_file, img_folder):
        assert os.path.exists(csv_file), f"CSV file: {csv_file} does not exist."
        assert os.path.exists(img_folder), f"Image folder: {img_folder} does not exist."
        
        # 读取CSV文件
        df = pd.read_csv(csv_file)
        
        # 提取第一列的文件名
        image_filenames = df.iloc[:, 0].tolist()
        
        # 构建完整的图像路径
        image_paths = []
        for filename in image_filenames:
            img_path = os.path.join(img_folder, filename)
            # 检查文件是否存在
            if not os.path.exists(img_path):
                print(f"⚠️ 警告: 图像文件不存在: {img_path}")
            image_paths.append(img_path)
        
        print(f"✓ 从 {csv_file} 中读取了 {len(image_paths)} 个图像文件名")
        # print(f"✓ 图像路径示例: {image_paths[0] if image_paths else 'None'}")
        
        return image_paths

    def extract_features_for_all(self, csv_file, img_folder, save_path="vitembeddings.npy"):
        # 从CSV中按顺序读取图像路径
        image_paths = self.extract_image_paths_from_csv(csv_file, img_folder)
        
        print(f"\n开始提取 {len(image_paths)} 张图像的特征...")
        
        # 对所有图像提取特征，并使用进度条显示进度
        train_features = []
        failed_count = 0
        
        for idx, path in enumerate(tqdm(image_paths, desc="Extracting features")):
            try:
                feature = self.extract_features(path)
                if feature is not None:
                    train_features.append(feature)
                else:
                    failed_count += 1
                    print(f"⚠️ 特征提取失败: {path} (索引: {idx})")
            except Exception as e:
                failed_count += 1
                print(f"❌ 提取时出错: {path} (索引: {idx}), 错误: {str(e)}")
                # 添加一个零向量占位，保持索引对齐
                # 注意：这里需要根据实际的特征维度来确定
                # 如果失败太多，建议中止
                if failed_count > 10:
                    print(f"❌ 失败次数过多 ({failed_count})，请检查数据")
                    raise

        # 将列表转换为NumPy数组
        train_features = np.array(train_features)
        train_features = train_features.squeeze(1)

        # 保存为npy文件
        np.save(save_path, train_features)
        
        print(f"\n{'='*60}")
        print(f"✓ 特征提取完成！")
        print(f"✓ 成功提取: {len(train_features)} 个特征")
        print(f"✓ 失败数量: {failed_count}")
        print(f"✓ 特征形状: {train_features.shape}")
        print(f"✓ 保存路径: {save_path}")
        print(f"{'='*60}\n")

        return train_features

if __name__ == "__main__":
    extractor_img = ViTFeatureExtractorModel()
    
    
    print("\n" + "=" * 60)
    print("开始提取训练集图像特征...")
    print("=" * 60)
    extractor_img.extract_features_for_all("dataset/train.csv", "dataset/train", "embedding/train_vitembeddings.npy")
    print("\n" + "=" * 60)
    print("开始提取测试集图像特征...")
    print("=" * 60)
    extractor_img.extract_features_for_all("dataset/test.csv", "dataset/test", "embedding/test_vitembeddings.npy")