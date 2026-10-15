import torch
from transformers import ViTModel, ViTImageProcessor
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
import pandas as pd


class ViTFeatureExtractorModel:
    def __init__(self, model_name='google/vit-base-patch16-224-in21k'):
        """加载 ViT 模型及其图像预处理器。"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ViTModel.from_pretrained(model_name)

        if torch.cuda.device_count() > 1:
            print(f"使用 {torch.cuda.device_count()} 个GPU进行并行计算!")
            self.model = torch.nn.DataParallel(self.model)

        self.model = self.model.to(self.device)
        self.image_processor = ViTImageProcessor.from_pretrained(model_name)
        self.model.eval()

    def preprocess_image(self, image_path):
        """把图像转换为 ViT 所需的张量输入。"""
        image = Image.open(image_path)
        inputs = self.image_processor(images=image, return_tensors="pt")
        return inputs

    def extract_features(self, image_path):
        """返回完整图像 token 嵌入（含 CLS），形状为 (B, N, D)。"""
        inputs = self.preprocess_image(image_path)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
        cls_embedding = outputs.last_hidden_state.cpu().numpy()
        return cls_embedding

    def extract_image_paths_from_csv(self, csv_file, img_folder):
        assert os.path.exists(csv_file), f"CSV file: {csv_file} does not exist."
        assert os.path.exists(img_folder), f"Image folder: {img_folder} does not exist."

        df = pd.read_csv(csv_file)

        # CSV 第一列为图像文件名。
        image_filenames = df.iloc[:, 0].tolist()

        image_paths = []
        for filename in image_filenames:
            img_path = os.path.join(img_folder, filename)
            if not os.path.exists(img_path):
                print(f"⚠️ 警告: 图像文件不存在: {img_path}")
            image_paths.append(img_path)

        print(f"✓ 从 {csv_file} 中读取了 {len(image_paths)} 个图像文件名")

        return image_paths

    def extract_features_for_all(self, csv_file, img_folder, save_path="vitembeddings.npy"):
        image_paths = self.extract_image_paths_from_csv(csv_file, img_folder)

        print(f"\n开始提取 {len(image_paths)} 张图像的特征...")

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
                if failed_count > 10:
                    print(f"❌ 失败次数过多 ({failed_count})，请检查数据")
                    raise

        train_features = np.array(train_features)
        train_features = train_features.squeeze(1)

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
