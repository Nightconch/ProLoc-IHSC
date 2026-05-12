import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report
from model import CrossAttentionModel  # 引入模型类
from tqdm import tqdm  # 导入 tqdm 用于显示进度条
import pandas as pd
from metrics import sample_based_metrics, location_based_metrics_micro, print_sample_based_metrics, print_location_based_metrics
import sys
import atexit
from vit import ViTFeatureExtractorModel
from prott5 import ProteinEmbeddingExtractor

# 定义一个自定义数据集
class CustomDataset(Dataset):
    def __init__(self, seq_features, attention_masks, img_features, labels=None):
        self.seq_features = seq_features
        self.attention_masks = attention_masks
        self.img_features = img_features
        self.labels = labels

    def __len__(self):
        return len(self.seq_features)

    def __getitem__(self, idx):
        seq_feat = self.seq_features[idx]
        attn_mask = self.attention_masks[idx]
        img_feat = self.img_features[idx]
        if self.labels is not None:
            label = self.labels[idx]
            return seq_feat, attn_mask, img_feat, label
        else:
            return seq_feat, attn_mask, img_feat




# 测试函数，增加整体准确率的计算，并保存预测概率
def test_model(model, test_loader, device, has_labels, threshold, save_dir=None):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", disable=True):
            if has_labels:
                seq_feat, attn_mask, img_feat, labels = batch
                labels = labels.to(device)
                all_labels.append(labels.cpu().numpy())
            else:
                seq_feat, attn_mask, img_feat = batch
            seq_feat = seq_feat.to(device)
            attn_mask = attn_mask.to(device)
            img_feat = img_feat.to(device)

            # 前向传播，获取模型输出
            preds, probs, logits = model.predict_with_adaptive_threshold(
                image_features=img_feat,
                sequence_features=seq_feat,
                attention_mask=attn_mask,
                threshold=threshold
            )

            # 使用最佳阈值对预测结果进行二进制化
            probs = probs.cpu().numpy()
            all_probs.append(probs)
            preds_np = preds.cpu().numpy()
            all_preds.append(preds_np)

    all_preds = np.vstack(all_preds)

    label_names = ['cytoplasm', 'endoplasmic_reticulum', 'mitochondria', 'nucleus', 'plasma_membrane']

    if len(all_probs) > 0:
        all_probs = np.vstack(all_probs)
    else:
        all_probs = np.array([])

    if has_labels:
        all_labels = np.vstack(all_labels)

    if all_probs.size > 0:
        if save_dir is not None:
            results_path = os.path.join(save_dir, "test_results.csv")
            prob_df = pd.DataFrame(all_probs, columns=[f"prob_{name}" for name in label_names])
            pred_df = pd.DataFrame(all_preds.astype(int), columns=[f"pred_{name}" for name in label_names])
            if has_labels:
                label_df = pd.DataFrame(all_labels.astype(int), columns=[f"label_{name}" for name in label_names])
                output_df = pd.concat([prob_df, pred_df, label_df], axis=1)
            else:
                output_df = pd.concat([prob_df, pred_df], axis=1)
            output_df.to_csv(results_path, index=False)
            print(f"预测概率与标签结果已保存到: {results_path}")

    if has_labels:

        # 基于样本的度量
        ATR, Acc, Pre, Rec, F1 = sample_based_metrics(all_preds, all_labels)
        print_sample_based_metrics(ATR, Acc, Pre, Rec, F1)

        # 基于位置的微观度量
        Acc_prime_micro, Pre_prime_micro, Rec_prime_micro, F1_prime_micro = location_based_metrics_micro(all_preds, all_labels)
        print_location_based_metrics(Acc_prime_micro, Pre_prime_micro, Rec_prime_micro, F1_prime_micro)

        cls_report = classification_report(all_labels, all_preds, target_names=label_names, zero_division=0)
        print("分类报告：")
        print(cls_report)

        if save_dir is not None:
            report_path = os.path.join(save_dir, "classification_report.txt")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(cls_report)
            print(f"分类报告已保存到: {report_path}")

        return ATR, Acc, Pre, Rec, F1, Acc_prime_micro, Pre_prime_micro, Rec_prime_micro, F1_prime_micro
    else:
        return all_preds


# 主函数
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    best_model_path = "results/best_model.pth"
    log_dir = os.path.dirname(best_model_path)
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "test.txt")
    threshold = 0.4
    class Tee:
        def __init__(self, *streams):
            self.streams = streams
            # 尝试继承首个底层流的常见属性，避免第三方库访问失败
            try:
                self.encoding = getattr(streams[0], "encoding", "utf-8")
            except Exception:
                self.encoding = "utf-8"
        def write(self, data):
            for s in self.streams:
                try:
                    s.write(data)
                    s.flush()
                except Exception:
                    # 在解释器关闭阶段，底层流可能已不可用，静默忽略
                    pass
        def flush(self):
            for s in self.streams:
                try:
                    s.flush()
                except Exception:
                    pass
        def isatty(self):
            try:
                return any(getattr(s, "isatty", lambda: False)() for s in self.streams)
            except Exception:
                return False
        def fileno(self):
            for s in self.streams:
                try:
                    return s.fileno()
                except Exception:
                    continue
            raise OSError("No valid fileno available")

    _log_fp = open(log_file_path, 'a', encoding='utf-8', buffering=1)
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(_orig_stdout, _log_fp)
    sys.stderr = Tee(_orig_stderr, _log_fp)

    def _restore_and_close():
        try:
            # 恢复原始流，避免关闭期写入导致异常
            sys.stdout, sys.stderr
        except Exception:
            # 在解释器关闭晚期，sys 模块属性可能不可用
            return
        try:
            sys.stdout = _orig_stdout
            sys.stderr = _orig_stderr
        except Exception:
            pass
        try:
            if _log_fp and not _log_fp.closed:
                _log_fp.flush()
                _log_fp.close()
        except Exception:
            pass


    print("\n" + "=" * 60)
    print("开始提取测试集序列特征...")
    print("=" * 60)
    extractor_seq = ProteinEmbeddingExtractor()
    extractor_seq.process_file("dataset_test/test.csv","embedding/seq_test_embeddings.npy","embedding/seq_test_attention_masks.npy")
    
    print("\n" + "=" * 60)
    print("开始提取测试集图像特征...")
    print("=" * 60)
    extractor_img = ViTFeatureExtractorModel()
    extractor_img.extract_features_for_all("dataset_test/test.csv", "dataset_test/test", "embedding/test_vitembeddings.npy")


    atexit.register(_restore_and_close)
    print(f"日志输出到: {log_file_path}")
    print(f"加载模型权重从: {best_model_path}")

    # 加载测试集特征
    print("加载测试集序列特征")
    seq_features = np.load("embedding/seq_test_embeddings.npy")
    print("加载测试集序列掩码")
    attention_masks = np.load("embedding/seq_test_attention_masks.npy")
    print("加载测试集图像特征")
    img_features = np.load("embedding/test_vitembeddings.npy")

    # 加载模型
    embedding_dim = 512
    # num_heads = 256
    num_heads = 8
    num_layers = 6
    num_classes = 5  # 假设测试集中有 5 个标签类别
    image_embedding_dim = img_features.shape[2]
    sequence_embedding_dim = seq_features.shape[2]

    model = CrossAttentionModel(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        num_classes=num_classes,
        image_embedding_dim=image_embedding_dim,
        sequence_embedding_dim=sequence_embedding_dim
    )

    # 加载保存的模型权重
    # 尝试加载模型
    # PyTorch 2.6+ 需要设置 weights_only=False 来加载包含 numpy 数组的 checkpoint
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    
    # 判断加载的是字典还是直接的state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # 加载带标签关系损失的模型（train_with_label_relation_prot5.py 保存的格式）
        model.load_state_dict(checkpoint['model_state_dict'])
        # print(f"加载模型成功！训练轮次: {checkpoint.get('epoch', 'N/A')}, 验证损失: {checkpoint.get('val_loss', 'N/A'):.4f}")
    else:
        # 加载基础模型（train_prot5.py 保存的格式）
        model.load_state_dict(checkpoint)
        print("加载模型成功！")
    
    model = model.to(device)

    # 手动定义最佳阈值
    # thresholds = np.array([[0.34 ,0.38 ,0.4 , 0.56, 0.48]])
    # thresholds = np.array([[0.5,0.5,0.5,0.5,0.5]])

    print(f"使用的阈值: {threshold}")
    # 如果有测试集标签，可以加载标签数据
    try:
        label_df = pd.read_csv("dataset_test/test.csv")
        label_columns = [2, 3, 4, 5, 6]  # 对应的标签列
        labels = label_df.iloc[:, label_columns].values
        labels = torch.tensor(labels, dtype=torch.float32)
        has_labels = True
    except FileNotFoundError:
        print("没有找到标签文件，将不会计算测试集的 F1 分数。")
        labels = None
        has_labels = False

    # 打印数据形状以检查一致性
    print(f"\n数据形状检查:")
    print(f"seq_features shape: {seq_features.shape}")
    print(f"attention_masks shape: {attention_masks.shape}")
    print(f"img_features shape: {img_features.shape}")
    if has_labels:
        print(f"labels shape: {labels.shape}")

    # 检查样本数量是否一致
    num_seq = seq_features.shape[0]
    num_attn = attention_masks.shape[0]
    num_img = img_features.shape[0]

    if has_labels:
        num_labels = labels.shape[0]
        if not (num_seq == num_attn == num_img == num_labels):
            print(f"\n错误：数据样本数量不一致！")
            print(f"序列特征: {num_seq}, 注意力掩码: {num_attn}, 图像特征: {num_img}, 标签: {num_labels}")
            sys.exit(1)
    else:
        if not (num_seq == num_attn == num_img):
            print(f"\n错误：数据样本数量不一致！")
            print(f"序列特征: {num_seq}, 注意力掩码: {num_attn}, 图像特征: {num_img}")
            sys.exit(1)

    print(f"数据一致性检查通过，共 {num_seq} 个样本\n")

    # 将 NumPy 数组转换为 PyTorch 张量
    seq_features = torch.tensor(seq_features, dtype=torch.float32)
    attention_masks = torch.tensor(attention_masks, dtype=torch.bool)
    img_features = torch.tensor(img_features, dtype=torch.float32)

    # 创建测试集的数据集对象
    if has_labels:
        test_dataset = CustomDataset(seq_features, attention_masks, img_features, labels)
    else:
        test_dataset = CustomDataset(seq_features, attention_masks, img_features)

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 在测试集上测试模型
    test_model(model, test_loader, device, has_labels, threshold, save_dir=log_dir)