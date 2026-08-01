import os
import random
import sys
import atexit
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from model import CrossAttentionModel
from metrics import compute_co_occurrence_matrix, compute_co_occurrence_matrix_cos
from vit import ViTFeatureExtractorModel
from prott5 import ProteinEmbeddingExtractor
from isc import _normalize_protein_ids, protein_aware_isc_loss, protein_positive_relation


# ==================== 设置随机种子函数 ====================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def split_by_protein(labels, protein_ids, n_splits=10, random_state=42):
    """Split sample indices while keeping every protein on only one side."""
    labels_array = np.asarray(labels)
    protein_series = pd.Series(protein_ids, dtype="object")

    if labels_array.ndim != 2:
        raise ValueError("labels must be a two-dimensional array")
    if len(labels_array) != len(protein_series):
        raise ValueError(
            "labels and protein_ids must contain the same number of samples"
        )
    if protein_series.isna().any():
        raise ValueError("Protein Id contains missing values")

    protein_series = protein_series.astype(str).str.strip()
    if protein_series.eq("").any():
        raise ValueError("Protein Id contains blank values")

    protein_count = protein_series.nunique()
    if protein_count < n_splits:
        raise ValueError(
            f"at least {n_splits} distinct proteins are required, "
            f"but found {protein_count}"
        )

    protein_array = protein_series.to_numpy()
    label_strings = np.asarray(
        ["".join(map(str, row.astype(int))) for row in labels_array]
    )
    indices = np.arange(len(labels_array))
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    train_indices, val_indices = next(
        splitter.split(
            indices,
            y=label_strings,
            groups=protein_array,
        )
    )

    train_proteins = set(protein_array[train_indices])
    val_proteins = set(protein_array[val_indices])
    if not train_proteins.isdisjoint(val_proteins):
        raise RuntimeError(
            "protein-group split leaked proteins across train and validation"
        )

    combined_indices = np.concatenate([train_indices, val_indices])
    if not np.array_equal(np.sort(combined_indices), indices):
        raise RuntimeError(
            "protein-group split did not preserve every sample exactly once"
        )

    return train_indices, val_indices


def _normalize_training_protein_ids(protein_ids):
    return np.asarray(_normalize_protein_ids(protein_ids), dtype=object)


class DeferredTee:

    def __init__(self, stream):
        self.terminal = stream
        self.log = None
        self.buffer = []

    def write(self, message):
        if self.terminal is not None:
            self.terminal.write(message)
        if self.log is not None:
            self.log.write(message)
        else:
            self.buffer.append(message)

    def flush(self):
        if self.terminal is not None:
            self.terminal.flush()
        if self.log is not None:
            self.log.flush()

    def bind(self, file_path):
        if self.log is not None:
            return
        self.log = open(file_path, 'w', encoding='utf-8')
        if self.buffer:
            self.log.write(''.join(self.buffer))
            self.buffer.clear()
        self.log.flush()

    def close(self):
        if self.log is not None and not self.log.closed:
            self.log.close()
            self.log = None


# ==================== 标签共现分析 ====================

class LabelRelationLoss(nn.Module):
    def __init__(self, co_occurrence_matrix, lambda_weight=0.5, high_threshold=0.5, low_threshold=0.1):
        super(LabelRelationLoss, self).__init__()
        self.register_buffer('C', torch.tensor(co_occurrence_matrix, dtype=torch.float32))
        self.lambda_weight = lambda_weight
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        
        # 预先计算高共现和互斥的标签对
        n_labels = co_occurrence_matrix.shape[0]
        self.high_pairs = []
        self.low_pairs = []
        
        for i in range(n_labels):
            for j in range(i+1, n_labels):
                if co_occurrence_matrix[i, j] > high_threshold:
                    self.high_pairs.append((i, j))
                elif co_occurrence_matrix[i, j] < low_threshold:
                    self.low_pairs.append((i, j))
        
        # print(f"标签关系统计: 高共现标签对 {len(self.high_pairs)} 个, 互斥标签对 {len(self.low_pairs)} 个")
        
    def forward(self, probs):
        """
        计算标签关系损失
        
        参数:
            probs: 模型输出的概率 (batch_size, num_labels)
        """
        relation_loss = 0.0
        # count = 0
        
        # 高共现标签对：惩罚概率差异
        for i, j in self.high_pairs:
            # 两个标签的概率应该接近
            pair_diff_loss = torch.mean((probs[:, i] - probs[:, j]) ** 2)
            relation_loss += pair_diff_loss
            # count += 1
        
        # 互斥标签对：惩罚同时为高概率
        for i, j in self.low_pairs:
            # 两个标签不应该同时为高概率
            pair_exclusive_loss = torch.sum(probs[:, i] * probs[:, j])
            relation_loss += pair_exclusive_loss
            # count += 1
        
        # # 归一化
        # if count > 0:
        #     relation_loss = relation_loss / count
        
        return self.lambda_weight * relation_loss


# ==================== 少数类 ====================

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


def compute_adaptive_soft_weights(probs, threshold=0.3, temperature=10.0):

    # 计算每个样本的最大概率
    max_probs = probs.max(dim=1, keepdim=True)[0]  # (batch_size, 1)
    
    # 计算差值
    diff = max_probs - probs  # (batch_size, num_labels)
    
    soft_weights = torch.sigmoid((threshold - diff) * temperature)
    
    return soft_weights


def multi_task_loss(logits, probs, labels, adaptive_soft_weights=None, label_relation_criterion=None, 
                    focal_weight=4, bce_weight=4, 
                    main_loss_weight=3, Minority_Class_loss_weight=10, Label_Relation_loss_weight=1,
                    adaptive_weight=1.0):
    # 主任务(强化标准学习策略)：
    if adaptive_soft_weights is not None:
        # 使用 logits 计算 BCE，数值更稳定
        base_loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        consistency_bonus = torch.abs(probs - adaptive_soft_weights)  # 使用 probs 计算一致性
        weighted_loss = base_loss * (1.0 + adaptive_weight * consistency_bonus)
        main_loss = weighted_loss.mean()
    else:
        main_loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)

    # 少数任务：
    focal_loss = focal_criterion(probs[:, 4], labels[:, 4])  # Focal Loss 仍使用 probs
    bce_loss = nn.functional.binary_cross_entropy_with_logits(logits[:, 4], labels[:, 4])  # BCE 使用 logits
    Minority_Class_loss = focal_weight * focal_loss + bce_weight * bce_loss

    # 标签关系损失
    if label_relation_criterion is not None:
        Label_Relation_loss = label_relation_criterion(probs)
    else:
        Label_Relation_loss = 0.0

    # 最终损失
    total_loss = (main_loss_weight * main_loss + 
                  Minority_Class_loss_weight * Minority_Class_loss + 
                  Label_Relation_loss_weight * Label_Relation_loss)
    
    # 返回总损失和各项损失的详细信息
    loss_dict = {
        'total': total_loss,
        'main': main_loss,
        'Minority_Class': Minority_Class_loss,
        'Label_Relation': Label_Relation_loss if isinstance(Label_Relation_loss, torch.Tensor) else torch.tensor(Label_Relation_loss)
    }
    
    return total_loss, loss_dict


class CustomDataset(Dataset):
    def __init__(
        self,
        seq_features,
        attention_masks,
        img_features,
        labels,
        protein_ids,
        indices=None,
    ):
        self.seq_features = seq_features
        self.attention_masks = attention_masks
        self.img_features = img_features
        self.labels = labels
        if len(protein_ids) != len(labels):
            raise ValueError("protein_ids and labels must contain the same number of samples")
        self.protein_ids = _normalize_protein_ids(protein_ids)
        self.indices = indices  # 如果提供了索引，使用索引访问数据
        
        # 如果提供了索引，使用索引的长度；否则使用标签的长度
        self.length = len(indices) if indices is not None else len(labels)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 获取实际索引
        actual_idx = self.indices[idx] if self.indices is not None else idx
        
        # 从内存映射数组或numpy数组按需加载数据
        # 注意：这里只加载单个样本，不会将整个数组加载到内存
        seq_feat = np.array(self.seq_features[actual_idx], dtype=np.float32)
        attn_mask = np.array(self.attention_masks[actual_idx], dtype=bool)
        img_feat = np.array(self.img_features[actual_idx], dtype=np.float32)
        label = self.labels[actual_idx]
        protein_id = self.protein_ids[actual_idx]
        
        # 转换为torch tensor
        seq_feat = torch.from_numpy(seq_feat).float()
        attn_mask = torch.from_numpy(attn_mask).bool()
        img_feat = torch.from_numpy(img_feat).float()
        
        # label已经是tensor，不需要转换
        return seq_feat, attn_mask, img_feat, label, protein_id


def training_isc_metrics(
    image_embeddings, sequence_embeddings, protein_ids, temperature=0.07
):
    positive_relation = protein_positive_relation(protein_ids)
    isc_loss = protein_aware_isc_loss(
        image_embeddings,
        sequence_embeddings,
        protein_ids,
        temperature=temperature,
    )
    mean_positives_per_anchor = positive_relation.sum(dim=1).float().mean()
    return isc_loss, mean_positives_per_anchor


def accumulate_validation_isc(
    weighted_loss,
    processed_samples,
    image_embeddings,
    sequence_embeddings,
    protein_ids,
    temperature=0.07,
):
    batch_loss = protein_aware_isc_loss(
        image_embeddings,
        sequence_embeddings,
        protein_ids,
        temperature=temperature,
    )
    batch_size = image_embeddings.size(0)
    return (
        batch_loss,
        weighted_loss + batch_loss.item() * batch_size,
        processed_samples + batch_size,
    )


class EarlyStopping:
    
    def __init__(self, patience=10, min_delta=0.0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, val_loss, epoch):
        is_best = False
        
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
            is_best = True
        elif val_loss < self.best_loss - self.min_delta:
            # 有改善
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            is_best = True
            if self.verbose:
                print(f"  ✓ 验证损失改善: {val_loss:.4f}")
        else:
            # 没有改善
            self.counter += 1
            if self.verbose:
                print(f"  ⚠ 验证损失未改善 ({self.counter}/{self.patience})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"  ⏹ 早停触发！最佳epoch: {self.best_epoch}, 最佳损失: {self.best_loss:.4f}")
        
        return is_best


# ==================== 主函数 ====================

if __name__ == '__main__':
    # 命令行参数
    parser = argparse.ArgumentParser(description="Train model with selectable GPU.")
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU id to use (e.g., 0 or 1). If no GPU is available, CPU will be used.",
    )
    args = parser.parse_args()

    # 设置随机种子
    seed = 42
    set_seed(seed)

    log_stream = DeferredTee(sys.stdout)
    sys.stdout = log_stream
    sys.stderr = log_stream
    atexit.register(log_stream.close)

    if torch.cuda.is_available():
        # 选择指定 GPU
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"使用设备: {device}\n")

    
    # 分开提取图像和序列特征
    
    # print("\n" + "=" * 60)
    # print("开始提取训练集图像特征...")
    # print("=" * 60)
    # extractor_img = ViTFeatureExtractorModel()
    # extractor_img.extract_features_for_all("dataset/train.csv", "dataset/train", "embedding/train_vitembeddings.npy")
    # print("=" * 60)

    # print("\n" + "=" * 60)
    # print("开始提取训练集序列特征...")
    # print("=" * 60)
    # extractor_seq = ProteinEmbeddingExtractor()
    # extractor_seq.process_file("dataset/train.csv", "embedding/seq_train_embeddings.npy", "embedding/seq_train_attention_masks.npy")
    # print("=" * 60)



    print("="*70)
    print("加载序列特征")
    seq_features = np.load("embedding/seq_train_embeddings.npy", mmap_mode='r')
    print("加载序列掩码")
    attention_masks = np.load("embedding/seq_train_attention_masks.npy", mmap_mode='r')
    print("加载图像特征")
    img_features = np.load("embedding/train_vitembeddings.npy", mmap_mode='r')
    print()

            # 加载标签（标签文件较小，可以正常加载）
    print("加载标签")
    label_df = pd.read_csv("dataset/train.csv")
    label_columns = ['cytoplasm', 'endoplasmic reticulum', 'mitochondria', 'nucleus', 'plasma membrane']
    labels = label_df[label_columns].values
    labels = torch.tensor(labels, dtype=torch.float32)

    # 保证样本数一致
    num_samples = labels.shape[0]
    # if len(seq_features) > num_samples:
    #     print(f"警告: 序列特征文件包含 {len(seq_features)} 个样本，标签文件包含 {num_samples} 个样本")
    #     print(f"将只使用前 {num_samples} 个样本")
    # elif len(seq_features) < num_samples:
    #     raise ValueError(f"序列特征文件包含 {len(seq_features)} 个样本，少于标签文件的 {num_samples} 个样本")
    
    # 获取数据维度信息（用于后续模型初始化）
    # 这里只读取第一个样本的维度，不会加载整个数组
    # 对于内存映射数组，直接访问第一个元素只会读取该元素，不会加载整个数组
    # 获取序列特征维度
    if len(seq_features.shape) == 3:
        # 形状为 (num_samples, seq_len, embedding_dim)
        sequence_embedding_dim = seq_features.shape[2]
    elif len(seq_features.shape) == 2:
        # 形状为 (num_samples, embedding_dim)
        sequence_embedding_dim = seq_features.shape[1]
    else:
        # 读取第一个样本获取维度
        first_seq = np.array(seq_features[0])
        sequence_embedding_dim = first_seq.shape[-1]
    
    # 获取图像特征维度
    if len(img_features.shape) == 3:
        # 形状为 (num_samples, num_patches, embedding_dim)
        image_embedding_dim = img_features.shape[2]
    elif len(img_features.shape) == 2:
        # 形状为 (num_samples, embedding_dim)
        image_embedding_dim = img_features.shape[1]
    else:
        # 读取第一个样本获取维度
        first_img = np.array(img_features[0])
        image_embedding_dim = first_img.shape[-1]
        

    # 保证样本数一致
    num_samples = labels.shape[0]
    if len(seq_features) > num_samples:
        print(f"警告: 序列特征文件包含 {len(seq_features)} 个样本，标签文件包含 {num_samples} 个样本")
        print(f"将只使用前 {num_samples} 个样本")
    elif len(seq_features) < num_samples:
        raise ValueError(f"序列特征文件包含 {len(seq_features)} 个样本，少于标签文件的 {num_samples} 个样本")
    

    # 按蛋白质分组进行分层划分，避免同一蛋白质泄漏到训练集和验证集
    protein_column = "Protein Id"
    if protein_column not in label_df.columns:
        raise ValueError(
            f"训练标签文件缺少必需列 {protein_column!r}；"
            f"现有列: {list(label_df.columns)}"
        )

    protein_ids = label_df[protein_column]
    train_indices, val_indices = split_by_protein(
        labels.numpy(),
        protein_ids,
        n_splits=10,
        random_state=seed,
    )

    normalized_protein_ids = _normalize_training_protein_ids(protein_ids)
    train_protein_count = len(set(normalized_protein_ids[train_indices]))
    val_protein_count = len(set(normalized_protein_ids[val_indices]))
    train_percentage = 100.0 * len(train_indices) / num_samples
    val_percentage = 100.0 * len(val_indices) / num_samples
    print(
        f"训练集: {len(train_indices)} 个样本, "
        f"{train_protein_count} 个蛋白质 ({train_percentage:.2f}%)"
    )
    print(
        f"验证集: {len(val_indices)} 个样本, "
        f"{val_protein_count} 个蛋白质 ({val_percentage:.2f}%)"
    )

    # ==================== 基于训练集样本计算标签共现矩阵 ====================
    labels_np_train = labels[train_indices].numpy()
    co_occurrence_matrix = compute_co_occurrence_matrix(labels_np_train)
    
    # print("\n标签共现矩阵:")
    label_names_short = ['cyto', 'ER', 'mito', 'nucl', 'PM']
    co_df = pd.DataFrame(co_occurrence_matrix, 
                        index=label_names_short, 
                        columns=label_names_short)
    # print(co_df.round(3))
    
    # 保存共现矩阵
    # co_df.to_csv("label_co_occurrence_matrix.csv")
    # print("\n标签共现矩阵已保存到: label_co_occurrence_matrix.csv")
    

    train_dataset = CustomDataset(
        seq_features, 
        attention_masks,  
        img_features,
        labels, 
        normalized_protein_ids,
        indices=train_indices  
    )
    
    val_dataset = CustomDataset(
        seq_features,
        attention_masks,
        img_features,  
        labels, 
        normalized_protein_ids,
        indices=val_indices  
    )
    
    # print("="*70)
    # print("数据集创建完成")
    # print("="*70)
    # print(f"训练集样本数: {len(train_dataset)}")
    # print(f"验证集样本数: {len(val_dataset)}")
    # print()

    
    BATCH_SIZE = 32
    NUM_WORKERS = 4  # 根据系统内存调整（建议4-8）
    PIN_MEMORY = True if device.type == 'cuda' else False
    PERSISTENT_WORKERS = True if NUM_WORKERS > 0 else False
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS
    )
    
    # print(f"DataLoader配置:")
    # print(f"  Batch Size: {BATCH_SIZE}")
    # print(f"  Num Workers: {NUM_WORKERS}")
    # print(f"  Pin Memory: {PIN_MEMORY}")
    # print(f"  Persistent Workers: {PERSISTENT_WORKERS}")
    # print()

    # 定义模型、损失函数和优化器
    embedding_dim = 512
    num_heads = 8
    num_layers = 6
    num_classes = labels.shape[1]


    model = CrossAttentionModel(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        num_classes=num_classes,
        image_embedding_dim=image_embedding_dim,
        sequence_embedding_dim=sequence_embedding_dim
    )
    model = model.to(device)

    # 定义损失函数
    focal_criterion = FocalLoss(alpha=1, gamma=2)
    
    lambda_weight=1.0
    high_threshold=0.1
    low_threshold=0.001

    # ==================== 标签关系损失 ====================
    label_relation_criterion = LabelRelationLoss(
        co_occurrence_matrix, 
        lambda_weight=lambda_weight,  # LabelRelationLoss内部的lambda权重（建议设为1.0，通过Label_Relation_loss_weight控制）
        high_threshold=high_threshold,  # 高共现判定阈值
        low_threshold=low_threshold    # 互斥判定阈值
    ).to(device)
    
    # ==================== 损失函数超参数配置 ====================
    
    # Focal Loss 和 BCE 的权重配置（用于次任务：第5个标签）
    focal_weight = 4            # Focal Loss权重
    bce_weight = 4              # BCE权重
    
    # 多任务损失的权重配置
    main_loss_weight = 3        # 主任务损失权重（所有标签的BCE）
    Minority_Class_loss_weight = 10  # 次任务损失权重（第5个标签的Focal+BCE）
    Label_Relation_loss_weight = 2   # 第三任务损失权重（标签关系损失）
    
    # 软化配置
    adaptive_weight = 0.8       # 融合权重（0表示不使用，1表示完全使用）
    temperature = 15.0          # 软化温度（大值接近硬阈值，小值更平滑）
    isc_temperature = 0.07
    isc_loss_weight = 1.0
    
    
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # ==================== 训练配置 ====================

    # 训练轮数和早停配置
    max_epochs = 10           # 最大训练轮数
    patience = 10              # 早停耐心值（多少个epoch验证损失不改善就停止）
    min_delta = 0.0            # 认为是改善的最小变化量

    threshold = 0.4

    run_id = f"1"

    # print("\n" + "="*70)
    # print("训练配置:")
    # print("="*70)
    # print(f"  最大训练轮数 (max_epochs):  {max_epochs}")
    # print(f"  早停耐心值 (patience):      {patience} epochs")
    # print(f"  最小改善量 (min_delta):     {min_delta}")
    # print(f"  阈值 (threshold):          {threshold}")
    # print("="*70 + "\n")
    
    # 初始化早停机制
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, verbose=True)
    
    best_val_loss = float('inf')
    results_dir = f"results/{run_id}"
    os.makedirs(results_dir, exist_ok=True)
    log_file_path = os.path.join(results_dir, "train.txt")
    log_stream.bind(log_file_path)
    best_model_path = f"{results_dir}/best_model.pth"
    
    # 记录每个epoch的结果
    train_losses = []
    val_losses = []
    train_main_losses = []
    train_Minority_Class_losses = []
    train_Label_Relation_losses = []
    train_isc_losses = []
    train_mean_positives_per_anchor = []
    val_isc_losses = []


    for epoch in range(max_epochs):
        # ==================== 训练阶段 ====================
        model.train()
        running_train_loss = 0.0
        running_main_loss = 0.0
        running_Minority_Class_loss = 0.0
        running_Label_Relation_loss = 0.0
        running_isc_loss = 0.0
        running_mean_positives_per_anchor = 0.0
        print(f"Epoch {epoch + 1}/{max_epochs}")

        for batch in tqdm(train_loader, desc="Training", leave=False, disable=True):
            seq_feat, attn_mask, img_feat, label, batch_protein_ids = batch
            seq_feat = seq_feat.to(device)
            attn_mask = attn_mask.to(device)
            img_feat = img_feat.to(device)
            label = label.to(device)

            # 前向传播
            preds, probs, logits = model.predict_with_adaptive_threshold(
                image_features=img_feat,
                sequence_features=seq_feat,
                attention_mask=attn_mask,
                threshold = threshold
            )

            # 计算软化权重
            soft_weights = compute_adaptive_soft_weights(probs, threshold=threshold, temperature=temperature)

            # 计算多任务损失（包含三个损失项，融合软化动态阈值策略）
            total_loss, loss_dict = multi_task_loss(
                logits,  # 传入 logits
                probs,   # 传入 probs
                label,
                adaptive_soft_weights=soft_weights,  # 传递软化权重
                label_relation_criterion=label_relation_criterion,
                focal_weight=focal_weight,
                bce_weight=bce_weight,
                main_loss_weight=main_loss_weight,
                Minority_Class_loss_weight=Minority_Class_loss_weight,
                Label_Relation_loss_weight=Label_Relation_loss_weight,
                adaptive_weight=adaptive_weight  # 融合权重
            )

            image_global, sequence_global = model.get_global_embeddings(
                image_features=img_feat,
                sequence_features=seq_feat,
                attention_mask=attn_mask
            )
            isc_loss, mean_positives_per_anchor = training_isc_metrics(
                image_global,
                sequence_global,
                batch_protein_ids,
                temperature=isc_temperature,
            )
            total_loss = total_loss + isc_loss_weight * isc_loss

            # 反向传播和优化
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_train_loss += total_loss.item() * seq_feat.size(0)
            running_isc_loss += isc_loss.item() * seq_feat.size(0)
            running_mean_positives_per_anchor += (
                mean_positives_per_anchor.item() * seq_feat.size(0)
            )
            running_main_loss += loss_dict['main'].item() * seq_feat.size(0)
            running_Minority_Class_loss += loss_dict['Minority_Class'].item() * seq_feat.size(0)
            running_Label_Relation_loss += loss_dict['Label_Relation'].item() * seq_feat.size(0)

        epoch_train_loss = running_train_loss / len(train_dataset)
        epoch_train_isc_loss = running_isc_loss / len(train_dataset)
        epoch_mean_positives_per_anchor = (
            running_mean_positives_per_anchor / len(train_dataset)
        )
        epoch_main_loss = running_main_loss / len(train_dataset)
        epoch_Minority_Class_loss = running_Minority_Class_loss / len(train_dataset)
        epoch_Label_Relation_loss = running_Label_Relation_loss / len(train_dataset)
        
        # ==================== 验证阶段 ====================
        model.eval()
        running_val_loss = 0.0
        running_val_isc_loss = 0.0
        processed_val_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False, disable=True):
                seq_feat, attn_mask, img_feat, label, batch_protein_ids = batch
                seq_feat = seq_feat.to(device)
                attn_mask = attn_mask.to(device)
                img_feat = img_feat.to(device)
                label = label.to(device)

                preds, probs, logits = model.predict_with_adaptive_threshold(
                    image_features=img_feat,
                    sequence_features=seq_feat,
                    attention_mask=attn_mask,
                    threshold=threshold
                )

                # 计算软化权重
                soft_weights = compute_adaptive_soft_weights(probs, threshold=threshold, temperature=temperature)

                # 验证时也使用完整损失（包含软化策略）
                total_loss, loss_dict = multi_task_loss(
                    logits,  # 传入 logits
                    probs,   # 传入 probs
                    label,
                    adaptive_soft_weights=soft_weights,  # 传递软化权重
                    label_relation_criterion=label_relation_criterion,
                    focal_weight=focal_weight,
                    bce_weight=bce_weight,
                    main_loss_weight=main_loss_weight,
                    Minority_Class_loss_weight=Minority_Class_loss_weight,
                    Label_Relation_loss_weight=Label_Relation_loss_weight,
                    adaptive_weight=adaptive_weight  # 融合权重
                )

                image_global, sequence_global = model.get_global_embeddings(
                    image_features=img_feat,
                    sequence_features=seq_feat,
                    attention_mask=attn_mask
                )
                batch_size = seq_feat.size(0)
                isc_loss, running_val_isc_loss, processed_val_samples = accumulate_validation_isc(
                    running_val_isc_loss,
                    processed_val_samples,
                    image_global,
                    sequence_global,
                    batch_protein_ids,
                    temperature=isc_temperature,
                )
                total_loss = total_loss + isc_loss_weight * isc_loss
                running_val_loss += total_loss.item() * batch_size

        epoch_val_loss = running_val_loss / len(val_dataset)
        epoch_val_isc_loss = running_val_isc_loss / processed_val_samples
        
        print(f"  Train Loss: {epoch_train_loss:.4f} (Main: {epoch_main_loss:.4f}, Minority_Class: {epoch_Minority_Class_loss:.4f}, Label_Relation: {epoch_Label_Relation_loss:.4f}, isc: {epoch_train_isc_loss:.4f}, positives/anchor: {epoch_mean_positives_per_anchor:.4f})")
        print(f"  Val Loss: {epoch_val_loss:.4f}")
        
        # 记录损失值
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_main_losses.append(epoch_main_loss)
        train_Minority_Class_losses.append(epoch_Minority_Class_loss)
        train_Label_Relation_losses.append(epoch_Label_Relation_loss)
        train_isc_losses.append(epoch_train_isc_loss)
        train_mean_positives_per_anchor.append(epoch_mean_positives_per_anchor)
        val_isc_losses.append(epoch_val_isc_loss)

        # 检查早停
        is_best = early_stopping(epoch_val_loss, epoch + 1)
        
        if is_best:
            best_val_loss = epoch_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'co_occurrence_matrix': co_occurrence_matrix
            }, best_model_path)
            print(f"  ★ 最佳模型已保存！Val Loss: {best_val_loss:.4f}")
        
        # 如果触发早停，退出训练
        if early_stopping.early_stop:
            print(f"\n早停于第 {epoch + 1} 轮")
            break

    print("\n训练完成！")
    
    # ==================== 保存训练历史 ====================
    
    actual_epochs = len(train_losses)  # 实际训练的轮数
    
    history_df = pd.DataFrame({
        'epoch': range(1, actual_epochs + 1),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_main_loss': train_main_losses,
        'train_Minority_Class_loss': train_Minority_Class_losses,
        'train_Label_Relation_loss': train_Label_Relation_losses,
        'train_isc_loss': train_isc_losses,
        'train_mean_positives_per_anchor': train_mean_positives_per_anchor,
        'val_isc_loss': val_isc_losses
    })
    history_csv_path = f"{results_dir}/training_history_label_relation_prot5.csv"
    history_df.to_csv(history_csv_path, index=False)
    print(f"\n训练历史已保存到: {history_csv_path}")
    print(f"实际训练轮数: {actual_epochs}/{max_epochs}")
    
    # ==================== 绘制训练曲线 ====================
    
    # 调整为 2x3 子图，将 isc 损失也整合到同一张图中
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 图1: 总损失（训练和验证）
    axes[0, 0].plot(range(1, actual_epochs + 1), train_losses, 'b-', linewidth=2, label='Train Loss')
    axes[0, 0].plot(range(1, actual_epochs + 1), val_losses, 'r-', linewidth=2, label='Val Loss')
    axes[0, 0].axvline(x=early_stopping.best_epoch, color='green', linestyle='--', alpha=0.5, label=f'Best Epoch: {early_stopping.best_epoch}')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training & Validation Loss', fontsize=14)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 图2: 所有标签的BCE
    axes[0, 1].plot(range(1, actual_epochs + 1), train_main_losses, 'g-', linewidth=2, marker='o', markersize=4)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Loss', fontsize=12)
    axes[0, 1].set_title('Main Loss', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 图3: 第5个标签的Focal+BCE
    axes[0, 2].plot(range(1, actual_epochs + 1), train_Minority_Class_losses, 'orange', linewidth=2, marker='s', markersize=4)
    axes[0, 2].set_xlabel('Epoch', fontsize=12)
    axes[0, 2].set_ylabel('Loss', fontsize=12)
    axes[0, 2].set_title('Minority_Class loss', fontsize=14)
    axes[0, 2].grid(True, alpha=0.3)
    
    # 图4: 标签关系损失
    axes[1, 0].plot(range(1, actual_epochs + 1), train_Label_Relation_losses, 'purple', linewidth=2, marker='D', markersize=4)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Loss', fontsize=12)
    axes[1, 0].set_title('Label_Relation Loss', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)

    # 图5: isc 损失
    axes[1, 1].plot(range(1, actual_epochs + 1), train_isc_losses, 'cyan', linewidth=2, marker='^', markersize=4)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Loss', fontsize=12)
    axes[1, 1].set_title('isc Loss', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)

    # 隐藏多余子图
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # 保存图像
    curves_path = f"{results_dir}/training_curves.png"
    plt.savefig(curves_path, dpi=300, bbox_inches='tight')
    print(f"训练曲线图已保存到: {curves_path}")
    plt.close()
    
    print(f"\n最终最佳验证损失: {best_val_loss:.4f}")

