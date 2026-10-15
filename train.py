import os
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
from model import CrossAttentionModel
from metrics import compute_co_occurrence_matrix
from isc import _normalize_protein_ids, protein_aware_isc_loss, protein_positive_relation


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


class LabelRelationLoss(nn.Module):
    def __init__(self, co_occurrence_matrix, lambda_weight=0.5, high_threshold=0.5, low_threshold=0.1):
        super(LabelRelationLoss, self).__init__()
        self.register_buffer('C', torch.tensor(co_occurrence_matrix, dtype=torch.float32))
        self.lambda_weight = lambda_weight
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold


        n_labels = co_occurrence_matrix.shape[0]
        self.high_pairs = []
        self.low_pairs = []

        for i in range(n_labels):
            for j in range(i+1, n_labels):
                if co_occurrence_matrix[i, j] > high_threshold:
                    self.high_pairs.append((i, j))
                elif co_occurrence_matrix[i, j] < low_threshold:
                    self.low_pairs.append((i, j))


    def forward(self, probs):
        """
        计算标签关系损失

        参数:
            probs: 模型输出的概率 (batch_size, num_labels)
        """
        relation_loss = 0.0
        count = 0
        
        for i, j in self.high_pairs:
            pair_diff_loss = torch.mean((probs[:, i] - probs[:, j]) ** 2)
            relation_loss += pair_diff_loss
            count += 1
        
        for i, j in self.low_pairs:
            pair_exclusive_loss = torch.mean(probs[:, i] * probs[:, j])
            relation_loss += pair_exclusive_loss
            count += 1
        
        if count > 0:
            relation_loss = relation_loss / count

        return self.lambda_weight * relation_loss


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


    max_probs = probs.max(dim=1, keepdim=True)[0]


    diff = max_probs - probs

    soft_weights = torch.sigmoid((threshold - diff) * temperature)

    return soft_weights


def multi_task_loss(logits, probs, labels, adaptive_soft_weights=None, label_relation_criterion=None,
                    focal_weight=4, bce_weight=4,
                    main_loss_weight=3, Minority_Class_loss_weight=10, Label_Relation_loss_weight=1,
                    adaptive_weight=1.0):

    if adaptive_soft_weights is not None:

        base_loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        consistency_bonus = torch.abs(probs - adaptive_soft_weights)
        weighted_loss = base_loss * (1.0 + adaptive_weight * consistency_bonus)
        main_loss = weighted_loss.mean()
    else:
        main_loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)


    focal_loss = focal_criterion(probs[:, 4], labels[:, 4])
    bce_loss = nn.functional.binary_cross_entropy_with_logits(logits[:, 4], labels[:, 4])
    Minority_Class_loss = focal_weight * focal_loss + bce_weight * bce_loss


    if label_relation_criterion is not None:
        Label_Relation_loss = label_relation_criterion(probs)
    else:
        Label_Relation_loss = 0.0


    total_loss = (main_loss_weight * main_loss +
                  Minority_Class_loss_weight * Minority_Class_loss +
                  Label_Relation_loss_weight * Label_Relation_loss)


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
        self.indices = indices


        self.length = len(indices) if indices is not None else len(labels)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        actual_idx = self.indices[idx] if self.indices is not None else idx


        seq_feat = np.array(self.seq_features[actual_idx], dtype=np.float32)
        attn_mask = np.array(self.attention_masks[actual_idx], dtype=bool)
        img_feat = np.array(self.img_features[actual_idx], dtype=np.float32)
        label = self.labels[actual_idx]
        protein_id = self.protein_ids[actual_idx]


        seq_feat = torch.from_numpy(seq_feat).float()
        attn_mask = torch.from_numpy(attn_mask).bool()
        img_feat = torch.from_numpy(img_feat).float()


        return seq_feat, attn_mask, img_feat, label, protein_id


def training_isc_metrics(
    image_embeddings,
    sequence_embeddings,
    image_protein_ids,
    sequence_protein_ids,
    image_labels,
    sequence_labels,
    temperature=0.07,
):
    positive_relation = protein_positive_relation(
        image_protein_ids,
        sequence_protein_ids,
        image_labels,
        sequence_labels,
        device=image_embeddings.device,
    )
    isc_loss = protein_aware_isc_loss(
        image_embeddings,
        sequence_embeddings,
        image_protein_ids,
        sequence_protein_ids,
        image_labels,
        sequence_labels,
        temperature=temperature,
    )
    mean_positives_per_anchor = positive_relation.sum(dim=1).float().mean()
    return isc_loss, mean_positives_per_anchor


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train model with selectable GPU.")
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU id to use (e.g., 0 or 1). If no GPU is available, CPU will be used.",
    )
    args = parser.parse_args()


    log_stream = DeferredTee(sys.stdout)
    sys.stdout = log_stream
    sys.stderr = log_stream
    atexit.register(log_stream.close)

    if torch.cuda.is_available():

        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"使用设备: {device}\n")


    print("="*70)
    print("加载序列特征")
    seq_features = np.load("embedding/seq_train_embeddings.npy", mmap_mode='r')
    print("加载序列掩码")
    attention_masks = np.load("embedding/seq_train_attention_masks.npy", mmap_mode='r')
    print("加载图像特征")
    img_features = np.load("embedding/train_vitembeddings.npy", mmap_mode='r')
    print()


    print("加载标签")
    label_df = pd.read_csv("dataset/train.csv")
    label_columns = ['cytoplasm', 'endoplasmic reticulum', 'mitochondria', 'nucleus', 'plasma membrane']
    labels = label_df[label_columns].values
    labels = torch.tensor(labels, dtype=torch.float32)


    num_samples = labels.shape[0]


    if len(seq_features.shape) == 3:

        sequence_embedding_dim = seq_features.shape[2]
    elif len(seq_features.shape) == 2:

        sequence_embedding_dim = seq_features.shape[1]
    else:

        first_seq = np.array(seq_features[0])
        sequence_embedding_dim = first_seq.shape[-1]


    if len(img_features.shape) == 3:

        image_embedding_dim = img_features.shape[2]
    elif len(img_features.shape) == 2:

        image_embedding_dim = img_features.shape[1]
    else:

        first_img = np.array(img_features[0])
        image_embedding_dim = first_img.shape[-1]


    num_samples = labels.shape[0]
    if len(seq_features) > num_samples:
        print(f"警告: 序列特征文件包含 {len(seq_features)} 个样本，标签文件包含 {num_samples} 个样本")
        print(f"将只使用前 {num_samples} 个样本")
    elif len(seq_features) < num_samples:
        raise ValueError(f"序列特征文件包含 {len(seq_features)} 个样本，少于标签文件的 {num_samples} 个样本")


    protein_column = "Protein Id"
    if protein_column not in label_df.columns:
        raise ValueError(
            f"训练标签文件缺少必需列 {protein_column!r}；"
            f"现有列: {list(label_df.columns)}"
        )

    protein_ids = label_df[protein_column]
    normalized_protein_ids = _normalize_protein_ids(protein_ids)
    train_protein_count = len(set(normalized_protein_ids))


    labels_np_train = labels.numpy()
    co_occurrence_matrix = compute_co_occurrence_matrix(labels_np_train)


    train_dataset = CustomDataset(
        seq_features,
        attention_masks,
        img_features,
        labels,
        normalized_protein_ids,
    )


    BATCH_SIZE = 32
    NUM_WORKERS = 4
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


    focal_criterion = FocalLoss(alpha=1, gamma=2)

    lambda_weight=1.0
    high_threshold=0.1
    low_threshold=0.001


    label_relation_criterion = LabelRelationLoss(
        co_occurrence_matrix,
        lambda_weight=lambda_weight,
        high_threshold=high_threshold,
        low_threshold=low_threshold
    ).to(device)


    focal_weight = 4
    bce_weight = 4


    main_loss_weight = 3
    Minority_Class_loss_weight = 10
    Label_Relation_loss_weight = 2


    adaptive_weight = 0.8
    temperature = 15.0
    isc_temperature = 0.07
    isc_loss_weight = 1.0


    optimizer = optim.Adam(model.parameters(), lr=1e-5)


    max_epochs = 100

    threshold = 0.4

    run_id = f"1"


    results_dir = f"results/{run_id}"
    os.makedirs(results_dir, exist_ok=True)
    log_file_path = os.path.join(results_dir, "train.txt")
    log_stream.bind(log_file_path)

    checkpoint_path = f"{results_dir}/best_model.pth"


    train_losses = []
    train_main_losses = []
    train_Minority_Class_losses = []
    train_Label_Relation_losses = []
    train_isc_losses = []
    train_mean_positives_per_anchor = []


    for epoch in range(max_epochs):

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


            preds, probs, logits = model.predict_with_adaptive_threshold(
                image_features=img_feat,
                sequence_features=seq_feat,
                attention_mask=attn_mask,
                threshold = threshold
            )


            soft_weights = compute_adaptive_soft_weights(probs, threshold=threshold, temperature=temperature)


            total_loss, loss_dict = multi_task_loss(
                logits,
                probs,
                label,
                adaptive_soft_weights=soft_weights,
                label_relation_criterion=label_relation_criterion,
                focal_weight=focal_weight,
                bce_weight=bce_weight,
                main_loss_weight=main_loss_weight,
                Minority_Class_loss_weight=Minority_Class_loss_weight,
                Label_Relation_loss_weight=Label_Relation_loss_weight,
                adaptive_weight=adaptive_weight
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
                batch_protein_ids,
                label,
                label,
                temperature=isc_temperature,
            )
            total_loss = total_loss + isc_loss_weight * isc_loss


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

        print(f"  Train Loss: {epoch_train_loss:.4f} (Main: {epoch_main_loss:.4f}, Minority_Class: {epoch_Minority_Class_loss:.4f}, Label_Relation: {epoch_Label_Relation_loss:.4f}, isc: {epoch_train_isc_loss:.4f}, positives/anchor: {epoch_mean_positives_per_anchor:.4f})")


        train_losses.append(epoch_train_loss)
        train_main_losses.append(epoch_main_loss)
        train_Minority_Class_losses.append(epoch_Minority_Class_loss)
        train_Label_Relation_losses.append(epoch_Label_Relation_loss)
        train_isc_losses.append(epoch_train_isc_loss)
        train_mean_positives_per_anchor.append(epoch_mean_positives_per_anchor)


        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        print(f"  第 {epoch + 1} 轮模型已保存到: {checkpoint_path}")

    print("\n训练完成！")


    actual_epochs = len(train_losses)

    history_df = pd.DataFrame({
        'epoch': range(1, actual_epochs + 1),
        'train_loss': train_losses,
        'train_main_loss': train_main_losses,
        'train_Minority_Class_loss': train_Minority_Class_losses,
        'train_Label_Relation_loss': train_Label_Relation_losses,
        'train_isc_loss': train_isc_losses,
        'train_mean_positives_per_anchor': train_mean_positives_per_anchor,
    })
    history_csv_path = f"{results_dir}/training_history_label_relation_prot5.csv"
    history_df.to_csv(history_csv_path, index=False)
    print(f"\n训练历史已保存到: {history_csv_path}")
    print(f"实际训练轮数: {actual_epochs}/{max_epochs}")


    fig, axes = plt.subplots(2, 3, figsize=(18, 10))


    axes[0, 0].plot(range(1, actual_epochs + 1), train_losses, 'b-', linewidth=2, label='Train Loss')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training Loss', fontsize=14)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)


    axes[0, 1].plot(range(1, actual_epochs + 1), train_main_losses, 'g-', linewidth=2, marker='o', markersize=4)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Loss', fontsize=12)
    axes[0, 1].set_title('Main Loss', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)


    axes[0, 2].plot(range(1, actual_epochs + 1), train_Minority_Class_losses, 'orange', linewidth=2, marker='s', markersize=4)
    axes[0, 2].set_xlabel('Epoch', fontsize=12)
    axes[0, 2].set_ylabel('Loss', fontsize=12)
    axes[0, 2].set_title('Minority_Class loss', fontsize=14)
    axes[0, 2].grid(True, alpha=0.3)


    axes[1, 0].plot(range(1, actual_epochs + 1), train_Label_Relation_losses, 'purple', linewidth=2, marker='D', markersize=4)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Loss', fontsize=12)
    axes[1, 0].set_title('Label_Relation Loss', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)


    axes[1, 1].plot(range(1, actual_epochs + 1), train_isc_losses, 'cyan', linewidth=2, marker='^', markersize=4)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Loss', fontsize=12)
    axes[1, 1].set_title('isc Loss', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)


    axes[1, 2].axis('off')

    plt.tight_layout()


    curves_path = f"{results_dir}/training_curves.png"
    plt.savefig(curves_path, dpi=300, bbox_inches='tight')
    print(f"训练曲线图已保存到: {curves_path}")
    plt.close()

    print(f"\n最终训练损失: {train_losses[-1]:.4f}")
    print(f"训练最终模型: {checkpoint_path}")

