import torch
import torch.nn as nn
class CrossAttentionModel(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers, image_embedding_dim, sequence_embedding_dim,num_classes, dropout=0.5):
        """图像与序列交叉注意力模型，自注意力和前馈模块重复使用 num_layers 次。"""
        super(CrossAttentionModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.image_projection = nn.Linear(image_embedding_dim, embedding_dim)
        self.sequence_projection = nn.Linear(sequence_embedding_dim, embedding_dim)

        self.cross_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout)
        self.self_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embedding_dim, embedding_dim)
        )

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, image_features, sequence_features, attention_mask=None):
        """接收图像 (B, P, D)、序列 (B, L, D) 和有效位置掩码，返回分类 logits。"""
        batch_size, seq_len, _ = sequence_features.size()
        image_features = self.image_projection(image_features)
        sequence_features = self.sequence_projection(sequence_features)
        sequence_features = sequence_features.permute(1, 0, 2)
        image_features = image_features.permute(1, 0, 2)
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None
        # 图像查询序列，忽略序列填充位置。
        attn_output, _ = self.cross_attention(
            query=image_features,
            key=sequence_features,
            value=sequence_features,
            key_padding_mask=key_padding_mask
        )
        # 重复使用同一套自注意力和前馈参数。
        for _ in range(self.num_layers):
            self_attn_output, _ = self.self_attention(query=attn_output, key=attn_output, value=attn_output)
            attn_output = self.layer_norm1(attn_output + self_attn_output)
            ff_output = self.feed_forward(attn_output)
            attn_output = self.layer_norm2(attn_output + ff_output)

        attn_output = attn_output.permute(1, 0, 2)

        pooled_output = attn_output.mean(dim=1)
        logits = self.classifier(pooled_output)

        return logits

    def get_global_embeddings(self, image_features, sequence_features, attention_mask=None):
        image_features = self.image_projection(image_features)
        sequence_features = self.sequence_projection(sequence_features)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            sequence_features = sequence_features * mask
            lengths = mask.sum(dim=1).clamp(min=1.0)
            sequence_global = sequence_features.sum(dim=1) / lengths
        else:
            sequence_global = sequence_features.mean(dim=1)
        image_global = image_features.mean(dim=1)
        return image_global, sequence_global

    def predict_with_adaptive_threshold(self, image_features, sequence_features, threshold=0.5, attention_mask=None):

        logits = self.forward(image_features, sequence_features, attention_mask)

        probabilities = torch.sigmoid(logits)

        batch_size, num_classes = probabilities.shape
        predictions = torch.zeros_like(probabilities, dtype=torch.float32)

        # 根据各类别概率与样本最高概率的差值预测标签。
        max_probs = probabilities.max(dim=1, keepdim=True)[0]
        diff = max_probs - probabilities
        predictions = (diff < threshold).float()

        return predictions, probabilities, logits
