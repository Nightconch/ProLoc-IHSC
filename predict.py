#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""蛋白质亚细胞定位预测，支持单样本和图像/FASTA 文件夹批量处理。"""

import os
import sys
import argparse
import torch
import pandas as pd
import re
from tqdm import tqdm

from model import CrossAttentionModel
from prott5 import ProteinEmbeddingExtractor
from vit import ViTFeatureExtractorModel


class ProteinLocalizationPredictor:
    """蛋白质亚细胞定位预测器"""

    def __init__(self, model_path, threshold=0.4, device=None):
        """按给定权重、预测阈值和设备初始化预测器。"""
        self.threshold = threshold
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        self.label_names = ['cytoplasm', 'endoplasmic_reticulum', 'mitochondria', 'nucleus', 'plasma_membrane']

        print("正在初始化序列特征提取器...")
        self.seq_extractor = ProteinEmbeddingExtractor(batch_size=1)

        print("正在初始化图像特征提取器...")
        self.img_extractor = ViTFeatureExtractorModel()

        print(f"正在加载模型: {model_path}")
        self.model = self._load_model(model_path)
        print("模型加载完成！")

    def _load_model(self, model_path):
        """加载训练好的模型"""
        embedding_dim = 512
        num_heads = 8
        num_layers = 6
        num_classes = 5

        image_embedding_dim = 768  # ViT-base 输出维度
        sequence_embedding_dim = 1024  # ProtT5 输出维度

        model = CrossAttentionModel(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_classes=num_classes,
            image_embedding_dim=image_embedding_dim,
            sequence_embedding_dim=sequence_embedding_dim
        )

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(self.device)
        model.eval()

        return model

    def _extract_sequence_features(self, sequence):
        """返回序列嵌入 (1, L, D) 和有效位置掩码 (1, L)。"""
        # U、Z、O、B 替换为 X，氨基酸之间用空格分隔。
        sequence = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))

        encoding = self.seq_extractor.tokenizer.batch_encode_plus(
            [sequence],
            add_special_tokens=True,
            padding=True,
            return_tensors='pt',
            truncation=True,
            max_length=6000
        )

        input_ids = encoding['input_ids'].to(self.seq_extractor.device)
        attention_mask = encoding['attention_mask'].to(self.seq_extractor.device)

        with torch.no_grad():
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = self.seq_extractor.model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state

        return embeddings.cpu(), attention_mask.cpu()

    def _extract_image_features(self, image_path):
        """返回图像 token 嵌入 (1, N, D)。"""
        return self.img_extractor.extract_features(image_path)

    def predict_single(self, image_path, sequence):
        """返回每个定位标签的 0/1 预测结果。"""
        seq_feat, attn_mask = self._extract_sequence_features(sequence)
        img_feat = self._extract_image_features(image_path)

        if isinstance(seq_feat, torch.Tensor):
            seq_feat = seq_feat.to(self.device, dtype=torch.float32)
        else:
            seq_feat = torch.from_numpy(seq_feat).to(self.device, dtype=torch.float32)

        if isinstance(attn_mask, torch.Tensor):
            attn_mask = attn_mask.to(self.device, dtype=torch.bool)
        else:
            attn_mask = torch.from_numpy(attn_mask).to(self.device, dtype=torch.bool)

        if isinstance(img_feat, torch.Tensor):
            img_feat = img_feat.to(self.device, dtype=torch.float32)
        else:
            img_feat = torch.from_numpy(img_feat).to(self.device, dtype=torch.float32)

        with torch.no_grad():
            preds, probs, logits = self.model.predict_with_adaptive_threshold(
                image_features=img_feat,
                sequence_features=seq_feat,
                attention_mask=attn_mask,
                threshold=self.threshold
            )

        preds_np = preds.cpu().numpy()[0]

        predictions_dict = {}
        for i, label_name in enumerate(self.label_names):
            predictions_dict[label_name] = int(preds_np[i])

        return predictions_dict

    def predict_batch_from_folders(self, img_folder, seq_folder, output_csv):
        """按相同文件名主干匹配图像与 FASTA，输出预测 CSV。"""
        image_files = [f for f in os.listdir(img_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        fasta_files = [f for f in os.listdir(seq_folder) if f.lower().endswith('.fasta')]

        if len(image_files) == 0:
            print(f"错误: 在 {img_folder} 中未找到图像文件")
            return

        if len(fasta_files) == 0:
            print(f"错误: 在 {seq_folder} 中未找到.fasta文件")
            return

        fasta_dict = {}
        for fasta_file in fasta_files:
            prefix = os.path.splitext(fasta_file)[0]
            fasta_dict[prefix] = fasta_file

        results = []
        matched_count = 0
        unmatched_images = []

        for img_file in tqdm(image_files, desc="预测中"):
            try:
                img_prefix = os.path.splitext(img_file)[0]

                if img_prefix not in fasta_dict:
                    unmatched_images.append(img_file)
                    result = {
                        'image': img_file,
                        'sequence_file': 'NOT_FOUND'
                    }
                    for label_name in self.label_names:
                        result[f'pred_{label_name}'] = -1
                    results.append(result)
                    continue

                matched_count += 1
                fasta_file = fasta_dict[img_prefix]
                image_path = os.path.join(img_folder, img_file)
                fasta_path = os.path.join(seq_folder, fasta_file)

                sequence = self.read_fasta(fasta_path)

                predictions_dict = self.predict_single(image_path, sequence)

                result = {
                    'image': img_file,
                    'sequence_file': fasta_file
                }
                for label_name in self.label_names:
                    result[f'pred_{label_name}'] = predictions_dict[label_name]
                results.append(result)

            except Exception as e:
                result = {
                    'image': img_file,
                    'sequence_file': 'ERROR'
                }
                for label_name in self.label_names:
                    result[f'pred_{label_name}'] = -1
                results.append(result)

        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv, index=False)

        print("\n预测完成！前5个样本的结果:")
        print("=" * 80)
        for i in range(min(5, len(results))):
            print(f"\n样本 {i+1}: {results[i]['image']} <-> {results[i]['sequence_file']}")
            for label_name in self.label_names:
                pred_val = results[i][f'pred_{label_name}']
                if pred_val == -1:
                    print(f"  ✗ {label_name}: ERROR")
                else:
                    status = "✓" if pred_val == 1 else "✗"
                    print(f"  {status} {label_name}: {pred_val}")
        print("\n" + "=" * 80)
        print(f"所有结果已保存到: {output_csv}")
        print(f"共处理 {len(image_files)} 个图像文件")
        print(f"成功匹配 {matched_count} 对")
        if unmatched_images:
            print(f"未匹配的图像 ({len(unmatched_images)}): {', '.join(unmatched_images[:5])}" +
                  ("..." if len(unmatched_images) > 5 else ""))


    def read_fasta(self, fasta_file):
        """跳过 FASTA 标题行，合并序列行。"""
        with open(fasta_file, 'r') as f:
            lines = f.readlines()

        sequence = ''.join([line.strip() for line in lines if not line.startswith('>')])
        return sequence


def main():
    parser = argparse.ArgumentParser(description='蛋白质亚细胞定位预测')
    parser.add_argument('--model', type=str, default='results/best_model.pth', help='模型权重文件路径')
    parser.add_argument('--ihc', type=str, required=True,
                        help='单个图像文件或包含图像的文件夹')
    parser.add_argument('--sequence', type=str, help='单个序列文件或包含序列的文件夹')
    parser.add_argument('--output', type=str, help='输出CSV文件路径（批量模式，可选，默认为predictions.csv）')

    args = parser.parse_args()

    threshold = 0.4

    predictor = ProteinLocalizationPredictor(
        model_path=args.model,
        threshold=threshold
    )

    input_path = args.ihc

    if os.path.isfile(input_path):
        if not args.sequence:
            print("错误: 单文件模式需要 --sequence 参数指定序列")
            sys.exit(1)

        if os.path.exists(args.sequence) and args.sequence.endswith('.fasta'):
            sequence = predictor.read_fasta(args.sequence)
        else:
            sequence = args.sequence

        predictions_dict = predictor.predict_single(input_path, sequence)

        print("\n预测结果:")
        print("=" * 60)
        for label_name, pred_value in predictions_dict.items():
            status = "✓" if pred_value == 1 else "✗"
            print(f"  {status} {label_name}: {pred_value}")
        print("=" * 60)

        if args.output:
            output_csv = args.output
        else:
            output_csv = "prediction_result.csv"

        result_df = pd.DataFrame([{
            'image': os.path.basename(input_path),
            **{f'pred_{k}': v for k, v in predictions_dict.items()}
        }])
        result_df.to_csv(output_csv, index=False)
        print(f"结果已保存到: {output_csv}")

    elif os.path.isdir(input_path):
        if args.sequence and os.path.isdir(args.sequence):
            img_folder = input_path
            seq_folder = args.sequence

            if args.output:
                output_csv = args.output
            else:
                output_csv = "predictions.csv"

            predictor.predict_batch_from_folders(img_folder, seq_folder, output_csv)

        else:
            print("错误: 批量模式需要 --sequence 参数指定 FASTA 序列文件夹")
            sys.exit(1)

    else:
        print(f"错误: 输入路径不存在: {input_path}")
        sys.exit(1)


if __name__ == '__main__':
    main()
