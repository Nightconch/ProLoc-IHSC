#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
蛋白质亚细胞定位预测脚本
支持单张图片+序列预测，以及批量文件夹预测
输出预测的标签（不包含概率）
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import re
from tqdm import tqdm

# 导入必要的模块
from model import CrossAttentionModel
from prott5 import ProteinEmbeddingExtractor
from vit import ViTFeatureExtractorModel


class ProteinLocalizationPredictor:
    """蛋白质亚细胞定位预测器"""

    def __init__(self, model_path, threshold=0.4, device=None):
        """
        初始化预测器

        参数:
            model_path: 模型权重文件路径
            threshold: 预测阈值
            device: 计算设备 ('cuda' 或 'cpu')
        """
        self.threshold = threshold
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # 标签名称
        self.label_names = ['cytoplasm', 'endoplasmic_reticulum', 'mitochondria', 'nucleus', 'plasma_membrane']

        # 初始化特征提取器
        print("正在初始化序列特征提取器...")
        self.seq_extractor = ProteinEmbeddingExtractor(batch_size=1)

        print("正在初始化图像特征提取器...")
        self.img_extractor = ViTFeatureExtractorModel()

        # 加载模型
        print(f"正在加载模型: {model_path}")
        self.model = self._load_model(model_path)
        print("模型加载完成！")

    def _load_model(self, model_path):
        """加载训练好的模型"""
        # 模型参数（需要与训练时一致）
        embedding_dim = 512
        num_heads = 8
        num_layers = 6
        num_classes = 5

        # 获取特征维度（通过提取一个样本来确定）
        # 这里使用固定值，如果需要可以动态获取
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

        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(self.device)
        model.eval()

        return model

    def _extract_sequence_features(self, sequence):
        """
        提取单个序列的特征

        参数:
            sequence: 蛋白质序列字符串

        返回:
            embedding: 序列嵌入 (1, seq_len, embedding_dim)
            attention_mask: 注意力掩码 (1, seq_len)
        """
        # 替换特殊字符并添加空格
        sequence = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))

        # 编码
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

        # 提取特征
        with torch.no_grad():
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = self.seq_extractor.model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state  # (1, seq_len, embedding_size)

        return embeddings.cpu(), attention_mask.cpu()

    def _extract_image_features(self, image_path):
        """
        提取单张图像的特征

        参数:
            image_path: 图像文件路径

        返回:
            features: 图像特征 (1, num_patches, embedding_dim)
        """
        return self.img_extractor.extract_features(image_path)

    def predict_single(self, image_path, sequence):
        """
        预测单个样本

        参数:
            image_path: 图像文件路径
            sequence: 蛋白质序列字符串

        返回:
            predictions_dict: 包含每个标签预测结果的字典
        """
        # 提取特征
        seq_feat, attn_mask = self._extract_sequence_features(sequence)
        img_feat = self._extract_image_features(image_path)

        # 转换为张量并移动到设备
        # 如果已经是张量，使用 .to() 直接转换；如果是numpy数组，使用 torch.from_numpy()
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

        # 预测
        with torch.no_grad():
            preds, probs, logits = self.model.predict_with_adaptive_threshold(
                image_features=img_feat,
                sequence_features=seq_feat,
                attention_mask=attn_mask,
                threshold=self.threshold
            )

        # 获取预测标签
        preds_np = preds.cpu().numpy()[0]  # (num_classes,)

        # 返回每个位置的预测结果
        predictions_dict = {}
        for i, label_name in enumerate(self.label_names):
            predictions_dict[label_name] = int(preds_np[i])

        return predictions_dict

    def predict_batch_from_folders(self, img_folder, seq_folder, output_csv):
        """
        从图像文件夹和序列文件夹批量预测

        参数:
            img_folder: 图像文件夹路径
            seq_folder: 序列文件夹路径（包含.fasta文件）
            output_csv: 输出CSV文件路径
        """
        # 获取所有图像文件
        image_files = [f for f in os.listdir(img_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # 获取所有fasta文件
        fasta_files = [f for f in os.listdir(seq_folder) if f.lower().endswith('.fasta')]

        if len(image_files) == 0:
            print(f"错误: 在 {img_folder} 中未找到图像文件")
            return

        if len(fasta_files) == 0:
            print(f"错误: 在 {seq_folder} 中未找到.fasta文件")
            return

        # 创建fasta文件的前缀映射（去掉扩展名）
        fasta_dict = {}
        for fasta_file in fasta_files:
            prefix = os.path.splitext(fasta_file)[0]  # 去掉.fasta扩展名
            fasta_dict[prefix] = fasta_file

        results = []
        matched_count = 0
        unmatched_images = []

        for img_file in tqdm(image_files, desc="预测中"):
            try:
                # 获取图像文件的前缀（去掉扩展名）
                img_prefix = os.path.splitext(img_file)[0]

                # 查找匹配的fasta文件
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

                # 读取序列
                sequence = self.read_fasta(fasta_path)

                # 预测
                predictions_dict = self.predict_single(image_path, sequence)

                # 保存结果
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

        # 保存结果
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv, index=False)

        # 打印前几个结果
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
        """
        读取FASTA文件

        参数:
            fasta_file: FASTA文件路径

        返回:
            sequence: 蛋白质序列字符串
        """
        with open(fasta_file, 'r') as f:
            lines = f.readlines()

        # 跳过标题行，合并序列行
        sequence = ''.join([line.strip() for line in lines if not line.startswith('>')])
        return sequence


def main():
    parser = argparse.ArgumentParser(description='蛋白质亚细胞定位预测')
    parser.add_argument('--model', type=str, required=True, help='模型权重文件路径')
    parser.add_argument('--ihc', type=str, required=True,
                        help='单个图像文件或包含图像的文件夹')
    parser.add_argument('--sequence', type=str, help='单个序列文件或包含序列的文件夹')
    parser.add_argument('--output', type=str, help='输出CSV文件路径（批量模式，可选，默认为predictions.csv）')

    args = parser.parse_args()

    # 固定阈值
    threshold = 0.4

    # 初始化预测器
    predictor = ProteinLocalizationPredictor(
        model_path=args.model,
        threshold=threshold
    )

    # 自动识别模式
    input_path = args.ihc

    # 判断输入是文件还是文件夹
    if os.path.isfile(input_path):
        # 单文件模式
        if not args.sequence:
            print("错误: 单文件模式需要 --sequence 参数指定序列")
            sys.exit(1)

        # 检查序列是否是文件路径
        if os.path.exists(args.sequence) and args.sequence.endswith('.fasta'):
            sequence = predictor.read_fasta(args.sequence)
        else:
            sequence = args.sequence

        predictions_dict = predictor.predict_single(input_path, sequence)

        # 打印结果
        print("\n预测结果:")
        print("=" * 60)
        for label_name, pred_value in predictions_dict.items():
            status = "✓" if pred_value == 1 else "✗"
            print(f"  {status} {label_name}: {pred_value}")
        print("=" * 60)

        # 保存到CSV
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
        # 批量模式 - 检查是否有序列文件夹
        if args.sequence and os.path.isdir(args.sequence):
            # 从图像文件夹和序列文件夹批量预测
            img_folder = input_path
            seq_folder = args.sequence

            # 设置输出路径
            if args.output:
                output_csv = args.output
            else:
                output_csv = "predictions.csv"

            predictor.predict_batch_from_folders(img_folder, seq_folder, output_csv)

        else:
            # 从CSV文件批量预测
            # 查找CSV文件
            csv_files = [f for f in os.listdir(input_path) if f.endswith('.csv')]

            if not csv_files:
                print(f"错误: 在文件夹 {input_path} 中未找到CSV文件")
                sys.exit(1)

            csv_file = os.path.join(input_path, csv_files[0])
            img_folder = input_path

            # 设置输出路径
            if args.output:
                output_csv = args.output
            else:
                output_csv = "result/predictions.csv"

            predictor.predict_batch(csv_file, img_folder, output_csv)

    else:
        print(f"错误: 输入路径不存在: {input_path}")
        sys.exit(1)


if __name__ == '__main__':
    main()
