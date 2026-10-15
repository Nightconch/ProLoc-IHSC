import os
import argparse
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from model import CrossAttentionModel
from tqdm import tqdm
import pandas as pd
from metrics import sample_based_metrics, location_based_metrics_micro, print_sample_based_metrics, print_location_based_metrics
import sys
from vit import ViTFeatureExtractorModel
from prott5 import ProteinEmbeddingExtractor

class CustomDataset(Dataset):
    def __init__(self, seq_features, attention_masks, img_features, labels=None):
        self.seq_features = seq_features
        self.attention_masks = attention_masks
        self.img_features = img_features
        self.labels = labels

    def __len__(self):
        return len(self.seq_features)

    def __getitem__(self, idx):
        seq_feat = torch.from_numpy(np.array(self.seq_features[idx], dtype=np.float32, copy=True))
        attn_mask = torch.from_numpy(np.array(self.attention_masks[idx], dtype=bool, copy=True))
        img_feat = torch.from_numpy(np.array(self.img_features[idx], dtype=np.float32, copy=True))
        if self.labels is not None:
            label = self.labels[idx]
            return seq_feat, attn_mask, img_feat, label
        else:
            return seq_feat, attn_mask, img_feat


def test_model(model, test_loader, device, has_labels, threshold, save_dir=None, file_names=None):
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

            preds, probs, logits = model.predict_with_adaptive_threshold(
                image_features=img_feat,
                sequence_features=seq_feat,
                attention_mask=attn_mask,
                threshold=threshold
            )

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
            if file_names is not None:
                output_df.insert(0, 'image', list(file_names))
            output_df.to_csv(results_path, index=False)
            print(f"预测概率与标签结果已保存到: {results_path}")

    if has_labels:

        ATR, Acc, Pre, Rec, F1 = sample_based_metrics(all_preds, all_labels)
        print_sample_based_metrics(ATR, Acc, Pre, Rec, F1)

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


class Tee:

    def __init__(self, *streams):
        self.streams = streams
        try:
            self.encoding = getattr(streams[0], 'encoding', 'utf-8')
        except Exception:
            self.encoding = 'utf-8'

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        try:
            return any((getattr(s, 'isatty', lambda: False)() for s in self.streams))
        except Exception:
            return False

    def fileno(self):
        for s in self.streams:
            try:
                return s.fileno()
            except Exception:
                continue
        raise OSError('No valid fileno available')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Evaluate a selected test dataset.')
    parser.add_argument('--data-dir', type=Path, default=Path('dataset_test'))
    parser.add_argument('--model', type=Path, default=Path('results/best_model.pth'))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--seq-features', type=Path, default=Path('embedding/seq_test_embeddings.npy'))
    parser.add_argument('--attention-masks', type=Path, default=Path('embedding/seq_test_attention_masks.npy'))
    parser.add_argument('--image-features', type=Path, default=Path('embedding/test_vitembeddings.npy'))
    parser.add_argument('--reuse-features', action='store_true', help='Read existing feature files without extracting them again.')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error('--batch-size must be positive')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = args.output_dir or args.model.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.data_dir / 'test.csv'

    with (output_dir / 'test.txt').open('w', encoding='utf-8', buffering=1) as log_file:
        with redirect_stdout(Tee(sys.stdout, log_file)), redirect_stderr(Tee(sys.stderr, log_file)):
            print(f'使用设备: {device}')
            print(f'测试数据: {csv_path.resolve()}')
            print(f'模型权重: {args.model.resolve()}')
            if not args.reuse_features:
                for feature_path in (args.seq_features, args.attention_masks, args.image_features):
                    feature_path.parent.mkdir(parents=True, exist_ok=True)
                extractor_seq = ProteinEmbeddingExtractor()
                extractor_seq.process_file(str(csv_path), str(args.seq_features), str(args.attention_masks))
                extractor_img = ViTFeatureExtractorModel()
                extractor_img.extract_features_for_all(str(csv_path), str(args.data_dir / 'test'), str(args.image_features))
            else:
                print('复用已提供的特征文件，按样本读取。')

            seq_features = np.load(args.seq_features, mmap_mode='r')
            attention_masks = np.load(args.attention_masks, mmap_mode='r')
            img_features = np.load(args.image_features, mmap_mode='r')
            label_columns = ['cytoplasm', 'endoplasmic reticulum', 'mitochondria', 'nucleus', 'plasma membrane']
            try:
                label_frame = pd.read_csv(csv_path)
                labels = torch.tensor(label_frame[label_columns].to_numpy(), dtype=torch.float32)
                file_names = label_frame['File Name'].tolist()
            except FileNotFoundError:
                print('没有找到标签文件，仅输出预测结果。')
                labels = None
                file_names = None
            has_labels = labels is not None
            counts = {
                'sequence': len(seq_features), 'masks': len(attention_masks), 'image': len(img_features),
            }
            if has_labels:
                counts['labels'] = len(labels)
            if len(set(counts.values())) != 1:
                raise ValueError(f'Feature and label sample counts differ: {counts}')
            if len(seq_features) == 0:
                raise ValueError('Test dataset must contain at least one sample')
            print(f'数据一致性检查通过，共 {len(seq_features)} 个样本')
            print(f'序列特征: {seq_features.shape}; 图像特征: {img_features.shape}')

            model = CrossAttentionModel(
                embedding_dim=512, num_heads=8, num_layers=6, num_classes=5,
                image_embedding_dim=img_features.shape[2],
                sequence_embedding_dim=seq_features.shape[2],
            )
            checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
            state = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
            model.load_state_dict(state)
            model = model.to(device)
            test_dataset = CustomDataset(seq_features, attention_masks, img_features, labels)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
            metrics = test_model(
                model, test_loader, device, has_labels, threshold=0.4,
                save_dir=str(output_dir), file_names=file_names,
            )
    return metrics


if __name__ == '__main__':
    main()
