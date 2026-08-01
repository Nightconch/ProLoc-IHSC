import importlib
import math
import sys
import types

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader


def import_train_without_feature_extractors():
    fake_vit = types.ModuleType("vit")
    fake_vit.ViTFeatureExtractorModel = object
    fake_prott5 = types.ModuleType("prott5")
    fake_prott5.ProteinEmbeddingExtractor = object

    previous_vit = sys.modules.get("vit")
    previous_prott5 = sys.modules.get("prott5")
    sys.modules["vit"] = fake_vit
    sys.modules["prott5"] = fake_prott5
    try:
        sys.modules.pop("train", None)
        return importlib.import_module("train")
    finally:
        if previous_vit is None:
            sys.modules.pop("vit", None)
        else:
            sys.modules["vit"] = previous_vit
        if previous_prott5 is None:
            sys.modules.pop("prott5", None)
        else:
            sys.modules["prott5"] = previous_prott5


train = import_train_without_feature_extractors()


def test_post_split_normalized_ids_support_numpy_index_selection():
    """Break caught: a normalized Python list cannot be fancy-indexed by split arrays."""
    protein_ids = np.repeat(
        np.array([" P0 ", "P1", " P2", "P3 "], dtype=object), 2
    )
    protein_labels = np.array(
        [
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ]
    )
    labels = np.repeat(protein_labels, 2, axis=0)
    train_indices, validation_indices = train.split_by_protein(
        labels, protein_ids, n_splits=2
    )
    normalize_for_training = getattr(
        train, "_normalize_training_protein_ids", train._normalize_protein_ids
    )

    normalized_ids = normalize_for_training(protein_ids)
    training_ids = normalized_ids[train_indices]
    validation_ids = normalized_ids[validation_indices]

    assert training_ids.tolist() == [
        str(protein_ids[index]).strip() for index in train_indices
    ]
    assert validation_ids.tolist() == [
        str(protein_ids[index]).strip() for index in validation_indices
    ]


def test_training_isc_uses_duplicate_ids_from_indexed_dataset_batch():
    """Break caught: dropping IDs or using diagonal ISC penalizes same-protein pairs."""
    sequence_features = np.zeros((3, 2, 2), dtype=np.float32)
    attention_masks = np.ones((3, 2), dtype=bool)
    image_features = np.zeros((3, 2), dtype=np.float32)
    labels = torch.zeros((3, 5), dtype=torch.float32)
    protein_ids = [" BRCA1 ", " P53 ", "P53"]

    dataset = train.CustomDataset(
        sequence_features,
        attention_masks,
        image_features,
        labels,
        protein_ids=protein_ids,
        indices=np.array([1, 2, 0]),
    )
    _, _, _, _, batch_protein_ids = next(iter(DataLoader(dataset, batch_size=3)))

    image_embeddings = torch.eye(3)
    sequence_embeddings = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    isc_loss, mean_positives_per_anchor = train.training_isc_metrics(
        image_embeddings, sequence_embeddings, batch_protein_ids
    )

    assert batch_protein_ids == ("P53", "P53", "BRCA1")
    assert isc_loss.item() < 1e-5
    assert mean_positives_per_anchor.item() == pytest.approx(5 / 3)


def test_validation_isc_accumulator_is_protein_aware_and_sample_weighted():
    """Break caught: diagonal ISC or omitted final-batch accounting skews validation ISC."""
    first_image_embeddings = torch.eye(3)
    first_sequence_embeddings = torch.eye(3)
    final_image_embeddings = torch.eye(2)
    final_sequence_embeddings = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0]]
    )

    _, weighted_loss, processed_samples = train.accumulate_validation_isc(
        0.0,
        0,
        first_image_embeddings,
        first_sequence_embeddings,
        ("P53", "BRCA1", "EGFR"),
        temperature=1.0,
    )
    _, weighted_loss, processed_samples = train.accumulate_validation_isc(
        weighted_loss,
        processed_samples,
        final_image_embeddings,
        final_sequence_embeddings,
        ("TP53", "TP53"),
        temperature=1.0,
    )

    expected_epoch_isc = 3 * math.log(1 + 2 * math.exp(-1)) / 5

    assert processed_samples == 5
    assert weighted_loss / processed_samples == pytest.approx(expected_epoch_isc)
