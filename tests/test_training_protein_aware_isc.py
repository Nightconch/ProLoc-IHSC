import importlib
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
