"""Label-gated image-sequence contrastive loss using protein identity."""

import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional


def _normalize_protein_ids(protein_ids):
    if protein_ids is None:
        raise ValueError("Protein IDs are missing")

    normalized_ids = []
    for index, protein_id in enumerate(protein_ids):
        is_missing = pd.isna(protein_id)
        if protein_id is None or (
            isinstance(is_missing, (bool, np.bool_)) and is_missing
        ):
            raise ValueError(f"Protein ID at index {index} is missing")
        normalized_id = str(protein_id).strip()
        if not normalized_id:
            raise ValueError(f"Protein ID at index {index} is blank")
        normalized_ids.append(normalized_id)
    return normalized_ids


def _normalize_localization_labels(labels, sample_count, name, device):
    try:
        label_tensor = torch.as_tensor(labels, device=device)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{name} labels must be a binary two-dimensional array") from error
    if label_tensor.ndim != 2:
        raise ValueError(f"{name} labels must be two-dimensional")
    if label_tensor.shape[0] != sample_count:
        raise ValueError(f"{name} labels and Protein IDs must have the same number of samples")
    if sample_count == 0:
        raise ValueError(f"{name} batch must contain at least one sample")
    if label_tensor.shape[1] == 0:
        raise ValueError(f"{name} labels must contain at least one class")
    if not torch.all((label_tensor == 0) | (label_tensor == 1)):
        raise ValueError(f"{name} labels must be binary (0 or 1)")
    return label_tensor.bool()


def _label_gated_relations(
    image_protein_ids, sequence_protein_ids, image_labels, sequence_labels, *, device=None,
):
    normalized_image_ids = _normalize_protein_ids(image_protein_ids)
    normalized_sequence_ids = _normalize_protein_ids(sequence_protein_ids)
    relation_device = torch.device("cpu") if device is None else torch.device(device)
    image_labels = _normalize_localization_labels(
        image_labels, len(normalized_image_ids), "Image", relation_device,
    )
    sequence_labels = _normalize_localization_labels(
        sequence_labels, len(normalized_sequence_ids), "Sequence", relation_device,
    )
    if image_labels.shape[1] != sequence_labels.shape[1]:
        raise ValueError("Image and sequence labels must have the same number of classes")

    same_protein = torch.tensor(
        [
            [image_id == sequence_id for sequence_id in normalized_sequence_ids]
            for image_id in normalized_image_ids
        ],
        dtype=torch.bool,
        device=relation_device,
    )
    same_labels = (image_labels[:, None, :] == sequence_labels[None, :, :]).all(dim=2)
    overlapping_labels = (image_labels[:, None, :] & sequence_labels[None, :, :]).any(dim=2)
    positive_relation = same_protein & same_labels
    negative_relation = ~same_protein & ~overlapping_labels
    return positive_relation, negative_relation


def protein_positive_relation(
    image_protein_ids, sequence_protein_ids, image_labels, sequence_labels, *, device=None,
):
    """Mark pairs with the same protein ID and exactly equal binary label sets."""
    positive_relation, _ = _label_gated_relations(
        image_protein_ids, sequence_protein_ids, image_labels, sequence_labels, device=device,
    )
    return positive_relation


def _multi_positive_cross_entropy(logits, positive_relation, candidate_relation):
    """Use positive mass over positive-plus-negative mass; exclude ignored pairs."""
    if logits.ndim != 2:
        raise ValueError("Logits must be two-dimensional")
    if not isinstance(positive_relation, torch.Tensor):
        raise ValueError("Positive relation must be a Tensor")
    if positive_relation.dtype is not torch.bool:
        raise ValueError("Positive relation must be boolean")
    if positive_relation.shape != logits.shape:
        raise ValueError("Positive relation must have the same shape as logits")
    if positive_relation.device != logits.device:
        raise ValueError("Positive relation must be on the logits device")
    if not torch.all(positive_relation.any(dim=1)):
        raise ValueError("Each sample must have at least one positive pair")
    if not isinstance(candidate_relation, torch.Tensor) or candidate_relation.dtype is not torch.bool:
        raise ValueError("Candidate relation must be a boolean Tensor")
    if candidate_relation.shape != logits.shape:
        raise ValueError("Candidate relation must have the same shape as logits")
    if candidate_relation.device != logits.device:
        raise ValueError("Candidate relation must be on the logits device")
    if torch.any(positive_relation & ~candidate_relation):
        raise ValueError("Candidate relation must include every positive pair")

    positive_logits = logits.masked_fill(~positive_relation, float("-inf"))
    candidate_logits = logits.masked_fill(~candidate_relation, float("-inf"))
    return (
        torch.logsumexp(candidate_logits, dim=1)
        - torch.logsumexp(positive_logits, dim=1)
    ).mean()


def protein_aware_isc_loss(
    image_embeddings,
    sequence_embeddings,
    image_protein_ids,
    sequence_protein_ids,
    image_labels,
    sequence_labels,
    temperature=0.07,
):
    """Compute symmetric label-gated ISC from aligned binary localization labels.

    Positives have identical IDs and label sets. Negatives have different IDs
    and disjoint label sets. All other pairs are excluded from both sums.
    Every anchor in both modalities must have at least one qualified positive.
    """
    if image_embeddings.ndim != 2 or sequence_embeddings.ndim != 2:
        raise ValueError("Image and sequence embeddings must be two-dimensional")
    if image_embeddings.shape[1] != sequence_embeddings.shape[1]:
        raise ValueError("Image and sequence embeddings must have the same feature width")
    if image_embeddings.device != sequence_embeddings.device:
        raise ValueError("Image and sequence embeddings must be on the same device")
    try:
        valid_temperature = math.isfinite(temperature) and temperature > 0
    except (TypeError, ValueError):
        valid_temperature = False
    if not valid_temperature:
        raise ValueError("Temperature must be finite and positive")

    normalized_image_ids = _normalize_protein_ids(image_protein_ids)
    normalized_sequence_ids = _normalize_protein_ids(sequence_protein_ids)
    if len(normalized_image_ids) != image_embeddings.shape[0]:
        raise ValueError("Image Protein IDs and embeddings must have the same number of samples")
    if len(normalized_sequence_ids) != sequence_embeddings.shape[0]:
        raise ValueError("sequence Protein IDs and embeddings must have the same number of samples")

    positive_relation, negative_relation = _label_gated_relations(
        normalized_image_ids,
        normalized_sequence_ids,
        image_labels,
        sequence_labels,
        device=image_embeddings.device,
    )

    image_normalized = functional.normalize(image_embeddings, p=2, dim=1)
    sequence_normalized = functional.normalize(sequence_embeddings, p=2, dim=1)
    logits = image_normalized @ sequence_normalized.T / temperature
    candidate_relation = positive_relation | negative_relation
    return (
        _multi_positive_cross_entropy(logits, positive_relation, candidate_relation)
        + _multi_positive_cross_entropy(logits.T, positive_relation.T, candidate_relation.T)
    ) * 0.5
