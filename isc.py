"""Protein-aware image-sequence contrastive loss."""

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


def protein_positive_relation(image_protein_ids, sequence_protein_ids, *, device=None):
    """Return the same-protein image--sequence positive relation."""
    normalized_image_ids = _normalize_protein_ids(image_protein_ids)
    normalized_sequence_ids = _normalize_protein_ids(sequence_protein_ids)
    return torch.tensor(
        [
            [image_id == sequence_id for sequence_id in normalized_sequence_ids]
            for image_id in normalized_image_ids
        ],
        dtype=torch.bool,
        device=device,
    )


def _multi_positive_cross_entropy(logits, positive_relation):
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

    positive_logits = logits.masked_fill(~positive_relation, float("-inf"))
    return (
        torch.logsumexp(logits, dim=1)
        - torch.logsumexp(positive_logits, dim=1)
    ).mean()


def protein_aware_isc_loss(
    image_embeddings,
    sequence_embeddings,
    image_protein_ids,
    sequence_protein_ids,
    temperature=0.07,
):
    """Compute symmetric ISC loss where every same-protein pair is positive."""
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

    positive_relation = protein_positive_relation(
        normalized_image_ids,
        normalized_sequence_ids,
        device=image_embeddings.device,
    )

    image_normalized = functional.normalize(image_embeddings, p=2, dim=1)
    sequence_normalized = functional.normalize(sequence_embeddings, p=2, dim=1)
    logits = image_normalized @ sequence_normalized.T / temperature
    return (
        _multi_positive_cross_entropy(logits, positive_relation)
        + _multi_positive_cross_entropy(logits.T, positive_relation.T)
    ) * 0.5
