"""Protein-aware image-sequence contrastive loss."""

import math

import torch
import torch.nn.functional as functional


def _normalize_protein_ids(protein_ids):
    if protein_ids is None:
        raise ValueError("Protein IDs are missing")

    normalized_ids = []
    for index, protein_id in enumerate(protein_ids):
        if protein_id is None or (
            isinstance(protein_id, float) and math.isnan(protein_id)
        ):
            raise ValueError(f"Protein ID at index {index} is missing")
        normalized_id = str(protein_id).strip()
        if not normalized_id:
            raise ValueError(f"Protein ID at index {index} is blank")
        normalized_ids.append(normalized_id)
    return normalized_ids


def protein_positive_relation(protein_ids):
    """Return the same-protein image--sequence positive relation."""
    normalized_ids = _normalize_protein_ids(protein_ids)
    return torch.tensor(
        [[left_id == right_id for right_id in normalized_ids] for left_id in normalized_ids],
        dtype=torch.bool,
    )


def _multi_positive_cross_entropy(logits, positive_relation):
    if not torch.all(positive_relation.any(dim=1)):
        raise ValueError("Each sample must have at least one positive pair")

    positive_logits = logits.masked_fill(~positive_relation, float("-inf"))
    return (
        torch.logsumexp(logits, dim=1)
        - torch.logsumexp(positive_logits, dim=1)
    ).mean()


def protein_aware_isc_loss(
    image_embeddings, sequence_embeddings, protein_ids, temperature=0.07
):
    """Compute symmetric ISC loss where every same-protein pair is positive."""
    if image_embeddings.ndim != 2 or sequence_embeddings.ndim != 2:
        raise ValueError("Image and sequence embeddings must be two-dimensional")
    if image_embeddings.shape[0] != sequence_embeddings.shape[0]:
        raise ValueError(
            "Image and sequence embeddings must have the same number of samples"
        )

    positive_relation = protein_positive_relation(protein_ids)
    batch_size = image_embeddings.shape[0]
    if positive_relation.shape[0] != batch_size:
        raise ValueError("Protein IDs and embeddings must have the same number of samples")

    image_normalized = functional.normalize(image_embeddings, p=2, dim=1)
    sequence_normalized = functional.normalize(sequence_embeddings, p=2, dim=1)
    logits = image_normalized @ sequence_normalized.T / temperature
    positive_relation = positive_relation.to(logits.device)
    return (
        _multi_positive_cross_entropy(logits, positive_relation)
        + _multi_positive_cross_entropy(logits.T, positive_relation.T)
    ) * 0.5
