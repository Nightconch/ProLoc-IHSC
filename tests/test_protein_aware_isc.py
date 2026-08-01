import math

import pytest
import torch
import torch.nn.functional as functional

from isc import protein_aware_isc_loss, protein_positive_relation


def test_positive_relation_trims_ids_and_marks_all_same_protein_pairs():
    """Break caught: comparing untrimmed IDs would lose the off-diagonal positive."""
    relation = protein_positive_relation([" P53 ", "P53", "p53", "BRCA1"])

    assert relation.dtype is torch.bool
    assert torch.equal(
        relation,
        torch.tensor(
            [
                [True, True, False, False],
                [True, True, False, False],
                [False, False, True, False],
                [False, False, False, True],
            ]
        ),
    )


def test_positive_relation_converts_numeric_ids_to_trimmed_text():
    """Break caught: rejecting numeric IDs loses valid same-protein positives."""
    relation = protein_positive_relation([123, " 123 ", 456])

    assert torch.equal(
        relation,
        torch.tensor(
            [[True, True, False], [True, True, False], [False, False, True]]
        ),
    )


@pytest.mark.parametrize(
    ("protein_ids", "message"),
    [
        (["P53", None], "missing"),
        (["P53", "   "], "blank"),
    ],
)
def test_positive_relation_rejects_invalid_protein_ids(protein_ids, message):
    """Break caught: invalid IDs silently becoming positive or negative labels."""
    with pytest.raises(ValueError, match=message):
        protein_positive_relation(protein_ids)


def test_loss_rejects_identifier_count_that_differs_from_batch_size():
    """Break caught: an ID list can be misaligned with the embedding rows."""
    embeddings = torch.eye(2)

    with pytest.raises(ValueError, match="same number of samples"):
        protein_aware_isc_loss(embeddings, embeddings, ["P53"])


def test_all_same_protein_batch_has_zero_loss():
    """Break caught: treating only the diagonal as positive penalizes valid matches."""
    image_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    sequence_embeddings = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    loss = protein_aware_isc_loss(
        image_embeddings, sequence_embeddings, ["P53", "P53"]
    )

    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_unique_ids_match_the_existing_diagonal_isc_objective():
    """Break caught: changing the unique-ID path changes established ISC behavior."""
    image_embeddings = torch.tensor(
        [[1.0, 0.0], [0.3, 0.9], [-0.8, 0.2]], dtype=torch.float64
    )
    sequence_embeddings = torch.tensor(
        [[0.8, 0.1], [0.2, 1.0], [-0.7, 0.4]], dtype=torch.float64
    )
    temperature = 0.07
    logits = functional.normalize(image_embeddings, dim=1) @ functional.normalize(
        sequence_embeddings, dim=1
    ).T / temperature
    expected = (functional.cross_entropy(logits, torch.arange(3)) + functional.cross_entropy(
        logits.T, torch.arange(3)
    )) * 0.5

    loss = protein_aware_isc_loss(
        image_embeddings, sequence_embeddings, ["P53", "BRCA1", "EGFR"], temperature
    )

    assert loss.item() == pytest.approx(expected.item(), abs=1e-12)


def test_mixed_ids_have_finite_gradients_and_are_permutation_invariant():
    """Break caught: missing non-diagonal positives or ordering-dependent aggregation."""
    image_embeddings = torch.tensor(
        [[1.0, 0.1], [0.9, 0.2], [0.0, 1.0]], requires_grad=True
    )
    sequence_embeddings = torch.tensor(
        [[0.8, 0.2], [1.0, 0.0], [0.1, 0.9]], requires_grad=True
    )
    protein_ids = ["P53", "P53", "BRCA1"]

    loss = protein_aware_isc_loss(image_embeddings, sequence_embeddings, protein_ids)
    loss.backward()
    permutation = torch.tensor([1, 0, 2])
    permuted_loss = protein_aware_isc_loss(
        image_embeddings.detach()[permutation],
        sequence_embeddings.detach()[permutation],
        [protein_ids[index] for index in permutation.tolist()],
    )

    assert math.isfinite(loss.item())
    assert torch.isfinite(image_embeddings.grad).all()
    assert torch.isfinite(sequence_embeddings.grad).all()
    assert permuted_loss.item() == pytest.approx(loss.item(), abs=1e-7)
