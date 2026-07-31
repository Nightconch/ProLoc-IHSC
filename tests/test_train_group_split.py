import importlib
import sys
import types
import unittest

import numpy as np


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


class SplitByProteinTests(unittest.TestCase):
    def setUp(self):
        self.protein_ids = np.repeat(
            np.array([f"ENSG{i:011d}" for i in range(20)]), 2
        )
        protein_labels = np.array(
            [[1, 0, 0, 1, 0]] * 10 + [[0, 0, 1, 0, 0]] * 10
        )
        self.labels = np.repeat(protein_labels, 2, axis=0)

    def test_keeps_each_protein_on_exactly_one_side(self):
        train_indices, val_indices = train.split_by_protein(
            self.labels, self.protein_ids
        )

        train_proteins = set(self.protein_ids[train_indices])
        val_proteins = set(self.protein_ids[val_indices])
        self.assertTrue(train_proteins.isdisjoint(val_proteins))
        np.testing.assert_array_equal(
            np.sort(np.concatenate([train_indices, val_indices])),
            np.arange(len(self.labels)),
        )
        self.assertEqual(len(val_indices), 4)

    def test_stratifies_by_the_existing_label_combination(self):
        _, val_indices = train.split_by_protein(
            self.labels, self.protein_ids
        )

        validation_labels = {
            tuple(row) for row in self.labels[val_indices]
        }
        self.assertEqual(
            validation_labels,
            {(1, 0, 0, 1, 0), (0, 0, 1, 0, 0)},
        )

    def test_same_seed_produces_same_split(self):
        first = train.split_by_protein(
            self.labels, self.protein_ids, random_state=42
        )
        second = train.split_by_protein(
            self.labels, self.protein_ids, random_state=42
        )

        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_normalizes_whitespace_before_grouping(self):
        protein_ids = self.protein_ids.astype(object)
        protein_ids[0] = f"  {protein_ids[0]}  "

        train_indices, val_indices = train.split_by_protein(
            self.labels, protein_ids
        )

        normalized = np.char.strip(protein_ids.astype(str))
        self.assertTrue(
            set(normalized[train_indices]).isdisjoint(
                set(normalized[val_indices])
            )
        )

    def test_rejects_mismatched_label_and_group_lengths(self):
        with self.assertRaisesRegex(ValueError, "same number of samples"):
            train.split_by_protein(self.labels[:-1], self.protein_ids)

    def test_rejects_missing_protein_ids(self):
        protein_ids = self.protein_ids.astype(object)
        protein_ids[0] = None

        with self.assertRaisesRegex(ValueError, "missing"):
            train.split_by_protein(self.labels, protein_ids)

    def test_rejects_blank_protein_ids(self):
        protein_ids = self.protein_ids.astype(object)
        protein_ids[0] = "   "

        with self.assertRaisesRegex(ValueError, "blank"):
            train.split_by_protein(self.labels, protein_ids)

    def test_rejects_fewer_proteins_than_folds(self):
        with self.assertRaisesRegex(ValueError, "at least 10 distinct proteins"):
            train.split_by_protein(
                self.labels[:18],
                self.protein_ids[:18],
            )


if __name__ == "__main__":
    unittest.main()
