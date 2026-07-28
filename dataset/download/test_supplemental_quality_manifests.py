import unittest

import pandas as pd

from dataset.download import supplemental_quality_manifests as manifests
from dataset.download.official_hq_manifests import (
    DOWNLOAD_LABEL_COLUMNS,
    OUTPUT_COLUMNS,
    SOURCE_COLUMNS,
    SOURCE_ROW_ID,
)
from dataset.download.supplemental_quality_manifests import (
    classify_quality,
    prepare_supplemental_rows,
)


def quality_row(
    source_row,
    protein_id,
    intensity,
    quantity,
    image_name,
    *,
    verification="enhanced",
    locations="nucleus",
):
    row = {column: "" for column in SOURCE_COLUMNS}
    row.update({column: 0 for column in DOWNLOAD_LABEL_COLUMNS})
    row.update(
        {
            SOURCE_ROW_ID: source_row,
            "Protein Name": f"Protein {protein_id}",
            "Protein Id": protein_id,
            "Antibody Id": "HPA000123",
            "Tissue": "Caudate",
            "Organ": "Brain",
            "Intensity Level": intensity,
            "Quantity": quantity,
            "URL": f"Brain/Caudate/HPA000123/{image_name}",
            "IF Verification": verification,
            "locations": locations,
            "IF Organ": "Brain",
            "nucleus": 1,
        }
    )
    return row


class SupplementalPreparationTest(unittest.TestCase):
    def test_quality_rules_keep_the_highest_matching_tier(self):
        cases = [
            ("strong", ">75%", "HQ"),
            ("moderate", ">75%", "MQ"),
            ("strong", "75%-25%", "MQ"),
            ("weak", ">75%", "LQ"),
            ("moderate", "75%-25%", "LQ"),
            ("weak", "75%-25%", "LQ"),
            ("weak;strong", ">75%;>75%", "HQ"),
            ("weak;moderate", ">75%;>75%", "MQ"),
            ("weak", "<25%", None),
        ]

        for intensity, quantity, expected in cases:
            with self.subTest(intensity=intensity, quantity=quantity):
                self.assertEqual(classify_quality(intensity, quantity), expected)

    def test_quality_rules_reject_mismatched_value_counts(self):
        with self.assertRaisesRegex(
            ValueError,
            "intensity and quantity must have the same number of values",
        ):
            classify_quality("moderate;weak", ">75%")

    def test_prepare_excludes_official_rows_demotes_hq_and_reports_bad_rows(self):
        source = pd.DataFrame(
            [
                quality_row(10, "P_OFFICIAL", "strong", ">75%", "official.jpg"),
                quality_row(11, " P_MQ ", "strong", ">75%", "mq.jpg"),
                quality_row(12, "P_LQ", "weak", ">75%", "lq.jpg"),
                quality_row(13, "   ", "moderate", ">75%", "blank.jpg"),
                quality_row(
                    14,
                    "P_BAD_PAIR",
                    "moderate;weak",
                    ">75%",
                    "bad-pair.jpg",
                ),
            ]
        )

        prepared, failures, stats = prepare_supplemental_rows(source, {"10"})

        self.assertEqual(prepared[SOURCE_ROW_ID].tolist(), [11, 12])
        self.assertEqual(prepared["Quality"].tolist(), ["MQ", "LQ"])
        self.assertEqual(prepared["__protein_id"].tolist(), ["P_MQ", "P_LQ"])
        self.assertEqual(prepared["__source_line"].tolist(), [3, 4])
        self.assertEqual(stats["supplemental_hq_demoted"], 1)
        self.assertEqual(stats["blank_protein_rows"], 1)
        self.assertEqual(stats["invalid_quality_rows"], 1)
        self.assertEqual(
            [
                (failure["source_row"], failure["stage"], failure["tier"])
                for failure in failures
            ],
            [("13", "protein_id", "MQ"), ("14", "quality", "")],
        )

    def test_prepare_applies_the_five_label_candidate_gate(self):
        valid = quality_row(20, "P_VALID", "moderate", ">75%", "valid.jpg")
        not_enhanced = quality_row(
            21,
            "P_NOT_ENHANCED",
            "moderate",
            ">75%",
            "not-enhanced.jpg",
            verification="supported",
        )
        blank_locations = quality_row(
            22,
            "P_BLANK_LOCATIONS",
            "moderate",
            ">75%",
            "blank-locations.jpg",
            locations=" ",
        )
        nonnumeric_label = quality_row(
            23,
            "P_NONNUMERIC",
            "moderate",
            ">75%",
            "nonnumeric.jpg",
        )
        nonnumeric_label["nucleus"] = "unknown"
        all_zero_labels = quality_row(
            24,
            "P_ALL_ZERO",
            "moderate",
            ">75%",
            "all-zero.jpg",
        )
        all_zero_labels.update({column: 0 for column in DOWNLOAD_LABEL_COLUMNS})

        prepared, failures, stats = prepare_supplemental_rows(
            pd.DataFrame(
                [
                    valid,
                    not_enhanced,
                    blank_locations,
                    nonnumeric_label,
                    all_zero_labels,
                ]
            ),
            set(),
        )

        self.assertEqual(prepared["Protein Id"].tolist(), ["P_VALID"])
        self.assertEqual(failures, [])
        self.assertEqual(stats["eligible_rows"], 1)


class ProteinAssignmentTest(unittest.TestCase):
    def _frames(self):
        official_train = pd.DataFrame(
            [
                quality_row(
                    100,
                    "P_KNOWN_TRAIN",
                    "strong",
                    ">75%",
                    "official-train.jpg",
                )
            ]
        )
        official_test = pd.DataFrame(
            [
                quality_row(
                    101,
                    "P_KNOWN_TEST",
                    "strong",
                    ">75%",
                    "official-test.jpg",
                )
            ]
        )
        supplemental_rows = [
            quality_row(
                200,
                " P_KNOWN_TRAIN ",
                "moderate",
                ">75%",
                "known-train-a.jpg",
            ),
            quality_row(
                201,
                "P_KNOWN_TRAIN",
                "weak",
                ">75%",
                "known-train-b.jpg",
            ),
            quality_row(
                202,
                "P_KNOWN_TEST",
                "moderate",
                ">75%",
                "known-test.jpg",
            ),
        ]
        for number in range(10):
            row = quality_row(
                300 + number,
                f"P_UNKNOWN_{number}",
                "moderate" if number % 2 == 0 else "weak",
                ">75%",
                f"unknown-{number}.jpg",
            )
            row.update({column: 0 for column in DOWNLOAD_LABEL_COLUMNS})
            row[DOWNLOAD_LABEL_COLUMNS[number % len(DOWNLOAD_LABEL_COLUMNS)]] = 1
            if number % 3 == 0:
                row["nucleus"] = 1
            supplemental_rows.append(row)
        supplemental, failures, _stats = prepare_supplemental_rows(
            pd.DataFrame(supplemental_rows), set()
        )
        self.assertEqual(failures, [])
        return official_train, official_test, supplemental

    def test_known_proteins_inherit_and_unknown_assignment_is_deterministic(self):
        official_train, official_test, supplemental = self._frames()

        mapping_a, stats_a = manifests.assign_protein_splits(
            supplemental, official_train, official_test, seed=73
        )
        mapping_b, stats_b = manifests.assign_protein_splits(
            supplemental, official_train, official_test, seed=73
        )

        unknown_ids = {f"P_UNKNOWN_{number}" for number in range(10)}
        self.assertEqual(mapping_a, mapping_b)
        self.assertEqual(stats_a, stats_b)
        self.assertEqual(mapping_a["P_KNOWN_TRAIN"], "train")
        self.assertEqual(mapping_a["P_KNOWN_TEST"], "test")
        self.assertNotIn(" P_KNOWN_TRAIN ", mapping_a)
        self.assertEqual(
            sum(mapping_a[protein_id] == "test" for protein_id in unknown_ids),
            1,
        )
        self.assertEqual(stats_a["known_train_proteins"], 1)
        self.assertEqual(stats_a["known_test_proteins"], 1)
        self.assertEqual(stats_a["unknown_proteins"], 10)
        self.assertEqual(stats_a["unknown_test_proteins"], 1)

    def test_official_protein_overlap_is_rejected_after_trimming(self):
        official_train, official_test, supplemental = self._frames()
        official_test.loc[official_test.index[0], "Protein Id"] = " P_KNOWN_TRAIN "

        with self.assertRaisesRegex(
            ValueError,
            "official train and test Protein Id overlap.*P_KNOWN_TRAIN",
        ):
            manifests.assign_protein_splits(
                supplemental, official_train, official_test, seed=73
            )


class ManifestAssemblyTest(unittest.TestCase):
    def test_official_order_and_cross_split_shared_urls_are_preserved(self):
        official_rows = [
            quality_row(10, "P_HQ_A", "strong", ">75%", "hq-a.jpg"),
            quality_row(20, " P_HQ_B ", "strong", ">75%", "hq-b.jpg"),
            quality_row(30, "P_HQ_TEST", "strong", ">75%", "hq-test.jpg"),
        ]
        official_train = pd.DataFrame([official_rows[1], official_rows[0]])
        official_test = pd.DataFrame([official_rows[2]])
        supplemental, failures, _stats = prepare_supplemental_rows(
            pd.DataFrame(
                [
                    quality_row(
                        40,
                        " P_MQ ",
                        "moderate",
                        ">75%",
                        "shared.jpg",
                    ),
                    quality_row(
                        50,
                        "P_LQ",
                        "weak",
                        ">75%",
                        "shared.jpg",
                    ),
                ]
            ),
            set(),
        )
        self.assertEqual(failures, [])
        sequences = {
            "P_HQ_A": "AAAA",
            "P_HQ_B": "BBBB",
            "P_HQ_TEST": "CCCC",
            "P_MQ": "DDDD",
            "P_LQ": "EEEE",
        }

        outputs = manifests.assemble_quality_outputs(
            official_train,
            official_test,
            supplemental,
            {"P_MQ": "train", "P_LQ": "test"},
            sequences,
        )

        self.assertEqual(outputs["HQ_train"][SOURCE_ROW_ID].tolist(), [20, 10])
        self.assertEqual(outputs["HQ_train"]["Protein Id"].tolist(), [" P_HQ_B ", "P_HQ_A"])
        self.assertEqual(outputs["MQ_train"]["Protein Id"].tolist(), [" P_MQ "])
        self.assertEqual(outputs["MQ_train"]["Sequence"].tolist(), ["DDDD"])
        self.assertEqual(outputs["LQ_test"]["Protein Id"].tolist(), ["P_LQ"])
        self.assertEqual(
            outputs["MQ_train"]["Modified URL"].iloc[0],
            outputs["LQ_test"]["Modified URL"].iloc[0],
        )
        self.assertTrue(
            all(frame.columns.tolist() == OUTPUT_COLUMNS for frame in outputs.values())
        )

    def test_global_invariant_rejects_trimmed_protein_overlap(self):
        empty = pd.DataFrame({"Protein Id": []})
        outputs = {
            "HQ_train": pd.DataFrame({"Protein Id": ["P1"]}),
            "HQ_test": pd.DataFrame({"Protein Id": [" P1 "]}),
            "MQ_train": empty.copy(),
            "MQ_test": empty.copy(),
            "LQ_train": empty.copy(),
            "LQ_test": empty.copy(),
        }

        with self.assertRaisesRegex(
            AssertionError, "Protein Id overlap between train and test.*P1"
        ):
            manifests.assert_protein_disjoint(outputs)


if __name__ == "__main__":
    unittest.main()
