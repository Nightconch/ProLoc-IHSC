import unittest

import pandas as pd

from dataset.download.official_hq_manifests import (
    DOWNLOAD_LABEL_COLUMNS,
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


if __name__ == "__main__":
    unittest.main()
