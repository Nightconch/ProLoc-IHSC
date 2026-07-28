import hashlib
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pandas as pd

from dataset.download import filter_quality_urls as source_module
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


FORMAL_FILENAMES = [
    "HQ_train_img_URL.csv",
    "HQ_test_img_URL.csv",
    "MQ_train_img_URL.csv",
    "MQ_test_img_URL.csv",
    "LQ_train_img_URL.csv",
    "LQ_test_img_URL.csv",
]


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


def _file_md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def cli_fixture_frames():
    official_a = quality_row(
        10, "P_HQ_A", "strong", ">75%", "official-a.jpg"
    )
    official_b = quality_row(
        20, " P_HQ_B ", "strong", ">75%", "official-b.jpg"
    )
    official_test_row = quality_row(
        30, "P_HQ_TEST", "strong", ">75%", "official-test.jpg"
    )
    supplemental_rows = [
        quality_row(40, "P_HQ_B", "strong", ">75%", "shared.jpg"),
        quality_row(41, "P_HQ_TEST", "weak", ">75%", "shared.jpg"),
    ]
    for number in range(10):
        row = quality_row(
            100 + number,
            f"P_UNKNOWN_{number}",
            "moderate" if number % 2 == 0 else "weak",
            ">75%",
            f"unknown-{number}.jpg",
        )
        row.update({column: 0 for column in DOWNLOAD_LABEL_COLUMNS})
        row[DOWNLOAD_LABEL_COLUMNS[number % len(DOWNLOAD_LABEL_COLUMNS)]] = 1
        if number % 4 == 0:
            row["nucleus"] = 1
        supplemental_rows.append(row)
    supplemental_rows.extend(
        [
            quality_row(
                210,
                "P_UNRESOLVED",
                "moderate",
                ">75%",
                "unresolved.jpg",
            ),
            quality_row(220, "   ", "weak", ">75%", "blank.jpg"),
        ]
    )
    invalid_image = quality_row(
        230,
        "P_BAD_IMAGE",
        "moderate",
        ">75%",
        "bad-image.jpg",
    )
    invalid_image["Antibody Id"] = "not-an-antibody"
    supplemental_rows.append(invalid_image)
    source = pd.DataFrame(
        [official_a, official_b, official_test_row, *supplemental_rows]
    )
    return {
        "normalLabeled.csv": source,
        "data_train.csv": pd.DataFrame([official_b, official_a]),
        "data_test.csv": pd.DataFrame([official_test_row]),
    }


def write_cli_fixture(cache_dir):
    cache_dir.mkdir(parents=True)
    fixture_md5 = {}
    for name, frame in cli_fixture_frames().items():
        path = cache_dir / name
        frame.to_csv(path, index=False)
        fixture_md5[name] = _file_md5(path)
    fixture_urls = {
        name: f"https://fixtures.invalid/{name}" for name in fixture_md5
    }
    return fixture_urls, fixture_md5


def fixture_sequence_resolver(protein_ids, _cache_path):
    requested = {str(protein_id).strip() for protein_id in protein_ids}
    unresolved = {"P_UNRESOLVED"} & requested
    return (
        {
            protein_id: f"SEQUENCE_{protein_id}"
            for protein_id in requested - unresolved
        },
        unresolved,
    )


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


class SupplementalManifestCliTest(unittest.TestCase):
    def _run(self, root, seed=73, sequence_resolver=fixture_sequence_resolver):
        cache_dir = root / "cache"
        output_dir = root / "output"
        source_urls, source_md5 = write_cli_fixture(cache_dir)
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = manifests.main(
                [
                    "--cache-dir",
                    str(cache_dir),
                    "--output-dir",
                    str(output_dir),
                    "--seed",
                    str(seed),
                ],
                source_urls=source_urls,
                source_md5=source_md5,
                sequence_resolver=sequence_resolver,
            )
        return exit_code, output_dir, stdout.getvalue(), stderr.getvalue()

    def test_cli_publishes_deterministic_six_manifest_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"

            exit_a, output_a, stdout_a, stderr_a = self._run(first)
            exit_b, output_b, stdout_b, stderr_b = self._run(second)

            self.assertEqual((exit_a, exit_b), (0, 0))
            self.assertEqual(stderr_a, "")
            self.assertEqual(stderr_b, "")
            self.assertEqual(json.loads(stdout_a), json.loads(stdout_b))

            for filename in [
                *FORMAL_FILENAMES,
                "manifest_failures.csv",
                "manifest_generation_report.json",
            ]:
                self.assertEqual(
                    (output_a / filename).read_bytes(),
                    (output_b / filename).read_bytes(),
                    filename,
                )

            outputs = {
                filename.removesuffix("_img_URL.csv"): pd.read_csv(
                    output_a / filename
                )
                for filename in FORMAL_FILENAMES
            }
            self.assertEqual(
                outputs["HQ_train"][SOURCE_ROW_ID].tolist(), [20, 10]
            )
            self.assertEqual(
                outputs["HQ_test"][SOURCE_ROW_ID].tolist(), [30]
            )
            supplemental = pd.concat(
                [
                    outputs[name]
                    for name in ("MQ_train", "MQ_test", "LQ_train", "LQ_test")
                ],
                ignore_index=True,
            )
            self.assertTrue({10, 20, 30}.isdisjoint(set(supplemental[SOURCE_ROW_ID])))
            self.assertIn(40, set(outputs["MQ_train"][SOURCE_ROW_ID]))
            self.assertIn(41, set(outputs["LQ_test"][SOURCE_ROW_ID]))

            shared_mq = outputs["MQ_train"].loc[
                outputs["MQ_train"][SOURCE_ROW_ID].eq(40), "Modified URL"
            ].iloc[0]
            shared_lq = outputs["LQ_test"].loc[
                outputs["LQ_test"][SOURCE_ROW_ID].eq(41), "Modified URL"
            ].iloc[0]
            self.assertEqual(shared_mq, shared_lq)

            unknown_ids = {f"P_UNKNOWN_{number}" for number in range(10)}
            observed_unknown = {
                str(protein_id).strip()
                for protein_id in supplemental["Protein Id"]
                if str(protein_id).strip() in unknown_ids
            }
            self.assertEqual(observed_unknown, unknown_ids)

            train_ids = {
                str(protein_id).strip()
                for name, frame in outputs.items()
                if name.endswith("_train")
                for protein_id in frame["Protein Id"]
            }
            test_ids = {
                str(protein_id).strip()
                for name, frame in outputs.items()
                if name.endswith("_test")
                for protein_id in frame["Protein Id"]
            }
            self.assertEqual(train_ids & test_ids, set())

            failures = pd.read_csv(output_a / "manifest_failures.csv").fillna("")
            self.assertEqual(
                {
                    (str(row["source_row"]), row["stage"], row["Protein Id"])
                    for row in failures.to_dict("records")
                },
                {
                    ("210", "sequence", "P_UNRESOLVED"),
                    ("220", "protein_id", ""),
                    ("230", "image_fields", "P_BAD_IMAGE"),
                },
            )
            self.assertNotIn("P_UNRESOLVED", set(supplemental["Protein Id"]))
            self.assertNotIn("P_BAD_IMAGE", set(supplemental["Protein Id"]))

            report = json.loads(
                (output_a / "manifest_generation_report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(report["status"], "ok")
            self.assertTrue(report["published"])
            self.assertEqual(report["protein_id_overlap"], 0)
            self.assertEqual(report["split"]["unknown_proteins"], 12)
            self.assertEqual(report["split"]["unknown_test_proteins"], 1)

    def test_unresolved_official_sequence_preserves_existing_manifests(self):
        def unresolved_official(protein_ids, cache_path):
            sequences, unresolved = fixture_sequence_resolver(
                protein_ids, cache_path
            )
            sequences.pop("P_HQ_A", None)
            return sequences, unresolved | {"P_HQ_A"}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "output"
            output_dir.mkdir(parents=True)
            sentinels = {}
            for filename in FORMAL_FILENAMES:
                payload = f"old-{filename}".encode()
                (output_dir / filename).write_bytes(payload)
                sentinels[filename] = payload

            cache_dir = root / "cache"
            source_urls, source_md5 = write_cli_fixture(cache_dir)
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                exit_code = manifests.main(
                    [
                        "--cache-dir",
                        str(cache_dir),
                        "--output-dir",
                        str(output_dir),
                        "--seed",
                        "73",
                    ],
                    source_urls=source_urls,
                    source_md5=source_md5,
                    sequence_resolver=unresolved_official,
                )

            self.assertEqual(exit_code, 1)
            for filename, payload in sentinels.items():
                self.assertEqual((output_dir / filename).read_bytes(), payload)
            report = json.loads(
                (output_dir / "manifest_generation_report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(report["status"], "error")
            self.assertFalse(report["published"])
            failures = pd.read_csv(output_dir / "manifest_failures.csv")
            official_failures = failures.loc[failures["tier"].eq("HQ")]
            self.assertIn("P_HQ_A", set(official_failures["Protein Id"]))
            self.assertIn('"status": "error"', stderr.getvalue())


class ManifestPublicationTest(unittest.TestCase):
    def _outputs(self):
        outputs = {}
        for tier in ("HQ", "MQ", "LQ"):
            for split in ("train", "test"):
                row = {column: "" for column in OUTPUT_COLUMNS}
                row["Protein Id"] = f"P_{split.upper()}_{tier}"
                outputs[f"{tier}_{split}"] = pd.DataFrame(
                    [row], columns=OUTPUT_COLUMNS
                )
        return outputs

    def test_global_overlap_is_rejected_before_staging(self):
        outputs = self._outputs()
        outputs["HQ_test"].loc[0, "Protein Id"] = " P_TRAIN_HQ "

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            with self.assertRaisesRegex(AssertionError, "Protein Id overlap"):
                manifests.publish_quality_bundle(outputs, output_dir)

            self.assertEqual(list(output_dir.iterdir()), [])

    def test_publish_rolls_back_all_six_manifests(self):
        outputs = self._outputs()
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            sentinels = {}
            for filename in FORMAL_FILENAMES:
                payload = f"old-{filename}".encode()
                (output_dir / filename).write_bytes(payload)
                sentinels[filename] = payload

            def fail_on_third_staged_replace(source, destination):
                source = Path(source)
                destination = Path(destination)
                if (
                    source.parent != output_dir
                    and destination.parent == output_dir
                    and destination.name == "MQ_train_img_URL.csv"
                ):
                    raise OSError("injected third publish failure")
                os.replace(source, destination)

            with self.assertRaisesRegex(
                OSError, "injected third publish failure"
            ):
                manifests.publish_quality_bundle(
                    outputs,
                    output_dir,
                    replace=fail_on_third_staged_replace,
                )

            for filename, payload in sentinels.items():
                self.assertEqual((output_dir / filename).read_bytes(), payload)
            self.assertEqual(
                sorted(path.name for path in output_dir.iterdir()),
                sorted(FORMAL_FILENAMES),
            )

    def test_publish_retains_backup_when_restore_fails(self):
        outputs = self._outputs()
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            sentinels = {}
            for filename in FORMAL_FILENAMES:
                payload = f"old-{filename}".encode()
                (output_dir / filename).write_bytes(payload)
                sentinels[filename] = payload

            def fail_publish_and_mq_train_restore(source, destination):
                source = Path(source)
                destination = Path(destination)
                if (
                    source.parent != output_dir
                    and destination.parent == output_dir
                    and destination.name == "MQ_train_img_URL.csv"
                ):
                    raise OSError("injected publish failure")
                if (
                    source.parent == output_dir
                    and source.name.startswith(
                        ".MQ_train_img_URL.csv.backup-"
                    )
                    and destination.name == "MQ_train_img_URL.csv"
                ):
                    raise OSError("injected restore failure")
                os.replace(source, destination)

            with self.assertRaisesRegex(
                RuntimeError, "rollback also failed.*injected restore failure"
            ):
                manifests.publish_quality_bundle(
                    outputs,
                    output_dir,
                    replace=fail_publish_and_mq_train_restore,
                )

            retained = list(
                output_dir.glob(".MQ_train_img_URL.csv.backup-*")
            )
            self.assertEqual(len(retained), 1)
            self.assertEqual(
                retained[0].read_bytes(),
                sentinels["MQ_train_img_URL.csv"],
            )
            for filename, payload in sentinels.items():
                if filename != "MQ_train_img_URL.csv":
                    self.assertEqual(
                        (output_dir / filename).read_bytes(), payload
                    )


if __name__ == "__main__":
    unittest.main()
