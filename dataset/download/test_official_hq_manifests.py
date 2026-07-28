import hashlib
import importlib
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd


SOURCE_ROW_ID = "Unnamed: 0"
SOURCE_COLUMNS = [
    SOURCE_ROW_ID,
    "Protein Name",
    "Protein Id",
    "Antibody Id",
    "Reliability Verification",
    "Tissue",
    "Organ",
    "Staining Level",
    "Intensity Level",
    "Quantity",
    "SnomedParameters",
    "URL",
    "IF Verification",
    "locations",
    "IF Organ",
    "cytoplasm",
    "cytoskeleton",
    "endoplasmic reticulum",
    "golgi apparatus",
    "lysosomes",
    "mitochondria",
    "nucleoli",
    "nucleus",
    "plasma membrane",
    "vesicles",
]


def official_row(source_row, protein_id, antibody_id, relative_url):
    row = {column: "" for column in SOURCE_COLUMNS}
    row.update(
        {
            SOURCE_ROW_ID: source_row,
            "Protein Name": f"Protein {protein_id.strip()}",
            "Protein Id": protein_id,
            "Antibody Id": antibody_id,
            "Reliability Verification": "enhanced",
            "Tissue": "brain",
            "Organ": "Brain",
            "Staining Level": "high",
            "Intensity Level": "strong",
            "Quantity": ">75%",
            "SnomedParameters": "caudate",
            "URL": relative_url,
            "IF Verification": "enhanced",
            "locations": "nucleus",
            "IF Organ": "Brain",
            "cytoplasm": 0,
            "cytoskeleton": 0,
            "endoplasmic reticulum": 0,
            "golgi apparatus": 0,
            "lysosomes": 0,
            "mitochondria": 0,
            "nucleoli": 0,
            "nucleus": 1,
            "plasma membrane": 0,
            "vesicles": 0,
        }
    )
    return row


def write_source_fixtures(cache_dir, source, train, test):
    frames = {
        "normalLabeled.csv": source,
        "data_train.csv": train,
        "data_test.csv": test,
    }
    source_md5 = {}
    for name, frame in frames.items():
        path = cache_dir / name
        frame.to_csv(path, index=False)
        source_md5[name] = hashlib.md5(path.read_bytes()).hexdigest()
    source_urls = {
        name: f"https://fixtures.invalid/records/10632698/{name}"
        for name in frames
    }
    return source_urls, source_md5


class JsonResponse:
    def __init__(self, payload, links=None):
        self.payload = payload
        self.links = {} if links is None else links

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class OfficialHqCliTest(unittest.TestCase):
    def test_cli_publishes_only_official_rows_in_official_order(self):
        try:
            module = importlib.import_module(
                "dataset.download.official_hq_manifests"
            )
        except ModuleNotFoundError as error:
            self.fail(f"official HQ CLI is missing: {error}")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_dir = root / "cache"
            output_dir = root / "output"
            cache_dir.mkdir()

            rows = [
                official_row(10, "P10", "HPA000010", "P10/Brain/a.jpg"),
                official_row(20, " P20 ", "HPA000020", "shared/Brain/shared.jpg"),
                official_row(30, "P30", "HPA000030", "P30/Brain/c.jpg"),
                official_row(40, "P40", "HPA000020", "shared/Brain/shared.jpg"),
                official_row(50, "P50", "HPA000050", "P50/Brain/extra.jpg"),
            ]
            source = pd.DataFrame(rows, columns=SOURCE_COLUMNS)
            official_train = source.iloc[[1, 0]].copy()
            official_test = source.iloc[[3, 2]].copy()
            source_urls, source_md5 = write_source_fixtures(
                cache_dir, source, official_train, official_test
            )
            sequences = {
                protein_id: f"SEQ{protein_id}"
                for protein_id in ("P10", "P20", "P30", "P40")
            }

            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = module.main(
                    [
                        "--cache-dir",
                        str(cache_dir),
                        "--output-dir",
                        str(output_dir),
                    ],
                    source_urls=source_urls,
                    source_md5=source_md5,
                    sequence_resolver=lambda _ids, _cache: (sequences, set()),
                )

            self.assertEqual(exit_code, 0, stderr.getvalue())
            train = pd.read_csv(output_dir / "HQ_train_img_URL.csv")
            test = pd.read_csv(output_dir / "HQ_test_img_URL.csv")
            self.assertEqual(train[SOURCE_ROW_ID].tolist(), [20, 10])
            self.assertEqual(test[SOURCE_ROW_ID].tolist(), [40, 30])
            self.assertEqual(train["Protein Id"].tolist(), [" P20 ", "P10"])
            self.assertEqual(test["Protein Id"].tolist(), ["P40", "P30"])
            self.assertNotIn("P50", set(train["Protein Id"]) | set(test["Protein Id"]))
            self.assertEqual(len(train), len(official_train))
            self.assertEqual(len(test), len(official_test))
            self.assertEqual(train.iloc[0]["Modified URL"], test.iloc[0]["Modified URL"])
            self.assertEqual(train.iloc[0]["Sequence"], "SEQP20")
            self.assertEqual(test.iloc[0]["Sequence"], "SEQP40")

            report = json.loads(
                (output_dir / "manifest_generation_report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(report["status"], "ok")
            self.assertTrue(report["published"])

    def test_historical_hq_files_do_not_affect_exact_occupied_rows(self):
        module = importlib.import_module(
            "dataset.download.official_hq_manifests"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_dir = root / "cache"
            output_dir = root / "output"
            cache_dir.mkdir()
            output_dir.mkdir()
            historical = {
                "train_img_URL.csv": b"poisoned historical train\n",
                "test_img_URL.csv": b"poisoned historical test\n",
                "HQ_only_generated.csv": b"obsolete audit output\n",
                "HQ_comparison_report.txt": b"obsolete comparison\n",
            }
            for name, payload in historical.items():
                (output_dir / name).write_bytes(payload)

            rows = [
                official_row(10.0, "P10", "HPA000010", "P10/a.jpg"),
                official_row(20.0, "P20", "HPA000020", "shared/shared.jpg"),
                official_row(30.0, "P30", "HPA000030", "P30/c.jpg"),
                official_row(40.0, "P40", "HPA000020", "shared/shared.jpg"),
            ]
            source = pd.DataFrame(rows, columns=SOURCE_COLUMNS)
            official_train = source.iloc[[1, 0]].copy()
            official_test = source.iloc[[3, 2]].copy()
            source_urls, source_md5 = write_source_fixtures(
                cache_dir, source, official_train, official_test
            )
            sequences = {
                protein_id: "AAAA"
                for protein_id in ("P10", "P20", "P30", "P40")
            }

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                exit_code = module.main(
                    [
                        "--cache-dir",
                        str(cache_dir),
                        "--output-dir",
                        str(output_dir),
                    ],
                    source_urls=source_urls,
                    source_md5=source_md5,
                    sequence_resolver=lambda _ids, _cache: (sequences, set()),
                )

            self.assertEqual(exit_code, 0)
            occupied = pd.read_csv(
                output_dir / "official_hq_occupied_rows.csv", dtype=str
            )
            self.assertEqual(
                occupied["split"].tolist(),
                ["train", "train", "test", "test"],
            )
            self.assertEqual(
                occupied["source_row"].tolist(),
                ["20", "10", "40", "30"],
            )
            self.assertEqual(
                pd.read_csv(output_dir / "HQ_train_img_URL.csv")[SOURCE_ROW_ID].tolist(),
                [20.0, 10.0],
            )
            self.assertEqual(
                pd.read_csv(output_dir / "HQ_test_img_URL.csv")[SOURCE_ROW_ID].tolist(),
                [40.0, 30.0],
            )
            for name, payload in historical.items():
                self.assertEqual((output_dir / name).read_bytes(), payload)


class SequenceResolutionTest(unittest.TestCase):
    def test_parser_accepts_only_one_unique_nonblank_sequence(self):
        module = importlib.import_module(
            "dataset.download.official_hq_manifests"
        )
        payload = {
            "results": [
                {"from": "P1", "to": {"sequence": {"value": "AAAA"}}},
                {"from": "P1", "to": {"sequence": {"value": "AAAA"}}},
                {"from": "P2", "to": {"sequence": {"value": "BBBB"}}},
                {"from": "P2", "to": {"sequence": {"value": "CCCC"}}},
                {"from": "P3", "to": {"sequence": {"value": "   "}}},
            ]
        }

        try:
            sequences, unresolved = module.parse_uniprot_results(payload)
        except AttributeError as error:
            self.fail(f"reviewed sequence parser is missing: {error}")

        self.assertEqual(sequences, {"P1": "AAAA"})
        self.assertEqual(unresolved, {"P2", "P3"})

    def test_conflicting_cached_sequences_remain_unresolved(self):
        module = importlib.import_module(
            "dataset.download.official_hq_manifests"
        )
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "uniprot_sequences.csv"
            cache_path.write_text(
                "Protein Id,Sequence\nP1,AAAA\nP1,BBBB\nP2,CCCC\n",
                encoding="utf-8",
            )
            with patch.object(module, "requests", create=True) as requests:
                requests.post.side_effect = AssertionError(
                    "conflicting cache entries must not be guessed or refetched"
                )
                requests.get.side_effect = AssertionError(
                    "conflicting cache entries must not be guessed or refetched"
                )
                sequences, unresolved = module.fetch_reviewed_sequences(
                    {"P1", "P2"}, cache_path
                )

        self.assertEqual(sequences, {"P2": "CCCC"})
        self.assertEqual(unresolved, {"P1"})

    def test_remote_mapping_accepts_only_unique_swiss_prot_sequences(self):
        module = importlib.import_module(
            "dataset.download.official_hq_manifests"
        )

        def post(url, data, timeout):
            self.assertEqual(url, module.UNIPROT_RUN)
            self.assertEqual(data["from"], "Ensembl")
            self.assertEqual(data["to"], "UniProtKB-Swiss-Prot")
            self.assertEqual(set(data["ids"].split(",")), {"P1", "P2"})
            self.assertEqual(timeout, 60)
            return JsonResponse({"jobId": "job-1"})

        def get(url, timeout):
            self.assertEqual(timeout, 60)
            if url == module.UNIPROT_STATUS.format(job_id="job-1"):
                return JsonResponse({"results": [], "failedIds": []})
            if url == module.UNIPROT_DETAILS.format(job_id="job-1"):
                return JsonResponse(
                    {"redirectURL": "https://fixtures.invalid/results?size=25"}
                )
            self.assertTrue(url.startswith("https://fixtures.invalid/results?"))
            return JsonResponse(
                {
                    "results": [
                        {"from": "P1", "to": {"sequence": {"value": "AAAA"}}},
                        {"from": "P2", "to": {"sequence": {"value": "BBBB"}}},
                        {"from": "P2", "to": {"sequence": {"value": "CCCC"}}},
                    ]
                }
            )

        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "uniprot_sequences.csv"
            with patch.object(module.requests, "post", side_effect=post), patch.object(
                module.requests, "get", side_effect=get
            ):
                sequences, unresolved = module.fetch_reviewed_sequences(
                    {"P1", "P2"}, cache_path
                )
            cached = pd.read_csv(cache_path, dtype=str)

        self.assertEqual(sequences, {"P1": "AAAA"})
        self.assertEqual(unresolved, {"P2"})
        self.assertEqual(cached.to_dict("records"), [{"Protein Id": "P1", "Sequence": "AAAA"}])


class OfficialHqFailureTest(unittest.TestCase):
    def _run_cli(self, root, source, train, test, sequence_resolver):
        module = importlib.import_module(
            "dataset.download.official_hq_manifests"
        )
        cache_dir = root / "cache"
        output_dir = root / "output"
        cache_dir.mkdir()
        source_urls, source_md5 = write_source_fixtures(
            cache_dir, source, train, test
        )
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = module.main(
                [
                    "--cache-dir",
                    str(cache_dir),
                    "--output-dir",
                    str(output_dir),
                ],
                source_urls=source_urls,
                source_md5=source_md5,
                sequence_resolver=sequence_resolver,
            )
        return exit_code, output_dir, stderr.getvalue()

    def _assert_no_formal_hq(self, output_dir):
        self.assertFalse((output_dir / "HQ_train_img_URL.csv").exists())
        self.assertFalse((output_dir / "HQ_test_img_URL.csv").exists())
        report = json.loads(
            (output_dir / "manifest_generation_report.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(report["status"], "error")
        self.assertFalse(report["published"])

    def test_normalized_protein_overlap_reports_both_official_sides(self):
        rows = [
            official_row(10, "P1", "HPA000010", "P1/train.jpg"),
            official_row(20, " P1 ", "HPA000020", "P1/test.jpg"),
        ]
        source = pd.DataFrame(rows, columns=SOURCE_COLUMNS)
        train = source.iloc[[0]].copy()
        test = source.iloc[[1]].copy()

        def sequence_resolver(_ids, _cache):
            raise AssertionError("split overlap must fail before sequence resolution")

        with tempfile.TemporaryDirectory() as directory:
            exit_code, output_dir, stderr = self._run_cli(
                Path(directory), source, train, test, sequence_resolver
            )
            self.assertEqual(exit_code, 1)
            self._assert_no_formal_hq(output_dir)
            failures = pd.read_csv(
                output_dir / "manifest_failures.csv", dtype=str
            )

        self.assertIn("Protein Id overlap", stderr)
        self.assertEqual(failures["stage"].tolist(), ["split_overlap", "split_overlap"])
        self.assertEqual(failures["split"].tolist(), ["train", "test"])
        self.assertEqual(failures["source_row"].tolist(), ["10", "20"])
        self.assertEqual(failures["Protein Id"].tolist(), ["P1", "P1"])

    def test_unresolved_sequence_reports_every_corresponding_source_row(self):
        rows = [
            official_row(10, "P1", "HPA000010", "P1/a.jpg"),
            official_row(20, "P1", "HPA000020", "P1/b.jpg"),
        ]
        source = pd.DataFrame(rows, columns=SOURCE_COLUMNS)
        train = source.copy()
        test = source.iloc[0:0].copy()

        with tempfile.TemporaryDirectory() as directory:
            exit_code, output_dir, _stderr = self._run_cli(
                Path(directory),
                source,
                train,
                test,
                lambda _ids, _cache: ({}, {"P1"}),
            )
            self.assertEqual(exit_code, 1)
            self._assert_no_formal_hq(output_dir)
            failures = pd.read_csv(
                output_dir / "manifest_failures.csv", dtype=str
            )

        self.assertEqual(failures["stage"].tolist(), ["sequence", "sequence"])
        self.assertEqual(failures["source_line"].tolist(), ["2", "3"])
        self.assertEqual(failures["source_row"].tolist(), ["10", "20"])

    def test_blank_sequence_mapping_is_fatal(self):
        source = pd.DataFrame(
            [official_row(10, "P1", "HPA000010", "P1/a.jpg")],
            columns=SOURCE_COLUMNS,
        )
        train = source.copy()
        test = source.iloc[0:0].copy()

        with tempfile.TemporaryDirectory() as directory:
            exit_code, output_dir, _stderr = self._run_cli(
                Path(directory),
                source,
                train,
                test,
                lambda _ids, _cache: ({"P1": "   "}, set()),
            )
            self.assertEqual(exit_code, 1)
            self._assert_no_formal_hq(output_dir)
            failures = pd.read_csv(
                output_dir / "manifest_failures.csv", dtype=str
            )

        self.assertEqual(failures["stage"].tolist(), ["sequence"])
        self.assertIn("unique nonblank reviewed sequence", failures.iloc[0]["reason"])

    def test_invalid_download_fields_report_all_official_rows(self):
        rows = [
            official_row(10, "P10", "HPA", "P10/a.jpg"),
            official_row(20, "P20", "HPA000020", "P20/folder/"),
            official_row(30, "P30", "HPA000030", "P30/c.jpg"),
            official_row(40, "P40", "HPA000040", "P40/d.jpg"),
            official_row(50, "   ", "HPA000050", "P50/e.jpg"),
        ]
        rows[2]["locations"] = "   "
        rows[3]["nucleus"] = ""
        source = pd.DataFrame(rows, columns=SOURCE_COLUMNS)
        train = source.copy()
        test = source.iloc[0:0].copy()
        sequences = {f"P{number}": "AAAA" for number in (10, 20, 30, 40)}

        with tempfile.TemporaryDirectory() as directory:
            exit_code, output_dir, _stderr = self._run_cli(
                Path(directory),
                source,
                train,
                test,
                lambda _ids, _cache: (sequences, set()),
            )
            self.assertEqual(exit_code, 1)
            self._assert_no_formal_hq(output_dir)
            failures = pd.read_csv(
                output_dir / "manifest_failures.csv", dtype=str
            )

        self.assertEqual(failures["source_row"].tolist(), ["10", "20", "30", "40", "50"])
        self.assertEqual(
            failures["stage"].tolist(),
            ["image_fields", "image_fields", "required_field", "required_field", "required_field"],
        )


class OfficialHqPublicationTest(unittest.TestCase):
    def test_bundle_publish_rolls_back_all_formal_artifacts(self):
        module = importlib.import_module(
            "dataset.download.official_hq_manifests"
        )
        outputs = {
            "HQ_train": pd.DataFrame([{"value": "new train"}]),
            "HQ_test": pd.DataFrame([{"value": "new test"}]),
        }
        occupied = pd.DataFrame(
            [
                {
                    "split": "train",
                    "source_position": 1,
                    "source_line": 2,
                    "source_row": "10",
                }
            ]
        )
        old_payloads = {
            "HQ_train_img_URL.csv": b"old train\n",
            "HQ_test_img_URL.csv": b"old test\n",
            "official_hq_occupied_rows.csv": b"old occupied\n",
        }

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            for name, payload in old_payloads.items():
                (output_dir / name).write_bytes(payload)
            failed = False

            def fail_second_staged_replace(source, destination):
                nonlocal failed
                source = Path(source)
                destination = Path(destination)
                if (
                    not failed
                    and destination.name == "HQ_test_img_URL.csv"
                    and ".official-hq-staging-" in source.parent.name
                ):
                    failed = True
                    raise OSError("injected second publish failure")
                os.replace(source, destination)

            try:
                publish = module.publish_official_bundle
            except AttributeError as error:
                self.fail(f"atomic official HQ publisher is missing: {error}")

            with self.assertRaisesRegex(
                OSError, "injected second publish failure"
            ):
                publish(
                    outputs,
                    occupied,
                    output_dir,
                    replace=fail_second_staged_replace,
                )

            for name, payload in old_payloads.items():
                self.assertEqual((output_dir / name).read_bytes(), payload)
            self.assertEqual(
                sorted(path.name for path in output_dir.iterdir()),
                sorted(old_payloads),
            )

    def test_bundle_publish_rolls_back_if_backup_creation_fails(self):
        module = importlib.import_module(
            "dataset.download.official_hq_manifests"
        )
        outputs = {
            "HQ_train": pd.DataFrame([{"value": "new train"}]),
            "HQ_test": pd.DataFrame([{"value": "new test"}]),
        }
        occupied = pd.DataFrame([{"source_row": "10"}])
        old_payloads = {
            "HQ_train_img_URL.csv": b"old train\n",
            "HQ_test_img_URL.csv": b"old test\n",
            "official_hq_occupied_rows.csv": b"old occupied\n",
        }

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            for name, payload in old_payloads.items():
                (output_dir / name).write_bytes(payload)
            failed = False

            def fail_second_backup(source, destination):
                nonlocal failed
                destination = Path(destination)
                if (
                    not failed
                    and destination.name.startswith(
                        ".HQ_test_img_URL.csv.backup-"
                    )
                ):
                    failed = True
                    raise OSError("injected backup failure")
                os.replace(source, destination)

            with self.assertRaisesRegex(OSError, "injected backup failure"):
                module.publish_official_bundle(
                    outputs,
                    occupied,
                    output_dir,
                    replace=fail_second_backup,
                )

            for name, payload in old_payloads.items():
                self.assertEqual((output_dir / name).read_bytes(), payload)
            self.assertEqual(
                sorted(path.name for path in output_dir.iterdir()),
                sorted(old_payloads),
            )


if __name__ == "__main__":
    unittest.main()
