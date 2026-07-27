import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parent))

import filter_quality_urls
from filter_quality_urls import main


SOURCE_FIXTURES = {
    "normalLabeled.csv": (
        b"Unnamed: 0,Protein Id,URL\n"
        b"1,P1,P1/a.jpg\n"
        b"2,P2,P2/b.jpg\n"
    ),
    "data_train.csv": b"Unnamed: 0,Protein Id,URL\n1,P1,P1/a.jpg\n",
    "data_test.csv": b"Unnamed: 0,Protein Id,URL\n2,P2,P2/b.jpg\n",
}
SOURCE_FIXTURE_MD5 = {
    "normalLabeled.csv": "d6a6399842fdaaf31ee17cadf39cdb9c",
    "data_train.csv": "d39d8a06db1bd7fc756de55420001c41",
    "data_test.csv": "dde3db2b815a24bfee6dd09394a8148f",
}
SOURCE_FIXTURE_URLS = {
    name: f"https://fixtures.invalid/records/10632698/{name}"
    for name in SOURCE_FIXTURES
}
IDENTITY_FIXTURES = {
    "missing_id_column": (
        "data_train.csv",
        b"Protein Id,URL\nP1,P1/a.jpg\n",
        "732452c3099f05a0258fcbbb7eca2c20",
        "data_train.csv missing required columns: Unnamed: 0",
    ),
    "blank_id": (
        "data_train.csv",
        b"Unnamed: 0,Protein Id,URL\n,P1,P1/a.jpg\n",
        "4a4e0b17c5a1930e20aaa4ef1c4d8d4c",
        "data_train.csv contains blank Unnamed: 0",
    ),
    "duplicate_source_id": (
        "normalLabeled.csv",
        (
            b"Unnamed: 0,Protein Id,URL\n"
            b"1,P1,P1/a.jpg\n"
            b"1,P9,P9/x.jpg\n"
        ),
        "ce52daa38652431cfe0fceee3c9b84a6",
        "normalLabeled.csv contains duplicate source rows",
    ),
    "duplicate_official_id": (
        "data_train.csv",
        (
            b"Unnamed: 0,Protein Id,URL\n"
            b"1,P1,P1/a.jpg\n"
            b"1,P1,P1/a.jpg\n"
        ),
        "f80162192a841d3303334b82d2f786f6",
        "data_train.csv contains duplicate source rows",
    ),
    "train_test_overlap": (
        "data_test.csv",
        SOURCE_FIXTURES["data_train.csv"],
        SOURCE_FIXTURE_MD5["data_train.csv"],
        "official train and test source rows overlap",
    ),
    "row_absent_from_source": (
        "data_train.csv",
        b"Unnamed: 0,Protein Id,URL\n3,P3,P3/c.jpg\n",
        "91111a44b37c05b3b2d21690add12c3e",
        "official row 3 is absent from normalLabeled.csv",
    ),
    "row_field_changed": (
        "data_train.csv",
        b"Unnamed: 0,Protein Id,URL\n1,P9,P1/a.jpg\n",
        "dc4010ae30e50142a0e8eb5839e83d2b",
        "official row 1 differs in Protein Id",
    ),
}


class BytesResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def iter_content(self, _chunk_size):
        yield self.payload


class InterruptedResponse:
    def raise_for_status(self):
        pass

    def iter_content(self, _chunk_size):
        yield b"partial-source"
        raise ConnectionError("connection dropped")


class FilterQualityUrlsTest(unittest.TestCase):
    def _write_cache(self, cache_dir, payloads=None):
        payloads = SOURCE_FIXTURES if payloads is None else payloads
        cache_dir.mkdir()
        for name, payload in payloads.items():
            (cache_dir / name).write_bytes(payload)

    def _run_cli(self, cache_dir, output_dir, source_md5=None):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(
                [
                    "--cache-dir",
                    str(cache_dir),
                    "--output-dir",
                    str(output_dir),
                ],
                source_urls=SOURCE_FIXTURE_URLS,
                source_md5=(
                    SOURCE_FIXTURE_MD5 if source_md5 is None else source_md5
                ),
            )
        report = json.loads(
            (output_dir / "source_validation_report.json").read_text(
                encoding="utf-8"
            )
        )
        return exit_code, report

    def test_pinned_source_catalog_uses_the_approved_record_and_md5_values(self):
        self.assertEqual(
            filter_quality_urls.SOURCE_URLS,
            {
                "normalLabeled.csv": (
                    "https://zenodo.org/api/records/10632698/files/"
                    "normalLabeled.csv/content"
                ),
                "data_train.csv": (
                    "https://zenodo.org/api/records/10632698/files/"
                    "data_train.csv/content"
                ),
                "data_test.csv": (
                    "https://zenodo.org/api/records/10632698/files/"
                    "data_test.csv/content"
                ),
            },
        )
        self.assertEqual(
            filter_quality_urls.SOURCE_MD5,
            {
                "normalLabeled.csv": "37dff5cc73458fe529eb860c9a2ab900",
                "data_train.csv": "0236eb02e2f906282ccea4cf47a84591",
                "data_test.csv": "3e9f1ddaf5e14d7a61354f0884b1f002",
            },
        )

    def test_source_cli_reuses_valid_cache_without_http_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_dir = root / "cache"
            output_dir = root / "output"
            self._write_cache(cache_dir)

            with patch.object(
                filter_quality_urls.requests,
                "get",
                side_effect=AssertionError("valid cache must not use HTTP"),
            ):
                exit_code, report = self._run_cli(cache_dir, output_dir)

            self.assertEqual(exit_code, 0)
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["record_id"], "10632698")
            self.assertEqual(report["official_source_rows"], 2)
            self.assertEqual(
                {name: details["action"] for name, details in report["sources"].items()},
                {name: "reused" for name in SOURCE_FIXTURES},
            )
            self.assertEqual(
                sorted(path.name for path in output_dir.iterdir()),
                ["source_validation_report.json"],
            )

    def test_source_cli_refetches_an_invalid_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_dir = root / "cache"
            output_dir = root / "output"
            self._write_cache(cache_dir)
            stale_path = cache_dir / "normalLabeled.csv"
            stale_path.write_bytes(b"stale")

            with patch.object(
                filter_quality_urls.requests,
                "get",
                return_value=BytesResponse(SOURCE_FIXTURES["normalLabeled.csv"]),
            ) as get:
                exit_code, report = self._run_cli(cache_dir, output_dir)

            self.assertEqual(exit_code, 0)
            self.assertEqual(stale_path.read_bytes(), SOURCE_FIXTURES["normalLabeled.csv"])
            self.assertEqual(
                report["sources"]["normalLabeled.csv"]["action"], "replaced"
            )
            get.assert_called_once_with(
                SOURCE_FIXTURE_URLS["normalLabeled.csv"],
                stream=True,
                timeout=60,
            )

    def test_source_cli_rejects_a_refetch_with_the_wrong_md5(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_dir = root / "cache"
            output_dir = root / "output"
            self._write_cache(cache_dir)
            stale_path = cache_dir / "normalLabeled.csv"
            stale_path.write_bytes(b"stale")

            with patch.object(
                filter_quality_urls.requests,
                "get",
                return_value=BytesResponse(b"wrong-download"),
            ):
                exit_code, report = self._run_cli(cache_dir, output_dir)

            self.assertEqual(exit_code, 1)
            self.assertEqual(stale_path.read_bytes(), b"stale")
            self.assertFalse((cache_dir / "normalLabeled.csv.part").exists())
            self.assertEqual(report["status"], "error")
            self.assertEqual(report["error"]["type"], "ValueError")
            self.assertIn(
                "MD5 mismatch for normalLabeled.csv",
                report["error"]["message"],
            )
            self.assertIn(
                SOURCE_FIXTURE_MD5["normalLabeled.csv"],
                report["error"]["message"],
            )

    def test_source_cli_keeps_cache_atomic_when_streaming_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_dir = root / "cache"
            output_dir = root / "output"
            self._write_cache(cache_dir)
            stale_path = cache_dir / "normalLabeled.csv"
            stale_path.write_bytes(b"stale")

            with patch.object(
                filter_quality_urls.requests,
                "get",
                return_value=InterruptedResponse(),
            ):
                exit_code, report = self._run_cli(cache_dir, output_dir)

            self.assertEqual(exit_code, 1)
            self.assertEqual(stale_path.read_bytes(), b"stale")
            self.assertFalse((cache_dir / "normalLabeled.csv.part").exists())
            self.assertEqual(report["status"], "error")
            self.assertIn("connection dropped", report["error"]["message"])

    def test_source_cli_rejects_official_row_identity_anomalies(self):
        for case_name, (
            changed_name,
            changed_payload,
            changed_md5,
            expected_message,
        ) in IDENTITY_FIXTURES.items():
            with self.subTest(case=case_name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                cache_dir = root / "cache"
                output_dir = root / "output"
                payloads = {**SOURCE_FIXTURES, changed_name: changed_payload}
                expected_md5 = {**SOURCE_FIXTURE_MD5, changed_name: changed_md5}
                self._write_cache(cache_dir, payloads)

                with patch.object(
                    filter_quality_urls.requests,
                    "get",
                    side_effect=AssertionError("matching fixtures must not use HTTP"),
                ):
                    exit_code, report = self._run_cli(
                        cache_dir, output_dir, expected_md5
                    )

                self.assertEqual(exit_code, 1)
                self.assertEqual(report["status"], "error")
                self.assertIn(expected_message, report["error"]["message"])
                self.assertEqual(
                    sorted(path.name for path in output_dir.iterdir()),
                    ["source_validation_report.json"],
                )


if __name__ == "__main__":
    unittest.main()
