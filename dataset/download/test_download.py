import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parent))

import download
from download import (
    ImageValidationError,
    is_blank_rgb,
    normalized_jpeg_bytes,
    write_validated_image,
)


def image_bytes(mode, color, image_format="PNG"):
    image = Image.new(mode, (3, 2), color)
    output = io.BytesIO()
    image.save(output, format=image_format)
    return output.getvalue()


def palette_image_bytes():
    image = Image.new("P", (3, 2), 0)
    image.putpalette([10, 20, 30] + [0, 0, 0] * 255)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def rgba_image_bytes():
    image = Image.new("RGBA", (32, 16), (255, 0, 0, 0))
    for x in range(16, 32):
        for y in range(16):
            image.putpixel((x, y), (20, 30, 40, 255))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def palette_transparency_image_bytes():
    image = Image.new("P", (32, 16), 0)
    image.putpalette([255, 0, 0, 20, 30, 40] + [0, 0, 0] * 254)
    for x in range(16, 32):
        for y in range(16):
            image.putpixel((x, y), 1)
    image.info["transparency"] = 0
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def manifest_row(**overrides):
    row = {
        "Unnamed: 0": 101,
        "Protein Id": "ENSG1",
        "Modified URL": "https://images.proteinatlas.org/123/a.jpg",
        "Antibody Id": "HPA000123",
        "locations": "nucleus",
        "cytoplasm": 0,
        "endoplasmic reticulum": 0,
        "mitochondria": 0,
        "nucleus": 1,
        "plasma membrane": 0,
        "Sequence": "AAAA",
    }
    row.update(overrides)
    return row


class FakeResponse:
    def __init__(self, content, status_code=200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class BodyReadFailureResponse:
    def raise_for_status(self):
        pass

    @property
    def content(self):
        raise OSError("response body read failed")


class DownloadFilenameContractTest(unittest.TestCase):
    def test_build_file_name_uses_existing_contract(self):
        row = {
            "Protein Id": "ENSG1",
            "Modified URL": "https://images.proteinatlas.org/123/a.jpg",
            "Antibody Id": "HPA000123",
            "locations": "nucleus;cytosol",
        }

        self.assertEqual(
            download.build_file_name(row),
            "ENSG1-a-HPA000123-nucleus;cytosol.jpg",
        )

    def test_build_file_name_rejects_windows_invalid_characters(self):
        row = {
            "Protein Id": "ENSG1",
            "Modified URL": "https://images.proteinatlas.org/123/a.jpg",
            "Antibody Id": "HPA000123",
            "locations": "nucleus/other",
        }

        with self.assertRaisesRegex(ValueError, "locations.*invalid"):
            download.build_file_name(row)

    def test_build_file_name_rejects_blank_components(self):
        row = {
            "Protein Id": "ENSG1",
            "Modified URL": "https://images.proteinatlas.org/123/a.jpg",
            "Antibody Id": "   ",
            "locations": "nucleus",
        }

        with self.assertRaisesRegex(ValueError, "Antibody Id.*blank"):
            download.build_file_name(row)

    def test_build_file_name_rejects_missing_components(self):
        row = {
            "Protein Id": None,
            "Modified URL": "https://images.proteinatlas.org/123/a.jpg",
            "Antibody Id": "HPA000123",
            "locations": "nucleus",
        }

        with self.assertRaisesRegex(ValueError, "Protein Id.*blank"):
            download.build_file_name(row)


class ManifestInputContractTest(unittest.TestCase):
    def test_process_manifest_rejects_missing_required_columns_before_http(self):
        row = manifest_row()
        del row["Sequence"]
        http_calls = []

        def fake_get(*args, **kwargs):
            http_calls.append((args, kwargs))
            raise AssertionError("HTTP must not run for an invalid manifest")

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "Sequence"):
                download.process_manifest(
                    pd.DataFrame([row]),
                    "MQ",
                    "train",
                    Path(temp_dir),
                    2,
                    fake_get,
                )

        self.assertEqual(http_calls, [])


class DownloadRowTest(unittest.TestCase):
    def test_download_task_writes_validated_image_and_returns_source_context(self):
        row = manifest_row()
        payload = image_bytes("L", 40)

        def fake_get(url, timeout):
            self.assertEqual(url, row["Modified URL"])
            self.assertEqual(timeout, 60)
            return FakeResponse(payload)

        with tempfile.TemporaryDirectory() as temp_dir:
            image_dir = Path(temp_dir)
            task = download.DownloadTask(3, "MQ", "train", row)

            result = download.download_task(task, image_dir, fake_get)

            target = image_dir / "ENSG1-a-HPA000123-nucleus.jpg"
            self.assertTrue(result.success)
            self.assertTrue(result.converted)
            self.assertEqual(result.ordinal, 3)
            self.assertEqual(result.row, row)
            self.assertEqual(result.file_name, target.name)
            with Image.open(target) as image:
                image.load()
                self.assertEqual(image.format, "JPEG")
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(len(image.getbands()), 3)

    def test_download_task_returns_structured_http_failure(self):
        row = manifest_row()
        task = download.DownloadTask(4, "MQ", "train", row)

        with tempfile.TemporaryDirectory() as temp_dir:
            image_dir = Path(temp_dir)
            result = download.download_task(
                task,
                image_dir,
                lambda _url, timeout: FakeResponse(b"unavailable", 503),
            )

            self.assertFalse(result.success)
            self.assertEqual(result.stage, "http")
            self.assertIn("HTTP 503", result.reason)
            self.assertEqual(result.ordinal, 4)
            self.assertEqual(result.row, row)
            self.assertEqual(
                result.file_name,
                "ENSG1-a-HPA000123-nucleus.jpg",
            )
            self.assertEqual(list(image_dir.iterdir()), [])

    def test_download_task_classifies_body_read_failure_as_http(self):
        row = manifest_row()
        task = download.DownloadTask(8, "MQ", "train", row)

        with tempfile.TemporaryDirectory() as temp_dir:
            result = download.download_task(
                task,
                Path(temp_dir),
                lambda _url, timeout: BodyReadFailureResponse(),
            )

            self.assertFalse(result.success)
            self.assertEqual(result.stage, "http")
            self.assertIn("body read failed", result.reason)

    def test_download_task_returns_structured_decode_failure(self):
        row = manifest_row()
        task = download.DownloadTask(5, "MQ", "train", row)

        with tempfile.TemporaryDirectory() as temp_dir:
            image_dir = Path(temp_dir)
            result = download.download_task(
                task,
                image_dir,
                lambda _url, timeout: FakeResponse(b"not an image"),
            )

            self.assertFalse(result.success)
            self.assertEqual(result.stage, "decode")
            self.assertIn("decode failed", result.reason)
            self.assertEqual(result.ordinal, 5)
            self.assertEqual(result.row, row)
            self.assertFalse(
                (image_dir / "ENSG1-a-HPA000123-nucleus.jpg").exists()
            )

    def test_download_task_returns_structured_blank_failure(self):
        row = manifest_row()
        task = download.DownloadTask(6, "MQ", "train", row)
        payload = image_bytes("RGB", (255, 255, 255))

        with tempfile.TemporaryDirectory() as temp_dir:
            image_dir = Path(temp_dir)
            result = download.download_task(
                task,
                image_dir,
                lambda _url, timeout: FakeResponse(payload),
            )

            self.assertFalse(result.success)
            self.assertEqual(result.stage, "blank")
            self.assertIn("blank", result.reason)
            self.assertEqual(result.ordinal, 6)
            self.assertEqual(result.row, row)
            self.assertFalse(
                (image_dir / "ENSG1-a-HPA000123-nucleus.jpg").exists()
            )

    def test_download_task_returns_structured_write_failure(self):
        row = manifest_row()
        task = download.DownloadTask(7, "MQ", "train", row)
        payload = image_bytes("RGB", (10, 20, 30), "JPEG")

        with tempfile.TemporaryDirectory() as temp_dir:
            image_dir = Path(temp_dir) / "not-a-directory"
            image_dir.write_bytes(b"obstruction")

            result = download.download_task(
                task,
                image_dir,
                lambda _url, timeout: FakeResponse(payload),
            )

            self.assertFalse(result.success)
            self.assertEqual(result.stage, "write")
            self.assertTrue(result.reason)
            self.assertEqual(result.ordinal, 7)
            self.assertEqual(result.row, row)
            self.assertEqual(image_dir.read_bytes(), b"obstruction")

    def test_download_task_classifies_staged_file_open_failure_as_write(self):
        row = manifest_row()
        task = download.DownloadTask(8, "MQ", "train", row)
        payload = image_bytes("RGB", (10, 20, 30), "JPEG")

        with tempfile.TemporaryDirectory() as temp_dir:
            image_dir = Path(temp_dir)
            image_path = image_dir / "ENSG1-a-HPA000123-nucleus.jpg"
            path_type = type(image_path)
            original_path_open = path_type.open
            original_image_open = download.Image.open

            def fail_staged_path_open(
                path,
                mode="r",
                buffering=-1,
                encoding=None,
                errors=None,
                newline=None,
            ):
                if mode == "rb" and path.name.endswith(".part"):
                    raise OSError("disk read failed")
                return original_path_open(
                    path, mode, buffering, encoding, errors, newline
                )

            def fail_pillow_path_open(source, mode="r", formats=None):
                if isinstance(source, Path):
                    raise OSError("disk read failed")
                return original_image_open(source, mode, formats)

            with patch.object(
                path_type,
                "open",
                autospec=True,
                side_effect=fail_staged_path_open,
            ), patch.object(
                download.Image,
                "open",
                side_effect=fail_pillow_path_open,
            ):
                result = download.download_task(
                    task,
                    image_dir,
                    lambda _url, timeout: FakeResponse(payload),
                )

            self.assertFalse(result.success)
            self.assertEqual(result.stage, "write")
            self.assertIn("disk read failed", result.reason)
            self.assertEqual(result.ordinal, 8)
            self.assertEqual(result.row, row)
            self.assertFalse(image_path.exists())
            self.assertFalse(image_path.with_name(f"{image_path.name}.part").exists())


class ManifestPreflightTest(unittest.TestCase):
    def test_process_manifest_rejects_blank_or_missing_values_before_http(self):
        invalid_rows = {
            "blank sequence": (manifest_row(Sequence="   "), "Sequence"),
            "missing label": (
                manifest_row(**{"plasma membrane": pd.NA}),
                "plasma membrane",
            ),
        }
        for case, (row, expected_column) in invalid_rows.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temp_dir:
                http_calls = []

                def fake_get(*args, **kwargs):
                    http_calls.append((args, kwargs))
                    raise AssertionError("HTTP must not run for invalid values")

                with self.assertRaisesRegex(ValueError, expected_column):
                    download.process_manifest(
                        pd.DataFrame([row]),
                        "MQ",
                        "train",
                        Path(temp_dir),
                        2,
                        fake_get,
                    )

                self.assertEqual(http_calls, [])

    def test_process_manifest_rejects_filename_collisions_before_http(self):
        rows = [
            manifest_row(**{"Unnamed: 0": 101}),
            manifest_row(**{"Unnamed: 0": 202}),
        ]
        http_calls = []

        def fake_get(*args, **kwargs):
            http_calls.append((args, kwargs))
            raise AssertionError("HTTP must not run for filename collisions")

        with tempfile.TemporaryDirectory() as temp_dir:
            image_dir = Path(temp_dir)
            with self.assertRaisesRegex(ValueError, "filename collision"):
                download.process_manifest(
                    pd.DataFrame(rows),
                    "MQ",
                    "train",
                    image_dir,
                    2,
                    fake_get,
                )

            self.assertEqual(list(image_dir.iterdir()), [])

        self.assertEqual(http_calls, [])

    def test_process_manifest_rejects_case_insensitive_collision_before_http(self):
        rows = [
            manifest_row(**{"Protein Id": "P1", "Unnamed: 0": 101}),
            manifest_row(**{"Protein Id": "p1", "Unnamed: 0": 202}),
        ]
        http_calls = []

        def fake_get(*args, **kwargs):
            http_calls.append((args, kwargs))
            raise AssertionError("HTTP must not run for filename collisions")

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "filename collision"):
                download.process_manifest(
                    pd.DataFrame(rows),
                    "MQ",
                    "train",
                    Path(temp_dir),
                    2,
                    fake_get,
                )

        self.assertEqual(http_calls, [])

    def test_process_manifest_rejects_stale_images_before_http(self):
        http_calls = []

        def fake_get(*args, **kwargs):
            http_calls.append((args, kwargs))
            raise AssertionError("HTTP must not run with stale formal images")

        with tempfile.TemporaryDirectory() as temp_dir:
            image_dir = Path(temp_dir)
            stale = image_dir / "stale.jpg"
            stale.write_bytes(b"existing-image")

            with self.assertRaisesRegex(FileExistsError, "existing files"):
                download.process_manifest(
                    pd.DataFrame([manifest_row()]),
                    "MQ",
                    "train",
                    image_dir,
                    2,
                    fake_get,
                )

            self.assertEqual(stale.read_bytes(), b"existing-image")

        self.assertEqual(http_calls, [])


class ManifestProcessingTest(unittest.TestCase):
    def test_process_manifest_restores_input_order_after_out_of_order_http(self):
        rows = [
            manifest_row(
                **{
                    "Unnamed: 0": 10,
                    "Protein Id": "P1",
                    "Modified URL": "https://fixtures.invalid/slow.jpg",
                    "Antibody Id": "HPA000001",
                }
            ),
            manifest_row(
                **{
                    "Unnamed: 0": 20,
                    "Protein Id": "P2",
                    "Modified URL": "https://fixtures.invalid/fast.jpg",
                    "Antibody Id": "HPA000002",
                }
            ),
        ]
        fast_response_started = threading.Event()
        payload = image_bytes("RGB", (10, 20, 30), "JPEG")

        def fake_get(url, timeout):
            self.assertEqual(timeout, 60)
            if url.endswith("slow.jpg"):
                if not fast_response_started.wait(timeout=2):
                    raise AssertionError("manifest downloads did not run concurrently")
            else:
                fast_response_started.set()
            return FakeResponse(payload)

        with tempfile.TemporaryDirectory() as temp_dir:
            image_dir = Path(temp_dir)

            successes, failures, stats = download.process_manifest(
                pd.DataFrame(rows),
                "MQ",
                "train",
                image_dir,
                2,
                fake_get,
            )

            self.assertEqual(successes["Protein Id"].tolist(), ["P1", "P2"])
            self.assertEqual(
                successes["File Name"].tolist(),
                [
                    "P1-slow-HPA000001-nucleus.jpg",
                    "P2-fast-HPA000002-nucleus.jpg",
                ],
            )
            self.assertTrue(failures.empty)
            self.assertEqual(
                sorted(path.name for path in image_dir.iterdir()),
                sorted(successes["File Name"].tolist()),
            )
            self.assertEqual(
                stats,
                {
                    "input_rows": 2,
                    "success_rows": 2,
                    "failure_rows": 0,
                    "converted_rows": 0,
                },
            )

    def test_process_manifest_retains_shared_url_and_content_across_proteins(self):
        shared_url = "https://fixtures.invalid/shared.jpg"
        rows = [
            manifest_row(
                **{
                    "Unnamed: 0": 51,
                    "Protein Id": "P10",
                    "Modified URL": shared_url,
                    "Antibody Id": "HPA000010",
                }
            ),
            manifest_row(
                **{
                    "Unnamed: 0": 52,
                    "Protein Id": "P20",
                    "Modified URL": shared_url,
                    "Antibody Id": "HPA000010",
                }
            ),
        ]
        payload = image_bytes("RGB", (10, 20, 30), "JPEG")
        requested_urls = []

        def fake_get(url, timeout):
            self.assertEqual(timeout, 60)
            requested_urls.append(url)
            return FakeResponse(payload)

        with tempfile.TemporaryDirectory() as temp_dir:
            image_dir = Path(temp_dir)

            successes, failures, _stats = download.process_manifest(
                pd.DataFrame(rows),
                "MQ",
                "train",
                image_dir,
                2,
                fake_get,
            )

            self.assertEqual(successes["Protein Id"].tolist(), ["P10", "P20"])
            self.assertEqual(requested_urls, [shared_url, shared_url])
            self.assertTrue(failures.empty)
            self.assertEqual(len(successes), 2)
            for file_name in successes["File Name"]:
                self.assertEqual((image_dir / file_name).read_bytes(), payload)

    def test_process_manifest_publishes_ordered_structured_failures(self):
        rows = [
            manifest_row(
                **{
                    "Unnamed: 0": 31,
                    "Protein Id": "P3",
                    "Modified URL": "https://fixtures.invalid/slow-http.jpg",
                    "Antibody Id": "HPA000003",
                }
            ),
            manifest_row(
                **{
                    "Unnamed: 0": 42,
                    "Protein Id": "P4",
                    "Modified URL": "https://fixtures.invalid/fast-decode.jpg",
                    "Antibody Id": "HPA000004",
                }
            ),
        ]
        fast_response_started = threading.Event()

        def fake_get(url, timeout):
            self.assertEqual(timeout, 60)
            if url.endswith("slow-http.jpg"):
                if not fast_response_started.wait(timeout=2):
                    raise AssertionError("manifest downloads did not run concurrently")
                return FakeResponse(b"unavailable", 503)
            fast_response_started.set()
            return FakeResponse(b"not an image")

        with tempfile.TemporaryDirectory() as temp_dir:
            image_dir = Path(temp_dir)

            successes, failures, stats = download.process_manifest(
                pd.DataFrame(rows),
                "MQ",
                "train",
                image_dir,
                2,
                fake_get,
            )

            self.assertTrue(successes.empty)
            self.assertEqual(
                failures.columns.tolist(),
                [
                    "tier",
                    "split",
                    "ordinal",
                    "source_row",
                    "Protein Id",
                    "URL",
                    "stage",
                    "reason",
                ],
            )
            self.assertEqual(failures["ordinal"].tolist(), [0, 1])
            self.assertEqual(failures["source_row"].tolist(), [31, 42])
            self.assertEqual(failures["Protein Id"].tolist(), ["P3", "P4"])
            self.assertEqual(failures["stage"].tolist(), ["http", "decode"])
            self.assertEqual(failures["tier"].tolist(), ["MQ", "MQ"])
            self.assertEqual(failures["split"].tolist(), ["train", "train"])
            self.assertTrue(all(failures["reason"].astype(str).str.len() > 0))
            self.assertEqual(list(image_dir.iterdir()), [])
            self.assertEqual(
                stats,
                {
                    "input_rows": 2,
                    "success_rows": 0,
                    "failure_rows": 2,
                    "converted_rows": 0,
                },
            )

    def test_process_manifest_never_overwrites_existing_images(self):
        row = manifest_row()
        http_calls = []

        def fake_get(*args, **kwargs):
            http_calls.append((args, kwargs))
            raise AssertionError("HTTP must not run for an existing target")

        with tempfile.TemporaryDirectory() as temp_dir:
            image_dir = Path(temp_dir)
            target = image_dir / "ENSG1-a-HPA000123-nucleus.jpg"
            target.write_bytes(b"existing-image")

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                download.process_manifest(
                    pd.DataFrame([row]),
                    "MQ",
                    "train",
                    image_dir,
                    2,
                    fake_get,
                )

            self.assertEqual(target.read_bytes(), b"existing-image")

        self.assertEqual(http_calls, [])


class ManifestOutputTest(unittest.TestCase):
    def test_download_manifest_publishes_images_success_csv_and_failures(self):
        rows = [
            manifest_row(
                **{
                    "Unnamed: 0": 11,
                    "Protein Id": "P1",
                    "Modified URL": "https://fixtures.invalid/slow.jpg",
                    "Antibody Id": "HPA000001",
                    "Sequence": "AAAA",
                }
            ),
            manifest_row(
                **{
                    "Unnamed: 0": 22,
                    "Protein Id": "P2",
                    "Modified URL": "https://fixtures.invalid/failure.jpg",
                    "Antibody Id": "HPA000002",
                    "Sequence": "BBBB",
                }
            ),
            manifest_row(
                **{
                    "Unnamed: 0": 33,
                    "Protein Id": "P3",
                    "Modified URL": "https://fixtures.invalid/fast.jpg",
                    "Antibody Id": "HPA000003",
                    "locations": "cytoplasm",
                    "cytoplasm": 1,
                    "nucleus": 0,
                    "Sequence": "CCCC",
                }
            ),
        ]
        fast_response_started = threading.Event()
        payload = image_bytes("RGB", (10, 20, 30), "JPEG")

        def fake_get(url, timeout):
            self.assertEqual(timeout, 60)
            if url.endswith("slow.jpg"):
                if not fast_response_started.wait(timeout=2):
                    raise AssertionError("manifest downloads did not run concurrently")
                return FakeResponse(payload)
            fast_response_started.set()
            if url.endswith("failure.jpg"):
                return FakeResponse(b"unavailable", 503)
            return FakeResponse(payload)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            outcome = download.download_manifest(
                pd.DataFrame(rows),
                "MQ",
                "train",
                output_dir,
                3,
                fake_get,
            )

            success_path = output_dir / "MQ_train.csv"
            failure_path = output_dir / "MQ_train_failures.csv"
            image_dir = output_dir / "MQ_train_img"
            successes = pd.read_csv(success_path)
            failures = pd.read_csv(failure_path)

            self.assertEqual(outcome.status, "completed_with_failures")
            self.assertEqual(outcome.success_csv, success_path)
            self.assertEqual(outcome.failure_csv, failure_path)
            self.assertEqual(
                successes.columns.tolist(),
                [
                    "File Name",
                    "locations",
                    "cytoplasm",
                    "endoplasmic reticulum",
                    "mitochondria",
                    "nucleus",
                    "plasma membrane",
                    "Sequence",
                    "Protein Id",
                ],
            )
            self.assertEqual(successes["Protein Id"].tolist(), ["P1", "P3"])
            self.assertEqual(
                successes["locations"].tolist(), ["nucleus", "cytoplasm"]
            )
            self.assertEqual(successes["cytoplasm"].tolist(), [0, 1])
            self.assertEqual(successes["nucleus"].tolist(), [1, 0])
            self.assertEqual(successes["Sequence"].tolist(), ["AAAA", "CCCC"])
            self.assertEqual(failures["Protein Id"].tolist(), ["P2"])
            self.assertEqual(failures["stage"].tolist(), ["http"])
            self.assertEqual(
                sorted(path.name for path in image_dir.glob("*.jpg")),
                sorted(successes["File Name"].tolist()),
            )
            self.assertEqual(
                outcome.stats,
                {
                    "input_rows": 3,
                    "success_rows": 2,
                    "failure_rows": 1,
                    "converted_rows": 0,
                },
            )


DATASETS = (
    ("HQ", "train"),
    ("HQ", "test"),
    ("MQ", "train"),
    ("MQ", "test"),
    ("LQ", "train"),
    ("LQ", "test"),
)


def write_pipeline_manifests(manifest_dir, rows_by_dataset):
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for tier, split in DATASETS:
        pd.DataFrame(rows_by_dataset[(tier, split)]).to_csv(
            manifest_dir / f"{tier}_{split}_img_URL.csv",
            index=False,
        )


class DownloadPipelineTest(unittest.TestCase):
    def test_main_publishes_six_rgb_datasets(self):
        payloads = [
            image_bytes("RGB", (10, 20, 30), "JPEG"),
            image_bytes("L", 40),
            image_bytes("CMYK", (0, 128, 255, 0), "JPEG"),
            rgba_image_bytes(),
            palette_image_bytes(),
            image_bytes("RGB", (30, 40, 50), "JPEG"),
        ]
        rows_by_dataset = {}
        payload_by_url = {}
        expected_proteins = {}
        for index, (tier, split) in enumerate(DATASETS, start=1):
            protein_id = f"P_{split.upper()}_{tier}"
            url = f"https://fixtures.invalid/{tier.lower()}-{split}.jpg"
            rows_by_dataset[(tier, split)] = [
                manifest_row(
                    **{
                        "Unnamed: 0": index * 10,
                        "Protein Id": protein_id,
                        "Modified URL": url,
                        "Antibody Id": f"HPA{index:06d}",
                        "Sequence": chr(64 + index) * 4,
                    }
                )
            ]
            payload_by_url[url] = payloads[index - 1]
            expected_proteins[(tier, split)] = protein_id

        shared_url = rows_by_dataset[("HQ", "train")][0]["Modified URL"]
        rows_by_dataset[("MQ", "test")][0]["Modified URL"] = shared_url
        requested_urls = []

        def fake_get(url, timeout):
            self.assertEqual(timeout, 60)
            requested_urls.append(url)
            return FakeResponse(payload_by_url[url])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_dir = root / "manifests"
            output_dir = root / "output"
            write_pipeline_manifests(manifest_dir, rows_by_dataset)
            upstream_report = {
                "status": "ok",
                "published": True,
                "record_id": "10632698",
                "seed": 73,
                "source_validation": {
                    "status": "ok",
                    "sources": {
                        "normalLabeled.csv": {"md5": "normal-md5"},
                        "data_train.csv": {"md5": "train-md5"},
                        "data_test.csv": {"md5": "test-md5"},
                    },
                },
            }
            (manifest_dir / "manifest_generation_report.json").write_text(
                json.dumps(upstream_report), encoding="utf-8"
            )
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = download.main(
                    [
                        "--manifest-dir",
                        str(manifest_dir),
                        "--output-dir",
                        str(output_dir),
                        "--workers",
                        "2",
                    ],
                    http_get=fake_get,
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads(stdout.getvalue())["protein_id_overlap"], 0)
            for tier, split in DATASETS:
                frame = pd.read_csv(output_dir / f"{tier}_{split}.csv")
                self.assertEqual(frame.columns.tolist(), download.FINAL_COLUMNS)
                self.assertEqual(
                    frame["Protein Id"].tolist(),
                    [expected_proteins[(tier, split)]],
                )
                image_path = output_dir / f"{tier}_{split}_img" / frame.loc[0, "File Name"]
                with Image.open(image_path) as image:
                    image.load()
                    self.assertEqual(image.format, "JPEG")
                    self.assertEqual(image.mode, "RGB")
                    self.assertEqual(len(image.getbands()), 3)

            train_ids = {
                expected_proteins[(tier, "train")] for tier in ("HQ", "MQ", "LQ")
            }
            test_ids = {
                expected_proteins[(tier, "test")] for tier in ("HQ", "MQ", "LQ")
            }
            self.assertEqual(train_ids & test_ids, set())
            self.assertEqual(requested_urls.count(shared_url), 2)

    def test_hq_success_preserves_official_count_and_order(self):
        rows_by_dataset = {}
        for index, (tier, split) in enumerate(DATASETS, start=1):
            rows_by_dataset[(tier, split)] = [
                manifest_row(
                    **{
                        "Unnamed: 0": index,
                        "Protein Id": f"P_{split}_{index}",
                        "Modified URL": f"https://fixtures.invalid/{index}.jpg",
                        "Antibody Id": f"HPA{index:06d}",
                    }
                )
            ]
        rows_by_dataset[("HQ", "train")] = [
            manifest_row(
                **{
                    "Unnamed: 0": 20,
                    "Protein Id": "P_OFFICIAL_SECOND",
                    "Modified URL": "https://fixtures.invalid/slow.jpg",
                    "Antibody Id": "HPA000020",
                }
            ),
            manifest_row(
                **{
                    "Unnamed: 0": 10,
                    "Protein Id": "P_OFFICIAL_FIRST",
                    "Modified URL": "https://fixtures.invalid/fast.jpg",
                    "Antibody Id": "HPA000010",
                }
            ),
        ]
        fast_started = threading.Event()
        payload = image_bytes("RGB", (10, 20, 30), "JPEG")

        def fake_get(url, timeout):
            self.assertEqual(timeout, 60)
            if url.endswith("slow.jpg"):
                if not fast_started.wait(timeout=2):
                    raise AssertionError("HQ rows were not downloaded concurrently")
            elif url.endswith("fast.jpg"):
                fast_started.set()
            return FakeResponse(payload)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_dir = root / "manifests"
            output_dir = root / "output"
            write_pipeline_manifests(manifest_dir, rows_by_dataset)

            with redirect_stdout(io.StringIO()):
                exit_code = download.main(
                    [
                        "--manifest-dir",
                        str(manifest_dir),
                        "--output-dir",
                        str(output_dir),
                        "--workers",
                        "2",
                    ],
                    http_get=fake_get,
                )

            self.assertEqual(exit_code, 0)
            hq_train = pd.read_csv(output_dir / "HQ_train.csv")
            self.assertEqual(len(hq_train), 2)
            self.assertEqual(
                hq_train["Protein Id"].tolist(),
                ["P_OFFICIAL_SECOND", "P_OFFICIAL_FIRST"],
            )
            self.assertEqual(
                len(list((output_dir / "HQ_train_img").glob("*.jpg"))),
                2,
            )

    def test_preflight_rejects_cross_manifest_protein_overlap_before_http(self):
        rows_by_dataset = {}
        for index, (tier, split) in enumerate(DATASETS, start=1):
            rows_by_dataset[(tier, split)] = [
                manifest_row(
                    **{
                        "Unnamed: 0": index,
                        "Protein Id": f"P{index}",
                        "Modified URL": f"https://fixtures.invalid/{index}.jpg",
                        "Antibody Id": f"HPA{index:06d}",
                    }
                )
            ]
        rows_by_dataset[("HQ", "train")][0]["Protein Id"] = "P_SHARED"
        rows_by_dataset[("LQ", "test")][0]["Protein Id"] = " P_SHARED "
        http_calls = []

        def fake_get(*args, **kwargs):
            http_calls.append((args, kwargs))
            return FakeResponse(image_bytes("RGB", (10, 20, 30), "JPEG"))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_dir = root / "manifests"
            output_dir = root / "output"
            write_pipeline_manifests(manifest_dir, rows_by_dataset)

            with self.assertRaisesRegex(
                AssertionError,
                "Protein Id overlap between train and test.*P_SHARED",
            ):
                download.main(
                    [
                        "--manifest-dir",
                        str(manifest_dir),
                        "--output-dir",
                        str(output_dir),
                    ],
                    http_get=fake_get,
                )

            self.assertEqual(http_calls, [])
            self.assertFalse(
                any((output_dir / f"{tier}_{split}.csv").exists() for tier, split in DATASETS)
            )

    def test_preflight_rejects_global_filename_collision_before_http(self):
        rows_by_dataset = {}
        for index, (tier, split) in enumerate(DATASETS, start=1):
            rows_by_dataset[(tier, split)] = [
                manifest_row(
                    **{
                        "Unnamed: 0": index,
                        "Protein Id": f"P_{split}_{index}",
                        "Modified URL": f"https://fixtures.invalid/{index}.jpg",
                        "Antibody Id": f"HPA{index:06d}",
                    }
                )
            ]
        colliding = manifest_row(
            **{
                "Protein Id": "P_COLLISION",
                "Modified URL": "https://fixtures.invalid/shared.jpg",
                "Antibody Id": "HPA999999",
                "locations": "nucleus",
            }
        )
        rows_by_dataset[("HQ", "train")][0].update(colliding)
        rows_by_dataset[("MQ", "train")][0].update(colliding)
        http_calls = []

        def fake_get(*args, **kwargs):
            http_calls.append((args, kwargs))
            return FakeResponse(image_bytes("RGB", (10, 20, 30), "JPEG"))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_dir = root / "manifests"
            output_dir = root / "output"
            write_pipeline_manifests(manifest_dir, rows_by_dataset)

            with self.assertRaisesRegex(ValueError, "global filename collision"):
                download.main(
                    [
                        "--manifest-dir",
                        str(manifest_dir),
                        "--output-dir",
                        str(output_dir),
                    ],
                    http_get=fake_get,
                )

            self.assertEqual(http_calls, [])
            self.assertFalse(
                any((output_dir / f"{tier}_{split}.csv").exists() for tier, split in DATASETS)
            )

    def test_preflight_checks_all_image_directories_before_http(self):
        rows_by_dataset = {}
        for index, (tier, split) in enumerate(DATASETS, start=1):
            rows_by_dataset[(tier, split)] = [
                manifest_row(
                    **{
                        "Unnamed: 0": index,
                        "Protein Id": f"P_{split}_{index}",
                        "Modified URL": f"https://fixtures.invalid/{index}.jpg",
                        "Antibody Id": f"HPA{index:06d}",
                    }
                )
            ]
        http_calls = []

        def fake_get(*args, **kwargs):
            http_calls.append((args, kwargs))
            return FakeResponse(image_bytes("RGB", (10, 20, 30), "JPEG"))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_dir = root / "manifests"
            output_dir = root / "output"
            write_pipeline_manifests(manifest_dir, rows_by_dataset)
            stale_dir = output_dir / "LQ_test_img"
            stale_dir.mkdir(parents=True)
            stale_image = stale_dir / "existing.jpg"
            stale_image.write_bytes(b"existing")

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                download.main(
                    [
                        "--manifest-dir",
                        str(manifest_dir),
                        "--output-dir",
                        str(output_dir),
                    ],
                    http_get=fake_get,
                )

            self.assertEqual(http_calls, [])
            self.assertEqual(stale_image.read_bytes(), b"existing")
            self.assertEqual(
                sorted(path.name for path in output_dir.iterdir()),
                [
                    "LQ_test_img",
                    "download_audit_report.json",
                    "download_failures.csv",
                    "zero_success_proteins.csv",
                ],
            )
            report = json.loads(
                (output_dir / "download_audit_report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(report["status"], "error")
            self.assertFalse(report["published"])

    def test_preflight_loads_all_six_schemas_before_http(self):
        rows_by_dataset = {}
        for index, (tier, split) in enumerate(DATASETS, start=1):
            rows_by_dataset[(tier, split)] = [
                manifest_row(
                    **{
                        "Unnamed: 0": index,
                        "Protein Id": f"P_{split}_{index}",
                        "Modified URL": f"https://fixtures.invalid/{index}.jpg",
                        "Antibody Id": f"HPA{index:06d}",
                    }
                )
            ]
        del rows_by_dataset[("LQ", "test")][0]["Sequence"]
        http_calls = []

        def fake_get(*args, **kwargs):
            http_calls.append((args, kwargs))
            return FakeResponse(image_bytes("RGB", (10, 20, 30), "JPEG"))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_dir = root / "manifests"
            output_dir = root / "output"
            write_pipeline_manifests(manifest_dir, rows_by_dataset)

            with self.assertRaisesRegex(ValueError, "Sequence"):
                download.main(
                    [
                        "--manifest-dir",
                        str(manifest_dir),
                        "--output-dir",
                        str(output_dir),
                    ],
                    http_get=fake_get,
                )

            self.assertEqual(http_calls, [])
            self.assertFalse(
                any(
                    (output_dir / f"{tier}_{split}.csv").exists()
                    for tier, split in DATASETS
                )
            )
            self.assertFalse(
                any(
                    (output_dir / f"{tier}_{split}_img").exists()
                    for tier, split in DATASETS
                )
            )
            report = json.loads(
                (output_dir / "download_audit_report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(report["status"], "error")
            self.assertFalse(report["published"])
            self.assertEqual(report["error"]["type"], "ValueError")
            self.assertIn("Sequence", report["error"]["message"])
            self.assertTrue(
                pd.read_csv(output_dir / "download_failures.csv").empty
            )

    def test_hq_failure_aborts_formal_publication(self):
        rows_by_dataset = {}
        for index, (tier, split) in enumerate(DATASETS, start=1):
            rows_by_dataset[(tier, split)] = [
                manifest_row(
                    **{
                        "Unnamed: 0": index * 10,
                        "Protein Id": f"P_{split}_{index}",
                        "Modified URL": f"https://fixtures.invalid/{tier}-{split}.jpg",
                        "Antibody Id": f"HPA{index:06d}",
                    }
                )
            ]
        blank_row = manifest_row(
            **{
                "Unnamed: 0": 99,
                "Protein Id": "P_HQ_BLANK",
                "Modified URL": "https://fixtures.invalid/hq-blank.jpg",
                "Antibody Id": "HPA999999",
            }
        )
        rows_by_dataset[("HQ", "train")].append(blank_row)
        requested_urls = []
        valid_payload = image_bytes("RGB", (10, 20, 30), "JPEG")
        blank_payload = image_bytes("RGB", (255, 255, 255), "JPEG")

        def fake_get(url, timeout):
            self.assertEqual(timeout, 60)
            requested_urls.append(url)
            return FakeResponse(
                blank_payload if url.endswith("hq-blank.jpg") else valid_payload
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_dir = root / "manifests"
            output_dir = root / "output"
            write_pipeline_manifests(manifest_dir, rows_by_dataset)
            output_dir.mkdir()
            old_hq = {
                "HQ_train.csv": b"old-hq-train\n",
                "HQ_test.csv": b"old-hq-test\n",
            }
            for filename, payload in old_hq.items():
                (output_dir / filename).write_bytes(payload)

            with self.assertRaisesRegex(RuntimeError, "HQ download failed"):
                download.main(
                    [
                        "--manifest-dir",
                        str(manifest_dir),
                        "--output-dir",
                        str(output_dir),
                        "--workers",
                        "2",
                    ],
                    http_get=fake_get,
                )

            for filename, payload in old_hq.items():
                self.assertEqual((output_dir / filename).read_bytes(), payload)
            failures = pd.read_csv(output_dir / "download_failures.csv")
            self.assertEqual(failures["tier"].tolist(), ["HQ"])
            self.assertEqual(failures["split"].tolist(), ["train"])
            self.assertEqual(failures["source_row"].tolist(), [99])
            self.assertEqual(failures["Protein Id"].tolist(), ["P_HQ_BLANK"])
            self.assertEqual(failures["stage"].tolist(), ["blank"])
            self.assertTrue(failures.loc[0, "reason"])
            self.assertFalse(
                any("MQ-" in url or "LQ-" in url for url in requested_urls)
            )
            report = json.loads(
                (output_dir / "download_audit_report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(report["status"], "error")
            self.assertFalse(report["published"])
            self.assertEqual(report["total_failures"], 1)

    def test_mq_lq_failures_are_logged_skipped_and_report_zero_success(self):
        rows_by_dataset = {}
        for index, (tier, split) in enumerate(DATASETS, start=1):
            rows_by_dataset[(tier, split)] = [
                manifest_row(
                    **{
                        "Unnamed: 0": index * 10,
                        "Protein Id": f"P_{split}_{index}",
                        "Modified URL": f"https://fixtures.invalid/{tier}-{split}.jpg",
                        "Antibody Id": f"HPA{index:06d}",
                    }
                )
            ]
        rows_by_dataset[("MQ", "train")] = [
            manifest_row(
                **{
                    "Unnamed: 0": 31,
                    "Protein Id": "P_MQ_FIRST",
                    "Modified URL": "https://fixtures.invalid/mq-first.jpg",
                    "Antibody Id": "HPA000031",
                }
            ),
            manifest_row(
                **{
                    "Unnamed: 0": 32,
                    "Protein Id": "P_MQ_ZERO",
                    "Modified URL": "https://fixtures.invalid/mq-http-failure.jpg",
                    "Antibody Id": "HPA000032",
                }
            ),
            manifest_row(
                **{
                    "Unnamed: 0": 33,
                    "Protein Id": "P_MQ_SECOND",
                    "Modified URL": "https://fixtures.invalid/mq-second.jpg",
                    "Antibody Id": "HPA000033",
                }
            ),
        ]
        rows_by_dataset[("LQ", "test")] = [
            manifest_row(
                **{
                    "Unnamed: 0": 61,
                    "Protein Id": "P_LQ_ZERO",
                    "Modified URL": "https://fixtures.invalid/lq-decode-failure.jpg",
                    "Antibody Id": "HPA000061",
                }
            )
        ]
        valid_payload = image_bytes("RGB", (10, 20, 30), "JPEG")

        def fake_get(url, timeout):
            self.assertEqual(timeout, 60)
            if url.endswith("mq-http-failure.jpg"):
                return FakeResponse(b"unavailable", 503)
            if url.endswith("lq-decode-failure.jpg"):
                return FakeResponse(b"not an image")
            return FakeResponse(valid_payload)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_dir = root / "manifests"
            output_dir = root / "output"
            write_pipeline_manifests(manifest_dir, rows_by_dataset)

            upstream_report = {
                "status": "ok",
                "published": True,
                "record_id": "10632698",
                "seed": 73,
                "source_validation": {
                    "status": "ok",
                    "sources": {
                        "normalLabeled.csv": {"md5": "normal-md5"},
                        "data_train.csv": {"md5": "train-md5"},
                        "data_test.csv": {"md5": "test-md5"},
                    },
                },
            }
            (manifest_dir / "manifest_generation_report.json").write_text(
                json.dumps(upstream_report), encoding="utf-8"
            )

            with redirect_stdout(io.StringIO()):
                exit_code = download.main(
                    [
                        "--manifest-dir",
                        str(manifest_dir),
                        "--output-dir",
                        str(output_dir),
                        "--workers",
                        "3",
                    ],
                    http_get=fake_get,
                )

            self.assertEqual(exit_code, 0)
            mq_train = pd.read_csv(output_dir / "MQ_train.csv")
            self.assertEqual(
                mq_train["Protein Id"].tolist(),
                ["P_MQ_FIRST", "P_MQ_SECOND"],
            )
            self.assertTrue(pd.read_csv(output_dir / "LQ_test.csv").empty)
            failures = pd.read_csv(output_dir / "download_failures.csv")
            self.assertEqual(
                list(zip(failures["tier"], failures["split"])),
                [("MQ", "train"), ("LQ", "test")],
            )
            self.assertEqual(
                failures["Protein Id"].tolist(),
                ["P_MQ_ZERO", "P_LQ_ZERO"],
            )
            self.assertEqual(failures["stage"].tolist(), ["http", "decode"])
            zero_success = pd.read_csv(
                output_dir / "zero_success_proteins.csv"
            )
            self.assertEqual(
                zero_success.to_dict("records"),
                [
                    {
                        "tier": "MQ",
                        "split": "train",
                        "Protein Id": "P_MQ_ZERO",
                        "input_rows": 1,
                    },
                    {
                        "tier": "LQ",
                        "split": "test",
                        "Protein Id": "P_LQ_ZERO",
                        "input_rows": 1,
                    },
                ],
            )
            report = json.loads(
                (output_dir / "download_audit_report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(report["status"], "ok")
            self.assertTrue(report["published"])
            self.assertEqual(report["total_failures"], 2)
            self.assertEqual(report["zero_success_proteins"], 2)
            self.assertEqual(
                report["upstream"]["manifest_generation_report"],
                upstream_report,
            )
            self.assertIsNone(
                report["upstream"]["source_validation_report"]
            )
            self.assertEqual(
                report["datasets"]["MQ_train"],
                {
                    "input_rows": 3,
                    "success_rows": 2,
                    "failure_rows": 1,
                    "converted_rows": 0,
                },
            )
            self.assertEqual(
                [failure["Protein Id"] for failure in report["failures"]],
                ["P_MQ_ZERO", "P_LQ_ZERO"],
            )
            self.assertEqual(
                report["zero_success_details"],
                zero_success.to_dict("records"),
            )
            self.assertTrue(report["protein_id_leakage"]["checked"])
            self.assertEqual(report["protein_id_leakage"]["overlap"], [])

    def test_hq_sequence_failure_is_reported_before_http(self):
        rows_by_dataset = {}
        for index, (tier, split) in enumerate(DATASETS, start=1):
            rows_by_dataset[(tier, split)] = [
                manifest_row(
                    **{
                        "Unnamed: 0": index * 10,
                        "Protein Id": f"P_{split}_{index}",
                        "Modified URL": f"https://fixtures.invalid/{tier}-{split}.jpg",
                        "Antibody Id": f"HPA{index:06d}",
                    }
                )
            ]
        rows_by_dataset[("HQ", "test")][0]["Sequence"] = "   "
        rows_by_dataset[("MQ", "train")][0]["Modified URL"] = "   "
        http_calls = []

        def fake_get(*args, **kwargs):
            http_calls.append((args, kwargs))
            return FakeResponse(image_bytes("RGB", (10, 20, 30), "JPEG"))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_dir = root / "manifests"
            output_dir = root / "output"
            write_pipeline_manifests(manifest_dir, rows_by_dataset)
            output_dir.mkdir()
            old_train = b"old-hq-train\n"
            old_test = b"old-hq-test\n"
            (output_dir / "HQ_train.csv").write_bytes(old_train)
            (output_dir / "HQ_test.csv").write_bytes(old_test)

            with self.assertRaisesRegex(RuntimeError, "HQ download failed"):
                download.main(
                    [
                        "--manifest-dir",
                        str(manifest_dir),
                        "--output-dir",
                        str(output_dir),
                    ],
                    http_get=fake_get,
                )

            self.assertEqual(http_calls, [])
            self.assertEqual(
                (output_dir / "HQ_train.csv").read_bytes(), old_train
            )
            self.assertEqual(
                (output_dir / "HQ_test.csv").read_bytes(), old_test
            )
            failures = pd.read_csv(output_dir / "download_failures.csv")
            self.assertEqual(len(failures), 2)
            self.assertEqual(failures["tier"].tolist(), ["HQ", "MQ"])
            self.assertEqual(failures["split"].tolist(), ["test", "train"])
            self.assertEqual(failures["stage"].tolist(), ["sequence", "url"])
            self.assertEqual(
                failures["Protein Id"].tolist(),
                ["P_test_2", "P_train_3"],
            )
            report = json.loads(
                (output_dir / "download_audit_report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(report["total_failures"], 2)
            self.assertEqual(len(report["failures"]), 2)

    def test_mq_lq_sequence_and_url_preflight_failures_skip_rows(self):
        rows_by_dataset = {}
        for index, (tier, split) in enumerate(DATASETS, start=1):
            rows_by_dataset[(tier, split)] = [
                manifest_row(
                    **{
                        "Unnamed: 0": index * 10,
                        "Protein Id": f"P_{split}_{index}",
                        "Modified URL": f"https://fixtures.invalid/{tier}-{split}.jpg",
                        "Antibody Id": f"HPA{index:06d}",
                    }
                )
            ]
        rows_by_dataset[("MQ", "train")].insert(
            0,
            manifest_row(
                **{
                    "Unnamed: 0": 31,
                    "Protein Id": "P_MQ_NO_SEQUENCE",
                    "Modified URL": "https://fixtures.invalid/mq-no-sequence.jpg",
                    "Antibody Id": "HPA000031",
                    "Sequence": " ",
                }
            ),
        )
        rows_by_dataset[("LQ", "test")].insert(
            0,
            manifest_row(
                **{
                    "Unnamed: 0": 61,
                    "Protein Id": "P_LQ_NO_URL",
                    "Modified URL": " ",
                    "Antibody Id": "HPA000061",
                }
            ),
        )
        requested_urls = []
        valid_payload = image_bytes("RGB", (10, 20, 30), "JPEG")

        def fake_get(url, timeout):
            self.assertEqual(timeout, 60)
            requested_urls.append(url)
            return FakeResponse(valid_payload)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_dir = root / "manifests"
            output_dir = root / "output"
            write_pipeline_manifests(manifest_dir, rows_by_dataset)

            with redirect_stdout(io.StringIO()):
                exit_code = download.main(
                    [
                        "--manifest-dir",
                        str(manifest_dir),
                        "--output-dir",
                        str(output_dir),
                    ],
                    http_get=fake_get,
                )

            self.assertEqual(exit_code, 0)
            self.assertNotIn(
                "https://fixtures.invalid/mq-no-sequence.jpg",
                requested_urls,
            )
            self.assertNotIn("", [url.strip() for url in requested_urls])
            failures = pd.read_csv(output_dir / "download_failures.csv")
            self.assertEqual(
                failures["Protein Id"].tolist(),
                ["P_MQ_NO_SEQUENCE", "P_LQ_NO_URL"],
            )
            self.assertEqual(failures["stage"].tolist(), ["sequence", "url"])
            zero_success = pd.read_csv(
                output_dir / "zero_success_proteins.csv"
            )
            self.assertEqual(
                list(zip(zero_success["tier"], zero_success["split"])),
                [("MQ", "train"), ("LQ", "test")],
            )
            self.assertEqual(
                zero_success["Protein Id"].tolist(),
                ["P_MQ_NO_SEQUENCE", "P_LQ_NO_URL"],
            )


class FinalManifestPublicationTest(unittest.TestCase):
    def _empty_frames(self):
        return {
            dataset: pd.DataFrame(columns=download.FINAL_COLUMNS)
            for dataset in DATASETS
        }

    def test_success_guard_rejects_normalized_protein_overlap(self):
        frames = self._empty_frames()
        base = {
            "File Name": "image.jpg",
            "locations": "nucleus",
            "cytoplasm": 0,
            "endoplasmic reticulum": 0,
            "mitochondria": 0,
            "nucleus": 1,
            "plasma membrane": 0,
            "Sequence": "AAAA",
        }
        frames[("HQ", "train")] = pd.DataFrame(
            [{**base, "Protein Id": "P1"}], columns=download.FINAL_COLUMNS
        )
        frames[("MQ", "test")] = pd.DataFrame(
            [{**base, "Protein Id": " P1 "}], columns=download.FINAL_COLUMNS
        )

        with self.assertRaisesRegex(
            AssertionError,
            "Protein Id overlap between successful train and test.*P1",
        ):
            download.assert_success_protein_disjoint(frames)

    def test_bundle_publish_rolls_back_all_six_csvs(self):
        frames = {}
        for index, dataset in enumerate(DATASETS, start=1):
            row = manifest_row(**{"Protein Id": f"P{index}"})
            frames[dataset] = pd.DataFrame(
                [
                    {
                        "File Name": f"image-{index}.jpg",
                        "locations": row["locations"],
                        "cytoplasm": row["cytoplasm"],
                        "endoplasmic reticulum": row[
                            "endoplasmic reticulum"
                        ],
                        "mitochondria": row["mitochondria"],
                        "nucleus": row["nucleus"],
                        "plasma membrane": row["plasma membrane"],
                        "Sequence": row["Sequence"],
                        "Protein Id": row["Protein Id"],
                    }
                ],
                columns=download.FINAL_COLUMNS,
            )

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            old_payloads = {}
            for tier, split in DATASETS:
                path = output_dir / f"{tier}_{split}.csv"
                payload = f"old-{tier}-{split}\n".encode()
                path.write_bytes(payload)
                old_payloads[path] = payload

            published_replacements = 0

            def fail_second_publish(source, destination):
                nonlocal published_replacements
                if source.parent.name.startswith(".final-csv-staging-"):
                    published_replacements += 1
                    if published_replacements == 2:
                        raise OSError("second publish failed")
                return os.replace(source, destination)

            with self.assertRaisesRegex(OSError, "second publish failed"):
                download.publish_final_manifests(
                    frames,
                    output_dir,
                    replace=fail_second_publish,
                )

            for path, payload in old_payloads.items():
                self.assertEqual(path.read_bytes(), payload)
            self.assertEqual(
                [
                    path.name
                    for path in output_dir.iterdir()
                    if path.name.startswith(".final-csv-staging-")
                    or ".backup-" in path.name
                    or path.name.endswith(".part")
                ],
                [],
            )


class DownloadCliTest(unittest.TestCase):
    def test_script_help_exposes_six_manifest_arguments(self):
        result = subprocess.run(
            [sys.executable, str(Path(download.__file__).resolve()), "--help"],
            cwd=Path(download.__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--manifest-dir", result.stdout)
        self.assertIn("--output-dir", result.stdout)
        self.assertIn("--workers", result.stdout)


@contextmanager
def written_image(payload):
    with tempfile.TemporaryDirectory() as temp_dir:
        target = Path(temp_dir) / "image.jpg"
        converted = write_validated_image(payload, target)
        yield target, converted


class DownloadImageContractTest(unittest.TestCase):
    def test_all_channels_at_threshold_are_blank(self):
        image = Image.new("RGB", (2, 2), (250, 250, 250))

        self.assertTrue(is_blank_rgb(image))

    def test_any_channel_below_threshold_is_not_blank(self):
        image = Image.new("RGB", (2, 2), (249, 255, 255))

        self.assertFalse(is_blank_rgb(image))

    def test_rejects_image_at_blank_threshold_after_rgb_conversion(self):
        payload = image_bytes("RGB", (250, 250, 250))

        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "image.jpg"
            with self.assertRaisesRegex(ImageValidationError, "blank"):
                write_validated_image(payload, target)
            self.assertFalse(target.exists())

    def test_rejects_conversion_that_becomes_blank_after_jpeg_encoding(self):
        payload = image_bytes("RGB", (249, 251, 252))

        with self.assertRaisesRegex(ImageValidationError, "final image is blank"):
            normalized_jpeg_bytes(payload)

    def test_rejects_unrecognized_content_as_decode_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "image.jpg"
            with self.assertRaisesRegex(
                ImageValidationError, "image decode failed"
            ):
                write_validated_image(b"not an image", target)
            self.assertFalse(target.exists())

    def test_rejects_truncated_content_as_decode_failure(self):
        payload = image_bytes("RGB", (10, 20, 30), "JPEG")
        truncated = payload[: len(payload) // 2]

        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "image.jpg"
            with self.assertRaisesRegex(
                ImageValidationError, "image decode failed"
            ):
                write_validated_image(truncated, target)
            self.assertFalse(target.exists())

    def test_preserves_nonblank_rgb_jpeg_bytes(self):
        payload = image_bytes("RGB", (10, 20, 30), "JPEG")

        normalized, converted = normalized_jpeg_bytes(payload)

        self.assertFalse(converted)
        self.assertEqual(normalized, payload)

    def test_converts_grayscale_to_rgb_jpeg(self):
        payload = image_bytes("L", 40)

        with written_image(payload) as (target, converted):
            self.assertTrue(converted)
            with Image.open(target) as image:
                image.load()
                self.assertEqual(image.format, "JPEG")
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(len(image.getbands()), 3)

    def test_converts_cmyk_jpeg_to_rgb_jpeg(self):
        payload = image_bytes("CMYK", (0, 128, 255, 0), "JPEG")

        with written_image(payload) as (target, converted):
            self.assertTrue(converted)
            self.assertNotEqual(target.read_bytes(), payload)
            with Image.open(target) as image:
                image.load()
                self.assertEqual(image.format, "JPEG")
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(len(image.getbands()), 3)

    def test_converts_palette_image_to_rgb_jpeg(self):
        with written_image(palette_image_bytes()) as (target, converted):
            self.assertTrue(converted)
            with Image.open(target) as image:
                image.load()
                self.assertEqual(image.format, "JPEG")
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(len(image.getbands()), 3)

    def test_composites_alpha_transparency_on_white(self):
        with written_image(rgba_image_bytes()) as (target, converted):
            self.assertTrue(converted)
            with Image.open(target) as image:
                image.load()
                self.assertTrue(
                    all(channel >= 245 for channel in image.getpixel((2, 8)))
                )

    def test_composites_palette_transparency_on_white(self):
        with written_image(palette_transparency_image_bytes()) as (
            target,
            converted,
        ):
            self.assertTrue(converted)
            with Image.open(target) as image:
                image.load()
                self.assertTrue(
                    all(channel >= 245 for channel in image.getpixel((2, 8)))
                )

    def test_atomically_writes_and_preserves_qualified_jpeg_bytes(self):
        payload = image_bytes("RGB", (10, 20, 30), "JPEG")
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "nested" / "image.jpg"

            converted = write_validated_image(payload, target)

            self.assertFalse(converted)
            self.assertEqual(target.read_bytes(), payload)
            self.assertFalse(target.with_name("image.jpg.part").exists())
            with Image.open(target) as image:
                image.load()
                self.assertEqual(image.format, "JPEG")
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(len(image.getbands()), 3)

    def test_atomic_replace_failure_preserves_target_and_cleans_part_file(self):
        original = image_bytes("RGB", (10, 20, 30), "JPEG")
        replacement = image_bytes("RGB", (40, 50, 60), "JPEG")
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "image.jpg"
            target.write_bytes(original)
            part_path = target.with_name("image.jpg.part")

            with patch.object(
                type(target), "replace", side_effect=OSError("replace failed")
            ):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    write_validated_image(replacement, target)

            self.assertEqual(target.read_bytes(), original)
            self.assertFalse(part_path.exists())

    def test_write_revalidation_rejects_invalid_staged_content(self):
        source = image_bytes("RGB", (10, 20, 30), "JPEG")
        staged_cases = {
            "non-JPEG": image_bytes("RGB", (10, 20, 30), "PNG"),
            "non-RGB": image_bytes("L", 40, "JPEG"),
            "corrupt": b"not an image",
        }
        for case, staged_payload in staged_cases.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temp_dir:
                target = Path(temp_dir) / "image.jpg"
                part_path = target.with_name("image.jpg.part")

                def write_staged(path, _candidate):
                    with path.open("wb") as output:
                        return output.write(staged_payload)

                with patch.object(
                    type(target),
                    "write_bytes",
                    autospec=True,
                    side_effect=write_staged,
                ):
                    with self.assertRaisesRegex(
                        ImageValidationError, "written image"
                    ):
                        write_validated_image(source, target)

                self.assertFalse(target.exists())
                self.assertFalse(part_path.exists())

    def test_write_revalidation_rejects_blank_staged_jpeg(self):
        source = image_bytes("RGB", (10, 20, 30), "JPEG")
        blank_jpeg = image_bytes("RGB", (255, 255, 255), "JPEG")
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "image.jpg"
            part_path = target.with_name("image.jpg.part")

            def write_blank(path, _candidate):
                with path.open("wb") as output:
                    return output.write(blank_jpeg)

            with patch.object(
                type(target),
                "write_bytes",
                autospec=True,
                side_effect=write_blank,
            ):
                with self.assertRaisesRegex(
                    ImageValidationError, "written image is blank"
                ):
                    write_validated_image(source, target)

            self.assertFalse(target.exists())
            self.assertFalse(part_path.exists())


if __name__ == "__main__":
    unittest.main()
