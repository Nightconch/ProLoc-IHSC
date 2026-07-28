import io
import sys
import tempfile
import threading
import unittest
from contextlib import contextmanager
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
