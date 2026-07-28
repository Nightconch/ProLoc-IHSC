import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pandas as pd
from PIL import Image

from dataset.download import download as downloader
from dataset.download import supplemental_quality_manifests as manifests


DATASETS = (
    ("HQ", "train"),
    ("HQ", "test"),
    ("MQ", "train"),
    ("MQ", "test"),
    ("LQ", "train"),
    ("LQ", "test"),
)
SOURCE_ROW_ID = "Unnamed: 0"
REQUIRED_SOURCE_COLUMNS = [
    "Unnamed: 0",
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
REQUIRED_DOWNLOAD_LABEL_COLUMNS = [
    "cytoplasm",
    "endoplasmic reticulum",
    "mitochondria",
    "nucleus",
    "plasma membrane",
]
MANIFEST_FILENAMES = [
    "HQ_train_img_URL.csv",
    "HQ_test_img_URL.csv",
    "MQ_train_img_URL.csv",
    "MQ_test_img_URL.csv",
    "LQ_train_img_URL.csv",
    "LQ_test_img_URL.csv",
]
EXPECTED_FINAL_COLUMNS = [
    "File Name",
    "locations",
    "cytoplasm",
    "endoplasmic reticulum",
    "mitochondria",
    "nucleus",
    "plasma membrane",
    "Sequence",
    "Protein Id",
]
EXPECTED_MANIFEST_SOURCE_IDS = {
    "HQ_train": [20, 10],
    "HQ_test": [30],
    "MQ_train": [40, 42, 44, 100, 102, 104, 106],
    "MQ_test": [108],
    "LQ_train": [101, 103, 105, 107, 109],
    "LQ_test": [41, 43],
}
EXPECTED_FINAL_FILE_NAMES = {
    "HQ_train": [
        "P_HQ_TRAIN-official-rgb-HPA000123-nucleus.jpg",
        "P_HQ_A-official-gray-HPA000123-nucleus.jpg",
    ],
    "HQ_test": ["P_HQ_TEST-official-rgba-HPA000123-nucleus.jpg"],
    "MQ_train": [
        "P_HQ_TRAIN-shared-HPA000777-nucleus.jpg",
        "P_UNKNOWN_0-unknown-0-HPA000123-nucleus.jpg",
        "P_UNKNOWN_2-unknown-2-HPA000123-nucleus.jpg",
        "P_UNKNOWN_4-unknown-4-HPA000123-nucleus.jpg",
        "P_UNKNOWN_6-unknown-6-HPA000123-nucleus.jpg",
    ],
    "MQ_test": ["P_UNKNOWN_8-unknown-8-HPA000123-nucleus.jpg"],
    "LQ_train": [
        "P_UNKNOWN_1-unknown-1-HPA000123-nucleus.jpg",
        "P_UNKNOWN_3-unknown-3-HPA000123-nucleus.jpg",
        "P_UNKNOWN_5-unknown-5-HPA000123-nucleus.jpg",
        "P_UNKNOWN_7-unknown-7-HPA000123-nucleus.jpg",
        "P_UNKNOWN_9-unknown-9-HPA000123-nucleus.jpg",
    ],
    "LQ_test": ["P_HQ_TEST-shared-HPA000777-nucleus.jpg"],
}
EXPECTED_FINAL_PROTEIN_IDS = {
    "HQ_train": ["P_HQ_TRAIN", "P_HQ_A"],
    "HQ_test": ["P_HQ_TEST"],
    "MQ_train": [
        "P_HQ_TRAIN",
        "P_UNKNOWN_0",
        "P_UNKNOWN_2",
        "P_UNKNOWN_4",
        "P_UNKNOWN_6",
    ],
    "MQ_test": ["P_UNKNOWN_8"],
    "LQ_train": [
        "P_UNKNOWN_1",
        "P_UNKNOWN_3",
        "P_UNKNOWN_5",
        "P_UNKNOWN_7",
        "P_UNKNOWN_9",
    ],
    "LQ_test": ["P_HQ_TEST"],
}


class FixtureResponse:
    def __init__(self, content, status_code=200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def image_payload(mode, color, image_format="PNG"):
    image = Image.new(mode, (8, 6), color)
    output = io.BytesIO()
    image.save(output, format=image_format)
    return output.getvalue()


def rgba_payload_with_transparency():
    image = Image.new("RGBA", (8, 6), (255, 0, 0, 0))
    for x in range(4, 8):
        for y in range(6):
            image.putpixel((x, y), (20, 30, 40, 255))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def source_row(
    source_id,
    protein_id,
    intensity,
    image_name,
    *,
    antibody_id="HPA000123",
):
    row = {column: "" for column in REQUIRED_SOURCE_COLUMNS}
    row.update({column: 0 for column in REQUIRED_DOWNLOAD_LABEL_COLUMNS})
    row.update(
        {
            SOURCE_ROW_ID: source_id,
            "Protein Name": f"Protein {str(protein_id).strip()}",
            "Protein Id": protein_id,
            "Antibody Id": antibody_id,
            "Reliability Verification": "approved",
            "Tissue": "Caudate",
            "Organ": "Brain",
            "Staining Level": "high",
            "Intensity Level": intensity,
            "Quantity": ">75%",
            "SnomedParameters": "fixture",
            "URL": f"Brain/Caudate/{antibody_id}/{image_name}",
            "IF Verification": "enhanced",
            "locations": "nucleus",
            "IF Organ": "Brain",
            "nucleus": 1,
        }
    )
    return row


def fixture_frames():
    official_a = source_row(10, "P_HQ_A", "strong", "official-gray.png")
    official_train = source_row(
        20, " P_HQ_TRAIN ", "strong", "official-rgb.jpg"
    )
    official_test = source_row(
        30, "P_HQ_TEST", "strong", "official-rgba.png"
    )
    supplemental_rows = [
        source_row(
            40,
            "P_HQ_TRAIN",
            "strong",
            "shared.jpg",
            antibody_id="HPA000777",
        ),
        source_row(
            41,
            "P_HQ_TEST",
            "weak",
            "shared.jpg",
            antibody_id="HPA000777",
        ),
        source_row(42, "P_HQ_TRAIN", "moderate", "white.png"),
        source_row(43, "P_HQ_TEST", "weak", "unavailable.jpg"),
        source_row(44, "P_HQ_TRAIN", "moderate", "corrupt.jpg"),
    ]
    for number in range(10):
        supplemental_rows.append(
            source_row(
                100 + number,
                f"P_UNKNOWN_{number}",
                "moderate" if number % 2 == 0 else "weak",
                f"unknown-{number}.png",
            )
        )
    source = pd.DataFrame(
        [official_a, official_train, official_test, *supplemental_rows],
        columns=REQUIRED_SOURCE_COLUMNS,
    )
    return {
        "normalLabeled.csv": source,
        "data_train.csv": pd.DataFrame(
            [official_train, official_a], columns=REQUIRED_SOURCE_COLUMNS
        ),
        "data_test.csv": pd.DataFrame(
            [official_test], columns=REQUIRED_SOURCE_COLUMNS
        ),
    }


def file_md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def write_fixture_sources(cache_dir, frames=None):
    cache_dir.mkdir(parents=True)
    source_md5 = {}
    frames = fixture_frames() if frames is None else frames
    for name, frame in frames.items():
        path = cache_dir / name
        frame.to_csv(path, index=False)
        source_md5[name] = file_md5(path)
    source_urls = {
        name: f"https://fixtures.invalid/sources/{name}"
        for name in source_md5
    }
    return source_urls, source_md5


def fixture_sequence_resolver(protein_ids, _cache_path):
    normalized = {str(value).strip() for value in protein_ids}
    return ({protein_id: f"SEQUENCE_{protein_id}" for protein_id in normalized}, set())


def controlled_http_get(url, timeout):
    if timeout != 60:
        raise AssertionError(f"unexpected timeout: {timeout}")
    image_name = url.rsplit("/", 1)[-1]
    payloads = {
        "official-rgb.jpg": FixtureResponse(
            image_payload("RGB", (12, 34, 56), "JPEG")
        ),
        "official-gray.png": FixtureResponse(image_payload("L", 70)),
        "official-rgba.png": FixtureResponse(rgba_payload_with_transparency()),
        "shared.jpg": FixtureResponse(
            image_payload("RGB", (40, 50, 60), "JPEG")
        ),
        "white.png": FixtureResponse(image_payload("RGB", (255, 255, 255))),
        "corrupt.jpg": FixtureResponse(
            image_payload("RGB", (11, 22, 33), "JPEG")[:100]
        ),
        "unavailable.jpg": FixtureResponse(b"unavailable", 503),
    }
    if image_name.startswith("unknown-"):
        number = int(image_name.removeprefix("unknown-").removesuffix(".png"))
        if number % 3 == 0:
            return FixtureResponse(image_payload("RGB", (70, 80, 90), "JPEG"))
        if number % 3 == 1:
            return FixtureResponse(image_payload("L", 90))
        return FixtureResponse(rgba_payload_with_transparency())
    return payloads[image_name]


class DatasetPipelineEndToEndTest(unittest.TestCase):
    def test_script_help_exposes_documented_arguments_without_pipeline_work(self):
        scripts = (
            (
                Path(manifests.__file__),
                {"--output-dir", "--cache-dir", "--seed"},
            ),
            (
                Path(downloader.__file__),
                {"--manifest-dir", "--output-dir", "--workers"},
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            working_dir = Path(directory)
            for script, documented_arguments in scripts:
                with self.subTest(script=script.name):
                    result = subprocess.run(
                        [sys.executable, str(script), "--help"],
                        cwd=working_dir,
                        capture_output=True,
                        text=True,
                        check=False,
                    )

                    self.assertEqual(result.returncode, 0)
                    self.assertEqual(result.stderr, "")
                    self.assertTrue(
                        documented_arguments.issubset(result.stdout.split())
                    )
            self.assertEqual(list(working_dir.iterdir()), [])

    def test_manifest_main_writes_error_diagnostics_for_trimmed_official_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_dir = root / "cache"
            output_dir = root / "manifest-output"
            frames = fixture_frames()
            frames["normalLabeled.csv"].loc[
                frames["normalLabeled.csv"][SOURCE_ROW_ID].eq(30), "Protein Id"
            ] = " P_HQ_TRAIN "
            frames["data_test.csv"].loc[:, "Protein Id"] = " P_HQ_TRAIN "
            fixture_urls, fixture_md5 = write_fixture_sources(cache_dir, frames)
            stderr = io.StringIO()

            with redirect_stderr(stderr):
                result = manifests.main(
                    [
                        "--cache-dir",
                        str(cache_dir),
                        "--output-dir",
                        str(output_dir),
                        "--seed",
                        "73",
                    ],
                    source_urls=fixture_urls,
                    source_md5=fixture_md5,
                    sequence_resolver=fixture_sequence_resolver,
                )

            self.assertEqual(result, 1)
            error_report = json.loads(stderr.getvalue())
            self.assertEqual(error_report["status"], "error")
            self.assertTrue(
                (output_dir / "manifest_generation_report.json").is_file()
            )
            self.assertTrue((output_dir / "manifest_failures.csv").is_file())

    def test_downloader_script_reports_missing_formal_manifest_before_http(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_dir = root / "manifests"
            output_dir = root / "download-output"
            manifest_dir.mkdir()
            for tier, split in DATASETS[1:]:
                pd.DataFrame(columns=downloader.REQUIRED_MANIFEST_COLUMNS).to_csv(
                    manifest_dir / f"{tier}_{split}_img_URL.csv", index=False
                )

            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(downloader.__file__)),
                    "--manifest-dir",
                    str(manifest_dir),
                    "--output-dir",
                    str(output_dir),
                    "--workers",
                    "1",
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            audit_report = json.loads(
                (output_dir / "download_audit_report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(audit_report["status"], "error")
            self.assertFalse(audit_report["published"])
            self.assertIn(
                "HQ_train_img_URL.csv", audit_report["error"]["message"]
            )

    def run_pipeline(self, root):
        cache_dir = root / "cache"
        manifest_dir = root / "manifests"
        output_dir = root / "output"
        fixture_urls, fixture_md5 = write_fixture_sources(cache_dir)
        manifest_stdout = io.StringIO()
        manifest_stderr = io.StringIO()
        with redirect_stdout(manifest_stdout), redirect_stderr(manifest_stderr):
            manifest_code = manifests.main(
                [
                    "--cache-dir",
                    str(cache_dir),
                    "--output-dir",
                    str(manifest_dir),
                    "--seed",
                    "73",
                ],
                source_urls=fixture_urls,
                source_md5=fixture_md5,
                sequence_resolver=fixture_sequence_resolver,
            )
        download_stdout = io.StringIO()
        download_stderr = io.StringIO()
        with redirect_stdout(download_stdout), redirect_stderr(download_stderr):
            download_code = downloader.main(
                [
                    "--manifest-dir",
                    str(manifest_dir),
                    "--output-dir",
                    str(output_dir),
                    "--workers",
                    "3",
                ],
                http_get=controlled_http_get,
            )
        self.assertEqual(manifest_stderr.getvalue(), "")
        self.assertEqual(download_stderr.getvalue(), "")
        return manifest_code, download_code, manifest_dir, output_dir

    def read_formal_outputs(self, output_dir):
        return {
            f"{tier}_{split}": pd.read_csv(output_dir / f"{tier}_{split}.csv")
            for tier, split in DATASETS
        }

    def test_two_cli_pipeline_is_deterministic_and_contract_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.run_pipeline(root / "first")
            second = self.run_pipeline(root / "second")
            first_manifest_code, first_download_code, first_manifests, first_output = first
            second_manifest_code, second_download_code, second_manifests, second_output = second

            self.assertEqual(
                (first_manifest_code, first_download_code), (0, 0)
            )
            self.assertEqual(
                (second_manifest_code, second_download_code), (0, 0)
            )

            for filename in MANIFEST_FILENAMES:
                self.assertEqual(
                    (first_manifests / filename).read_bytes(),
                    (second_manifests / filename).read_bytes(),
                    filename,
                )
            for filename in [f"{tier}_{split}.csv" for tier, split in DATASETS]:
                self.assertEqual(
                    (first_output / filename).read_bytes(),
                    (second_output / filename).read_bytes(),
                    filename,
                )
            for filename, directory_a, directory_b in (
                (
                    "manifest_generation_report.json",
                    first_manifests,
                    second_manifests,
                ),
                ("download_audit_report.json", first_output, second_output),
            ):
                self.assertEqual(
                    (directory_a / filename).read_bytes(),
                    (directory_b / filename).read_bytes(),
                    filename,
                )

            generated_hq_train = pd.read_csv(first_manifests / MANIFEST_FILENAMES[0])
            generated_hq_test = pd.read_csv(first_manifests / MANIFEST_FILENAMES[1])
            self.assertEqual(generated_hq_train["Unnamed: 0"].tolist(), [20, 10])
            self.assertEqual(generated_hq_test["Unnamed: 0"].tolist(), [30])

            manifest_frames = {
                filename.removesuffix("_img_URL.csv"): pd.read_csv(
                    first_manifests / filename
                )
                for filename in MANIFEST_FILENAMES
            }
            for dataset, expected_source_ids in EXPECTED_MANIFEST_SOURCE_IDS.items():
                self.assertEqual(
                    manifest_frames[dataset][SOURCE_ROW_ID].tolist(),
                    expected_source_ids,
                )
            self.assertIn(40, manifest_frames["MQ_train"]["Unnamed: 0"].tolist())
            self.assertIn(41, manifest_frames["LQ_test"]["Unnamed: 0"].tolist())
            shared_url = manifest_frames["MQ_train"].loc[
                manifest_frames["MQ_train"]["Unnamed: 0"].eq(40), "Modified URL"
            ].iloc[0]
            self.assertEqual(
                shared_url,
                manifest_frames["LQ_test"].loc[
                    manifest_frames["LQ_test"]["Unnamed: 0"].eq(41), "Modified URL"
                ].iloc[0],
            )
            self.assertEqual(
                sum(frame["Modified URL"].eq(shared_url).sum() for name, frame in manifest_frames.items() if name.endswith("_train")),
                1,
            )
            self.assertEqual(
                sum(frame["Modified URL"].eq(shared_url).sum() for name, frame in manifest_frames.items() if name.endswith("_test")),
                1,
            )

            manifest_report = json.loads(
                (first_manifests / "manifest_generation_report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest_report["split"]["unknown_proteins"], 10)
            self.assertEqual(manifest_report["split"]["unknown_test_proteins"], 1)

            final_outputs = self.read_formal_outputs(first_output)
            repeated_final_outputs = self.read_formal_outputs(second_output)
            for dataset, final_frame in final_outputs.items():
                self.assertEqual(final_frame.columns.tolist(), EXPECTED_FINAL_COLUMNS)
                self.assertEqual(
                    final_frame["File Name"].tolist(),
                    repeated_final_outputs[dataset]["File Name"].tolist(),
                )
                self.assertEqual(
                    final_frame["File Name"].tolist(),
                    EXPECTED_FINAL_FILE_NAMES[dataset],
                )
                self.assertEqual(
                    final_frame["Protein Id"].tolist(),
                    EXPECTED_FINAL_PROTEIN_IDS[dataset],
                )

            failures = pd.read_csv(first_output / "download_failures.csv").fillna("")
            self.assertEqual(
                set(failures["source_row"].astype(str)), {"42", "43", "44"}
            )
            self.assertEqual(set(failures["stage"]), {"blank", "decode", "http"})
            failed_file_names = {
                "P_HQ_TRAIN-white-HPA000123-nucleus.jpg",
                "P_HQ_TRAIN-corrupt-HPA000123-nucleus.jpg",
                "P_HQ_TEST-unavailable-HPA000123-nucleus.jpg",
            }
            all_final_file_names = set().union(
                *(set(frame["File Name"]) for frame in final_outputs.values())
            )
            self.assertEqual(failed_file_names & all_final_file_names, set())

            train_protein_ids = set().union(
                *(
                    set(frame["Protein Id"].map(str.strip))
                    for name, frame in final_outputs.items()
                    if name.endswith("_train")
                )
            )
            test_protein_ids = set().union(
                *(
                    set(frame["Protein Id"].map(str.strip))
                    for name, frame in final_outputs.items()
                    if name.endswith("_test")
                )
            )
            self.assertEqual(train_protein_ids & test_protein_ids, set())

            for tier, split in DATASETS:
                dataset = f"{tier}_{split}"
                frame = final_outputs[dataset]
                image_dir = first_output / f"{dataset}_img"
                image_names = [path.name for path in image_dir.glob("*.jpg")]
                self.assertEqual(len(frame["File Name"]), len(set(frame["File Name"])))
                self.assertEqual(sorted(image_names), sorted(frame["File Name"].tolist()))
                for image_name in image_names:
                    with Image.open(image_dir / image_name) as image:
                        image.load()
                        self.assertEqual(image.format, "JPEG")
                        self.assertEqual(image.mode, "RGB")
                        self.assertEqual(len(image.getbands()), 3)


if __name__ == "__main__":
    unittest.main()
