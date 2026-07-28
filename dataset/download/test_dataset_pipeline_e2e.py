import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pandas as pd
from PIL import Image

from dataset.download import download as downloader
from dataset.download import supplemental_quality_manifests as manifests
from dataset.download.official_hq_manifests import (
    DOWNLOAD_LABEL_COLUMNS,
    SOURCE_COLUMNS,
    SOURCE_ROW_ID,
)


DATASETS = (
    ("HQ", "train"),
    ("HQ", "test"),
    ("MQ", "train"),
    ("MQ", "test"),
    ("LQ", "train"),
    ("LQ", "test"),
)
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
    row = {column: "" for column in SOURCE_COLUMNS}
    row.update({column: 0 for column in DOWNLOAD_LABEL_COLUMNS})
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
        columns=SOURCE_COLUMNS,
    )
    return {
        "normalLabeled.csv": source,
        "data_train.csv": pd.DataFrame(
            [official_train, official_a], columns=SOURCE_COLUMNS
        ),
        "data_test.csv": pd.DataFrame([official_test], columns=SOURCE_COLUMNS),
    }


def file_md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def write_fixture_sources(cache_dir):
    cache_dir.mkdir(parents=True)
    source_md5 = {}
    for name, frame in fixture_frames().items():
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
                expected_proteins = manifest_frames[dataset].loc[
                    ~manifest_frames[dataset]["Image Name"].isin(
                        {"white.png", "corrupt.jpg", "unavailable.jpg"}
                    ),
                    "Protein Id",
                ].map(str.strip).tolist()
                self.assertEqual(final_frame["Protein Id"].tolist(), expected_proteins)

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
