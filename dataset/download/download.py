import argparse
import io
import json
import os
import shutil
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from urllib.parse import urlsplit

import pandas as pd
from PIL import Image


FINAL_COLUMNS = [
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
DOWNLOAD_FAILURE_COLUMNS = [
    "tier",
    "split",
    "ordinal",
    "source_row",
    "Protein Id",
    "URL",
    "stage",
    "reason",
]
ZERO_SUCCESS_COLUMNS = ["tier", "split", "Protein Id", "input_rows"]
REQUIRED_MANIFEST_COLUMNS = [
    "Modified URL",
    "Protein Id",
    "Antibody Id",
    "locations",
    "cytoplasm",
    "endoplasmic reticulum",
    "mitochondria",
    "nucleus",
    "plasma membrane",
    "Sequence",
]
DATASETS = (
    ("HQ", "train"),
    ("HQ", "test"),
    ("MQ", "train"),
    ("MQ", "test"),
    ("LQ", "train"),
    ("LQ", "test"),
)


class ImageValidationError(ValueError):
    pass


@dataclass(frozen=True)
class DownloadTask:
    ordinal: int
    tier: str
    split: str
    row: dict


@dataclass(frozen=True)
class DownloadResult:
    ordinal: int
    row: dict
    file_name: str
    success: bool
    converted: bool = False
    stage: str = ""
    reason: str = ""


@dataclass(frozen=True)
class ManifestDownloadOutcome:
    status: str
    successes: pd.DataFrame
    failures: pd.DataFrame
    stats: dict
    image_dir: Path
    success_csv: Path
    failure_csv: Path


def _validated_filename_component(name: str, value: str) -> str:
    invalid_characters = '<>:"/\\|?*\0'
    if not value.strip():
        raise ValueError(f"{name} is blank")
    if any(
        character in invalid_characters or ord(character) < 32
        for character in value
    ):
        raise ValueError(f"{name} contains invalid filename characters")
    return value


def _required_text(name: str, value, *, strip: bool) -> str:
    if value is None or pd.isna(value):
        raise ValueError(f"{name} is blank")
    text = str(value)
    if strip:
        text = text.strip()
    if not text.strip():
        raise ValueError(f"{name} is blank")
    return text


def build_file_name(row: dict) -> str:
    protein_id = _validated_filename_component(
        "Protein Id",
        _required_text("Protein Id", row["Protein Id"], strip=True),
    )
    antibody_id = _validated_filename_component(
        "Antibody Id",
        _required_text("Antibody Id", row["Antibody Id"], strip=True),
    )
    image_url = _required_text(
        "Modified URL", row["Modified URL"], strip=True
    )
    image_stem = _validated_filename_component(
        "Modified URL image stem",
        PurePosixPath(urlsplit(image_url).path).stem,
    )
    locations = _validated_filename_component(
        "locations",
        _required_text("locations", row["locations"], strip=False),
    )
    return f"{protein_id}-{image_stem}-{antibody_id}-{locations}.jpg"


def _default_http_get(url: str, **kwargs):
    import requests

    return requests.get(url, **kwargs)


def download_task(
    task: DownloadTask,
    image_dir: Path,
    http_get=_default_http_get,
) -> DownloadResult:
    file_name = build_file_name(task.row)
    try:
        response = http_get(task.row["Modified URL"], timeout=60)
        response.raise_for_status()
        payload = response.content
    except Exception as error:
        return DownloadResult(
            ordinal=task.ordinal,
            row=task.row,
            file_name=file_name,
            success=False,
            stage="http",
            reason=str(error) or type(error).__name__,
        )
    try:
        converted = write_validated_image(
            payload, Path(image_dir) / file_name
        )
    except ImageValidationError as error:
        reason = str(error)
        return DownloadResult(
            ordinal=task.ordinal,
            row=task.row,
            file_name=file_name,
            success=False,
            stage="blank" if "blank" in reason.lower() else "decode",
            reason=reason,
        )
    except OSError as error:
        return DownloadResult(
            ordinal=task.ordinal,
            row=task.row,
            file_name=file_name,
            success=False,
            stage="write",
            reason=str(error) or type(error).__name__,
        )
    return DownloadResult(
        ordinal=task.ordinal,
        row=task.row,
        file_name=file_name,
        success=True,
        converted=converted,
    )


def _reusable_image_names(frame: pd.DataFrame, image_dir: Path) -> set[str]:
    image_dir = Path(image_dir)
    if image_dir.exists() and not image_dir.is_dir():
        raise FileExistsError(
            f"output image directory already exists as a file: {image_dir}"
        )
    if not image_dir.exists():
        return set()

    expected_names = {
        build_file_name(row) for row in frame.to_dict(orient="records")
    }
    allowed_part_names = {f"{name}.part" for name in expected_names}
    unexpected = sorted(
        entry.name
        for entry in image_dir.iterdir()
        if not entry.is_file()
        or entry.name not in expected_names | allowed_part_names
    )
    if unexpected:
        raise FileExistsError(
            "output image already exists; directory contains existing files "
            "not in manifest: "
            f"{', '.join(unexpected)}"
        )

    reusable = set()
    for name in sorted(expected_names):
        path = image_dir / name
        if not path.exists():
            continue
        try:
            with path.open("rb") as cached_file:
                _validate_nonblank_jpeg_rgb(cached_file, "cached")
        except (ImageValidationError, OSError) as error:
            raise FileExistsError(
                "output image already exists but is not a reusable RGB JPEG: "
                f"{path}: {error}"
            ) from error
        reusable.add(name)
    return reusable


def process_manifest(
    frame: pd.DataFrame,
    tier: str,
    split: str,
    image_dir: Path,
    workers: int,
    http_get,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    missing = [
        column for column in REQUIRED_MANIFEST_COLUMNS if column not in frame
    ]
    if missing:
        raise ValueError(
            f"manifest missing required columns: {', '.join(missing)}"
        )
    for column in REQUIRED_MANIFEST_COLUMNS:
        for ordinal, value in enumerate(frame[column].tolist()):
            if pd.isna(value) or (
                isinstance(value, str) and not value.strip()
            ):
                raise ValueError(
                    f"manifest row {ordinal} has blank required field {column}"
                )
    rows = frame.to_dict(orient="records")
    file_names = [build_file_name(row) for row in rows]
    collision_keys = {
        name
        for name, count in Counter(
            file_name.casefold() for file_name in file_names
        ).items()
        if count > 1
    }
    collisions = sorted(
        file_name
        for file_name in file_names
        if file_name.casefold() in collision_keys
    )
    if collisions:
        raise ValueError(
            f"filename collision in manifest: {', '.join(collisions)}"
        )
    image_dir = Path(image_dir)
    reusable_names = _reusable_image_names(frame, image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        DownloadTask(ordinal, tier, split, row)
        for ordinal, row in enumerate(rows)
    ]
    ordered_results = [None] * len(tasks)
    pending_tasks = []
    for task, file_name in zip(tasks, file_names):
        if file_name in reusable_names:
            ordered_results[task.ordinal] = DownloadResult(
                ordinal=task.ordinal,
                row=task.row,
                file_name=file_name,
                success=True,
            )
        else:
            pending_tasks.append(task)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(download_task, task, image_dir, http_get)
            for task in pending_tasks
        ]
        for future in as_completed(futures):
            result = future.result()
            ordered_results[result.ordinal] = result

    success_records = []
    failure_records = []
    converted_rows = 0
    for result in ordered_results:
        if result is None:
            raise RuntimeError("download worker returned no result")
        row = result.row
        if result.success:
            if not (image_dir / result.file_name).is_file():
                raise RuntimeError(
                    f"successful download is missing image: {result.file_name}"
                )
            success_records.append(
                {
                    "File Name": result.file_name,
                    "locations": row["locations"],
                    "cytoplasm": row["cytoplasm"],
                    "endoplasmic reticulum": row[
                        "endoplasmic reticulum"
                    ],
                    "mitochondria": row["mitochondria"],
                    "nucleus": row["nucleus"],
                    "plasma membrane": row["plasma membrane"],
                    "Sequence": row["Sequence"],
                    "Protein Id": str(row["Protein Id"]).strip(),
                }
            )
            converted_rows += int(result.converted)
            continue

        task = tasks[result.ordinal]
        source_row = row.get("Unnamed: 0", result.ordinal)
        if pd.isna(source_row) or (
            isinstance(source_row, str) and not source_row.strip()
        ):
            source_row = result.ordinal
        failure_records.append(
            {
                "tier": task.tier,
                "split": task.split,
                "ordinal": result.ordinal,
                "source_row": source_row,
                "Protein Id": str(row["Protein Id"]).strip(),
                "URL": row["Modified URL"],
                "stage": result.stage,
                "reason": result.reason,
            }
        )

    successes = pd.DataFrame(success_records, columns=FINAL_COLUMNS)
    failures = pd.DataFrame(
        failure_records, columns=DOWNLOAD_FAILURE_COLUMNS
    )
    return successes, failures, {
        "input_rows": len(frame),
        "success_rows": len(successes),
        "failure_rows": len(failures),
        "converted_rows": converted_rows,
    }


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part_path = path.with_name(f"{path.name}.part")
    part_path.unlink(missing_ok=True)
    try:
        frame.to_csv(part_path, index=False)
        part_path.replace(path)
    finally:
        part_path.unlink(missing_ok=True)


def _write_json_atomic(payload: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part_path = path.with_name(f"{path.name}.part")
    part_path.unlink(missing_ok=True)
    try:
        part_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        part_path.replace(path)
    finally:
        part_path.unlink(missing_ok=True)


def download_manifest(
    frame: pd.DataFrame,
    tier: str,
    split: str,
    output_dir: Path,
    workers: int = 8,
    http_get=_default_http_get,
) -> ManifestDownloadOutcome:
    output_dir = Path(output_dir)
    stem = f"{tier}_{split}"
    image_dir = output_dir / f"{stem}_img"
    success_csv = output_dir / f"{stem}.csv"
    failure_csv = output_dir / f"{stem}_failures.csv"
    successes, failures, stats = process_manifest(
        frame,
        tier,
        split,
        image_dir,
        workers,
        http_get,
    )
    _write_csv_atomic(successes, success_csv)
    _write_csv_atomic(failures, failure_csv)
    status = "completed" if failures.empty else "completed_with_failures"
    return ManifestDownloadOutcome(
        status=status,
        successes=successes,
        failures=failures,
        stats=stats,
        image_dir=image_dir,
        success_csv=success_csv,
        failure_csv=failure_csv,
    )


def load_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [
        column for column in REQUIRED_MANIFEST_COLUMNS if column not in frame
    ]
    if missing:
        raise ValueError(
            f"manifest missing required columns: {', '.join(missing)}"
        )
    return frame


def _manifest_failure(
    tier: str,
    split: str,
    ordinal: int,
    row: dict,
    stage: str,
    reason: str,
) -> dict:
    source_row = row.get("Unnamed: 0", ordinal)
    if pd.isna(source_row) or (
        isinstance(source_row, str) and not source_row.strip()
    ):
        source_row = ordinal
    protein_id = row.get("Protein Id", "")
    if pd.isna(protein_id):
        protein_id = ""
    url = row.get("Modified URL", "")
    if pd.isna(url):
        url = ""
    return {
        "tier": tier,
        "split": split,
        "ordinal": ordinal,
        "source_row": source_row,
        "Protein Id": str(protein_id).strip(),
        "URL": str(url).strip(),
        "stage": stage,
        "reason": reason,
    }


def _row_preflight_failure(tier, split, ordinal, row):
    for column in REQUIRED_MANIFEST_COLUMNS:
        value = row.get(column)
        if not (pd.isna(value) or (isinstance(value, str) and not value.strip())):
            continue
        if column == "Sequence":
            stage = "sequence"
        elif column == "Modified URL":
            stage = "url"
        else:
            stage = "required_field"
        return _manifest_failure(
            tier,
            split,
            ordinal,
            row,
            stage,
            f"blank required field {column}",
        )
    try:
        build_file_name(row)
    except (KeyError, TypeError, ValueError) as error:
        return _manifest_failure(
            tier,
            split,
            ordinal,
            row,
            "filename",
            str(error),
        )
    return None


def preflight_manifests(manifests: dict) -> tuple[dict, pd.DataFrame]:
    expected = set(DATASETS)
    if set(manifests) != expected:
        raise ValueError(f"unexpected manifest keys: {sorted(manifests)}")

    protein_ids = {"train": set(), "test": set()}
    named_rows = []
    valid_frames = {}
    failure_records = []
    for tier, split in DATASETS:
        frame = manifests[(tier, split)]
        valid_positions = []
        for ordinal, row in enumerate(frame.to_dict(orient="records")):
            protein_value = row.get("Protein Id")
            if not (
                pd.isna(protein_value)
                or (isinstance(protein_value, str) and not protein_value.strip())
            ):
                protein_ids[split].add(str(protein_value).strip())
            failure = _row_preflight_failure(
                tier, split, ordinal, row
            )
            if failure is not None:
                failure_records.append(failure)
                continue
            valid_positions.append(ordinal)
            named_rows.append(
                (tier, split, ordinal, build_file_name(row))
            )
        valid_frames[(tier, split)] = frame.iloc[valid_positions].reset_index(
            drop=True
        )

    overlap = sorted(protein_ids["train"] & protein_ids["test"])
    if overlap:
        raise AssertionError(
            "Protein Id overlap between train and test: "
            f"{', '.join(overlap)}"
        )

    collision_keys = {
        name.casefold()
        for name, count in Counter(
            name.casefold() for _tier, _split, _ordinal, name in named_rows
        ).items()
        if count > 1
    }
    collisions = [
        f"{tier}_{split}[{ordinal}]={name}"
        for tier, split, ordinal, name in named_rows
        if name.casefold() in collision_keys
    ]
    if collisions:
        raise ValueError(
            f"global filename collision: {', '.join(collisions)}"
        )
    return valid_frames, pd.DataFrame(
        failure_records, columns=DOWNLOAD_FAILURE_COLUMNS
    )


def preflight_image_directories(output_dir: Path, manifests: dict) -> None:
    output_dir = Path(output_dir)
    for tier, split in DATASETS:
        image_dir = output_dir / f"{tier}_{split}_img"
        _reusable_image_names(manifests[(tier, split)], image_dir)


def collect_zero_success_proteins(
    manifests: dict, success_frames: dict
) -> pd.DataFrame:
    records = []
    for tier, split in DATASETS[2:]:
        input_ids = [
            _required_text("Protein Id", value, strip=True)
            for value in manifests[(tier, split)]["Protein Id"].tolist()
        ]
        successful_ids = {
            _required_text("Protein Id", value, strip=True)
            for value in success_frames[(tier, split)]["Protein Id"].tolist()
        }
        counts = Counter(input_ids)
        seen = set()
        for protein_id in input_ids:
            if protein_id in seen:
                continue
            seen.add(protein_id)
            if protein_id not in successful_ids:
                records.append(
                    {
                        "tier": tier,
                        "split": split,
                        "Protein Id": protein_id,
                        "input_rows": counts[protein_id],
                    }
                )
    return pd.DataFrame(records, columns=ZERO_SUCCESS_COLUMNS)


def _frame_records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records"))


def order_failures(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=DOWNLOAD_FAILURE_COLUMNS)
    dataset_order = {
        dataset: index for index, dataset in enumerate(DATASETS)
    }
    ordered = frame.copy()
    ordered["__dataset_order"] = [
        dataset_order[(tier, split)]
        for tier, split in zip(ordered["tier"], ordered["split"])
    ]
    ordered = ordered.sort_values(
        ["__dataset_order", "ordinal"], kind="stable"
    )
    return ordered[DOWNLOAD_FAILURE_COLUMNS].reset_index(drop=True)


def load_upstream_summary(manifest_dir: Path) -> dict:
    manifest_dir = Path(manifest_dir)
    reports = {}
    for filename in (
        "manifest_generation_report.json",
        "source_validation_report.json",
    ):
        path = manifest_dir / filename
        reports[filename.removesuffix(".json")] = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.is_file()
            else None
        )
    return reports


def assert_success_protein_disjoint(frames: dict) -> dict:
    if set(frames) != set(DATASETS):
        raise ValueError(f"unexpected success frame keys: {sorted(frames)}")
    split_ids = {"train": set(), "test": set()}
    for tier, split in DATASETS:
        split_ids[split].update(
            str(value).strip()
            for value in frames[(tier, split)]["Protein Id"].tolist()
            if not pd.isna(value) and str(value).strip()
        )
    overlap = sorted(split_ids["train"] & split_ids["test"])
    if overlap:
        raise AssertionError(
            "Protein Id overlap between successful train and test: "
            f"{', '.join(overlap)}"
        )
    return {
        "checked": True,
        "train_proteins": len(split_ids["train"]),
        "test_proteins": len(split_ids["test"]),
        "overlap": [],
    }


def publish_final_manifests(
    frames: dict, output_dir: Path, replace=os.replace
) -> None:
    if set(frames) != set(DATASETS):
        raise ValueError(f"unexpected final manifest keys: {sorted(frames)}")
    invalid_schemas = [
        f"{tier}_{split}"
        for tier, split in DATASETS
        if frames[(tier, split)].columns.tolist() != FINAL_COLUMNS
    ]
    if invalid_schemas:
        raise AssertionError(
            f"final manifest schema mismatch: {', '.join(invalid_schemas)}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    transaction_id = uuid.uuid4().hex
    staging_dir = output_dir / f".final-csv-staging-{transaction_id}"
    staging_dir.mkdir()
    artifacts = []
    backups = {}
    published = []
    rollback_errors = []
    retained_backups = set()
    publish_succeeded = False
    try:
        for tier, split in DATASETS:
            filename = f"{tier}_{split}.csv"
            staged = staging_dir / filename
            frames[(tier, split)].to_csv(staged, index=False)
            artifacts.append((staged, output_dir / filename))

        try:
            for _staged, destination in artifacts:
                if destination.exists():
                    backup = output_dir / (
                        f".{destination.name}.backup-{transaction_id}"
                    )
                    replace(destination, backup)
                    backups[destination] = backup
            for staged, destination in artifacts:
                replace(staged, destination)
                published.append(destination)
            publish_succeeded = True
        except Exception as publish_error:
            for destination in reversed(published):
                try:
                    destination.unlink(missing_ok=True)
                except Exception as rollback_error:
                    rollback_errors.append(rollback_error)
            for destination, backup in reversed(list(backups.items())):
                if not backup.exists():
                    continue
                try:
                    replace(backup, destination)
                except Exception as rollback_error:
                    retained_backups.add(backup)
                    rollback_errors.append(rollback_error)
            if rollback_errors:
                details = "; ".join(map(str, rollback_errors))
                raise RuntimeError(
                    f"final manifest publish failed ({publish_error}); "
                    f"rollback also failed: {details}"
                ) from publish_error
            raise
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
        for backup in backups.values():
            if publish_succeeded or backup not in retained_backups:
                backup.unlink(missing_ok=True)


def run_download_pipeline(
    manifest_dir: Path,
    output_dir: Path,
    workers: int = 8,
    http_get=_default_http_get,
) -> dict:
    manifest_dir = Path(manifest_dir)
    output_dir = Path(output_dir)
    upstream = {
        "manifest_generation_report": None,
        "source_validation_report": None,
    }
    manifests = {}
    try:
        upstream = load_upstream_summary(manifest_dir)
        manifests = {
            (tier, split): load_manifest(
                manifest_dir / f"{tier}_{split}_img_URL.csv"
            )
            for tier, split in DATASETS
        }
        prepared_manifests, preflight_failures = preflight_manifests(
            manifests
        )
        preflight_image_directories(output_dir, prepared_manifests)
    except Exception as error:
        _write_csv_atomic(
            pd.DataFrame(columns=DOWNLOAD_FAILURE_COLUMNS),
            output_dir / "download_failures.csv",
        )
        _write_csv_atomic(
            pd.DataFrame(columns=ZERO_SUCCESS_COLUMNS),
            output_dir / "zero_success_proteins.csv",
        )
        report = {
            "status": "error",
            "published": False,
            "upstream": upstream,
            "datasets": {},
            "total_failures": 0,
            "failures": [],
            "zero_success_proteins": 0,
            "zero_success_details": [],
            "protein_id_overlap": 0,
            "protein_id_leakage": {
                "checked": False,
                "train_proteins": 0,
                "test_proteins": 0,
                "overlap": [],
            },
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
        _write_json_atomic(report, output_dir / "download_audit_report.json")
        raise
    dataset_stats = {
        f"{tier}_{split}": {
            "input_rows": len(manifests[(tier, split)]),
            "success_rows": 0,
            "failure_rows": int(
                (
                    preflight_failures["tier"].eq(tier)
                    & preflight_failures["split"].eq(split)
                ).sum()
            ),
            "converted_rows": 0,
        }
        for tier, split in DATASETS
    }
    success_frames = {}
    failure_frames = [preflight_failures]

    hq_preflight_failures = preflight_failures.loc[
        preflight_failures["tier"].eq("HQ")
    ]
    if not hq_preflight_failures.empty:
        preflight_failures = order_failures(preflight_failures)
        _write_csv_atomic(
            preflight_failures, output_dir / "download_failures.csv"
        )
        _write_csv_atomic(
            pd.DataFrame(columns=ZERO_SUCCESS_COLUMNS),
            output_dir / "zero_success_proteins.csv",
        )
        report = {
            "status": "error",
            "published": False,
            "upstream": upstream,
            "datasets": dataset_stats,
            "total_failures": len(preflight_failures),
            "failures": _frame_records(preflight_failures),
            "zero_success_proteins": 0,
            "zero_success_details": [],
            "protein_id_overlap": 0,
            "protein_id_leakage": {
                "checked": False,
                "train_proteins": 0,
                "test_proteins": 0,
                "overlap": [],
            },
        }
        _write_json_atomic(report, output_dir / "download_audit_report.json")
        raise RuntimeError(
            f"HQ download failed for {len(hq_preflight_failures)} row(s)"
        )

    def process_dataset(tier, split):
        frame = prepared_manifests[(tier, split)]
        successes, failures, stats = process_manifest(
            frame,
            tier,
            split,
            output_dir / f"{tier}_{split}_img",
            workers,
            http_get,
        )
        success_frames[(tier, split)] = successes
        failure_frames.append(failures)
        current_stats = dataset_stats[f"{tier}_{split}"]
        current_stats["success_rows"] = stats["success_rows"]
        current_stats["failure_rows"] += stats["failure_rows"]
        current_stats["converted_rows"] = stats["converted_rows"]
        return failures

    hq_failures = []
    for tier, split in DATASETS[:2]:
        failures = process_dataset(tier, split)
        if not failures.empty:
            hq_failures.append(failures)

    if hq_failures:
        failures = order_failures(
            pd.concat(
                [preflight_failures, *hq_failures], ignore_index=True
            )
        )
        _write_csv_atomic(failures, output_dir / "download_failures.csv")
        _write_csv_atomic(
            pd.DataFrame(columns=ZERO_SUCCESS_COLUMNS),
            output_dir / "zero_success_proteins.csv",
        )
        report = {
            "status": "error",
            "published": False,
            "upstream": upstream,
            "datasets": dataset_stats,
            "total_failures": len(failures),
            "failures": _frame_records(failures),
            "zero_success_proteins": 0,
            "zero_success_details": [],
            "protein_id_overlap": 0,
            "protein_id_leakage": {
                "checked": False,
                "train_proteins": 0,
                "test_proteins": 0,
                "overlap": [],
            },
        }
        _write_json_atomic(report, output_dir / "download_audit_report.json")
        fatal_rows = sum(len(frame) for frame in hq_failures)
        raise RuntimeError(f"HQ download failed for {fatal_rows} row(s)")

    for tier, split in DATASETS[2:]:
        process_dataset(tier, split)

    failures = order_failures(
        pd.concat(failure_frames, ignore_index=True)
    )
    zero_success = collect_zero_success_proteins(
        manifests, success_frames
    )
    leakage = assert_success_protein_disjoint(success_frames)
    publish_final_manifests(success_frames, output_dir)
    _write_csv_atomic(failures, output_dir / "download_failures.csv")
    _write_csv_atomic(
        zero_success, output_dir / "zero_success_proteins.csv"
    )
    report = {
        "status": "ok",
        "published": True,
        "upstream": upstream,
        "datasets": dataset_stats,
        "total_failures": len(failures),
        "failures": _frame_records(failures),
        "zero_success_proteins": len(zero_success),
        "zero_success_details": _frame_records(zero_success),
        "protein_id_overlap": 0,
        "protein_id_leakage": leakage,
    }
    _write_json_atomic(report, output_dir / "download_audit_report.json")
    return report


def main(argv=None, *, http_get=_default_http_get) -> int:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Download HQ/MQ/LQ train/test RGB datasets"
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=script_dir / "generated_quality",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "generated_quality",
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    summary = run_download_pipeline(
        args.manifest_dir,
        args.output_dir,
        args.workers,
        http_get,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def is_blank_rgb(image: Image.Image, threshold: int = 250) -> bool:
    if image.mode != "RGB" or len(image.getbands()) != 3:
        raise ValueError("blank detection requires an RGB three-band image")
    return all(low >= threshold for low, _high in image.getextrema())


def _is_rgb_jpeg(image: Image.Image) -> bool:
    return (
        image.format == "JPEG"
        and image.mode == "RGB"
        and len(image.getbands()) == 3
    )


def _validate_nonblank_jpeg_rgb(source, stage: str) -> None:
    try:
        with Image.open(source) as checked:
            checked.load()
            if not _is_rgb_jpeg(checked):
                raise ImageValidationError(
                    f"{stage} image is not JPEG RGB with three bands"
                )
            if is_blank_rgb(checked):
                raise ImageValidationError(f"{stage} image is blank")
    except ImageValidationError:
        raise
    except Exception as error:
        raise ImageValidationError(
            f"{stage} image decode failed: {error}"
        ) from error


def normalized_jpeg_bytes(payload: bytes) -> tuple[bytes, bool]:
    try:
        with Image.open(io.BytesIO(payload)) as source:
            source.load()
            preserves_original = _is_rgb_jpeg(source)
            if preserves_original:
                rgb = source
            elif (
                "A" in source.getbands()
                or "transparency" in source.info
            ):
                rgba = source.convert("RGBA")
                background = Image.new(
                    "RGBA", rgba.size, (255, 255, 255, 255)
                )
                background.alpha_composite(rgba)
                rgb = background.convert("RGB")
            else:
                rgb = source.convert("RGB")
            if is_blank_rgb(rgb):
                raise ImageValidationError("blank image")
            if preserves_original:
                candidate = payload
                converted = False
            else:
                output = io.BytesIO()
                rgb.save(output, format="JPEG", quality=95)
                candidate = output.getvalue()
                converted = True
    except ImageValidationError:
        raise
    except Exception as error:
        raise ImageValidationError(f"image decode failed: {error}") from error

    _validate_nonblank_jpeg_rgb(io.BytesIO(candidate), "final")

    return candidate, converted


def write_validated_image(payload: bytes, path: Path) -> bool:
    path = Path(path)
    candidate, converted = normalized_jpeg_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    part_path = path.with_name(f"{path.name}.part")
    part_path.unlink(missing_ok=True)
    try:
        part_path.write_bytes(candidate)
        with part_path.open("rb") as staged_file:
            _validate_nonblank_jpeg_rgb(staged_file, "written")
        part_path.replace(path)
        return converted
    finally:
        part_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
