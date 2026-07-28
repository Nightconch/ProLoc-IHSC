import io
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
    if image_dir.exists() and not image_dir.is_dir():
        raise FileExistsError(
            f"output image directory already exists as a file: {image_dir}"
        )
    existing = (
        sorted(path.name for path in image_dir.iterdir())
        if image_dir.exists()
        else []
    )
    if existing:
        raise FileExistsError(
            "output image already exists; directory contains existing files: "
            f"{', '.join(existing)}"
        )
    image_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        DownloadTask(ordinal, tier, split, row)
        for ordinal, row in enumerate(rows)
    ]
    ordered_results = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(download_task, task, image_dir, http_get)
            for task in tasks
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
