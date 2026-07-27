import argparse
import hashlib
import json
import sys
from decimal import Decimal, InvalidOperation
from numbers import Number
from pathlib import Path

import pandas as pd
import requests


ZENODO_RECORD_ID = "10632698"
SOURCE_FILENAMES = (
    "normalLabeled.csv",
    "data_train.csv",
    "data_test.csv",
)
ZENODO = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files"
SOURCE_URLS = {
    name: f"{ZENODO}/{name}/content"
    for name in SOURCE_FILENAMES
}
SOURCE_MD5 = {
    "normalLabeled.csv": "37dff5cc73458fe529eb860c9a2ab900",
    "data_train.csv": "0236eb02e2f906282ccea4cf47a84591",
    "data_test.csv": "3e9f1ddaf5e14d7a61354f0884b1f002",
}
SOURCE_ROW_ID = "Unnamed: 0"
DERIVED_OFFICIAL_COLUMNS = {
    "Sequence",
    "NonZeroDigits",
    "Image Name",
    "Modified URL",
    "Quality",
}


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified_source(url: str, path: Path, expected_md5: str) -> bool:
    path = Path(path)
    if path.exists() and file_md5(path) == expected_md5:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    part_path = path.with_name(f"{path.name}.part")
    part_path.unlink(missing_ok=True)
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        with part_path.open("wb") as output:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    output.write(chunk)
        actual_md5 = file_md5(part_path)
        if actual_md5 != expected_md5:
            raise ValueError(
                f"MD5 mismatch for {path.name}: expected {expected_md5}, "
                f"got {actual_md5}"
            )
        part_path.replace(path)
        return True
    finally:
        part_path.unlink(missing_ok=True)


def _missing_columns(frame: pd.DataFrame, required, context: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{context} missing required columns: {', '.join(missing)}")


def _normalized_value(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, Number) and not isinstance(value, bool):
        try:
            return format(Decimal(str(value)).normalize(), "f")
        except InvalidOperation:
            pass
    return str(value)


def validate_official_rows(
    source: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> set[str]:
    frames = (
        ("normalLabeled.csv", source),
        ("data_train.csv", train),
        ("data_test.csv", test),
    )
    for name, frame in frames:
        _missing_columns(frame, [SOURCE_ROW_ID], name)

    normalized_ids = {}
    for name, frame in frames:
        ids = frame[SOURCE_ROW_ID].map(
            lambda value: _normalized_value(value).strip()
        )
        if ids.eq("").any():
            raise ValueError(f"{name} contains blank {SOURCE_ROW_ID}")
        if ids.duplicated().any():
            duplicates = sorted(set(ids[ids.duplicated(keep=False)]))
            raise ValueError(f"{name} contains duplicate source rows: {duplicates}")
        normalized_ids[name] = ids

    train_ids = set(normalized_ids["data_train.csv"])
    test_ids = set(normalized_ids["data_test.csv"])
    overlap = sorted(train_ids & test_ids)
    if overlap:
        raise ValueError(f"official train and test source rows overlap: {overlap}")

    source_copy = source.copy()
    source_copy["__source_row_id"] = normalized_ids["normalLabeled.csv"]
    source_by_id = source_copy.set_index("__source_row_id", drop=False)
    for split, official in (("train", train), ("test", test)):
        ids = normalized_ids[f"data_{split}.csv"]
        for position, (_, official_row) in enumerate(official.iterrows()):
            row_id = ids.iloc[position]
            if row_id not in source_by_id.index:
                raise ValueError(
                    f"official row {row_id} is absent from normalLabeled.csv"
                )
            source_row = source_by_id.loc[row_id]
            for column in official.columns:
                if (
                    column not in source.columns
                    or column in DERIVED_OFFICIAL_COLUMNS
                ):
                    continue
                if _normalized_value(official_row[column]) != _normalized_value(
                    source_row[column]
                ):
                    raise ValueError(f"official row {row_id} differs in {column}")

    return train_ids | test_ids


def prepare_verified_sources(cache_dir, source_urls=None, source_md5=None):
    source_urls = SOURCE_URLS if source_urls is None else source_urls
    source_md5 = SOURCE_MD5 if source_md5 is None else source_md5
    expected_names = set(SOURCE_FILENAMES)
    if set(source_urls) != expected_names or set(source_md5) != expected_names:
        raise ValueError(
            "source catalog must contain only normalLabeled.csv, "
            "data_train.csv, and data_test.csv"
        )

    cache_dir = Path(cache_dir)
    sources = {}
    for name in SOURCE_FILENAMES:
        replaced = download_verified_source(
            source_urls[name], cache_dir / name, source_md5[name]
        )
        sources[name] = {
            "action": "replaced" if replaced else "reused",
            "expected_md5": source_md5[name],
            "url": source_urls[name],
        }

    source = pd.read_csv(cache_dir / "normalLabeled.csv")
    official_train = pd.read_csv(cache_dir / "data_train.csv")
    official_test = pd.read_csv(cache_dir / "data_test.csv")
    official_source_rows = validate_official_rows(
        source, official_train, official_test
    )
    return {
        "status": "ok",
        "record_id": ZENODO_RECORD_ID,
        "sources": sources,
        "official_source_rows": len(official_source_rows),
    }


def _write_source_report(report, output_dir) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "source_validation_report.json"
    part_path = report_path.with_name(f"{report_path.name}.part")
    part_path.unlink(missing_ok=True)
    try:
        part_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        part_path.replace(report_path)
    finally:
        part_path.unlink(missing_ok=True)


def main(argv=None, *, source_urls=None, source_md5=None) -> int:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Prepare pinned Vislocas source tables"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=script_dir / "generated_quality"
    )
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args(argv)

    cache_dir = args.cache_dir or args.output_dir / "source"
    try:
        report = prepare_verified_sources(cache_dir, source_urls, source_md5)
    except Exception as error:
        report = {
            "status": "error",
            "record_id": ZENODO_RECORD_ID,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
        _write_source_report(report, args.output_dir)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 1

    _write_source_report(report, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
