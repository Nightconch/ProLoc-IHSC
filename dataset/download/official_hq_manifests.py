import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
import uuid
from decimal import Decimal, InvalidOperation
from numbers import Number
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd
import requests

if __package__:
    from .filter_quality_urls import (
        SOURCE_MD5,
        SOURCE_ROW_ID,
        SOURCE_URLS,
        prepare_verified_sources,
        validate_official_rows,
    )
else:
    from filter_quality_urls import (
        SOURCE_MD5,
        SOURCE_ROW_ID,
        SOURCE_URLS,
        prepare_verified_sources,
        validate_official_rows,
    )


ZENODO_RECORD_ID = "10632698"
UNIPROT_RUN = "https://rest.uniprot.org/idmapping/run"
UNIPROT_STATUS = "https://rest.uniprot.org/idmapping/status/{job_id}"
UNIPROT_DETAILS = "https://rest.uniprot.org/idmapping/details/{job_id}"
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
OUTPUT_COLUMNS = [
    *SOURCE_COLUMNS,
    "Sequence",
    "NonZeroDigits",
    "Image Name",
    "Modified URL",
]
MANIFEST_FAILURE_COLUMNS = [
    "stage",
    "tier",
    "split",
    "source_line",
    "source_row",
    "Protein Id",
    "URL",
    "reason",
]
DOWNLOAD_LABEL_COLUMNS = [
    "cytoplasm",
    "endoplasmic reticulum",
    "mitochondria",
    "nucleus",
    "plasma membrane",
]
FORMAL_ARTIFACTS = (
    ("HQ_train", "HQ_train_img_URL.csv"),
    ("HQ_test", "HQ_test_img_URL.csv"),
    ("occupied", "official_hq_occupied_rows.csv"),
)


def _clean_text(value):
    return "" if pd.isna(value) else str(value).strip()


def _source_row_identity(value):
    if pd.isna(value):
        return ""
    if isinstance(value, Number) and not isinstance(value, bool):
        try:
            return format(Decimal(str(value)).normalize(), "f")
        except InvalidOperation:
            pass
    return str(value).strip()


def _require_columns(frame, columns, source_name):
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{source_name} missing required columns: {', '.join(missing)}"
        )


def build_image_fields(antibody_id, relative_url):
    antibody = _clean_text(antibody_id)
    match = re.search(r"\d+", antibody)
    if match is None:
        raise ValueError("Antibody Id must contain a number")
    relative = _clean_text(relative_url).replace("\\", "/")
    image_name = relative.rsplit("/", 1)[-1]
    if not relative or not image_name:
        raise ValueError("URL must contain an image name")
    number = str(int(match.group()))
    return (
        number,
        image_name,
        f"http://images.proteinatlas.org/{number}/{image_name}",
    )


def _normalized_protein_ids(frame):
    _require_columns(frame, ["Protein Id"], "official rows")
    return frame["Protein Id"].map(_clean_text)


def assemble_official_hq(train, test, sequences):
    train_ids = _normalized_protein_ids(train)
    test_ids = _normalized_protein_ids(test)
    overlap = sorted((set(train_ids) - {""}) & (set(test_ids) - {""}))
    if overlap:
        raise ValueError(
            f"Protein Id overlap between official train and test: {overlap}"
        )

    outputs = {}
    for split, frame, protein_ids in (
        ("train", train, train_ids),
        ("test", test, test_ids),
    ):
        _require_columns(frame, SOURCE_COLUMNS, f"data_{split}.csv")
        result = frame.copy()
        result["Protein Id"] = protein_ids
        missing_sequences = sorted(
            {
                protein_id or "<blank>"
                for protein_id in protein_ids
                if not protein_id or not _clean_text(sequences.get(protein_id))
            }
        )
        if missing_sequences:
            raise ValueError(
                "missing reviewed sequences for: " + ", ".join(missing_sequences)
            )
        result["Sequence"] = protein_ids.map(sequences)
        image_fields = [
            build_image_fields(antibody_id, relative_url)
            for antibody_id, relative_url in result[["Antibody Id", "URL"]].itertuples(
                index=False, name=None
            )
        ]
        result[["NonZeroDigits", "Image Name", "Modified URL"]] = pd.DataFrame(
            image_fields,
            columns=["NonZeroDigits", "Image Name", "Modified URL"],
            index=result.index,
        )
        outputs[f"HQ_{split}"] = result.loc[:, OUTPUT_COLUMNS]
    return outputs


def parse_uniprot_results(payload):
    values_by_id = {}
    seen_ids = set()
    for result in payload.get("results", []):
        protein_id = _clean_text(result.get("from"))
        if not protein_id:
            continue
        seen_ids.add(protein_id)
        target = result.get("to")
        sequence = (
            _clean_text((target.get("sequence") or {}).get("value"))
            if isinstance(target, dict)
            else ""
        )
        if sequence:
            values_by_id.setdefault(protein_id, set()).add(sequence)

    sequences = {
        protein_id: next(iter(values))
        for protein_id, values in values_by_id.items()
        if len(values) == 1
    }
    unresolved = {
        protein_id
        for protein_id in seen_ids
        if len(values_by_id.get(protein_id, set())) != 1
    }
    return sequences, unresolved


def fetch_reviewed_sequences(protein_ids, cache_path):
    requested = {_clean_text(protein_id) for protein_id in protein_ids} - {""}
    cache_path = Path(cache_path)
    cached_values = {}
    if cache_path.exists():
        with cache_path.open(newline="", encoding="utf-8") as cache_file:
            for row in csv.DictReader(cache_file):
                protein_id = _clean_text(row.get("Protein Id"))
                sequence = _clean_text(row.get("Sequence"))
                if protein_id and sequence:
                    cached_values.setdefault(protein_id, set()).add(sequence)

    cached = {
        protein_id: next(iter(values))
        for protein_id, values in cached_values.items()
        if len(values) == 1
    }
    ambiguous_cached = {
        protein_id
        for protein_id, values in cached_values.items()
        if len(values) != 1
    }
    needed = requested - cached.keys() - ambiguous_cached
    fetched = {}
    if needed:
        response = requests.post(
            UNIPROT_RUN,
            data={
                "from": "Ensembl",
                "to": "UniProtKB-Swiss-Prot",
                "ids": ",".join(sorted(needed)),
            },
            timeout=60,
        )
        response.raise_for_status()
        job_id = _clean_text(response.json().get("jobId"))
        if not job_id:
            raise RuntimeError("UniProt mapping response did not include a jobId")

        deadline = time.monotonic() + 10 * 60
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"UniProt mapping job {job_id} timed out")
            response = requests.get(
                UNIPROT_STATUS.format(job_id=job_id),
                timeout=min(60, remaining),
            )
            response.raise_for_status()
            status_payload = response.json()
            job_status = _clean_text(status_payload.get("jobStatus")).upper()
            if job_status in {"FAILED", "ERROR"}:
                raise RuntimeError(
                    f"UniProt mapping job {job_id} {job_status.lower()}"
                )
            if "results" in status_payload or "failedIds" in status_payload:
                break
            time.sleep(min(2, max(0, deadline - time.monotonic())))

        response = requests.get(
            UNIPROT_DETAILS.format(job_id=job_id), timeout=60
        )
        response.raise_for_status()
        redirect_url = _clean_text(response.json().get("redirectURL"))
        if not redirect_url:
            raise RuntimeError(
                f"UniProt mapping job {job_id} did not include a redirectURL"
            )
        parsed_url = urlsplit(redirect_url)
        query = [
            (key, value)
            for key, value in parse_qsl(
                parsed_url.query, keep_blank_values=True
            )
            if key not in {"format", "fields", "size"}
        ]
        query.extend(
            (("format", "json"), ("fields", "sequence"), ("size", "500"))
        )
        page_url = urlunsplit(parsed_url._replace(query=urlencode(query)))

        results = []
        while page_url:
            response = requests.get(page_url, timeout=60)
            response.raise_for_status()
            results.extend(response.json().get("results", []))
            page_url = response.links.get("next", {}).get("url")
        fetched, _ = parse_uniprot_results({"results": results})

    merged = {**cached, **fetched}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = cache_path.with_name(f"{cache_path.name}.part")
    part_path.unlink(missing_ok=True)
    try:
        with part_path.open("w", newline="", encoding="utf-8") as cache_file:
            writer = csv.DictWriter(
                cache_file, fieldnames=["Protein Id", "Sequence"]
            )
            writer.writeheader()
            writer.writerows(
                {"Protein Id": protein_id, "Sequence": merged[protein_id]}
                for protein_id in sorted(merged)
            )
        part_path.replace(cache_path)
    finally:
        part_path.unlink(missing_ok=True)

    sequences = {
        protein_id: merged[protein_id]
        for protein_id in requested
        if protein_id in merged
    }
    unresolved = requested - sequences.keys()
    return sequences, unresolved


def _write_csv_atomic(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part_path = path.with_name(f"{path.name}.part")
    part_path.unlink(missing_ok=True)
    try:
        frame.to_csv(part_path, index=False)
        part_path.replace(path)
    finally:
        part_path.unlink(missing_ok=True)


def _write_json_atomic(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part_path = path.with_name(f"{path.name}.part")
    part_path.unlink(missing_ok=True)
    try:
        part_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        part_path.replace(path)
    finally:
        part_path.unlink(missing_ok=True)


def build_occupied_rows(train, test):
    records = []
    for split, frame in (("train", train), ("test", test)):
        for position, source_row in enumerate(frame[SOURCE_ROW_ID], start=1):
            records.append(
                {
                    "split": split,
                    "source_position": position,
                    "source_line": position + 1,
                    "source_row": _source_row_identity(source_row),
                }
            )
    return pd.DataFrame(
        records,
        columns=["split", "source_position", "source_line", "source_row"],
    )


def publish_official_bundle(outputs, occupied_rows, output_dir, replace=os.replace):
    if set(outputs) != {"HQ_train", "HQ_test"}:
        raise ValueError(f"unexpected official HQ output keys: {sorted(outputs)}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    transaction_id = uuid.uuid4().hex
    staging_dir = output_dir / f".official-hq-staging-{transaction_id}"
    staging_dir.mkdir()
    frames = {**outputs, "occupied": occupied_rows}
    artifacts = []
    backups = {}
    published = []
    rollback_errors = []
    try:
        for key, filename in FORMAL_ARTIFACTS:
            staged_path = staging_dir / filename
            frames[key].to_csv(staged_path, index=False)
            artifacts.append((staged_path, output_dir / filename))

        try:
            for _staged_path, destination in artifacts:
                if destination.exists():
                    backup = (
                        output_dir
                        / f".{destination.name}.backup-{transaction_id}"
                    )
                    replace(destination, backup)
                    backups[destination] = backup

            for staged_path, destination in artifacts:
                replace(staged_path, destination)
                published.append(destination)
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
                    rollback_errors.append(rollback_error)
            if rollback_errors:
                details = "; ".join(str(error) for error in rollback_errors)
                raise RuntimeError(
                    f"official HQ publish failed ({publish_error}); "
                    f"rollback also failed: {details}"
                ) from publish_error
            raise

        for backup in backups.values():
            backup.unlink(missing_ok=True)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def _failure_record(split, position, row, stage, reason):
    return {
        "stage": stage,
        "tier": "HQ",
        "split": split,
        "source_line": position + 2,
        "source_row": _source_row_identity(row.get(SOURCE_ROW_ID)),
        "Protein Id": _clean_text(row.get("Protein Id")),
        "URL": _clean_text(row.get("URL")),
        "reason": reason,
    }


def _official_frames(train, test):
    return (("train", train), ("test", test))


def collect_split_overlap_failures(train, test):
    train_ids = _normalized_protein_ids(train)
    test_ids = _normalized_protein_ids(test)
    overlap = sorted((set(train_ids) - {""}) & (set(test_ids) - {""}))
    if not overlap:
        return [], []
    overlap_set = set(overlap)
    failures = []
    for split, frame in _official_frames(train, test):
        for position, row in enumerate(frame.to_dict("records")):
            protein_id = _clean_text(row.get("Protein Id"))
            if protein_id in overlap_set:
                failures.append(
                    _failure_record(
                        split,
                        position,
                        row,
                        "split_overlap",
                        "normalized Protein Id appears in official train and test",
                    )
                )
    return failures, overlap


def collect_download_field_failures(train, test):
    failures = []
    required_columns = [
        SOURCE_ROW_ID,
        "Protein Id",
        "Antibody Id",
        "URL",
        "locations",
        *DOWNLOAD_LABEL_COLUMNS,
    ]
    for split, frame in _official_frames(train, test):
        _require_columns(frame, required_columns, f"data_{split}.csv")
        for position, row in enumerate(frame.to_dict("records")):
            required_error = None
            for column in ["Protein Id", "locations", *DOWNLOAD_LABEL_COLUMNS]:
                if not _clean_text(row.get(column)):
                    required_error = f"blank required field {column}"
                    break
            if required_error is not None:
                failures.append(
                    _failure_record(
                        split,
                        position,
                        row,
                        "required_field",
                        required_error,
                    )
                )
                continue
            try:
                build_image_fields(row.get("Antibody Id"), row.get("URL"))
            except (TypeError, ValueError) as error:
                failures.append(
                    _failure_record(
                        split,
                        position,
                        row,
                        "image_fields",
                        str(error),
                    )
                )
    return failures


def collect_sequence_failures(train, test, sequences, unresolved):
    unresolved = {_clean_text(protein_id) for protein_id in unresolved}
    failures = []
    for split, frame in _official_frames(train, test):
        for position, row in enumerate(frame.to_dict("records")):
            protein_id = _clean_text(row.get("Protein Id"))
            sequence = _clean_text(sequences.get(protein_id))
            if protein_id in unresolved or not sequence:
                failures.append(
                    _failure_record(
                        split,
                        position,
                        row,
                        "sequence",
                        "Protein Id did not resolve to one unique nonblank reviewed sequence",
                    )
                )
    return failures


def _failure_result(output_dir, failures, message, error_type="ValueError"):
    failure_frame = pd.DataFrame(failures, columns=MANIFEST_FAILURE_COLUMNS)
    _write_csv_atomic(failure_frame, output_dir / "manifest_failures.csv")
    report = {
        "status": "error",
        "published": False,
        "record_id": ZENODO_RECORD_ID,
        "failure_rows": len(failure_frame),
        "error": {"type": error_type, "message": message},
    }
    _write_json_atomic(report, output_dir / "manifest_generation_report.json")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), file=sys.stderr)
    return 1


def main(
    argv=None,
    *,
    source_urls=None,
    source_md5=None,
    sequence_resolver=None,
):
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Publish official Vislocas HQ URL manifests"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=script_dir / "generated_quality"
    )
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    cache_dir = args.cache_dir or output_dir / "source"
    source_urls = SOURCE_URLS if source_urls is None else source_urls
    source_md5 = SOURCE_MD5 if source_md5 is None else source_md5
    sequence_resolver = (
        fetch_reviewed_sequences if sequence_resolver is None else sequence_resolver
    )
    try:
        source_report = prepare_verified_sources(
            cache_dir, source_urls=source_urls, source_md5=source_md5
        )
        _write_json_atomic(
            {"record_id": ZENODO_RECORD_ID, **source_report},
            output_dir / "source_validation_report.json",
        )
        source = pd.read_csv(cache_dir / "normalLabeled.csv")
        official_train = pd.read_csv(cache_dir / "data_train.csv")
        official_test = pd.read_csv(cache_dir / "data_test.csv")
        validate_official_rows(source, official_train, official_test)

        overlap_failures, overlap = collect_split_overlap_failures(
            official_train, official_test
        )
        if overlap_failures:
            return _failure_result(
                output_dir,
                overlap_failures,
                f"Protein Id overlap between official train and test: {overlap}",
            )

        field_failures = collect_download_field_failures(
            official_train, official_test
        )
        if field_failures:
            return _failure_result(
                output_dir,
                field_failures,
                f"{len(field_failures)} official HQ rows have invalid download fields",
            )

        official_ids = (
            set(_normalized_protein_ids(official_train))
            | set(_normalized_protein_ids(official_test))
        ) - {""}
        try:
            sequences, unresolved = sequence_resolver(
                official_ids, output_dir / "uniprot_sequences.csv"
            )
        except Exception as error:
            failures = [
                _failure_record(
                    split,
                    position,
                    row,
                    "sequence",
                    f"reviewed sequence resolution failed: {error}",
                )
                for split, frame in _official_frames(
                    official_train, official_test
                )
                for position, row in enumerate(frame.to_dict("records"))
            ]
            return _failure_result(
                output_dir,
                failures,
                f"reviewed sequence resolution failed: {error}",
                type(error).__name__,
            )
        sequence_failures = collect_sequence_failures(
            official_train, official_test, sequences, unresolved
        )
        if sequence_failures:
            return _failure_result(
                output_dir,
                sequence_failures,
                f"{len(sequence_failures)} official HQ rows lack a unique nonblank reviewed sequence",
            )
        outputs = assemble_official_hq(
            official_train, official_test, sequences
        )

        publish_official_bundle(
            outputs,
            build_occupied_rows(official_train, official_test),
            output_dir,
        )
        _write_csv_atomic(
            pd.DataFrame(columns=MANIFEST_FAILURE_COLUMNS),
            output_dir / "manifest_failures.csv",
        )
        report = {
            "status": "ok",
            "published": True,
            "record_id": ZENODO_RECORD_ID,
            "output_rows": {
                "HQ_train": len(outputs["HQ_train"]),
                "HQ_test": len(outputs["HQ_test"]),
            },
            "protein_id_overlap": 0,
        }
        _write_json_atomic(report, output_dir / "manifest_generation_report.json")
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as error:
        return _failure_result(
            output_dir, [], str(error), type(error).__name__
        )


if __name__ == "__main__":
    raise SystemExit(main())
