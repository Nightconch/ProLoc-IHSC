import argparse
import heapq
import json
import os
import random
import shutil
import sys
import uuid
from pathlib import Path

import pandas as pd

if __package__:
    from .official_hq_manifests import (
        DOWNLOAD_LABEL_COLUMNS,
        OUTPUT_COLUMNS,
        SOURCE_COLUMNS,
        SOURCE_MD5,
        SOURCE_ROW_ID,
        SOURCE_URLS,
        ZENODO_RECORD_ID,
        _clean_text,
        _require_columns,
        _source_row_identity,
        _write_csv_atomic,
        _write_json_atomic,
        assemble_official_hq,
        build_image_fields,
        collect_download_field_failures,
        collect_sequence_failures,
        collect_split_overlap_failures,
        fetch_reviewed_sequences,
        prepare_verified_sources,
        validate_official_rows,
    )
else:
    from official_hq_manifests import (
        DOWNLOAD_LABEL_COLUMNS,
        OUTPUT_COLUMNS,
        SOURCE_COLUMNS,
        SOURCE_MD5,
        SOURCE_ROW_ID,
        SOURCE_URLS,
        ZENODO_RECORD_ID,
        _clean_text,
        _require_columns,
        _source_row_identity,
        _write_csv_atomic,
        _write_json_atomic,
        assemble_official_hq,
        build_image_fields,
        collect_download_field_failures,
        collect_sequence_failures,
        collect_split_overlap_failures,
        fetch_reviewed_sequences,
        prepare_verified_sources,
        validate_official_rows,
    )


LABEL_COLUMNS = list(DOWNLOAD_LABEL_COLUMNS)
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
REQUIRED_OUTPUTS = {
    "HQ_train",
    "HQ_test",
    "MQ_train",
    "MQ_test",
    "LQ_train",
    "LQ_test",
}
MANIFEST_FILENAMES = {
    "HQ_train": "HQ_train_img_URL.csv",
    "HQ_test": "HQ_test_img_URL.csv",
    "MQ_train": "MQ_train_img_URL.csv",
    "MQ_test": "MQ_test_img_URL.csv",
    "LQ_train": "LQ_train_img_URL.csv",
    "LQ_test": "LQ_test_img_URL.csv",
}


class ManifestGenerationError(ValueError):
    def __init__(self, message, failures):
        super().__init__(message)
        self.failures = list(failures)


def split_quality_values(value):
    if not isinstance(value, str):
        raise ValueError("quality values must be strings")
    return [part.strip().lower() for part in value.split(";")]


def classify_quality(intensity, quantity):
    intensities = split_quality_values(intensity)
    quantities = split_quality_values(quantity)
    if len(intensities) != len(quantities):
        raise ValueError(
            "intensity and quantity must have the same number of values"
        )

    quality = None
    for pair in zip(intensities, quantities):
        if pair == ("strong", ">75%"):
            return "HQ"
        if pair in (("moderate", ">75%"), ("strong", "75%-25%")):
            quality = "MQ"
        elif quality is None and pair in (
            ("weak", ">75%"),
            ("moderate", "75%-25%"),
            ("weak", "75%-25%"),
        ):
            quality = "LQ"
    return quality


def _failure_record(row, stage, reason, *, tier="", split=""):
    return {
        "stage": stage,
        "tier": tier,
        "split": split,
        "source_line": int(row["__source_line"]),
        "source_row": _source_row_identity(row.get(SOURCE_ROW_ID)),
        "Protein Id": _clean_text(row.get("Protein Id")),
        "URL": _clean_text(row.get("URL")),
        "reason": reason,
    }


def prepare_supplemental_rows(source, official_row_ids):
    required = [
        SOURCE_ROW_ID,
        "Protein Id",
        "IF Verification",
        "locations",
        "Intensity Level",
        "Quantity",
        *LABEL_COLUMNS,
    ]
    _require_columns(source, required, "normalLabeled.csv")

    working = source.copy()
    working["__source_line"] = range(2, len(working) + 2)
    normalized_official_ids = {
        _source_row_identity(source_row) for source_row in official_row_ids
    }
    source_row_ids = working[SOURCE_ROW_ID].map(_source_row_identity)
    supplemental = working.loc[
        ~source_row_ids.isin(normalized_official_ids)
    ].copy()

    labels = supplemental[LABEL_COLUMNS].apply(pd.to_numeric, errors="coerce")
    eligible = supplemental.loc[
        supplemental["IF Verification"].eq("enhanced")
        & supplemental["locations"].notna()
        & supplemental["locations"].astype(str).str.strip().ne("")
        & labels.notna().all(axis=1)
        & labels.gt(0).any(axis=1)
    ].copy()

    failures = []
    keep_positions = []
    qualities = []
    invalid_quality_rows = 0
    for position, (_, row) in enumerate(eligible.iterrows()):
        try:
            quality = classify_quality(
                row["Intensity Level"], row["Quantity"]
            )
        except ValueError as error:
            invalid_quality_rows += 1
            failures.append(_failure_record(row, "quality", str(error)))
            continue
        if quality is not None:
            keep_positions.append(position)
            qualities.append(quality)

    classified = eligible.iloc[keep_positions].copy()
    classified["Quality"] = pd.Series(
        qualities, index=classified.index, dtype=object
    )
    classified["__protein_id"] = classified["Protein Id"].map(_clean_text)

    demoted = int(classified["Quality"].eq("HQ").sum())
    classified.loc[classified["Quality"].eq("HQ"), "Quality"] = "MQ"

    blank_mask = classified["__protein_id"].eq("")
    blank_rows = classified.loc[blank_mask]
    for _, row in blank_rows.iterrows():
        failures.append(
            _failure_record(
                row,
                "protein_id",
                "Protein Id is blank after trimming leading/trailing whitespace",
                tier=row["Quality"],
            )
        )
    classified = classified.loc[~blank_mask].copy()

    failures.sort(key=lambda failure: failure["source_line"])
    stats = {
        "source_rows": len(source),
        "supplemental_source_rows": len(supplemental),
        "eligible_rows": len(eligible),
        "classified_rows": len(classified),
        "invalid_quality_rows": invalid_quality_rows,
        "blank_protein_rows": len(blank_rows),
        "supplemental_hq_demoted": demoted,
    }
    return classified, failures, stats


def _normalized_protein_set(frame, source_name):
    _require_columns(frame, ["Protein Id"], source_name)
    return {
        protein_id
        for protein_id in frame["Protein Id"].map(_clean_text)
        if protein_id
    }


def _select_test_proteins(protein_labels, unknown, seed):
    if not unknown:
        return set()

    test_count = min(len(unknown), max(1, round(len(unknown) * 0.1)))
    target = protein_labels.loc[unknown].sum() * 0.1
    totals = pd.Series(0.0, index=protein_labels.columns)
    rng = random.Random(seed)
    tie_values = {protein_id: rng.random() for protein_id in unknown}
    buckets = {}
    vectors = {}
    for protein_id in unknown:
        signature = tuple(
            bool(value) for value in protein_labels.loc[protein_id]
        )
        vectors.setdefault(
            signature,
            pd.Series(signature, index=protein_labels.columns, dtype=float),
        )
        heapq.heappush(
            buckets.setdefault(signature, []),
            (tie_values[protein_id], protein_id),
        )

    selected = set()
    for _ in range(test_count):
        _, _, protein_id, signature = min(
            (
                float(((totals + vectors[signature] - target) ** 2).sum()),
                bucket[0][0],
                bucket[0][1],
                signature,
            )
            for signature, bucket in buckets.items()
            if bucket
        )
        heapq.heappop(buckets[signature])
        selected.add(protein_id)
        totals += vectors[signature]
    return selected


def assign_protein_splits(supplemental, official_train, official_test, seed):
    _require_columns(
        supplemental,
        ["__protein_id", *LABEL_COLUMNS],
        "supplemental rows",
    )
    official_train_ids = _normalized_protein_set(
        official_train, "data_train.csv"
    )
    official_test_ids = _normalized_protein_set(
        official_test, "data_test.csv"
    )
    overlap = sorted(official_train_ids & official_test_ids)
    if overlap:
        raise ValueError(
            f"official train and test Protein Id overlap: {overlap}"
        )

    supplemental_ids = set(supplemental["__protein_id"])
    known_train = supplemental_ids & official_train_ids
    known_test = supplemental_ids & official_test_ids
    unknown = sorted(
        supplemental_ids - official_train_ids - official_test_ids
    )

    mapping = {
        **{protein_id: "train" for protein_id in official_train_ids},
        **{protein_id: "test" for protein_id in official_test_ids},
    }
    if unknown:
        grouped = supplemental.loc[
            supplemental["__protein_id"].isin(unknown),
            ["__protein_id", *LABEL_COLUMNS],
        ].copy()
        grouped[LABEL_COLUMNS] = grouped[LABEL_COLUMNS].apply(
            pd.to_numeric, errors="coerce"
        )
        protein_labels = (
            grouped.groupby("__protein_id", sort=True)[LABEL_COLUMNS]
            .max()
            .gt(0)
        )
        selected_test = _select_test_proteins(
            protein_labels, unknown, seed
        )
        mapping.update(
            {
                protein_id: (
                    "test" if protein_id in selected_test else "train"
                )
                for protein_id in unknown
            }
        )
    else:
        selected_test = set()

    stats = {
        "known_train_proteins": len(known_train),
        "known_test_proteins": len(known_test),
        "unknown_proteins": len(unknown),
        "unknown_test_proteins": len(selected_test),
    }
    return mapping, stats


def _manifest_rows(frame, sequences, source_name):
    _require_columns(frame, SOURCE_COLUMNS, source_name)
    result = frame.copy()
    protein_ids = result["Protein Id"].map(_clean_text)
    missing_sequences = sorted(
        {
            protein_id or "<blank>"
            for protein_id in protein_ids
            if not protein_id or not _clean_text(sequences.get(protein_id))
        }
    )
    if missing_sequences:
        raise ValueError(
            "missing reviewed sequences for: "
            + ", ".join(missing_sequences)
        )

    result["Sequence"] = protein_ids.map(sequences)
    image_fields = [
        build_image_fields(antibody_id, relative_url)
        for antibody_id, relative_url in result[
            ["Antibody Id", "URL"]
        ].itertuples(index=False, name=None)
    ]
    result[["NonZeroDigits", "Image Name", "Modified URL"]] = pd.DataFrame(
        image_fields,
        columns=["NonZeroDigits", "Image Name", "Modified URL"],
        index=result.index,
    )
    return result.loc[:, OUTPUT_COLUMNS]


def assert_protein_disjoint(outputs):
    if set(outputs) != REQUIRED_OUTPUTS:
        raise ValueError(f"unexpected output keys: {sorted(outputs)}")
    train_proteins = set().union(
        *(
            _normalized_protein_set(frame, name)
            for name, frame in outputs.items()
            if name.endswith("_train")
        )
    )
    test_proteins = set().union(
        *(
            _normalized_protein_set(frame, name)
            for name, frame in outputs.items()
            if name.endswith("_test")
        )
    )
    overlap = sorted(train_proteins & test_proteins)
    if overlap:
        raise AssertionError(
            f"Protein Id overlap between train and test: {overlap}"
        )


def assemble_quality_outputs(
    official_train,
    official_test,
    supplemental,
    split_mapping,
    sequences,
):
    _require_columns(
        supplemental,
        ["Quality", "__protein_id"],
        "supplemental rows",
    )
    missing_splits = sorted(
        set(supplemental["__protein_id"]) - set(split_mapping)
    )
    if missing_splits:
        raise ValueError(
            f"missing split assignments for: {', '.join(missing_splits)}"
        )
    row_splits = supplemental["__protein_id"].map(split_mapping)
    invalid_splits = sorted(set(row_splits) - {"train", "test"})
    if invalid_splits:
        raise ValueError(f"invalid split assignments: {invalid_splits}")

    outputs = assemble_official_hq(
        official_train, official_test, sequences
    )
    for tier in ("MQ", "LQ"):
        for split in ("train", "test"):
            subset = supplemental.loc[
                supplemental["Quality"].eq(tier) & row_splits.eq(split)
            ]
            outputs[f"{tier}_{split}"] = _manifest_rows(
                subset, sequences, f"{tier}_{split} supplemental rows"
            )
    assert_protein_disjoint(outputs)
    return outputs


def publish_quality_bundle(outputs, output_dir, replace=os.replace):
    if set(outputs) != REQUIRED_OUTPUTS:
        raise ValueError(f"unexpected output keys: {sorted(outputs)}")
    invalid_schemas = sorted(
        name
        for name, frame in outputs.items()
        if frame.columns.tolist() != OUTPUT_COLUMNS
    )
    if invalid_schemas:
        raise AssertionError(
            f"generated output schema mismatch: {invalid_schemas}"
        )
    assert_protein_disjoint(outputs)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    transaction_id = uuid.uuid4().hex
    staging_dir = output_dir / f".quality-manifest-staging-{transaction_id}"
    staging_dir.mkdir()
    artifacts = []
    backups = {}
    published = []
    rollback_errors = []
    retained_backups = set()
    publish_succeeded = False
    try:
        for name, filename in MANIFEST_FILENAMES.items():
            staged = staging_dir / filename
            outputs[name].to_csv(staged, index=False)
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
                    if backup.exists():
                        retained_backups.add(backup)
                    rollback_errors.append(rollback_error)
            if rollback_errors:
                details = "; ".join(map(str, rollback_errors))
                raise RuntimeError(
                    f"quality manifest publish failed ({publish_error}); "
                    f"rollback also failed: {details}"
                ) from publish_error
            raise

    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
        for backup in backups.values():
            if publish_succeeded or backup not in retained_backups:
                backup.unlink(missing_ok=True)


def _sorted_failures(failures):
    return sorted(
        failures,
        key=lambda failure: (
            int(failure.get("source_line") or 0),
            str(failure.get("stage") or ""),
            str(failure.get("Protein Id") or ""),
        ),
    )


def write_manifest_failures(failures, output_dir):
    frame = pd.DataFrame(
        _sorted_failures(failures), columns=MANIFEST_FAILURE_COLUMNS
    )
    _write_csv_atomic(frame, Path(output_dir) / "manifest_failures.csv")
    return frame


def _supplemental_sequence_failures(
    supplemental, split_mapping, sequences, unresolved
):
    unresolved = {_clean_text(protein_id) for protein_id in unresolved}
    sequence_values = {
        _clean_text(protein_id): _clean_text(sequence)
        for protein_id, sequence in sequences.items()
    }
    failed_ids = {
        protein_id
        for protein_id in set(supplemental["__protein_id"])
        if protein_id in unresolved or not sequence_values.get(protein_id)
    }
    failures = [
        _failure_record(
            row,
            "sequence",
            "Protein Id did not resolve to one unique nonblank reviewed sequence",
            tier=row["Quality"],
            split=split_mapping[row["__protein_id"]],
        )
        for _, row in supplemental.loc[
            supplemental["__protein_id"].isin(failed_ids)
        ].iterrows()
    ]
    return (
        supplemental.loc[
            ~supplemental["__protein_id"].isin(failed_ids)
        ].copy(),
        failures,
    )


def _supplemental_image_field_failures(supplemental, split_mapping):
    keep = []
    failures = []
    for _, row in supplemental.iterrows():
        try:
            build_image_fields(row.get("Antibody Id"), row.get("URL"))
        except (TypeError, ValueError) as error:
            keep.append(False)
            failures.append(
                _failure_record(
                    row,
                    "image_fields",
                    str(error),
                    tier=row["Quality"],
                    split=split_mapping[row["__protein_id"]],
                )
            )
        else:
            keep.append(True)
    return supplemental.loc[keep].copy(), failures


def _raise_official_failures(failures, message):
    if failures:
        raise ManifestGenerationError(message, failures)


def generate_quality_manifests(
    output_dir,
    cache_dir,
    seed,
    source_urls=None,
    source_md5=None,
    sequence_resolver=None,
):
    output_dir = Path(output_dir)
    cache_dir = Path(cache_dir)
    source_urls = SOURCE_URLS if source_urls is None else source_urls
    source_md5 = SOURCE_MD5 if source_md5 is None else source_md5
    sequence_resolver = (
        fetch_reviewed_sequences
        if sequence_resolver is None
        else sequence_resolver
    )

    source_validation = prepare_verified_sources(
        cache_dir, source_urls=source_urls, source_md5=source_md5
    )
    source = pd.read_csv(cache_dir / "normalLabeled.csv")
    official_train = pd.read_csv(cache_dir / "data_train.csv")
    official_test = pd.read_csv(cache_dir / "data_test.csv")
    official_row_ids = validate_official_rows(
        source, official_train, official_test
    )

    overlap_failures, overlap = collect_split_overlap_failures(
        official_train, official_test
    )
    _raise_official_failures(
        overlap_failures,
        f"Protein Id overlap between official train and test: {overlap}",
    )
    field_failures = collect_download_field_failures(
        official_train, official_test
    )
    _raise_official_failures(
        field_failures,
        f"{len(field_failures)} official HQ rows have invalid download fields",
    )

    supplemental, failures, preparation_stats = prepare_supplemental_rows(
        source, official_row_ids
    )
    split_mapping, split_stats = assign_protein_splits(
        supplemental, official_train, official_test, seed
    )
    official_ids = _normalized_protein_set(
        official_train, "data_train.csv"
    ) | _normalized_protein_set(official_test, "data_test.csv")
    requested_ids = official_ids | set(supplemental["__protein_id"])
    sequences, unresolved = sequence_resolver(
        requested_ids, output_dir / "uniprot_sequences.csv"
    )

    official_sequence_failures = collect_sequence_failures(
        official_train, official_test, sequences, unresolved
    )
    supplemental, sequence_failures = _supplemental_sequence_failures(
        supplemental, split_mapping, sequences, unresolved
    )
    failures.extend(sequence_failures)
    supplemental, image_failures = _supplemental_image_field_failures(
        supplemental, split_mapping
    )
    failures.extend(image_failures)
    _raise_official_failures(
        [*failures, *official_sequence_failures]
        if official_sequence_failures
        else [],
        f"{len(official_sequence_failures)} official HQ rows lack a unique "
        "nonblank reviewed sequence",
    )

    outputs = assemble_quality_outputs(
        official_train,
        official_test,
        supplemental,
        split_mapping,
        sequences,
    )
    assert_protein_disjoint(outputs)
    failure_frame = pd.DataFrame(
        _sorted_failures(failures), columns=MANIFEST_FAILURE_COLUMNS
    )
    report = {
        "status": "ok",
        "published": False,
        "record_id": ZENODO_RECORD_ID,
        "seed": seed,
        "source_validation": source_validation,
        "preparation": preparation_stats,
        "split": split_stats,
        "output_rows": {
            name: len(outputs[name]) for name in MANIFEST_FILENAMES
        },
        "failure_rows": len(failure_frame),
        "protein_id_overlap": 0,
    }
    return outputs, report, failure_frame


def _error_report(error, seed, failure_rows):
    return {
        "status": "error",
        "published": False,
        "record_id": ZENODO_RECORD_ID,
        "seed": seed,
        "failure_rows": failure_rows,
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
    }


def main(
    argv=None,
    *,
    source_urls=None,
    source_md5=None,
    sequence_resolver=None,
):
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Publish official HQ and supplemental MQ/LQ URL manifests"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=script_dir / "generated_quality"
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    cache_dir = args.cache_dir or output_dir / "source"
    try:
        outputs, report, failures = generate_quality_manifests(
            output_dir,
            cache_dir,
            args.seed,
            source_urls=source_urls,
            source_md5=source_md5,
            sequence_resolver=sequence_resolver,
        )
        publish_quality_bundle(outputs, output_dir)
        report["published"] = True
        _write_csv_atomic(
            failures, output_dir / "manifest_failures.csv"
        )
        _write_json_atomic(
            report, output_dir / "manifest_generation_report.json"
        )
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as error:
        failure_rows = (
            error.failures
            if isinstance(error, ManifestGenerationError)
            else []
        )
        failure_frame = write_manifest_failures(failure_rows, output_dir)
        report = _error_report(error, args.seed, len(failure_frame))
        _write_json_atomic(
            report, output_dir / "manifest_generation_report.json"
        )
        print(
            json.dumps(report, ensure_ascii=False, sort_keys=True),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
